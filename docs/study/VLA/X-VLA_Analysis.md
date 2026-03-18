# X-VLA 분석 가이드

> 작성일: 2026-03-18
> 논문: X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VLA Model (arXiv:2510.10274, ICLR 2026)
> 코드: https://github.com/2toinf/X-VLA
> LeRobot 통합: 네이티브 지원

---

## 1. X-VLA 개요

X-VLA는 **Soft Prompt**를 이용해 cross-embodiment 문제를 해결하는 0.9B VLA 모델.
ICLR 2026에 accept되었으며, 6개 시뮬레이션 + 3개 실제 로봇에서 SOTA 달성.

**핵심 아이디어:**
- 로봇마다 다른 하드웨어/카메라/action space → "heterogeneity 문제"
- 기존 방법: action head를 로봇마다 분리 (GR00T의 embodiment tag 등)
- X-VLA: **Soft Prompt** — 로봇별 learnable embedding을 모델 입력에 추가
- 최소한의 파라미터 추가만으로 다양한 로봇에 적응 가능

**핵심 특징:**
- 0.9B 파라미터
- Flow matching 기반 action 생성 (SmolVLA, GR00T과 동일 계열)
- Self-attention only (CA+SA interleave 아님, standard Transformer encoder)
- 290K episodes로 사전학습 (7개 하드웨어, 5종 로봇)
- LoRA로 1% 파라미터(9M)만 튜닝해도 LIBERO 93% 달성

---

## 2. 아키텍처 상세

```
┌──────────────────────────────────────────────────────────┐
│                         X-VLA                             │
│                                                          │
│  ┌─────────────────────────────────────────┐             │
│  │ High-dim Stream (관찰)                    │             │
│  │                                         │             │
│  │  Fixed Camera + Language                │             │
│  │  → Florence-Large VLM (pretrained)      │             │
│  │                                         │             │
│  │  Wrist Camera (보조)                     │             │
│  │  → Shared ViT (별도 인코딩)              │             │
│  └──────────────┬──────────────────────────┘             │
│                 │                                        │
│  ┌──────────────▼──────────────────────────┐             │
│  │ Low-dim Stream (행동)                     │             │
│  │                                         │             │
│  │  Proprioception (R_t) + Noisy Action (A_t)            │
│  │  + Time embedding (T)                   │             │
│  │  → concat → Linear projection           │             │
│  └──────────────┬──────────────────────────┘             │
│                 │                                        │
│  ┌──────────────▼──────────────────────────┐             │
│  │ ★ Soft Prompt Library                    │             │
│  │                                         │             │
│  │  Dataset/Embodiment ID로 쿼리:           │             │
│  │  AGIBOT → [learnable tokens A]          │             │
│  │  Franka → [learnable tokens B]          │             │
│  │  UR5    → [learnable tokens C]          │             │
│  │  새로봇  → [learnable tokens NEW]         │             │
│  └──────────────┬──────────────────────────┘             │
│                 │                                        │
│  ┌──────────────▼──────────────────────────┐             │
│  │ Transformer Encoder (x N layers)         │             │
│  │ Standard Self-Attention only             │             │
│  │                                         │             │
│  │  입력: [VLM features | Soft Prompt |     │             │
│  │         Proprio+Action+Time tokens]      │             │
│  │                                         │             │
│  │  학습: Flow Matching loss                │             │
│  └──────────────┬──────────────────────────┘             │
│                 │                                        │
│                 ▼                                        │
│  Action Chunk (EEF pose: xyz + Rotate6D + gripper)       │
└──────────────────────────────────────────────────────────┘
```

### 2.1 Soft Prompt가 해결하는 문제

기존 cross-embodiment 접근법과 비교:

| 방법 | 설명 | 한계 |
|------|------|------|
| (a) Action Head 분리 | 로봇별 다른 출력 head | action space만 처리, 카메라/도메인 차이 무시 |
| (b) HPT-style Projection | 입력에 로봇별 projection layer | VLM 표현 손상, 학습 불안정 |
| (c) Language Prompt | "이 로봇은 Franka이고..." 텍스트 설명 | 수작업 필요, 확장성 낮음 |
| **(d) Soft Prompt (X-VLA)** | **로봇별 learnable embedding** | **자동 학습, 안정적, 확장 가능** |

### 2.2 카메라 인코딩 전략 (GR00T, SmolVLA와 다른 점)

X-VLA는 카메라를 **역할별로 분리 인코딩**:
- **Fixed camera + Language** → Florence-Large VLM (high-level reasoning용)
- **Wrist camera** → Shared ViT (fine-grained manipulation용, 별도)

vs SmolVLA: 모든 카메라를 동일하게 SigLIP으로 인코딩
vs GR00T: Cosmos-Reason VLM에 모든 카메라 입력

### 2.3 Action 표현

X-VLA는 **EEF (End-Effector) pose**를 표준 action으로 사용:
- xyz position (3D)
- Rotate6D rotation (6D) — Euler/quaternion의 불연속성 문제 회피
- Binary gripper state

Loss: MSE (position, rotation) + BCE (gripper)

---

## 3. 학습 파이프라인

### Phase I: Pretraining

```
데이터: 290K episodes, 7 hardware configs, 5 robot types
  - AGIBOT-Beta (48.8%)
  - Droid-Franka (31.6%)
  - RoboMind-Franka/UR5/Agilex (19.6%)

학습:
  - backbone (π_θ) + Soft Prompts (P_H) 공동 최적화
  - Flow matching objective
  - 로봇별 Soft Prompt가 hardware configuration을 자동 학습
  - Custom LR: VLM/Soft Prompt에 낮은 학습률 → pretrained 표현 보존
```

### Phase II: Domain Adaptation (새 로봇 적응)

```
Step 1: Prompt Warm-up
  - 새 로봇용 learnable prompt p_new 생성 (랜덤 초기화)
  - backbone frozen, prompt만 학습
  - pretrained features를 활용하도록 prompt를 먼저 정렬

Step 2: Joint Policy Adaptation
  - backbone + warmed-up prompt 함께 fine-tune
  - 또는 LoRA로 1% 파라미터만 튜닝 (9M params)
```

**LoRA 적응 결과:**
- 1% 파라미터(9M)만 튜닝
- LIBERO: 93% 성공률
- Simpler-WidowX: 54%
- π0 (3B 파라미터 튜닝) 대비 300배 적은 파라미터로 비슷한 성능

---

## 4. GR00T, SmolVLA와 비교

| | X-VLA | SmolVLA | GR00T N1.6 |
|---|-------|---------|------------|
| 파라미터 | 0.9B | 0.45B | 3B |
| VLM | Florence-Large | SmolVLM-2 (500M) | Cosmos-Reason-2B |
| Action Head | Transformer Encoder (SA only) | Flow Matching Transformer (CA+SA) | DiT (32 layers) |
| Cross-embodiment | ★ Soft Prompt | 카메라 rename | Embodiment Tag |
| Action 표현 | EEF pose (Rotate6D) | Continuous (joint) | State-relative |
| 학습 데이터 | 290K episodes (7 configs) | 22.9K episodes (SO100) | 10K+ hours (diverse) |
| 추론 VRAM | ~4GB (추정) | ~2GB | ~8GB+ |
| 적응 방법 | LoRA (1% params) | Full fine-tune | Full fine-tune (48GB+) |
| 데이터 포맷 | LeRobot (네이티브) | LeRobot | LeRobot v2 (GR00T flavor) |

### 핵심 차이점

**X-VLA vs SmolVLA:**
- X-VLA는 cross-embodiment에 특화 (Soft Prompt), SmolVLA는 경량/효율에 특화
- X-VLA는 SA only, SmolVLA는 CA+SA interleave
- X-VLA는 EEF pose 표준화, SmolVLA는 joint-level action

**X-VLA vs GR00T:**
- X-VLA는 Soft Prompt로 적응, GR00T은 Embodiment Tag + full fine-tune
- X-VLA는 LoRA로 9M만 튜닝 가능, GR00T은 48GB+ 필요
- X-VLA는 카메라를 역할별 분리 인코딩, GR00T은 통합 인코딩

---

## 5. 코드 구조 (GitHub)

```
X-VLA/
├── xvla/
│   ├── model/
│   │   ├── xvla_model.py          # ★ X-VLA 모델 정의
│   │   ├── soft_prompt.py         # ★ Soft Prompt 구현
│   │   ├── florence_encoder.py    # Florence VLM 인코더
│   │   └── flow_matching.py       # Flow matching 학습/추론
│   │
│   ├── data/
│   │   ├── dataset.py             # 데이터 로딩
│   │   └── preprocessing.py       # Action alignment, temporal downsampling
│   │
│   └── configs/
│       └── *.yaml                 # 학습/추론 설정
│
├── scripts/
│   ├── train.py                   # 학습 스크립트
│   ├── eval.py                    # 평가 스크립트
│   └── deploy.py                  # 배포 스크립트
│
└── examples/                      # 벤치마크별 설정
```

### LeRobot 통합

X-VLA는 LeRobot에 네이티브 통합되어 있어서 LeRobot CLI로도 사용 가능:

```bash
# LeRobot에서 X-VLA 학습
python -m lerobot.scripts.lerobot_train \
  --policy.type=xvla \
  --dataset.repo_id=<DATASET> \
  ...
```

---

## 6. 실행 방법

### 6.1 Server-Client 구조 (독립 레포 사용 시)

```bash
# Server (GPU)
python scripts/server.py \
  --model-path 2toINF/X-VLA-Pt \
  --device cuda:0

# Client (로봇)
python scripts/client.py \
  --server-address localhost:5555
```

### 6.2 LeRobot 통합 사용 시

```bash
# Fine-tuning
python -m lerobot.scripts.lerobot_train \
  --policy.type=xvla \
  --policy.path=2toINF/X-VLA-Pt \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --output_dir=outputs/xvla_train

# LoRA fine-tuning (추정)
python -m lerobot.scripts.lerobot_train \
  --policy.type=xvla \
  --policy.path=2toINF/X-VLA-Pt \
  --peft.method_type=lora \
  --peft.r=16 \
  ...
```

---

## 7. 벤치마크 성능

### 7.1 시뮬레이션

| Benchmark | X-VLA-0.9B | π0 (3.3B) | OpenVLA (7B) | SmolVLA (0.45B) |
|-----------|-----------|-----------|-------------|----------------|
| LIBERO (avg) | **95.8%** | 86.0% | 76.5% | 87.3% |
| Simpler-WidowX | **89.6%** | - | - | - |
| MetaWorld | ✅ tested | - | - | 57.3% |
| CALVIN | ✅ tested | - | - | - |
| VLABench | ✅ tested | - | - | - |
| RoboTwin2 | ✅ tested | - | - | - |

### 7.2 LoRA 적응 (1% params only)

| Benchmark | Full Fine-tune | LoRA (9M params) |
|-----------|---------------|-----------------|
| LIBERO | 95.8% | 93% |
| Simpler-WidowX | 89.6% | 54% |

→ 9M 파라미터만 튜닝해도 대부분 성능 유지

---

## 8. 연구 관점에서 주목할 점

### 8.1 Soft Prompt의 가능성

Soft Prompt는 NLP에서 검증된 기법이지만 로봇에 적용한 건 X-VLA가 처음:
- 로봇별 하드웨어 정보를 자동 학습
- 새 로봇 추가 시 prompt만 추가하면 됨 (backbone 재학습 불필요)
- prompt 간 유사도 분석으로 로봇 간 관계 파악 가능

### 8.2 Intention Abstraction (Temporal Downsampling)

논문의 독특한 전처리:
- 원래 action trajectory를 시간축으로 다운샘플링
- 30개 anchor point로 4초 trajectory 요약
- "저수준 노이즈 제거 + 고수준 의도 학습" 효과
- 이 전처리가 성능에 큰 영향 (Tab. 1에서 validation error 0.11 → 0.077)

### 8.3 카메라 분리 인코딩

Fixed camera와 wrist camera를 분리하는 설계:
- Fixed camera: 안정적 장면 이해 (high-level reasoning)
- Wrist camera: 빠르게 변하는 manipulation cue (fine-grained)
- 이 분리가 성능 향상에 기여 (encoding pipeline 개선으로 error 0.071 → 0.053)

---

## 9. 스터디 체크리스트

- [ ] X-VLA 레포 fork & clone
- [ ] 논문 Figure 2 (heterogeneity 해결 방법 비교) 이해
- [ ] Soft Prompt 코드 확인 (soft_prompt.py)
- [ ] Florence-Large VLM 구조 이해
- [ ] Flow matching 구현 비교 (SmolVLA vs X-VLA)
- [ ] LeRobot 통합 사용법 테스트
- [ ] LoRA fine-tuning 실험
- [ ] SmolVLA와 같은 데이터셋으로 성능 비교
- [ ] Soft Prompt embedding 시각화/분석
