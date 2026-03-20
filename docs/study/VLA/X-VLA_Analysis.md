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

## 5. 코드 구조 (실제 확인 기준, 2026-03-19)

X-VLA는 두 가지 형태로 존재하며, **역할이 다르므로 둘 다 사용**:

### 5.1 독립 레포 (`~/X-VLA/`) — 시뮬레이션 eval / LoRA / 배포용

```
X-VLA/
├── models/
│   ├── modeling_xvla.py       # ★ FastAPI 서버 기반 XVLA 클래스 (배포용)
│   ├── modeling_florence2.py  # Florence2 VLM
│   ├── transformer.py         # SoftPromptedTransformer
│   ├── action_hub.py          # Action space 정의
│   ├── configuration_xvla.py
│   └── configuration_florence2.py
│
├── datasets/
│   ├── dataset.py             # 데이터 로딩
│   ├── domain_config.py       # 도메인별 설정
│   └── domain_handler/        # Droid, RoboMind, AGIBOT, LeRobot 등 핸들러
│
├── evaluation/                # ★ 시뮬레이션 벤치마크 eval (여기만 있음)
│   ├── libero/                # LIBERO eval 클라이언트
│   ├── calvin/                # CALVIN eval 클라이언트
│   ├── simpler/               # SimplerEnv
│   ├── vlabench/
│   ├── robotwin-2.0/
│   └── SoftFold-Agilex/
│
├── train.py                   # 독자 학습 스크립트
├── peft_train.py              # ★ LoRA fine-tuning 스크립트
├── deploy.py                  # 실제 로봇 배포 (FastAPI server-client)
├── environment.yml            # 독립 conda 환경
└── requirements.txt
```

**핵심:** `modeling_xvla.py`가 `FastAPI` 기반 서버로 구현됨 → 실제 로봇과 분리된 server-client 구조.

### 5.2 LeRobot 통합 (`~/lerobot/src/lerobot/policies/xvla/`) — 학습/추론 파이프라인용

```
lerobot/src/lerobot/policies/xvla/
├── modeling_xvla.py       # ★ PreTrainedPolicy 기반 (lerobot 표준 인터페이스)
├── soft_transformer.py    # ★ Soft Prompt + Transformer 구현
├── modeling_florence2.py  # Florence2 VLM
├── action_hub.py          # Action space 정의
├── processor_xvla.py      # 데이터 전처리
├── configuration_xvla.py
├── configuration_florence2.py
└── utils.py
```

**핵심:** SmolVLA와 동일한 `PreTrainedPolicy` 인터페이스 → lerobot CLI로 바로 사용 가능.

### 5.3 두 레포 역할 분담

| 목적 | 어디서 |
|------|--------|
| 코드 구조 분석 / SmolVLA와 비교 | lerobot xvla |
| lerobot 데이터셋으로 fine-tuning | lerobot xvla |
| LIBERO / CALVIN 시뮬레이션 eval | 독립 X-VLA 레포 (`evaluation/`) |
| LoRA fine-tuning | 독립 X-VLA 레포 (`peft_train.py`) |
| 실제 로봇 배포 | 독립 X-VLA 레포 (`deploy.py`) |

---

## 6. 실행 방법

### 6.1 LeRobot 통합 사용 시 (fine-tuning)

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=xvla \
  --policy.path=2toINF/X-VLA-Pt \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --output_dir=outputs/xvla_train
```

### 6.2 독립 레포 — LoRA fine-tuning

```bash
# peft_train.py 사용
python peft_train.py \
  --model-path 2toINF/X-VLA-Pt \
  ...
```

### 6.3 독립 레포 — 시뮬레이션 eval (LIBERO 예시)

```bash
# evaluation/libero/ 참고
python evaluation/libero/libero_client.py \
  --model-path 2toINF/X-VLA-Pt \
  ...
```

### 6.4 독립 레포 — 실제 로봇 배포 (FastAPI server-client)

```bash
# Server (GPU 머신)
python deploy.py --model-path 2toINF/X-VLA-Pt --device cuda:0

# Client (로봇 측)
# HTTP로 서버와 통신
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

## 9. `soft_transformer.py` 코드 분석 (lerobot 버전)

### 9.1 클래스 구조

```
SoftPromptedTransformer
├── DomainAwareLinear (action_encoder)   — 도메인별 action 임베딩
├── DomainAwareLinear (action_decoder)   — 도메인별 action 디코딩
├── vlm_proj          (Linear 또는 DomainAwareLinear) — VLM feature 투영
├── aux_visual_proj   (Linear 또는 DomainAwareLinear) — wrist camera 투영
├── soft_prompt_hub   (nn.Embedding)     — ★ 도메인별 learnable soft prompt
├── TransformerBlock x N                — standard SA only (bidirectional)
└── pos_emb           (nn.Parameter)    — learnable positional embedding
```

### 9.2 핵심 1 — `DomainAwareLinear`

cross-embodiment의 실제 구현. 일반 `nn.Linear` 대신 `nn.Embedding`으로 **로봇마다 별도 weight matrix**를 저장:

```python
class DomainAwareLinear(nn.Module):
    def __init__(self, input_size, output_size, num_domains=20):
        self.fc   = nn.Embedding(num_domains, output_size * input_size)  # 도메인별 weight
        self.bias = nn.Embedding(num_domains, output_size)               # 도메인별 bias
```

`forward` 시 `domain_id`로 해당 로봇의 weight를 꺼내서 matmul.
→ action space가 로봇마다 달라도 각자 맞는 weight로 처리 가능.

`action_encoder` / `action_decoder` 둘 다 `DomainAwareLinear` → 입력/출력 모두 도메인별로 적응.

### 9.3 핵심 2 — `soft_prompt_hub`

```python
self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
```

`domain_id`로 로봇별 learnable token 묶음을 꺼내서 시퀀스 **끝에 append**:

```
[action tokens | VLM tokens | wrist tokens] + pos_emb
→ append → [... | soft prompt tokens]
→ Transformer blocks (bidirectional SA)
→ 앞 num_actions 개만 decode → action 출력
```

Soft prompt가 끝에 붙어도 bidirectional attention이라 앞의 action 토큰들이 soft prompt를 attend 가능.
→ "이 로봇은 이런 하드웨어 특성이 있다"는 도메인 정보를 action 토큰이 참조하는 구조.

### 9.4 `forward()` 전체 흐름

```python
# 1. action + proprio + timestep → 하나의 토큰으로 합침
time_emb      = sinusoidal_embedding(t)                          # [B, dim_time]
time_tokens   = time_emb.expand(B, num_actions, ...)             # 모든 action에 동일 timestep
action_tokens = concat([noisy_action, proprio, time_tokens])     # [B, N_act, dim_a+dim_p+dim_t]
x = action_encoder(action_tokens, domain_id)                     # DomainAwareLinear → [B, N_act, H]

# 2. 시각 정보 concat
x = concat([x, vlm_proj(vlm_features), aux_visual_proj(wrist_features)])  # [B, N_total, H]

# 3. 위치 임베딩
x = x + pos_emb[:, :seq_len]

# 4. soft prompt append
soft_prompts = soft_prompt_hub(domain_id).view(B, len_soft_prompts, H)
x = concat([x, soft_prompts])                                    # [B, N_total + len_prompts, H]

# 5. Transformer (SA only, bidirectional, causal mask 없음)
for block in blocks:
    x = block(x)

# 6. 앞 num_actions 개만 decode
return action_decoder(norm(x[:, :num_actions]), domain_id)
```

### 9.5 SmolVLA와 핵심 구조 비교

| | SmolVLA | X-VLA |
|---|---------|-------|
| Attention | CA+SA interleaved | SA only (bidirectional) |
| 도메인 적응 | rename_map (수동) | DomainAwareLinear + Soft Prompt |
| VLM features 참조 방식 | Cross-Attention | 시퀀스에 concat 후 SA |
| Action 토큰 위치 | suffix (VLM 뒤) | prefix (시퀀스 앞) |
| Soft prompt 위치 | 없음 | 시퀀스 끝에 append |
| 시간 정보 주입 | sinusoidal → MLP fusion | sinusoidal → action 토큰에 concat |
| Action 디코딩 | `nn.Linear` (action_out_proj) | `DomainAwareLinear` |
| Causal mask | ✅ (action chunk 내부) | ❌ (완전 bidirectional) |

---

## 10. `modeling_xvla.py` 코드 분석

### 10.1 클래스 구조

```
XVLAPolicy (PreTrainedPolicy — lerobot 인터페이스)
  └── XVLAModel
        ├── vlm (Florence2, encoder-only)
        │    ├── vision_tower          (이미지 인코딩)
        │    └── language_model.encoder (텍스트+이미지 fusion)
        │         ※ decoder, lm_head는 __init__에서 삭제됨
        └── transformer (SoftPromptedTransformer)
```

### 10.2 Florence-2를 encoder-only로 사용

```python
del lm.model.decoder
del lm.lm_head
```

Florence-2는 원래 encoder-decoder 구조지만 **decoder와 lm_head를 삭제**하고 encoder만 사용.
텍스트 생성이 목적이 아니라 이미지+텍스트의 fused feature를 뽑는 게 목적이라서.

### 10.3 `forward_vlm()` — 카메라 역할 분리

```python
# view[0] → Florence-2 encoder에서 텍스트와 merge → vlm_features
merged_embeds = vlm._merge_input_ids_with_image_features(image_features[:, 0], text_embeds)
enc_out = vlm.language_model.model.encoder(merged_embeds)   # vlm_features

# view[1:] → 그대로 aux_visual_inputs (wrist camera 등)
aux_visual_inputs = image_features[:, 1:].reshape(B, -1, hidden_dim)
```

- **view[0]** (fixed cam): Florence-2 encoder에서 텍스트와 통합 → high-level reasoning
- **view[1:]** (wrist cam): VLM 없이 바로 Transformer에 전달 → fine-grained manipulation

### 10.4 Flow Matching — SmolVLA와 구현 방식이 다름

**학습 시 (`forward`):**

```python
# Stratified timestep 샘플링 (배치 내 균등 커버리지 보장)
t = (rand(1) + arange(batch_size) / batch_size) % (1 - 1e-5)

# noisy action (SmolVLA와 동일한 interpolation)
action_noisy = noise * t + action * (1 - t)

# 예측 target: clean action 직접 예측
pred_action = transformer(action_noisy, t, ...)
loss = compute_loss(pred_action, action)   # GT clean action과 비교
```

**추론 시 (`generate_actions`):**

```python
x1 = randn(...)    # 순수 노이즈
action = zeros(...)

for i in range(steps, 0, -1):    # t: 1.0 → 1/steps
    t = i / steps
    x_t = x1 * t + action * (1-t)   # 현재 action 추정으로 재보간
    action = transformer(x_t, t, ...)  # clean action 직접 예측 → 다음 반복에 사용
```

**SmolVLA와 핵심 차이:**

| | SmolVLA | X-VLA |
|---|---------|-------|
| 예측 target | velocity (`noise - action`) | clean action 직접 |
| 추론 방식 | Euler 적분 (`x += dt * v`) | 직접 예측 후 재보간 |
| timestep 샘플링 | Beta(1.5, 1.0) | Stratified (균등 커버리지) |
| denoising steps | 10 (고정) | config (`num_denoising_steps`) |

X-VLA는 velocity가 아닌 **clean action을 직접 예측**. 매 step마다 현재 clean 추정값으로 재보간해서 노이즈를 점진적으로 줄여가는 방식.

### 10.5 Fine-tuning 유연성 (`_apply_freezing`)

SmolVLA의 `train_expert_only` 하나와 달리 4가지를 독립 제어:

| config 옵션 | 대상 |
|---|---|
| `freeze_vision_encoder` | Florence-2 vision tower |
| `freeze_language_encoder` | Florence-2 language encoder |
| `train_policy_transformer` | SoftPromptedTransformer 전체 |
| `train_soft_prompts` | soft prompt만 독립 학습 |

조합 예시:
- soft prompt만 학습 → Prompt Warm-up (Phase I, 새 로봇 적응 1단계)
- transformer + soft prompt 학습 → full fine-tuning (Phase II)
- LoRA 적용 → transformer 내부 일부만 학습 (peft_train.py)

또한 `get_optim_params()`에서 VLM 컴포넌트에 **1/10 LR** 적용 → pretrained 표현 보존.

### 10.6 Cross-embodiment Padding

```python
pad_vector(state, max_state_dim)           # 로봇마다 다른 state dim → max로 zero-padding
pad_tensor_along_dim(action, chunk_size)   # action chunk도 동일하게 패딩
```

action/state 차원이 달라도 `max_action_dim`으로 zero-padding 후 `DomainAwareLinear`가 domain별로 맞게 처리. 하나의 모델이 다양한 로봇을 수용하는 구조의 핵심.

---

## 11. 나머지 파일 역할 요약

### `modeling_florence2.py`
Microsoft Florence-2 모델의 PyTorch 구현. X-VLA가 외부 의존성 없이 자체 포함하기 위해 코드를 직접 가져온 것. 내부 구조는 DaViT(vision encoder) + BART-style language encoder로 구성된 표준 Florence-2 그대로. X-VLA는 이 중 **encoder만 사용**하고 decoder/lm_head는 삭제함 (10.2 참고). Florence-2 자체를 깊이 공부하려면 별도로 보면 됨.

### `action_hub.py`
로봇별 action space를 등록/관리하는 레지스트리. 각 action space는 `preprocess` (action → 모델 입력 정규화), `compute_loss` (예측 vs GT), `postprocess` (모델 출력 → 실제 action) 를 정의함.

등록된 action space:

| 이름 | 대상 로봇 |
|------|---------|
| `ee6d` | EEF 6D (xyz + Rotate6D + gripper) — 표준 |
| `joint` | Joint angle 기반 |
| `agibot_ee6d` | AGIBOT 로봇 특화 |
| `franka_joint7` | Franka 7-DoF joint |
| `auto` | 데이터셋 action dim을 자동 감지 |
| `so101_bimanual` | SO101 양팔 로봇 |

`auto` 모드는 데이터셋의 실제 action dim을 읽어서 자동으로 action space를 설정 → fine-tuning 시 별도 설정 없이 바로 사용 가능.

### `processor_xvla.py`
이미지/텍스트 전처리 담당. 이미지 리사이즈/정규화, 텍스트 토크나이징을 Florence-2 입력 형식에 맞게 처리.

### `configuration_xvla.py` / `configuration_florence2.py`
모델 하이퍼파라미터 정의 (`hidden_size`, `depth`, `num_heads`, `num_domains`, `len_soft_prompts` 등). `from_pretrained` 시 자동으로 로드됨.

---

## 12. lerobot Fine-tuning 설정 기록

### 12.1 config.json 호환성 수정

HuggingFace에서 다운로드한 `pretrained/xvla_pt/config.json`은 lerobot의 draccus 파싱 시스템과 호환되지 않아 수동 수정 필요.

**제거한 필드** (HF 전용, XVLAConfig에 없음):
- `_name_or_path`, `model_type`, `architectures`, `auto_map`, `num_actions`, `soft_prompt_length`

**추가/수정한 필드**:
- `"type": "xvla"` — draccus가 policy type 식별에 필수
- `"max_action_dim": 30` — pretrained weight shape에서 역산
- `"max_state_dim": 20` — weight shape 역산: 73728/1024=72, 72-20(ee6d)-32(dim_time)=20

### 12.2 weight 로딩 수정 (`modeling_xvla.py`)

pretrained weight의 키가 `model.` prefix 없이 저장되어 있어서 `from_pretrained` 시 mismatch 발생.

**Fix 1** — prefix 추가:
```python
sample_key = next(iter(state_dict))
if not sample_key.startswith("model."):
    state_dict = {"model." + k: v for k, v in state_dict.items()}
```

**Fix 2** — embed_tokens/shared 양방향 복사:
```python
if encoder_key in state_dict and shared_key not in state_dict:
    state_dict[shared_key] = state_dict[encoder_key]
elif shared_key in state_dict and encoder_key not in state_dict:
    state_dict[encoder_key] = state_dict[shared_key]
```

### 12.3 processor 파일 생성

lerobot은 pretrained path에 `policy_preprocessor.json` / `policy_postprocessor.json`이 있어야 함. HuggingFace pretrained 모델에는 이 파일이 없으므로 수동 생성.

**preprocessor 단계 순서:**
1. `rename_observations_processor` — observation key rename
2. `to_batch_processor` — batch dim 추가
3. `tokenizer_processor` — BART tokenizer (`facebook/bart-large`, max_length=64)
4. `xvla_image_to_float` — [0,255] → [0,1]
5. `xvla_imagenet_normalize` — ImageNet mean/std 정규화
6. `xvla_add_domain_id` — domain_id 텐서 추가
7. `device_processor` — GPU 이동
8. `normalizer_processor` — stats는 학습 시 dataset stats로 override됨

**postprocessor 단계:**
1. `unnormalizer_processor`
2. `device_processor` (cpu)

### 12.4 이미지 리사이즈 필요

데이터셋 이미지: **480×640** (non-square)

Florence-2 DaViT backbone은 feature map이 정사각형이어야 함 (`assert h * w == num_tokens`). 480×640 → 15×20 feature map → assertion 실패.

**해결**: `--policy.resize_imgs_with_padding="[224,224]"`
- 224×224 입력 → DaViT stride [4,2,2,2] → **7×7=49 tokens** (√49=7 ✓)
- config `max_pos_embeddings: 50` 범위 내

### 12.5 최종 학습 커맨드

```bash
cd /home/jake/lerobot/src && PYTORCH_ALLOC_CONF=expandable_segments:True \
conda run -n smolvla python -m lerobot.scripts.lerobot_train \
  --policy.path=pretrained/xvla_pt \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --output_dir=outputs/xvla_train_100k \
  --policy.push_to_hub=false \
  --job_name=xvla_finetune \
  --steps=100000 \
  --batch_size=8 \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=true \
  --policy.freeze_language_encoder=true \
  --policy.resize_imgs_with_padding="[224,224]"
```

**메모리 설정 근거 (RTX 5070 Ti 12GB):**
- `dtype=bfloat16` — 모델 전체 bf16 (~50% 절감)
- `freeze_vision_encoder + freeze_language_encoder` — Florence-2 gradient 없음, 311M만 학습
- 결과: 6.3GB VRAM 사용, GPU 94% 활용, ~3.5 step/s

**SmolVLA 비교 조건:**
- 동일 데이터셋: `lerobot/svla_so100_pickplace` (50 episodes, 19,631 frames)
- 동일 steps: 100K
- 동일 batch_size: 8 (총 ~40 epoch)
- 예상 학습 시간: ~8시간

---

## 13. lerobot Fine-tuning 결과 분석

### 13.0 실험 이력 요약

| 실험 | 설정 | 결과 |
|------|------|------|
| Run 1: 100K steps (실패) | `pretrained/xvla_pt` + IDENTITY 정규화 | 완전 Mode Collapse |
| Run 2: 20K steps (개선) | `lerobot/xvla-base` + MEAN_STD + action_mode=auto | Home Position Collapse |

---

### 13.1 Run 1 — 추론 결과 (100K steps, 실패)

```
Frame   0  | Mean Error: 77.26  | Predicted: [12.125, 13.0, 13.125, 13.25, 13.125, 7.969]
Frame  50  | Mean Error: 77.22  | Predicted: [12.125, 13.0, 13.125, 13.25, 13.125, 7.969]  ← 동일
Frame 100  | Mean Error: 59.79  | Predicted: [12.125, 13.0, 13.125, 13.25, 13.125, 7.969]  ← 동일
Frame 400  | Mean Error: 78.65  | Predicted: [12.125, 13.0, 13.125, 13.25, 13.125, 7.969]  ← 동일
```

**9개 프레임 전부 동일한 값 출력 → 완전한 Mode Collapse.**
예측값 `[12.125, 13.0, ...]`은 실제 home position(`[0.6, 177, 164, 72, 82, 0.1]`)과도 무관.
Training loss: step 2K = 6,242,631 / step 100K = 6,275,072 → **100K 동안 loss 변화 없음**

### 13.2 Run 2 — 추론 결과 (20K steps, lerobot/xvla-base)

**학습 설정:**
```
pretrained: lerobot/xvla-base (HF 공식)
steps: 20,000
batch_size: 4, bfloat16
action_mode: auto (SO-100 6DoF 자동 감지)
freeze: false (전체 879M 학습)
rename_map: top→image, wrist→image2
최종 loss: 0.012  ←  Run 1 대비 약 5억 배 개선
```

**추론 결과:**
```
Frame   0  | Pred: [0.375, 178.0, 165.0, 72.5, 82.5, 0.0625]  | Mean Error:  0.28
Frame  50  | Pred: [0.375, 178.0, 165.0, 72.5, 82.0, 0.0625]  | Mean Error:  0.39
Frame 100  | Pred: [0.375, 178.0, 165.0, 72.5, 82.0, 0.0625]  | Mean Error: 18.04
Frame 150  | Pred: [0.375, 178.0, 165.0, 72.5, 82.0, 0.0625]  | Mean Error: 29.99  ← 피크
Frame 200  | Pred: [0.375, 178.0, 165.0, 72.0, 82.0, 0.0625]  | Mean Error: 22.14
Frame 250  | Pred: [0.375, 178.0, 165.0, 72.0, 82.0, 0.0625]  | Mean Error: 20.70
Frame 300  | Pred: [0.375, 178.0, 164.0, 72.0, 82.0, 0.0625]  | Mean Error: 17.37
Frame 350  | Pred: [0.375, 178.0, 164.0, 72.0, 82.0, 0.0625]  | Mean Error: 12.81
Frame 400  | Pred: [0.25,  178.0, 164.0, 72.0, 82.0, 0.0625]  | Mean Error:  2.67
```

**에러 패턴 (bell-shape):**
```
 Frame:   0    50   100   150   200   250   300   350   400
 Error: 0.28  0.39  18.0  30.0  22.1  20.7  17.4  12.8  2.67
          ↑                 ↑                              ↑
        home 시작        task 중간 (피크)             home 복귀
```

**진단: Home Position Collapse (Mode Collapse와 다름)**
- Mode Collapse(Run 1): 모든 프레임에 의미없는 고정값 출력
- Home Position Collapse(Run 2): home 자세(평균)를 외워서 모든 프레임에 출력
- Run 2는 task 중간 프레임(100~350)에서 큰 에러 → 실제 task 동작을 학습하지 못함
- Frame 0, 400은 실제로 home 자세 → 우연히 에러가 낮음

이 패턴은 **SmolVLA fine-tuning 결과와 동일** (SmolVLA_Analysis.md Section 12.4 참고)

---

### 13.3 Run 1 vs Run 2 비교

| 항목 | Run 1 (100K, 실패) | Run 2 (20K, 개선) |
|------|-------------------|------------------|
| pretrained | `pretrained/xvla_pt` | `lerobot/xvla-base` ✓ |
| action 정규화 | IDENTITY ✗ | MEAN_STD ✓ |
| action_mode | 미설정 | auto ✓ |
| 학습 가능 파라미터 | 311M | 879M |
| 최종 loss | 6,275,072 | **0.012** |
| 붕괴 유형 | 완전 Mode Collapse | Home Position Collapse |
| Frame 0 에러 | 77.26 | **0.28** |
| Frame 150 에러 | ~60 | 29.99 |
| Frame 400 에러 | 78.65 | **2.67** |

**결론: MEAN_STD 정규화 수정만으로 loss가 정상 수렴. 학습 자체는 성공.**
**남은 문제는 Home Position Collapse — 데이터/스텝 부족 문제.**

---

### 13.4 학습 실패 원인 분석 (Run 1)

**[원인 1: ACTION 정규화 미적용 — 가장 결정적]**

```
SmolVLA preprocessor: ACTION: MEAN_STD  → action을 평균 0, 표준편차 1로 정규화
X-VLA preprocessor:   ACTION: IDENTITY  → raw degree 값 그대로 사용
```

X-VLA 데이터셋 action 통계 (학습 중 계산됐지만 IDENTITY라 미사용):
- mean: `[14.5, 146.4, 143.3, 62.9, 85.8, 7.8]`
- std:  `[27.9, 34.9, 21.4, 16.9, 12.4, 9.5]`

Flow Matching 손실: `u_t = noise - action` (noise ~N(0,1), action ~0~180°)
→ `u_t` 스케일이 수십~수백 → MSE loss 수백만
→ 실제: step 2K = 6,242,631 / step 100K = 6,275,072 → **100K 스텝 동안 loss 변화 없음**

SmolVLA는 MEAN_STD로 action이 [-1,1] 범위 → loss 정상 수렴.

**[원인 2: VLM encoder 동결 — 단독으로는 문제 없음]**

```
freeze_vision_encoder=true + freeze_language_encoder=true
→ 학습 가능: 311M / 879M (Florence-2 backbone 전체 동결)
```

동결 자체는 합리적인 전략:
- X-VLA 사전학습에서 이미 manipulation 데이터로 학습된 Florence-2 피처를 그대로 재활용
- SoftPromptedTransformer만 SO-100 태스크에 적응시키는 것이 목표

**단, 원인 1(IDENTITY 정규화)과 결합되면 치명적:**
- 사전학습 시 SoftPromptedTransformer는 normalized action space([-1,1] 근처)에서 학습됨
- IDENTITY 정규화로 fine-tuning하면 raw degree(0~180) 스케일의 완전히 다른 target을 학습 강요
- 사전학습 weight가 있어도 오히려 이를 망가뜨리는 방향으로 작용

→ **동결 + MEAN_STD** 조합이었다면 사전학습 피처 재활용 + 정상 학습이 가능했을 것

**[원인 3: 이미지 강제 리사이즈 480×640 → 224×224]**

- DaViT square 제약으로 불가피했지만 (Section 12.4), pre-training distribution과 괴리
- 동결된 encoder에 축소된 이미지 입력 → encoder 출력 피처 품질 저하

**[원인 4: 12GB VRAM 제약의 복합 효과]**

원인 2, 3은 모두 12GB VRAM 한계를 맞추기 위한 불가피한 선택:

| 제약 | VRAM 절감 | 학습 품질 영향 |
|------|----------|--------------|
| bfloat16 | ~50% 절감 | 미미 |
| VLM 동결 | 대폭 절감 | 피처 미적응 |
| 224×224 리사이즈 | 소폭 절감 | pre-training 분포 괴리 |
| ACTION: IDENTITY | 없음 | **치명적 — loss 발산** |

**[원인 5: 데이터/파라미터 불균형]**

- 879M 모델 × 50 에피소드: SmolVLA(450M)보다 파라미터/데이터 비율이 더 불리
- X-VLA 논문의 fine-tuning은 충분한 태스크 데이터 가정

### 13.5 SmolVLA vs X-VLA 비교

| 항목 | SmolVLA | X-VLA |
|------|---------|-------|
| 모델 크기 | 450M | 879M |
| 학습 가능 파라미터 | 전체 | 311M (encoder 동결) |
| Action 정규화 | MEAN_STD ✓ | IDENTITY ✗ |
| 이미지 크기 | 원본 유지 | 480×640 → 224×224 강제 리사이즈 |
| 최종 loss | 수렴 | 6,275,072 (실패) |
| Mode collapse | home position 수렴 | 무의미한 고정값 |
| Frame 0 에러 | 0.14 | 77.26 |

### 13.6 Home Position Collapse 해결 방안

Run 2에서 Home Position Collapse가 발생한 원인과 해결 방향:

**원인:**
- 879M 파라미터 모델 × 50 에피소드 × 20K steps = 데이터/스텝 부족
- SmolVLA(450M)도 동일 조건에서 같은 패턴 → X-VLA 고유 문제가 아닌 데이터 규모 문제
- 에피소드 구조: home→task→home → 전체 프레임의 상당 비율이 home 자세 → 모델이 평균을 외움

**노트북(12GB)에서 가능한 개선:**

| 개선 방향 | 기대 효과 |
|----------|----------|
| steps 100K+ 로 증가 | 더 많은 학습 → task 중간 동작 학습 가능성 |
| 에피소드 수 100~200개로 증가 | 데이터 다양성 확보 |
| **둘 다 적용** | 가장 효과적 |

**워크스테이션(96GB)에서 추가 가능:**

| 개선 방향 | 기대 효과 |
|----------|----------|
| 원본 해상도 480×640 유지 | pre-training 분포 유지 |
| batch_size 32+ | gradient 안정화 |
| 에피소드 500+ | X-VLA 논문 수준 데이터 규모 |

→ **Run 1 실패는 IDENTITY 정규화 단 하나의 오류였음. Run 2에서 학습 자체는 정상 확인. 이제 데이터/스텝 규모 문제만 남음.**

### 13.7 공식 lerobot 문서와의 차이 (Run 1 근본 원인)

참고: https://huggingface.co/docs/lerobot/xvla

| 항목 | 공식 docs | 이번 실험 | 문제 여부 |
|------|-----------|----------|---------|
| pretrained model | `lerobot/xvla-base` (HF 공식) | `pretrained/xvla_pt` (원본 레포 수동 패치) | **핵심 문제** |
| action_mode | `--policy.action_mode=auto` | 미설정 | **핵심 문제** |
| freeze 설정 | `false` 권장 | `true` (VRAM 때문에 강제) | 불가피 |
| pip install | `pip install -e .[xvla]` | 완료됨 (transformers 5.3.0) | 문제 없음 |

**`lerobot/xvla-base`가 핵심:**
- `pretrained/xvla_pt`는 원본 X-VLA 레포 weight를 lerobot에 억지로 맞춘 것 → processor JSON 수동 생성, config 수동 패치 필요
- `lerobot/xvla-base`는 lerobot 통합용으로 새로 준비된 weight → processor/config 올바르게 포함

**`action_mode=auto`를 안 쓴 것:**
- SO-100은 6 DOF인데 기본 action_mode에서 차원 불일치 발생 가능
- `auto` 모드: dataset의 실제 action 차원 감지 + loss를 real_dim에만 계산

### 13.8 다음 실험 커맨드

Run 2 커맨드 (기준, 완료):
```bash
cd /home/jake/lerobot/src && PYTORCH_ALLOC_CONF=expandable_segments:True \
conda run -n smolvla python -u -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/xvla-base \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --output_dir=outputs/xvla_proper \
  --policy.push_to_hub=false \
  --job_name=xvla_proper \
  --steps=20000 \
  --batch_size=4 \
  --policy.dtype=bfloat16 \
  --policy.action_mode=auto \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --policy.resize_imgs_with_padding="[224,224]" \
  --rename_map='{"observation.images.top": "observation.images.image", "observation.images.wrist": "observation.images.image2"}'
```

Run 3 예정 (steps + 데이터 확대):
```bash
# steps=100000, 에피소드 수 100~200개 수집 후
cd /home/jake/lerobot/src && PYTORCH_ALLOC_CONF=expandable_segments:True \
conda run -n smolvla python -u -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/xvla-base \
  --dataset.repo_id=<새 데이터셋 repo_id> \
  --output_dir=outputs/xvla_100k_v2 \
  --policy.push_to_hub=false \
  --job_name=xvla_100k_v2 \
  --steps=100000 \
  --batch_size=4 \
  --policy.dtype=bfloat16 \
  --policy.action_mode=auto \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --policy.resize_imgs_with_padding="[224,224]" \
  --rename_map='{"observation.images.top": "observation.images.image", "observation.images.wrist": "observation.images.image2"}'
```

---

## 14. 스터디 체크리스트

- [x] X-VLA 레포 fork & clone (`~/X-VLA/`)
- [x] lerobot 내 xvla 통합 확인 (`lerobot/src/lerobot/policies/xvla/`)
- [x] 두 레포 역할 분담 파악 (Section 5.3 참고)
- [x] `soft_transformer.py` 코드 분석 (Section 9)
- [x] `modeling_xvla.py` 코드 분석 (Section 10)
- [x] 나머지 파일 역할 파악 (`modeling_florence2.py`, `action_hub.py` 등, Section 11)
- [x] lerobot fine-tuning 환경 설정 (config 호환성, processor, 이미지 리사이즈, Section 12)
- [x] lerobot fine-tuning 실험 시작 (so100_pickplace 100K steps → 실패, 원인 분석)
- [x] lerobot/xvla-base + MEAN_STD + action_mode=auto 로 재실험 (20K steps, Run 2)
- [x] Run 2 결과 분석 — loss 0.012 수렴, Home Position Collapse 확인 (Section 13)
- [ ] 논문 Figure 2 (heterogeneity 해결 방법 비교) 이해
- [ ] LoRA fine-tuning 실험 (`peft_train.py`)
- [ ] LIBERO 시뮬레이션 eval 환경 설치 및 실행
- [ ] Soft Prompt embedding 시각화/분석
