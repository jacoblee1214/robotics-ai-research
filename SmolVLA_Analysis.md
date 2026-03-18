# SmolVLA 스크립트 분석 & 아키텍처 가이드

> 작성일: 2026-03-18
> 논문: SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics (arXiv:2506.01844)
> 코드: https://github.com/huggingface/lerobot (LeRobot 라이브러리 내 포함)

---

## 1. SmolVLA 개요

SmolVLA는 HuggingFace에서 만든 **450M 파라미터의 경량 VLA 모델**로, 소비자용 GPU(12GB)에서도 학습/추론이 가능하며 10배 큰 VLA와 비슷한 성능을 달성함.

**핵심 특징:**
- 450M 파라미터 (Action Expert ~100M + VLM ~350M)
- SmolVLM-2 (500M) 기반, 상위 절반 레이어만 사용 (16/32 layers)
- Flow Matching 기반 continuous action 생성 (not discrete tokenization)
- Community 데이터셋 22.9K 에피소드로 사전학습 (다른 VLA 대비 10배 적음)
- 비동기 추론 지원 → 동기 대비 ~2배 속도 향상
- 추론 시 ~2GB VRAM만 사용

---

## 2. 아키텍처 상세

```
┌──────────────────────────────────────────────────────┐
│                      SmolVLA                          │
│                                                      │
│  입력: [카메라 이미지 (최대 3개)] + [언어 명령] + [로봇 상태]  │
│                                                      │
│  ┌────────────────────────────────────────────┐      │
│  │  SmolVLM-2 (Vision-Language Model)          │      │
│  │                                            │      │
│  │  ┌──────────┐                              │      │
│  │  │ SigLIP    │ → 이미지당 64 토큰           │      │
│  │  │ (Vision   │   (PixelShuffle로 축소)      │      │
│  │  │  Encoder) │                              │      │
│  │  └────┬─────┘                              │      │
│  │       │                                    │      │
│  │  ┌────▼──────────────────────────────┐     │      │
│  │  │ SmolLM2 (Language Decoder)         │     │      │
│  │  │ 32 layers 중 앞 16개만 사용 ✂️       │     │      │
│  │  │                                    │     │      │
│  │  │ 입력: [visual tokens] + [text] +   │     │      │
│  │  │       [state token (linear proj)]  │     │      │
│  │  └────┬──────────────────────────────┘     │      │
│  │       │ VLM features (frozen)              │      │
│  └───────┼────────────────────────────────────┘      │
│          │                                           │
│  ┌───────▼────────────────────────────────────┐      │
│  │  Action Expert (~100M params)               │      │
│  │  Flow Matching Transformer                  │      │
│  │                                            │      │
│  │  ┌─────────────────────────┐               │      │
│  │  │ Cross-Attention (CA)     │ ← VLM features│      │
│  │  │ Self-Attention (SA)      │ ← 액션 간 관계 │      │
│  │  │ Cross-Attention (CA)     │   (causal)    │      │
│  │  │ Self-Attention (SA)      │               │      │
│  │  │ ...interleaved...        │               │      │
│  │  └────────┬────────────────┘               │      │
│  │           │                                │      │
│  │  학습: flow matching loss                   │      │
│  │  추론: 10 denoising steps                   │      │
│  └───────────┼────────────────────────────────┘      │
│              │                                       │
│              ▼                                       │
│  Action Chunk: [a_t, a_{t+1}, ..., a_{t+n}]          │
│  (n=50 actions, continuous)                          │
└──────────────────────────────────────────────────────┘
```

### 2.1 핵심 설계 결정 (Ablation 결과)

| 설계 선택 | 결과 (LIBERO Avg SR%) | 비고 |
|-----------|----------------------|------|
| **CA+SA interleaved (채택)** | **85.5%** | CA만 79.0%, SA만 74.5% |
| Causal attention mask (채택) | 74.5% | Bidirectional 67.5% |
| Flow matching (채택) | 80.3% | Regression(L1) 75.3% |
| Layer skip N=16 (채택) | 78.5% | N=32(전체) 80.3%이지만 2배 느림 |
| Expert width 0.75x (채택) | 77.5% | 1.0x 82.3%이지만 더 무거움 |
| State를 VLM에 입력 (채택) | 80.3% | Expert에 입력 시 73.3% |
| Chunk size 50 (채택) | 80.3% | 10이 84.0%으로 최고, 1은 50.0% |

### 2.2 GR00T N1.6과 비교

| | SmolVLA | GR00T N1.6 |
|---|---------|------------|
| 파라미터 | 450M | 3B |
| VLM | SmolVLM-2 (500M) | Cosmos-Reason-2B |
| Action Head | Flow Matching Transformer | Diffusion Transformer (DiT) |
| Attention | CA+SA interleaved | Cross-attention to VLM |
| Action 표현 | Continuous (flow matching) | State-relative (flow matching) |
| 학습 데이터 | 22.9K episodes (community) | 10K+ hours (diverse) |
| 추론 VRAM | ~2GB | ~8GB+ |
| 데이터 포맷 | LeRobot v2/v3 | LeRobot v2 (GR00T flavored) |
| VLM 학습 | Frozen | Top 4 layers unfrozen |

---

## 3. 코드 구조 (LeRobot 내부)

```
lerobot/
├── src/lerobot/
│   ├── policies/
│   │   └── smolvla/
│   │       ├── modeling_smolvla.py     # ★ SmolVLAPolicy 모델 정의
│   │       └── configuration_smolvla.py # 모델 config
│   │
│   ├── scripts/
│   │   ├── lerobot_train.py            # ★ 학습 엔트리포인트
│   │   └── server/
│   │       ├── policy_server.py        # 비동기 추론 서버
│   │       └── robot_client.py         # 로봇 클라이언트
│   │
│   ├── datasets/
│   │   └── lerobot_dataset.py          # LeRobot 데이터셋 로더
│   │
│   └── configs/
│       ├── train.py                    # 학습 설정
│       └── parser.py                   # CLI 파서
│
├── examples/                           # 예시 스크립트
└── docs/source/
    ├── smolvla.mdx                     # SmolVLA 공식 문서
    └── async.mdx                       # 비동기 추론 문서
```

---

## 4. 실행 명령어

### 4.1 모델 로드 & 추론 테스트

```python
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 모델 로드
device = torch.device("cuda")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).eval()
preprocess, postprocess = make_pre_post_processors(
    policy.config, "lerobot/smolvla_base",
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# 데이터셋 로드 & 추론
dataset = LeRobotDataset("lerobot/libero")
episode_index = 0
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
frame = dict(dataset[from_idx])
# frame을 preprocess 후 policy에 넣어 action 얻기
```

### 4.2 Fine-tuning

```bash
# 기본 fine-tuning
python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --output_dir=outputs/train \
  --policy.repo_id=myuser/smolvla_test \
  --job_name=smolvla_finetune \
  --policy.push_to_hub=false

# 카메라 이름이 다를 때 rename_map 사용
python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --output_dir=outputs/train \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

### 4.3 비동기 추론 (실제 로봇)

```bash
# Terminal 1: Policy Server (GPU 머신)
python -m lerobot.scripts.server.policy_server \
  --host=0.0.0.0 --port=9000

# Terminal 2: Robot Client (로봇 머신)
python -m lerobot.scripts.server.robot_client \
  --config_path=async_evaluate/template.yaml
```

### 4.4 lerobot-record (추론 + 기록)

```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=myuser/eval_test \
  --dataset.single_task="Pick up the cube and place it in the box" \
  --policy.path=lerobot/smolvla_base
```

---

## 5. Fine-tuning 시 수정 포인트

### 5.1 카메라 매핑 (가장 흔한 수정)

SmolVLA는 `camera1`, `camera2`, `camera3`을 기대함. 데이터셋 카메라 이름이 다르면 `--rename_map`으로 매핑:

```bash
--rename_map='{"observation.images.내카메라이름": "observation.images.camera1"}'
```

논문 기준 카메라 우선순위: **top > wrist > side**

### 5.2 Action Chunk Size

- 기본값: 50
- 논문 ablation 결과: 10~50이 최적, 1은 성능 급락
- `--policy.chunk_size=50`으로 조절

### 5.3 학습 하이퍼파라미터

| 파라미터 | Pretrain | Fine-tune (sim) | Fine-tune (real) |
|---------|----------|-----------------|------------------|
| Steps | 200,000 | 100,000 | 200,000 (or 20,000) |
| Batch size | 256 | 64 | 64 |
| LR | 1e-4 → 2.5e-6 (cosine) | - | - |
| Optimizer | AdamW (β1=0.9, β2=0.95) | - | - |
| Image size | 512×512 | 512×512 | 512×512 |

- 실제로는 20K steps 정도면 충분히 수렴한다고 보고됨
- A100 기준 20K steps ≈ 4시간

### 5.4 VLM 관련

- **기본: VLM은 frozen** (학습하지 않음)
- Action Expert만 학습됨
- VLM을 fine-tune하고 싶으면 코드에서 freeze 해제 필요
- Visual tokens: 이미지당 64개 (PixelShuffle), tiling 미사용

### 5.5 실전 학습 설정 (RTX 5070 Ti 12GB 실측)

실제 `lerobot/svla_so100_pickplace` 데이터셋으로 학습 시 로그에서 확인된 값들:

```
num_learnable_params = 99,880,992  (100M, Action Expert만)
num_total_params     = 450,046,176 (450M)
dataset.num_frames   = 19,631      (20K frames)
dataset.num_episodes = 50
VRAM 사용량           ≈ 3.5GB
```

| 설정 | 실제 적용값 | 비고 |
|------|-----------|------|
| `resize_imgs_with_padding` | [512, 512] | 모든 입력 이미지를 512×512로 리사이즈 |
| `steps` | 100,000 | 기본 10만 스텝 (줄여도 됨) |
| `save_freq` | 20,000 | 2만 스텝마다 체크포인트 자동 저장 |
| `batch_size` | 8 | 12GB GPU에서 자동 설정된 값 |
| `scheduler` | cosine_decay_with_warmup | warmup 1000 steps → peak LR 1e-4 → decay to 2.5e-6 |
| `scheduler_decay_steps` | 30,000 | 3만 스텝에 걸쳐 decay |
| `log_freq` | 200 | 200 스텝마다 loss 출력 |
| `eval_freq` | 20,000 | 2만 스텝마다 evaluation |
| `camera3` | config에 존재하나 데이터 없으면 자동 무시 | rename_map에 2개만 매핑해도 동작 |
| `train_expert_only` | True | VLM은 frozen, Action Expert만 학습 |
| `train_state_proj` | True | state projection layer도 같이 학습 |

**배치 사이즈 조절 팁:**
- 12GB에서 batch_size=8이 기본으로 잡힘 (VRAM ~3.5GB)
- 여유가 있으므로 `--batch_size=16` 또는 `32`로 올려서 학습 속도 향상 가능
- 워크스테이션(96GB)에서는 `--batch_size=64` 이상 가능

---

## 6. 비동기 추론 상세

### 6.1 Sync vs Async

```
[Sync 추론]
관찰 → 추론(idle) → 50 actions 실행 → 관찰 → 추론(idle) → ...
         ↑ 로봇이 멈춤

[Async 추론]
관찰 → 추론(병렬) → actions 실행하면서 동시에 다음 추론 시작
         ↑ 로봇이 멈추지 않음
```

### 6.2 핵심 파라미터

- **actions_per_chunk**: 한 chunk에서 실행할 action 수
- **chunk_size_threshold (g)**: 큐가 이 비율 이하로 떨어지면 새 추론 요청
  - g=0: sync와 동일 (chunk 다 소진 후 요청)
  - g=0.7: 30% 소진 시점에 미리 요청 (권장)
  - g=1.0: 매 timestep 요청 (가장 반응적이나 연산 부담)

### 6.3 성능 비교 (논문 결과)

| | Sync | Async |
|---|------|-------|
| 평균 완료 시간 | 13.75s | 9.70s (**30% 빠름**) |
| 60초 내 성공 횟수 | 9개 | 19개 (**2배**) |
| 성공률 (Pick-Place) | 75% | 80% |

---

## 7. 성능 벤치마크

### 7.1 시뮬레이션 (LIBERO)

| Model | Params | Spatial | Object | Goal | Long | Avg |
|-------|--------|---------|--------|------|------|-----|
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 (pretrained) | 3.3B | 90 | 86 | 95 | 73 | 86.0 |
| **SmolVLA** | **0.45B** | **90** | **96** | **92** | **71** | **87.3** |

### 7.2 Real-world (SO100)

| Model | Pick-Place | Stacking | Sorting | Avg |
|-------|-----------|----------|---------|-----|
| ACT (single-task) | 70 | 50 | 25 | 48.3 |
| π0 (3.5B) | 100 | 40 | 45 | 61.7 |
| **SmolVLA (0.45B)** | **75** | **90** | **70** | **78.3** |

### 7.3 Pretrain 효과

| 설정 | Pick-Place | Stacking | Sorting | Avg |
|------|-----------|----------|---------|-----|
| No pretrain, single-task | 55 | 45 | 20 | 40.0 |
| No pretrain, multi-task | 80 | 40 | 35 | 51.7 |
| **Pretrain + multi-task** | **75** | **90** | **70** | **78.3** |

→ 사전학습이 큰 차이를 만들고, multi-task 학습이 추가 이득을 줌

---

## 8. 한계점 (논문에서 밝힌 것)

1. **데이터 다양성**: SO100 단일 로봇으로만 사전학습. cross-embodiment 데이터가 부족
2. **데이터 규모**: 22.9K episodes (OpenVLA는 100만+). 스케일업 여지 있음
3. **VLM 백본**: SmolVLM-2는 원래 문서/OCR용으로 학습됨. 로봇 특화 VLM이면 더 나을 수 있음
4. **장기 horizon 태스크**: 비교적 단순한 short-horizon 태스크에서만 검증됨
5. **학습 방식**: 모방학습만 사용. 강화학습 통합 시 개선 가능성

---

## 9. GR00T과 SmolVLA를 함께 활용하는 전략

| 용도 | SmolVLA | GR00T N1.6 |
|------|---------|------------|
| 빠른 프로토타이핑 | ✅ 12GB에서 전체 파이프라인 | ❌ 추론만 가능 |
| Fine-tuning 실험 | ✅ 단일 GPU, 몇 시간 | 48GB+, 더 오래 |
| 대형 모델 성능 | 제한적 | ✅ 3B, 다양한 로봇 데이터 |
| 실제 로봇 배포 | ✅ edge 디바이스 가능 | Jetson 이상 필요 |
| 데이터 포맷 | LeRobot v2/v3 | LeRobot v2 (호환) |

**추천 워크플로우:**
1. SmolVLA로 데이터 수집 → fine-tune → 빠른 검증 (노트북)
2. 같은 LeRobot 데이터를 GR00T에 적용 → 성능 비교 (워크스테이션)
3. 두 모델의 action 출력 비교 분석 → 연구 인사이트 도출
