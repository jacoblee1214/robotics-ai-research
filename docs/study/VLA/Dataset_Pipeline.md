# VLA 데이터셋 & 입력 파이프라인 정리

## 1. 데이터 모달리티 개요

VLA 모델의 입력은 크게 4가지 모달리티로 구성된다.

```
┌──────────────────────────────────────────────────────────────┐
│                      VLA 입력 모달리티                        │
├──────────────┬───────────────────────────────────────────────┤
│ Vision       │ RGB 이미지 (카메라 1~3개), 선택적으로 Depth   │
│ Language     │ 자연어 태스크 지시 (task instruction)         │
│ State        │ 로봇 고유 감각 (joint angle, EEF pose 등)     │
│ Action       │ 출력 대상 (joint/EEF 이동값, gripper)         │
└──────────────┴───────────────────────────────────────────────┘
```

---

## 2. 데이터 포맷 표준 (LeRobot HuggingFace 포맷)

직접 실험에 사용하는 데이터셋은 모두 이 포맷 기준.

### 2.1 디렉터리 구조

```
dataset_root/
├── meta/
│   ├── info.json          # fps, 카메라 이름, action/state dim, 총 episode 수
│   ├── episodes.jsonl     # 각 episode의 시작/끝 프레임 index
│   └── stats.json         # action/state 정규화용 mean, std, min, max
├── data/
│   └── chunk-000/
│       └── episode_*.parquet   # 타임스텝별 state, action, timestamp
└── videos/
    └── chunk-000/
        └── observation.images.<cam_name>/
            └── episode_*.mp4  # 카메라별 비디오
```

### 2.2 핵심 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `observation.images.<cam>` | uint8 [H, W, 3] | 카메라 이미지 (cam 이름은 커스텀) |
| `observation.state` | float32 [state_dim] | 로봇 현재 상태 (joint angle or EEF pose) |
| `action` | float32 [action_dim] | 해당 timestep의 GT action |
| `timestamp` | float64 | 에피소드 내 경과 시간 (초) |
| `frame_index` | int | 에피소드 내 프레임 번호 |
| `episode_index` | int | 전체 데이터셋 내 에피소드 번호 |
| `task` | str | 자연어 지시 (e.g. "Pick up the cube") |

### 2.3 info.json 핵심 항목

```json
{
  "fps": 30,
  "video.fps": 30,
  "features": {
    "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
    "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
    "observation.state": {"dtype": "float32", "shape": [6]},
    "action": {"dtype": "float32", "shape": [6]}
  },
  "total_episodes": 50,
  "total_frames": 19631
}
```

---

## 3. 모달리티별 파이프라인

### 3.1 Vision (이미지)

```
카메라 원본 (640×480 or 1280×720)
    ↓
리사이즈 (모델별 상이)
    ↓
ImageNet 정규화 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ↓
Vision Encoder (ViT 계열)
    ↓
이미지 토큰 시퀀스
```

| 모델 | 입력 해상도 | Vision Encoder | 이미지 토큰 수 |
|------|------------|---------------|----------------|
| SmolVLA | 224×224 (패딩) | SigLIP (SmolVLM) | 64/256 per cam |
| GR00T N1.6 | 224×224 | Eagle2 ViT | 가변 |
| X-VLA | 224×224 (패딩) | Florence-L ViT (fixed cam) + Shared ViT (wrist) | 256 per cam |
| π0 | 224×224 | SigLIP (PaliGemma) | 256 per cam |

**카메라 구성 권장:**
- **최소**: top(overhead) 1개 + wrist 1개 = 2대
- **이유**: top은 전체 장면/물체 위치 파악, wrist는 manipulation 직전 세밀한 제어용
- **선택적**: side 카메라 추가 시 depth 정보 보완 가능

### 3.2 Language (텍스트)

```
자연어 태스크 지시 (e.g. "Pick the red cube and place it in the box")
    ↓
토크나이저 (모델별 LLM 토크나이저)
    ↓
언어 토큰 시퀀스
    ↓
LLM/VLM 텍스트 임베딩과 이미지 토큰 결합
```

- 지시 길이: 일반적으로 1~3문장, 너무 길면 성능 저하 가능
- 동일 태스크도 여러 표현으로 augmentation 가능 (robustness 향상)

### 3.3 State (로봇 상태)

로봇의 현재 자세를 수치로 표현. 크게 두 종류:

| 종류 | 내용 | 차원 수 |
|------|------|---------|
| Joint angle | 각 관절의 각도 (deg or rad) | DOF 수 (SO-100: 6) |
| EEF pose | 엔드이펙터 xyz + 자세 (Euler or Rotation6D) | 6~9 |

- SO-100 (6-DOF manipulator): state_dim = 6 (joint angles)
- 정규화 필수: MEAN_STD로 [-1, 1] 범위로 맞춤 (raw degree 그대로 쓰면 학습 불안정)

### 3.4 Action

모델이 예측하는 대상. State와 같은 공간이지만 "다음에 취할 동작"을 의미.

| 종류 | 내용 | 비고 |
|------|------|------|
| Absolute joint | 목표 관절 각도 절댓값 | 직관적, 초기 학습 쉬움 |
| Relative joint | 현재 대비 이동량 (delta) | State 변화에 robust |
| EEF absolute | 목표 엔드이펙터 위치 절댓값 | task-space 제어 |
| EEF relative | 현재 EEF 대비 이동량 | X-VLA anchor point 방식 |

**SO-100 기준**: action_dim = 6 (관절 6개, 마지막이 gripper open/close)

---

## 4. 모델별 입력 파이프라인 비교

### 4.1 SmolVLA

```
Vision:    [top cam, wrist cam] → 224×224 리사이즈+패딩 → SigLIP ViT → 이미지 토큰
Language:  task text → SmolLM2 토크나이저 → 언어 토큰
State:     joint 6dim → MEAN_STD 정규화 → Linear proj → state 토큰

결합: [이미지 토큰 | 언어 토큰 | state 토큰] → SmolVLM (450M)
                                                    ↓
출력: chunk_size=50 × action_dim=6 (Action Expert)
실행: 50 actions을 30Hz로 순서대로 실행 (약 1.7초)
```

### 4.2 GR00T N1.6

```
Vision:    [top cam] → 224×224 → Eagle2 ViT → 이미지 토큰 시퀀스
Language:  task text → 언어 토큰
State:     joint/EEF dim → Linear proj

결합: [이미지 토큰 | 언어 토큰 | state] → LLM backbone (3B)
                                            ↓
출력: 8~16 actions (Flow Matching, 8~16 denoising steps)
실행: 30Hz로 실행 (약 0.27~0.53초)
```

### 4.3 X-VLA

```
Vision:    [fixed cam] → 224×224 → Florence-L ViT → 이미지 토큰 (high-level)
           [wrist cam] → 224×224 → Shared ViT     → aux 토큰 (fine-grained)
Language:  task text → Florence tokenizer → 언어 토큰
State:     proprio (EEF pose + gripper) → max_state_dim=20으로 zero-padding

결합: [이미지 토큰 | 언어 토큰] → Florence VLM
     [aux 토큰 | proprio] → SoftPromptedTransformer

출력: 30 anchor points (keyframe, EEF relative pose)
     → 중간 자세는 보간 → 약 4초 치 궤적
실행: keyframe 순서대로 IK(역기구학)으로 joint 변환 후 실행
```

### 4.4 π0

```
Vision:    [cam1, cam2, cam3] → 224×224 → SigLIP ViT → 이미지 토큰 (256 per cam)
Language:  task text → PaliGemma 토크나이저 → 언어 토큰
State:     joint/EEF dim → state_proj Linear → state 토큰 1개

결합: [이미지 토큰 × N | 언어 토큰 | state 토큰] → PaliGemma 3B
                                                      ↓ (Blockwise Causal Attention)
      [Action Expert: noisy action + timestep] ←→ PaliGemma 출력

출력: 50 actions × action_dim (Flow Matching, 10 denoising steps)
실행: 50Hz로 실행 (약 1.0초)
```

### 4.5 π0.5

```
π0 파이프라인과 동일하되:
출력 단계에서 FAST 토크나이저 사용
  → action 시퀀스 → DCT 변환 → BPE 토큰 (700 → 53 tokens)
  → 자기회귀 디코딩으로 action token 생성
  → 역변환으로 continuous action 복원
```

---

## 5. 공개 데이터셋 목록

### 5.1 LeRobot HuggingFace Hub (직접 사용 가능)

| 데이터셋 | 로봇 | 태스크 | Episodes | 비고 |
|----------|------|--------|---------|------|
| `lerobot/svla_so100_pickplace` | SO-100 | pick & place | 50 | 직접 실험에 사용 중 |
| `lerobot/pusht` | 2D 시뮬 | T 블록 밀기 | 200 | 입문용 benchmark |
| `lerobot/aloha_sim_insertion_scripted` | ALOHA (시뮬) | 부품 삽입 | 50 | bimanual |
| `lerobot/aloha_sim_transfer_cube_scripted` | ALOHA (시뮬) | 큐브 전달 | 50 | bimanual |
| `lerobot/libero_*` | LIBERO (시뮬) | 다양 | 다수 | Medium-horizon 평가용 |
| `lerobot/droid_wipe` | DROID | 테이블 닦기 | 다수 | real-world |

### 5.2 대규모 공개 데이터셋

| 데이터셋 | 규모 | 특징 | 사용 모델 |
|----------|------|------|-----------|
| Open X-Embodiment (OXE) | 22개 로봇, 100만+ 에피소드 | 다종 로봇 혼합, 표준 포맷 | π0, GR00T 사전학습 |
| BridgeData V2 | 60,096 에피소드 | 5Hz, kitchen manipulation | π0 fine-tuning 평가 |
| DROID | 76,000 에피소드 | 15Hz, 562개 환경 | π0 사전학습 |
| LIBERO | 시뮬 130개 태스크 | 표준 Medium-horizon 벤치마크 | MoDE-VLA 평가 |
| RoboAgent | 7,500 에피소드 | 다양한 manipulation | - |
| LASA | - | Language-annotated robot demos | X-VLA 사전학습 |

### 5.3 데이터 로드 예시 (LeRobot)

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/svla_so100_pickplace")
print(dataset.meta.info)          # fps, features, episode 수
print(dataset.meta.stats)         # 정규화 통계

frame = dict(dataset[0])          # 단일 프레임
# frame.keys(): observation.images.top, observation.images.wrist,
#               observation.state, action, timestamp, task, ...
```

---

## 6. 데이터 수집 가이드라인

### 6.1 제어 주파수 (fps) 선택

| 상황 | 권장 fps | 이유 |
|------|---------|------|
| 일반 manipulation (pick & place) | **30Hz** | 대부분 VLA의 기본, 데이터 크기 적당 |
| 고정밀 조작 (나사 조이기, 봉합) | 50Hz | π0 스타일, 세밀한 제어 필요 |
| 느린 태스크 (옷 접기, 물 붓기) | 10~20Hz | 고fps 불필요, 데이터 절약 |

> **실용 권장: 30Hz 고정** — SmolVLA, GR00T, X-VLA 모두 30fps 데이터 학습 가능.
> π0 파인튜닝 시에는 50Hz 권장.

### 6.2 비디오/이미지 해상도

| 단계 | 권장값 |
|------|--------|
| 수집 해상도 | 640×480 이상 (나중에 다운샘플 가능) |
| 모델 입력 해상도 | 224×224 (대부분 모델이 이 크기로 리사이즈) |
| 권장 수집 해상도 | 480×480 (정사각형) 또는 640×480 |

- 수집할 때 고해상도로 저장해두면 나중에 더 큰 모델에도 사용 가능
- 원본 보존 후 전처리 시 리사이즈/패딩

### 6.3 에피소드 길이 (Horizon) 설계

| 태스크 유형 | 권장 에피소드 길이 | 프레임 수 (30Hz) |
|-----------|-----------------|-----------------|
| Single-step | 1~5초 | 30~150 프레임 |
| Short-horizon | 5~20초 | 150~600 프레임 |
| Medium-horizon | 20~120초 | 600~3600 프레임 |

**SO-100 pick & place 기준**: 약 400프레임 @ 30Hz = 약 13초

> 에피소드는 항상 **home position 시작 → 태스크 → home position 복귀** 구조 권장.
> 단, SmolVLA 학습 시 home collapse 주의 (frame 0 = frame -1 문제 → 다양한 초기 자세 포함).

### 6.4 Action Chunk 크기 설계

```
Chunk 크기 결정 기준:
  너무 작음 (< 10): 추론 오버헤드 증가, 반응성 과잉
  너무 큼  (> 100): open-loop 구간 길어짐, 오차 누적

권장:
  - 단순 조작 태스크: chunk_size = 20~50
  - 정밀 조작: chunk_size = 10~20
  - 현재 연구 표준: 50 (SmolVLA, π0)
```

### 6.5 에피소드 수

| 목적 | 최소 에피소드 수 | 권장 |
|------|----------------|------|
| Fine-tuning (단순 태스크) | 20~50 | 50~100 |
| Fine-tuning (복잡 태스크) | 100 | 200~500 |
| 사전학습 | 수천~수만 | - |

> 50 에피소드로 시작 → 성능 불충분 시 100, 200으로 늘리는 게 효율적.
> 데이터 **다양성** (환경 변화, 조작 속도 변화)이 에피소드 수보다 중요할 때 많음.

### 6.6 State/Action 정규화

```
반드시 MEAN_STD 정규화 적용:
  normalized = (x - mean) / std  →  대략 [-1, 1] 범위

IDENTITY(정규화 없음) 사용 시:
  - Joint angle (0~180deg) → Flow Matching loss scale = 수백만
  - 학습 불가 (X-VLA 100K 실패 사례)

LeRobot은 dataset stats에서 자동으로 계산:
  dataset.meta.stats["action"]["mean"]
  dataset.meta.stats["action"]["std"]
```

---

## 7. 직접 모델 개발 시 입력 파이프라인 설계

### 7.1 최소 구성 (SO-100 기준)

```python
# 입력 텐서 구성
{
  "observation.images.top":   Tensor[B, 3, 224, 224],   # top 카메라
  "observation.images.wrist": Tensor[B, 3, 224, 224],   # wrist 카메라
  "observation.state":        Tensor[B, 6],              # joint 6DoF
  "task":                     List[str],                 # 자연어 지시
}

# 출력
{
  "action": Tensor[B, chunk_size, 6]   # chunk_size × action_dim
}
```

### 7.2 전처리 체크리스트

- [ ] 이미지 `[0, 255]` → `[0.0, 1.0]` float 변환
- [ ] ImageNet mean/std 정규화
- [ ] 224×224 리사이즈 (aspect ratio 유지 후 패딩 권장)
- [ ] State MEAN_STD 정규화 (dataset stats 사용)
- [ ] Action MEAN_STD 정규화 (학습 시 loss 안정화)
- [ ] 카메라 이름 rename_map 확인 (모델 config와 일치 여부)

### 7.3 레이어 설계 참고

```
[이미지] → ViT → 이미지 토큰 (frozen 혹은 fine-tuning)
[텍스트] → LLM tokenizer → 언어 토큰
[state]  → Linear(state_dim, hidden_dim) → state 토큰
     ↓ concat
[VLM backbone] → context 표현
     ↓
[Action Head: Flow Matching or Chunking]
  - Flow Matching: noisy action → velocity field 예측 → 10 step denoising
  - Chunking:      context → chunk_size × action_dim 직접 회귀
     ↓
[Action 출력] → MEAN_STD 역정규화 → 로봇 실행
```

### 7.4 학습 시 주요 파라미터

| 파라미터 | 권장값 | 비고 |
|----------|--------|------|
| batch_size | 4~16 | VRAM에 맞게 (12GB: 4) |
| learning_rate | 1e-4 ~ 5e-5 | Adam/AdamW |
| dtype | bfloat16 | 학습 안정성 + 메모리 절약 |
| freeze_vision_encoder | False (fine-tuning 시) | VRAM 부족 시 True |
| chunk_size | 50 | 태스크 길이에 맞게 조절 |
| action normalization | MEAN_STD | IDENTITY 절대 금지 |

---

## 8. 데이터 흐름 전체 요약

```
[데이터 수집]
  └─ 텔레오퍼레이션 → LeRobot 포맷으로 저장
       ├── videos/ (카메라 영상)
       ├── data/ (state, action, timestamp parquet)
       └── meta/ (fps, stats, info)

[학습 전처리]
  ├── 이미지: 리사이즈 → 정규화
  ├── state: MEAN_STD 정규화
  ├── action: MEAN_STD 정규화
  └── text: 토크나이저

[모델 입력]
  ├── 이미지 토큰 (ViT)
  ├── 언어 토큰 (LLM tokenizer)
  └── state 토큰 (Linear proj)

[모델 출력]
  └── action chunk → MEAN_STD 역정규화 → 로봇 실행 (30~50Hz)

[평가]
  ├── Open-loop: dataset 프레임에서 직접 추론 (Ground Truth 비교)
  └── Closed-loop: 실제 로봇 or 시뮬레이터 실행 (성공률)
```

---

## 9. 참고 링크

- [LeRobot HuggingFace Hub](https://huggingface.co/lerobot) — 공개 데이터셋 목록
- [Open X-Embodiment](https://robotics-transformer-x.github.io/) — 대규모 다종 로봇 데이터
- [LeRobot 데이터 수집 가이드](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md)
- [LIBERO 벤치마크](https://libero-project.github.io/)
