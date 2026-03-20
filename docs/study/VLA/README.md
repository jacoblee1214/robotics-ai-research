# VLA (Vision-Language-Action) 스터디 노트

## 핵심 개념
- Vision-Language Model + Action output → 로봇 제어
- 이미지 + 자연어 명령 → 로봇 행동 시퀀스 생성

## 주요 논문 & 모델

### 초기 모델

| 모델 | 핵심 | 아키텍처 |
|------|------|----------|
| RT-2 | VLM을 로봇 행동 생성에 직접 활용 | PaLI-X / PaLM-E 기반 |
| OpenVLA | 오픈소스 VLA | Prismatic VLM + Action head |
| Octo | Generalist robot policy | Transformer, 다양한 로봇/태스크 범용 |
| ACT | Action Chunking Transformer | bimanual manipulation 특화 |

### 현재 주요 모델 (직접 분석)

| 모델 | 크기 | Action 방식 | 분석 문서 |
|------|------|-------------|-----------|
| SmolVLA | 450M | Action Chunking (chunk=50) | [SmolVLA_Analysis.md](SmolVLA_Analysis.md) |
| GR00T N1.6 | 3B | Flow Matching | [GR00T_N1.6_Script_Analysis.md](GR00T_N1.6_Script_Analysis.md) |
| X-VLA | 0.9B | Flow Matching (anchor points) | [X-VLA_Analysis.md](X-VLA_Analysis.md) |
| π0 | PaliGemma 3B | Flow Matching | [pi0_Analysis.md](pi0_Analysis.md) |
| π0.5 | π0 기반 | FAST + Flow Matching | [pi0_Analysis.md](pi0_Analysis.md) |
| MoDE-VLA | - | Diffusion (MoDE) | - |

## Action 표현 방식

VLA 모델이 action을 어떻게 생성하는지는 성능과 태스크 적합성에 직결됨.

### Autoregressive (토큰 기반)
- **대표**: RT-2, OpenVLA
- action을 이산 토큰으로 변환 → 언어 모델처럼 순차 생성
- 장점: VLM과 자연스럽게 통합 / 단점: 연속 action 표현에 부자연스러움, 느림

### Action Chunking + Transformer (ACT 방식)
- **대표**: SmolVLA, ACT
- 한 번에 N개 action을 chunk로 예측 → open-loop 실행
- 장점: 빠른 추론, 구현 단순 / 단점: chunk 내 에러 누적, 재계획 불가

### Diffusion Policy
- **대표**: MoDE-VLA, Diffusion Policy
- 노이즈에서 action을 점진적으로 복원 (수십~수백 denoising step)
- 장점: 복잡한 multimodal 분포 표현 가능 / 단점: 추론 느림

### Flow Matching
- **대표**: X-VLA, GR00T, π0, π0.5
- Diffusion의 개선판 — 직선 경로로 노이즈 → action 변환 (더 적은 step)
- 장점: Diffusion보다 빠르고 안정적, 연속 action에 적합
- X-VLA: 30 anchor points(keyframe) 기반 / GR00T: 8~16 step chunk

### 정리

| 방식 | 속도 | 표현력 | 주요 사용처 |
|------|------|--------|-------------|
| Autoregressive | 중간 | 낮음 | 초기 VLA |
| Action Chunking | 빠름 | 중간 | Short-horizon |
| Diffusion | 느림 | 높음 | 정밀 조작 |
| Flow Matching | 빠름 | 높음 | 현재 주류 |

## 데이터셋 & 입력 파이프라인

→ [Dataset_Pipeline.md](Dataset_Pipeline.md) 별도 정리

## 실험 계획
- [ ] OpenVLA 코드 클론 및 환경 세팅
- [ ] 사전학습 모델 다운로드 및 inference 테스트
- [ ] 데이터셋 구조 분석
- [ ] Fine-tuning 실험

## Task Horizon 분류

VLA 모델이 다루는 태스크를 소요 시간 기준으로 분류.

### 분류 기준

| 분류 | 소요 시간 | 예시 | 대표 모델 |
|------|-----------|------|-----------|
| Single-step | 1~5초 | 버튼 누르기, 물건 밀기 (MetaWorld easy) | - |
| Short-horizon | 5~20초 | pick-and-place, 서랍 열기 | SmolVLA, OpenVLA |
| Medium-horizon | 20초~2분 | LIBERO-Long, 사과 깎기, 옷 접기 | π0, MoDE-VLA |
| Long-horizon | 2분~수십 분 | 주방 전체 정리 (여러 방 이동) | π0.5 |

### 시간을 결정하는 핵심 개념

#### Control Frequency (제어 주파수)

로봇 관절에 새로운 명령(각도/속도)을 보내는 속도. 단위: Hz (초당 횟수).

- **30Hz** = 1초에 30번 명령 전송 = 33ms마다 한 번
- 높을수록 움직임이 부드럽고 외란에 빠르게 반응 가능
- 상한선은 ① 하드웨어 통신 속도 + ② 모델 추론 속도 중 느린 쪽

| 모델 | 제어 주파수 | 비고 |
|------|------------|------|
| SmolVLA | 30Hz | dataset fps=30 기준 |
| GR00T N1.6 | 30Hz | dataset fps=30 기준 (추론 속도에 따라 실효 낮아질 수 있음) |
| π0 | 50Hz | 논문 명시 |
| X-VLA | keyframe 기반 | 연속 Hz 방식 아님 |

#### Action Chunk (액션 청크)

한 번의 모델 추론으로 **N개의 연속 action을 한꺼번에 예측**하는 방식.

```
[추론 1번] → [a0, a1, a2, ..., a49] 50개를 한 번에 출력
              ↓ 이걸 로봇이 순서대로 실행 (30Hz = 1.67초 동안)
[추론 2번] → [a50, a51, ..., a99] 다음 chunk
```

- **왜 쓰나**: 추론(~수백ms)이 제어주기(33ms)보다 느리기 때문에, 1번 추론 → N번 실행으로 속도 보완
- **open-loop 실행**: chunk 실행 중에는 새 관찰을 반영하지 않음 → 중간에 상황이 변해도 chunk 다 끝날 때까지 그냥 실행
- **chunk 크기 trade-off**: 클수록 추론 빈도↓(효율적), 하지만 환경 변화 반응 늦음

| 모델 | Chunk 크기 | 실행 시간 (30Hz 기준) |
|------|-----------|----------------------|
| SmolVLA | 50 actions | 약 1.7초 |
| GR00T N1.6 | 8~16 actions | 약 0.27~0.53초 |
| X-VLA | 30 anchor points | 약 4초 (keyframe 기반) |
| π0 | 50 actions | 약 1.0초 (50Hz 기준) |

> **X-VLA의 anchor point 방식**: joint angle을 매 timestep 예측하는 게 아니라,
> 궤적 상의 핵심 자세(keyframe) 30개를 예측 → 중간은 보간. 4초 치 움직임을 30개 점으로 압축 표현.

**Chunk 실행 구조**: task 소요시간 = (chunk 실행시간) × (필요 chunk 수)
chunk 실행 중 다음 chunk 추론이 병렬 진행됨 → 이론상 지연 없음.
→ Short-horizon에서는 충분하지만, Long-horizon에서는 취약점이 됨.

### Long-horizon이 어려운 핵심 이유: 메모리 부재

현재 VLA 대부분은 현재 observation만 보고 행동을 결정함.
"아까 서랍을 열었다"는 사실을 기억하지 못함 → 순차적 sub-task 처리 불가.

π0.5가 long-horizon을 다룰 수 있는 건 **hierarchical 구조** 덕분:
- High-level planner: 전체 작업 순서 계획
- Low-level controller: 개별 동작 실행

현재 대부분의 VLA는 이 구조 없이 단일 정책만 사용 → Medium-horizon(1분) 이상은 초기 단계.

### 현실과의 격차

| 일상 작업 | 소요 시간 | 현재 VLA 가능 여부 |
|-----------|-----------|-------------------|
| 사과 깎기 | 1~2분 | 겨우 시작 (MoDE-VLA) |
| 커피 만들기 | 3~5분 | 불가능 |
| 설거지 | 5~10분 | 불가능 |
| 요리 한 끼 | 30분+ | 매우 먼 미래 |

## 메모
(스터디하면서 추가)
