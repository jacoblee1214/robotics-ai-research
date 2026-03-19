# π0 / π0.5 분석

**논문**: π0: A Vision-Language-Action Flow Model for General Robot Control
**기관**: Physical Intelligence (π)
**arxiv**: 2410.24164 (π0), 2504.16054 (π0.5)

---

## 1. 핵심 아이디어

Pre-trained VLM(PaliGemma) 위에 Flow Matching 기반 Action Expert를 붙여,
인터넷 규모의 의미 지식을 그대로 활용하면서 **다양한 로봇 플랫폼의 대규모 데이터로 사전학습** 후
태스크별 파인튜닝으로 고도의 조작 작업을 수행하는 범용 로봇 파운데이션 모델.

---

## 2. 아키텍처

### VLM Backbone: PaliGemma (3B)

- 입력: 여러 RGB 카메라(로봇당 2~3개) + 자연어 명령 + 관절각도(고유감각)
- 이미지를 언어 토큰과 동일한 임베딩 공간에 late fusion으로 통합
- **Blockwise Causal Attention Mask**: 이미지/언어 토큰이 로봇 특이적 정보에 주의를 기울이지 못하게 차단 → 분포 이동 방지

### Action Expert (300M 추가)

- Mixture-of-Experts 설계:
  - 이미지/언어 토큰 → 메인 VLM 백본
  - 로봇 상태/행동 → 별도 Action Expert 모듈
- 입력: 선형 프로젝션된 상태 벡터 + 노이즈 섞인 액션 시퀀스 + Flow 타임스텝 임베딩
- 출력: 미래 50스텝 action chunk (연속값, 이산 토큰 아님)

```
[Camera Images] [Language] [Robot State]
      ↓               ↓           ↓
  VLM Backbone    VLM Backbone  Action Expert
        ↘              ↓           ↙
         [Cross-Attention via Blockwise Mask]
                       ↓
              Flow Matching Denoising
                       ↓
              Action Chunk (50 steps)
```

---

## 3. Flow Matching 사용 방식

이산 토큰으로 action을 예측하는 기존 VLA(RT-2, OpenVLA)와 달리,
**연속 행동 분포를 직접 모델링**.

### 학습

- 깨끗한 action에 가우시안 노이즈를 섞어 노이즈 action 생성
- 모델은 노이즈를 제거하는 벡터 필드(denoising vector) 예측
- 손실: 예측 벡터 vs 실제 denoising 타겟 간 차이 최소화
- **고노이즈 구간을 강조하는 커스텀 타임스텝 샘플링** 사용
  (이유: 행동 예측은 이미지 생성과 달리 고노이즈 구간이 더 중요)

### 추론

- 순수 노이즈에서 시작 → **10번의 적분 스텝**으로 유효한 action 생성
- 최대 50Hz 고주파 제어 가능
- 멀티모달 행동 분포 표현 가능 (같은 상황에서 여러 유효한 행동)

---

## 4. 학습 방식

### Pre-Training

- **10,000시간 이상** 데이터 혼합
  - 자체 수집 dexterous 조작 데이터 9억 타임스텝 (단팔 + 쌍팔)
  - 오픈소스: Open X-Embodiment, Bridge v2, DROID
- 7가지 로봇 구성, 68개 태스크 cross-embodiment 학습
- 데이터 불균형 보정: 멱함수 가중치(n^0.43)로 다운샘플
- 액션 공간: 가장 큰 로봇 기준(18차원)으로 zero-padding 통일

### Fine-Tuning (Post-Training)

- 태스크별 고품질 데이터로 정제 (태스크 복잡도에 따라 5~100시간)
- 핵심 통찰: **사전학습 데이터는 실수에서 회복하는 법을, 파인튜닝 데이터는 태스크를 잘 수행하는 법을 가르친다**
- 복잡한 다단계 태스크: 별도 VLM이 중간 언어 명령 생성 → 계층적 접근

---

## 5. 주요 실험 결과

- **Zero-shot**: 사전학습만으로 OpenVLA(7B), Octo(93M) 대비 압도적 성능
- **파인튜닝 효율**: 처음부터 학습하는 모델 대비 최대 2배 성능, 소량 데이터 상황에서 격차 더 큼
- **복잡 다단계**: 세탁물 접기, 모바일 세탁, 테이블 정리, 박스 조립 등에서 50%+ 성공률
  → "엔드투엔드 로봇 학습 문헌에서 가장 긴 dexterous 태스크"

---

## 6. π0 vs π0.5 비교

| 구분 | π0 | π0.5 |
|------|-----|-------|
| 목적 | 다양한 조작 태스크 범용 학습 | 미지 가정환경에서 장기 일반화 |
| Action 표현 | Flow Matching | 사전학습: FAST(이산토큰), 파인튜닝: Flow Matching 추가 |
| 데이터 소스 | 로봇 데이터 중심 | 로봇 + 웹 데이터 + 의미 주석 + VQA 이질적 혼합 |
| 태스크 범위 | 통제 환경 단일 태스크 | 미지 가정환경 10~15분 장기 태스크 |
| 고수준 추론 | 별도 VLM 계층 정책 | 단일 모델 내 서브태스크 예측 + 액션 생성 동시 수행 |
| 추론 구조 | 단일 스테이지 | 이중 스테이지: 서브태스크 예측 → 액션 생성 |
| Attention | Causal | 이미지/텍스트에 양방향 어텐션 |
| Co-training | 미적용 | 5가지 데이터 범주 동시 학습 |

### π0.5 5가지 학습 데이터 범주 (Co-training)

| 범주 | 내용 |
|------|------|
| MM (Mobile Manipulator) | 약 100개 실제 가정환경, 모바일 로봇 ~400시간 |
| ME (Multi-Environment) | 다양한 가정환경, 비모바일 팔 |
| CE (Cross-Embodiment Lab) | 실험실 시연 + 오픈소스 데이터 |
| HL (High-Level) | 수동 의미 레이블 (서브태스크 분해 + 바운딩 박스) |
| WD (Web Data) | 이미지 캡셔닝, VQA, 객체 로컬라이제이션 |

### π0.5 주요 결과

- 미지 가정환경 3곳에서 주방/침실 장기 청소 태스크 성공 (학습 중 해당 환경 미노출)
- 학습 환경 수 증가(3→104곳)에 따라 성능 단조 증가
- **학습된 서브태스크 분해**가 인간 전문가 수동 선택보다 우수
- GPT-4 제로샷 고수준 정책 대비 크게 우수 → 로봇 특화 적응의 중요성

---

## 7. 다른 모델과 비교

| 모델 | Backbone | Action 방식 | 사전학습 규모 | 특이점 |
|------|----------|-------------|--------------|--------|
| π0 | PaliGemma 3B | Flow Matching | 10,000h+ | Action Expert 분리 |
| π0.5 | π0 기반 | FAST + Flow Matching | + 웹 데이터 | Co-training, 장기 태스크 |
| SmolVLA | SmolVLM 500M | Action Chunking | 소규모 | 단일 GPU 학습 가능 |
| X-VLA | Florence-2 0.9B | Flow Matching | 중규모 | anchor points 방식 |
| GR00T N1.6 | Eagle2 3B | Flow Matching | 대규모 | 인간형 로봇 특화 |

---

## 8. 코드 분석 (openpi)

레포: `~/openpi` — JAX(공식) + PyTorch 버전 모두 존재.

### Flow Matching 실제 구현 (`pi0_pytorch.py`)

**학습 시 noisy action 생성 (forward, 326-328번 줄):**
```python
time = Beta(1.5, 1.0).sample() * 0.999 + 0.001  # 고노이즈 구간 편향
x_t = time * noise + (1 - time) * actions        # 노이즈 섞기
u_t = noise - actions                             # 타겟 벡터
loss = MSE(predicted_v_t, u_t)
```

**추론 시 10번 denoising (sample_actions, 401-418번 줄):**
```python
dt = -1.0 / num_steps  # t: 1.0 → 0.0 (노이즈 → 클린 action)
x_t = noise
while time >= 0:
    v_t = denoise_step(x_t, time)  # 벡터 예측
    x_t = x_t + dt * v_t          # Euler step
    time += dt
```

### Attention 구조

`embed_prefix` (이미지+언어): `att_masks = [0, 0, ..., 0]` → 서로 다 봄 (양방향)
`embed_suffix` (상태+액션): `att_masks = [1, 0, ..., 0]` → prefix는 suffix 못 봄

이게 Blockwise Causal Attention의 실제 구현.

### KV Cache 최적화 (추론 시)

```python
# prefix (이미지+언어) 1번만 인코딩 → KV cache 저장
past_key_values = forward(prefix_embs, use_cache=True)

# 10번 denoising에서 cached KV 재사용 → prefix 재계산 안 함
for step in range(10):
    v_t = denoise_step(..., past_key_values=past_key_values)
```

### π0 vs π0.5 코드 차이

| | π0 | π0.5 |
|-|----|----|
| 상태 임베딩 | `state_proj` (별도 state 토큰) | 없음 |
| 시간 임베딩 | `action_time_mlp` (action과 concat) | `time_mlp` (adaRMS conditioning) |
| `pi05` 플래그 | `False` | `True` |

### PyTorch 버전 사용 시 주의사항

1. **`transformers_replace` 필수**: 수정된 transformers를 직접 덮어씌워야 함
   ```bash
   uv pip install transformers==4.53.2
   cp -r ./src/openpi/models_pytorch/transformers_replace/* \
     .venv/lib/python3.11/site-packages/transformers/
   ```
2. **JAX 의존성 잔존**: `train_pytorch.py`가 데이터 로딩에 JAX import (34번 줄) → JAX 설치 필요
3. **action 차원 하드코딩**: `action_in_proj = nn.Linear(32, ...)` — 다른 로봇 사용 시 수정 필요
4. **`torch.compile` 자동 적용**: `sample_actions`에 `mode="max-autotune"` → 첫 실행 시 컴파일 시간 필요
