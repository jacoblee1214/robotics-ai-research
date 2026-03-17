# Robotics AI Research

로봇 관련 AI 연구 프로젝트 — VLA, VLM, LLM 기반 로봇 지능 연구

## Project Structure

```
robotics-ai-research/
├── experiments/       # 실험 코드 및 학습 스크립트
├── models/            # 모델 구현 (custom architectures)
├── papers/            # 논문 재현 코드 (git submodule)
├── notebooks/         # Jupyter 분석 및 프로토타이핑
├── configs/           # 학습/실험 설정 파일 (YAML)
├── scripts/           # 유틸리티 스크립트
├── docs/study/        # 스터디 노트 (VLA, VLM, LLM)
├── data/              # 데이터 (대용량은 .gitignore)
└── results/           # 실험 결과 로그 및 시각화
```

## Setup

```bash
# 1. Clone
git clone git@github.com:YOUR_USERNAME/robotics-ai-research.git
cd robotics-ai-research

# 2. Python 환경
conda create -n robot-ai python=3.10 -y
conda activate robot-ai
pip install -r requirements.txt

# 3. (선택) 논문 코드 submodule 초기화
git submodule update --init --recursive
```

## Research Areas

| Area | Description | Status |
|------|-------------|--------|
| VLA  | Vision-Language-Action models for robotics | 🟡 스터디 중 |
| VLM  | Vision-Language Models | 🔴 예정 |
| LLM  | Large Language Models | 🔴 예정 |

## References & Key Papers

- [OpenVLA](https://github.com/openvla/openvla) — Open Vision-Language-Action model
- [RT-2](https://github.com/google-deepmind/rt-2) — Robotic Transformer 2
- [Octo](https://github.com/octo-models/octo) — Generalist robot policy

## Dev Environment

- Ubuntu 24.04 / macOS (M4 Pro)
- Python 3.10+
- PyTorch, Transformers, JAX (as needed)
- VSCode + Terminal
