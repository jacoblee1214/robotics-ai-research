# VLA (Vision-Language-Action) 스터디 노트

## 핵심 개념
- Vision-Language Model + Action output → 로봇 제어
- 이미지 + 자연어 명령 → 로봇 행동 시퀀스 생성

## 주요 논문 & 모델

### RT-2 (Robotic Transformer 2)
- **핵심**: VLM을 로봇 행동 생성에 직접 활용
- **아키텍처**: PaLI-X / PaLM-E 기반
- **링크**: https://arxiv.org/abs/2307.15818

### OpenVLA
- **핵심**: 오픈소스 VLA 모델
- **아키텍처**: Prismatic VLM + Action head
- **링크**: https://github.com/openvla/openvla

### Octo
- **핵심**: Generalist robot policy
- **특징**: 다양한 로봇/태스크에 범용 적용
- **링크**: https://github.com/octo-models/octo

## 실험 계획
- [ ] OpenVLA 코드 클론 및 환경 세팅
- [ ] 사전학습 모델 다운로드 및 inference 테스트
- [ ] 데이터셋 구조 분석
- [ ] Fine-tuning 실험

## 메모
(스터디하면서 추가)
