"""
실험 템플릿
Usage: python experiments/template_experiment.py --config configs/example.yaml
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment template")
    parser.add_argument("--config", type=str, default="configs/example.yaml")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def setup_experiment(args):
    """실험 디렉토리 및 로깅 설정"""
    exp_name = args.exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("results") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir


def main():
    args = parse_args()
    exp_dir = setup_experiment(args)

    logger.info("=== Experiment Start ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {args.device}")

    # TODO: 여기에 실험 코드 작성
    # 1. 데이터 로드
    # 2. 모델 초기화
    # 3. 학습 루프
    # 4. 평가
    # 5. 결과 저장

    logger.info("=== Experiment Done ===")


if __name__ == "__main__":
    main()
