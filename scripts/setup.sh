#!/bin/bash
# ============================================
# robotics-ai-research 초기 세팅 스크립트
# ============================================
# 사용법:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
# ============================================

set -e

echo "=== Robotics AI Research Setup ==="

# 1. Git 초기화 (이미 초기화된 경우 스킵)
if [ ! -d ".git" ]; then
    git init
    echo "✅ Git initialized"
else
    echo "ℹ️  Git already initialized"
fi

# 2. Python 환경 (conda 사용 시)
if command -v conda &> /dev/null; then
    echo "Setting up conda environment..."
    conda create -n robot-ai python=3.10 -y 2>/dev/null || true
    echo "✅ Conda env 'robot-ai' ready"
    echo "   활성화: conda activate robot-ai"
    echo "   패키지 설치: pip install -r requirements.txt"
else
    echo "ℹ️  conda not found. venv로 대체:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
fi

# 3. placeholder 파일 생성 (빈 폴더 Git 추적용)
for dir in data results notebooks; do
    if [ ! -f "$dir/.gitkeep" ]; then
        touch "$dir/.gitkeep"
    fi
done
echo "✅ Directory placeholders created"

# 4. pre-commit hook (선택)
echo ""
echo "=== Setup Complete ==="
echo ""
echo "다음 단계:"
echo "  1. GitHub에서 새 레포 생성 (robotics-ai-research)"
echo "  2. 리모트 연결:"
echo "     git remote add origin git@github.com:YOUR_USERNAME/robotics-ai-research.git"
echo "  3. 첫 커밋 & 푸시:"
echo "     git add ."
echo "     git commit -m 'Initial project structure'"
echo "     git branch -M main"
echo "     git push -u origin main"
