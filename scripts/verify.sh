#!/bin/bash
# Verification script - check environment is ready for training

set -e

echo "=========================================="
echo "Environment Verification Script"
echo "=========================================="

ERRORS=0

# Function to check and report
check() {
    if [ $? -eq 0 ]; then
        echo "✓ $1"
    else
        echo "✗ $1"
        ERRORS=$((ERRORS + 1))
    fi
}

echo ""
echo "[1/6] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    check "GPU available"
else
    echo "! No nvidia-smi found (CPU-only training)"
fi

echo ""
echo "[2/6] Checking CUDA..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
check "PyTorch CUDA working"

echo ""
echo "[3/6] Checking dependencies..."
uv sync --frozen 2>/dev/null
check "Dependencies installed"

echo ""
echo "[4/6] Checking teacher API..."
if [ -f configs/teacher.env ]; then
    source configs/teacher.env
    echo "Teacher API configured: ${TEACHER_API_URL:-not set}"
    check "Teacher config exists"
else
    echo "! No teacher.env found (will use mock or skip)"
fi

echo ""
echo "[5/6] Checking data directory..."
if [ -d data/raw ] && [ "$(ls -A data/raw 2>/dev/null)" ]; then
    echo "Data files: $(ls data/raw | wc -l) files"
    check "Data directory ready"
else
    echo "! No data in data/raw"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "[6/6] Checking write permissions..."
touch checkpoints/.test 2>/dev/null && rm checkpoints/.test
touch logs/.test 2>/dev/null && rm logs/.test
check "Write permissions OK"

echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed! Ready for training."
    echo "=========================================="
    exit 0
else
    echo "✗ $ERRORS check(s) failed. Please fix issues."
    echo "=========================================="
    exit 1
fi