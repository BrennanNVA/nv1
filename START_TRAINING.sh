#!/bin/bash
# Quick script to start training with proper environment setup
# Usage:
#   ./START_TRAINING.sh AAPL 3          # Single symbol
#   ./START_TRAINING.sh AAPL MSFT 5     # Multiple symbols
#   ./START_TRAINING.sh all 5           # All symbols from config (25 symbols)

set -e  # Exit on error

# Check if first argument is "all"
if [ "$1" = "all" ]; then
    TRAIN_ALL=true
    YEARS=${2:-5}
    SYMBOLS=""
    echo "📊 Training ALL symbols from config.toml"
else
    TRAIN_ALL=false
    # Find the last argument - if it's a number, it's years; otherwise default to 3
    LAST_ARG="${@: -1}"
    if [[ "$LAST_ARG" =~ ^[0-9]+$ ]]; then
        # Last argument is a number (years)
        YEARS=$LAST_ARG
        # All arguments except the last are symbols
        SYMBOLS="${@:1:$(($#-1))}"
    else
        # No years specified, use default
        YEARS=3
        # All arguments are symbols
        SYMBOLS="$@"
    fi
fi

echo "============================================================"
echo "NOVA AETUS - TRAINING LAUNCHER"
echo "============================================================"
echo ""

# Get project directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "📁 Project directory: $PROJECT_ROOT"
if [ "$TRAIN_ALL" = true ]; then
    echo "📊 Symbols: ALL (from config.toml)"
else
    if [ -z "$SYMBOLS" ]; then
        SYMBOLS="AAPL"  # Default if no symbols provided
    fi
    echo "📊 Symbols: $SYMBOLS"
fi
echo "📅 Years: $YEARS"
echo ""

# Setup PATH for RMM support
export PATH="$HOME/miniconda3/envs/nova_aetus/bin:$HOME/miniconda3/bin:$PATH"

# Initialize conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "⚠️  Warning: conda.sh not found. Make sure conda is installed."
    exit 1
fi

# Activate environment
echo "🔧 Activating conda environment..."
if conda env list | grep -q "^nova_aetus "; then
    conda activate nova_aetus
    echo "✓ Environment activated"
else
    echo "❌ Error: nova_aetus environment not found"
    echo "   Create it with: conda create -n nova_aetus python=3.12 -y"
    exit 1
fi

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $PYTHON_VERSION"

# Verify RMM (optional)
echo "🔍 Checking RMM..."
if python -c "import rmm" 2>/dev/null; then
    RMM_VERSION=$(python -c "import rmm; print(rmm.__version__)" 2>/dev/null)
    echo "✓ RMM available: version $RMM_VERSION"
else
    echo "⚠️  RMM not available (optional - training works without it)"
fi

echo ""
echo "============================================================"
echo "STARTING TRAINING"
echo "============================================================"
echo ""

# Run training
if [ "$TRAIN_ALL" = true ]; then
    python scripts/train_models.py --all --years $YEARS
else
    # Convert space-separated symbols to array for Python script
    python scripts/train_models.py --symbols $SYMBOLS --years $YEARS
fi

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "✅ Check models directory for results:"
echo "   ls -lh models/"
echo ""
