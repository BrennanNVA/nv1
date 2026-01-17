#!/bin/bash
# Setup script for Conda + RMM + Training Environment
# This script installs Miniconda (if needed), sets up conda environment, and installs RMM

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "NOVA AETUS - CONDA + RMM SETUP"
echo "============================================================"
echo ""

# Check if conda is already installed
if command -v conda &> /dev/null; then
    echo "✓ Conda is already installed"
    conda --version
else
    echo "📦 Installing Miniconda..."

    # Download Miniconda installer
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    INSTALLER_PATH="/tmp/miniconda.sh"

    echo "Downloading Miniconda installer..."
    curl -L "$MINICONDA_URL" -o "$INSTALLER_PATH"

    echo "Installing Miniconda to ~/miniconda3..."
    bash "$INSTALLER_PATH" -b -p "$HOME/miniconda3"

    # Initialize conda for bash
    "$HOME/miniconda3/bin/conda" init bash

    # Add to PATH for current session
    export PATH="$HOME/miniconda3/bin:$PATH"

    echo "✓ Miniconda installed successfully"
    conda --version
fi

# Initialize conda (if not already done)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -d "$HOME/anaconda3" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

echo ""
echo "============================================================"
echo "SETTING UP CONDA ENVIRONMENT"
echo "============================================================"
echo ""

# Create or activate conda environment
ENV_NAME="nova_aetus"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "✓ Conda environment '$ENV_NAME' already exists"
    echo "Activating environment..."
    conda activate "$ENV_NAME"
else
    echo "📦 Creating new conda environment '$ENV_NAME' with Python 3.12..."
    conda create -n "$ENV_NAME" python=3.12 -y
    conda activate "$ENV_NAME"
    echo "✓ Environment created and activated"
fi

echo ""
echo "============================================================"
echo "INSTALLING RMM"
echo "============================================================"
echo ""

# Install RMM from rapidsai channel
echo "Installing RMM (RAPIDS Memory Manager)..."
conda install -c rapidsai -c conda-forge rmm cuda-version=12.0 -y

echo ""
echo "Verifying RMM installation..."
python3 -c "import rmm; print(f'✓ RMM version: {rmm.__version__}')" || echo "⚠️  RMM import failed"

echo ""
echo "============================================================"
echo "INSTALLING PROJECT DEPENDENCIES"
echo "============================================================"
echo ""

# Install project dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found"
fi

echo ""
echo "============================================================"
echo "VERIFICATION"
echo "============================================================"
echo ""

# Verify key packages
echo "Checking key packages..."
python3 -c "import xgboost; print(f'✓ XGBoost: {xgboost.__version__}')"
python3 -c "import polars; print(f'✓ Polars: {polars.__version__}')"
python3 -c "import rmm; print(f'✓ RMM: {rmm.__version__}')" 2>/dev/null || echo "⚠️  RMM not available (optional)"

echo ""
echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "IMPORTANT: Set PATH correctly to use Python 3.12 with RMM:"
echo "  export PATH=\"\$HOME/miniconda3/envs/$ENV_NAME/bin:\$HOME/miniconda3/bin:\$PATH\""
echo "  source ~/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify RMM:"
echo "  python -c \"import rmm; print(f'RMM: {rmm.__version__}')\""
echo ""
echo "To run training:"
echo "  python scripts/train_models.py --symbols AAPL --years 3"
echo ""
