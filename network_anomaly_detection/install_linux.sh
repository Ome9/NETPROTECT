#!/bin/bash
# Linux/Mac installation script

echo "Installing PyTorch with CUDA support for Linux/Mac..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA support (adjust version as needed)
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Verify installation
echo "Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available')"

echo ""
echo "Installation completed!"
echo "To activate the environment in the future, run: source venv/bin/activate"
echo "To start training, run: python train.py --config configs/baseline_config.yaml"