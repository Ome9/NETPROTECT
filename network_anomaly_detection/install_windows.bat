@echo off
REM Windows batch script to install PyTorch with CUDA support and other requirements

echo Installing PyTorch with CUDA support for Windows...

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\\Scripts\\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA 11.8 support (adjust version as needed)
echo Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install other requirements
echo Installing other requirements...
pip install -r requirements.txt

REM Verify installation
echo Verifying PyTorch CUDA installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available')"

echo.
echo Installation completed!
echo To activate the environment in the future, run: venv\\Scripts\\activate
echo To start training, run: python train.py --config configs/baseline_config.yaml
pause