#!/usr/bin/env python3
"""
Test script to verify the installation and basic functionality.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA devices: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    return True


def test_modules():
    """Test if custom modules can be imported."""
    print("\\nTesting custom modules...")
    
    # Add src to path
    script_dir = Path(__file__).parent
    src_dir = script_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    try:
        from utils.config import ConfigManager  # type: ignore
        print("✓ Configuration module")
    except ImportError as e:
        print(f"✗ Configuration module failed: {e}")
        return False
    
    try:
        from data.preprocessing import DataProcessor  # type: ignore
        print("✓ Data processing module")
    except ImportError as e:
        print(f"✗ Data processing module failed: {e}")
        return False
    
    try:
        from models.autoencoder import AutoEncoder, create_model  # type: ignore
        print("✓ Model module")
    except ImportError as e:
        print(f"✗ Model module failed: {e}")
        return False
    
    try:
        from training.trainer import AutoEncoderTrainer  # type: ignore
        print("✓ Training module")
    except ImportError as e:
        print(f"✗ Training module failed: {e}")
        return False
    
    try:
        from evaluation.metrics import AnomalyDetectionEvaluator  # type: ignore
        print("✓ Evaluation module")
    except ImportError as e:
        print(f"✗ Evaluation module failed: {e}")
        return False
    
    return True


def test_config_loading():
    """Test configuration loading."""
    print("\\nTesting configuration loading...")
    
    script_dir = Path(__file__).parent
    src_dir = script_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    try:
        from utils.config import ConfigManager  # type: ignore
        
        config_path = script_dir / 'configs' / 'baseline_config.yaml'
        if config_path.exists():
            config_manager = ConfigManager()
            config = config_manager.load_config(config_path)
            print("✓ Configuration loaded successfully")
            print(f"  Experiment: {config.experiment.name}")
            print(f"  Model type: {config.model.type}")
            return True
        else:
            print(f"✗ Config file not found: {config_path}")
            return False
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\\nTesting model creation...")
    
    script_dir = Path(__file__).parent
    src_dir = script_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    try:
        import torch
        from models.autoencoder import create_model, count_parameters  # type: ignore
        
        # Test autoencoder creation
        model = create_model(
            model_type='autoencoder',
            input_dim=41,
            hidden_dims=[32, 16],
            latent_dim=8
        )
        
        param_count = count_parameters(model)
        print(f"✓ Autoencoder created with {param_count:,} parameters")
        
        # Test forward pass
        dummy_input = torch.randn(10, 41)
        with torch.no_grad():
            output, latent = model(dummy_input)
        
        print(f"✓ Forward pass successful: {dummy_input.shape} -> {output.shape}")
        print(f"  Latent shape: {latent.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("NETWORK ANOMALY DETECTION - INSTALLATION TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test custom modules
    if not test_modules():
        all_passed = False
    
    # Test configuration
    if not test_config_loading():
        all_passed = False
    
    # Test model creation
    if not test_model_creation():
        all_passed = False
    
    print("\\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your installation is working correctly.")
        print("\\nYou can now run:")
        print("python train.py --config configs/baseline_config.yaml")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above and fix the issues.")
        print("\\nCommon solutions:")
        print("- Make sure you activated the virtual environment")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- For PyTorch CUDA issues, visit: https://pytorch.org/get-started/locally/")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())