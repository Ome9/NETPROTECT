# Network Anomaly Detection with Autoencoders

A comprehensive PyTorch implementation of autoencoder-based anomaly detection for network traffic analysis using the NSL-KDD dataset. This project implements unsupervised learning to detect network intrusions by training autoencoders on normal traffic patterns and identifying anomalies through reconstruction error analysis.

## 🚀 Features

- **Multiple Autoencoder Architectures**: Standard Autoencoder, Variational Autoencoder (VAE), and Deep Autoencoder
- **Memory Optimized**: Designed for 4GB VRAM GPU setups with efficient batching and small hidden layers
- **Comprehensive Evaluation**: Binary classification (Normal vs Attack) and multi-class attack type analysis
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Advanced Training**: Multiple optimizers (Adam, RMSProp, SGD), learning rate scheduling, and early stopping
- **Threshold Optimization**: Statistical and supervised threshold determination methods
- **Rich Visualization**: Error distribution plots, ROC curves, and confusion matrices
- **Modular Design**: Clean, extensible codebase following best practices

## 📋 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (CUDA-compatible)
- **RAM**: 8GB+ system memory recommended
- **Storage**: ~1GB for dataset and models

### Software Requirements
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- See `requirements.txt` for complete dependency list

## 🛠️ Installation

1. **Clone or create the project**:
```bash
cd network_anomaly_detection
```

2. **Create and activate virtual environment**:
```bash
# Windows
python -m venv venv
venv\\Scripts\\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install PyTorch with CUDA support**:
```bash
# For CUDA 11.8 (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. **Install other requirements**:
```bash
pip install -r requirements.txt
```

5. **Verify CUDA installation**:
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Devices: {torch.cuda.device_count()}")
```

## 📊 Dataset

This project uses the **NSL-KDD dataset**, an improved version of the KDD Cup 99 dataset for network intrusion detection.

### Dataset Structure
- **Training**: 125,973 samples with 41 features
- **Testing**: 22,544 samples with 41 features
- **Features**: Network connection features (duration, protocol, service, etc.)
- **Labels**: Binary (normal/attack) and multi-class (specific attack types)

### Attack Categories
- **DoS**: Denial of Service attacks (neptune, smurf, back, etc.)
- **Probe**: Surveillance and probing attacks (portsweep, nmap, etc.)
- **R2L**: Remote to Local attacks (warezclient, guess_passwd, etc.)
- **U2R**: User to Root attacks (buffer_overflow, rootkit, etc.)

## 🎯 Project Structure

```
network_anomaly_detection/
├── configs/                    # Configuration files
│   ├── baseline_config.yaml   # Standard autoencoder config
│   ├── deep_config.yaml      # Deep autoencoder config
│   └── vae_config.yaml       # Variational autoencoder config
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   │   ├── __init__.py
│   │   └── preprocessing.py   # Data loading and preprocessing
│   ├── models/                # Model architectures
│   │   ├── __init__.py
│   │   └── autoencoder.py     # Autoencoder implementations
│   ├── training/              # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py         # Training loop and utilities
│   ├── evaluation/            # Evaluation and metrics
│   │   ├── __init__.py
│   │   └── metrics.py         # Evaluation metrics and visualization
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       └── config.py          # Configuration management
├── experiments/               # Experiment tracking
├── results/                   # Training results and reports
├── models/                    # Saved model checkpoints
├── plots/                     # Generated visualizations
├── train.py                   # Main training script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Basic Training

Train a baseline autoencoder:

```bash
python train.py --config configs/baseline_config.yaml
```

### 2. Advanced Training Options

```bash
# Train with custom parameters
python train.py --config configs/baseline_config.yaml \\
    --epochs 150 \\
    --batch-size 128 \\
    --lr 0.0005

# Train deep autoencoder
python train.py --config configs/deep_config.yaml

# Train variational autoencoder
python train.py --config configs/vae_config.yaml

# Force CPU training
python train.py --config configs/baseline_config.yaml --device cpu
```

### 3. Configuration Customization

Create custom configurations by modifying the YAML files:

```yaml
# Example: Memory-optimized configuration
model:
  type: "autoencoder"
  hidden_dims: [24, 16, 8]  # Smaller layers
  latent_dim: 4             # Smaller bottleneck
  dropout_rate: 0.3         # Higher regularization

training:
  batch_size: 64            # Smaller batches
  learning_rate: 0.0001     # Lower learning rate
```

## 📈 Model Architectures

### 1. Standard Autoencoder
- **Architecture**: 41 → [32, 24, 16] → 8 → [16, 24, 32] → 41
- **Features**: BatchNorm, Dropout, configurable activation functions
- **Use Case**: General-purpose anomaly detection

### 2. Variational Autoencoder (VAE)
- **Architecture**: Encoder → μ, σ → Latent → Decoder
- **Features**: Probabilistic latent space, KL divergence regularization
- **Use Case**: Capturing data distribution, generating synthetic samples

### 3. Deep Autoencoder
- **Architecture**: Predefined deep structure optimized for memory
- **Features**: Enhanced regularization, optimized for 4GB VRAM
- **Use Case**: Complex pattern learning with limited memory

## 📊 Evaluation Metrics

### Binary Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Specificity**: True negatives / (True negatives + False positives)

### Threshold Determination Methods
- **Percentile-based**: 95th, 99th percentile of normal reconstruction errors
- **Statistical**: Mean + k×σ of normal errors
- **Supervised**: Optimize for F1-score, precision, or recall

### Multi-class Analysis
Per-attack type performance analysis for detailed insights into model effectiveness against different attack categories.

## 🔧 Advanced Usage

### Custom Model Architecture

```python
from src.models.autoencoder import AutoEncoder

# Create custom autoencoder
model = AutoEncoder(
    input_dim=41,
    hidden_dims=[64, 32, 16],
    latent_dim=8,
    activation='leaky_relu',
    use_batch_norm=True,
    dropout_rate=0.25
)
```

### Custom Training Loop

```python
from src.training.trainer import AutoEncoderTrainer

trainer = AutoEncoderTrainer(
    model=model,
    device=device,
    optimizer_name='adamw',
    learning_rate=0.001,
    scheduler_name='cosine'
)
```

### Custom Evaluation

```python
from src.evaluation.metrics import AnomalyDetectionEvaluator

evaluator = AnomalyDetectionEvaluator()
results = evaluator.comprehensive_evaluation(
    errors=reconstruction_errors,
    binary_labels=true_labels,
    threshold_methods=['percentile_95', 'optimal_f1']
)
```

## 📊 Results and Analysis

### Expected Performance
Based on NSL-KDD benchmarks, you can expect:

- **Binary Classification**:
  - Accuracy: 85-95%
  - ROC-AUC: 0.90-0.98
  - F1-Score: 0.80-0.92

- **Per-Attack Type**:
  - DoS attacks: High detection rate (>95%)
  - Probe attacks: Good detection rate (85-90%)
  - R2L/U2R attacks: Moderate detection rate (70-85%)

### Output Files
After training, the following files are generated:

```
results/experiment_name/
├── experiment_name_results.json      # Detailed metrics
├── experiment_name_report.md         # Human-readable report
├── experiment_name_config.yaml       # Used configuration
└── experiment_name_training.log      # Training logs

models/experiment_name/
├── experiment_name_model_final.pt    # Trained model
├── experiment_name_preprocessor.pkl  # Data preprocessor
└── experiment_name_model_epoch_*.pt  # Checkpoints

plots/experiment_name/
├── experiment_name_error_distribution.png
├── experiment_name_roc_curve.png
└── experiment_name_confusion_matrix.png
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   python train.py --config configs/baseline_config.yaml --batch-size 64
   ```

2. **Import Errors**:
   ```bash
   # Ensure you're in the project root directory
   cd network_anomaly_detection
   python train.py --config configs/baseline_config.yaml
   ```

3. **Dataset Not Found**:
   ```bash
   # Update dataset path in config or use CLI override
   python train.py --config configs/baseline_config.yaml --data-path /path/to/NSL-KDD_Dataset
   ```

### Performance Optimization

1. **Memory Optimization**:
   - Reduce batch size: `batch_size: 64` or `32`
   - Use smaller models: Reduce `hidden_dims` and `latent_dim`
   - Enable memory efficient mode in config

2. **Training Speed**:
   - Increase batch size if memory allows
   - Use `num_workers: 0` on Windows if DataLoader issues
   - Consider mixed precision training for larger models

## 🔬 Experiment Ideas

1. **Architecture Comparison**: Compare different autoencoder types
2. **Hyperparameter Tuning**: Optimize learning rate, latent dimensions
3. **Threshold Analysis**: Compare threshold determination methods
4. **Feature Analysis**: Study impact of different feature subsets
5. **Ensemble Methods**: Combine multiple autoencoders

## 📚 References

1. **NSL-KDD Dataset**: [UNB-CIC Dataset](https://www.unb.ca/cic/datasets/nsl.html)
2. **Autoencoders**: Hinton, G. E., & Salakhutdinov, R. R. (2006)
3. **Anomaly Detection**: Chandola, V., Banerjee, A., & Kumar, V. (2009)

## 🤝 Contributing

Feel free to contribute by:
- Adding new model architectures
- Implementing additional evaluation metrics
- Improving documentation
- Reporting bugs and issues

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

---

**Note**: This implementation is optimized for the NSL-KDD dataset but can be adapted for other network traffic datasets with minimal modifications to the preprocessing pipeline.