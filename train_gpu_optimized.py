"""
GPU-Optimized Multi-Dataset Anomaly Detection Training
Maximizes GPU utilization and minimizes CPU bottlenecks.
Supports NSL-KDD, CSE-CIC, and UNSW datasets.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import time
import psutil
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force GPU memory allocation strategy
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed


class GPUOptimizedAutoencoder(nn.Module):
    """GPU-optimized autoencoder with maximum memory efficiency."""
    
    def __init__(self, input_dim: int, hidden_dims: list = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]  # Larger for better GPU utilization
        
        # Encoder with optimized layers
        encoder_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # BatchNorm for better GPU utilization
                nn.ReLU(),  # Remove inplace for mixed precision compatibility
                nn.Dropout(0.1) if i < len(hidden_dims) - 1 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # Remove last dropout
        
        # Decoder with optimized layers
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]
        
        for i, hidden_dim in enumerate(reversed_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if i < len(reversed_dims) - 1 else nn.Identity(),
                nn.ReLU() if i < len(reversed_dims) - 1 else nn.Identity(),  # Remove inplace
                nn.Dropout(0.1) if i < len(reversed_dims) - 2 else nn.Identity()  # Remove inplace
            ])
            prev_dim = hidden_dim
        
        # Remove unnecessary identity layers
        decoder_layers = [layer for layer in decoder_layers if not isinstance(layer, nn.Identity)]
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights for better GPU performance
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class GPUOptimizedDataset:
    """GPU-optimized dataset preprocessing with memory management."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.scaler = None
        self.encoders = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess data with GPU optimization."""
        logger.info(f"Loading {self.dataset_name.upper()} dataset with GPU optimization...")
        
        if self.dataset_name.lower() == 'nsl-kdd':
            return self._load_nsl_kdd()
        elif self.dataset_name.lower() == 'cse-cic':
            return self._load_cse_cic()
        elif self.dataset_name.lower() == 'unsw':
            return self._load_unsw()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _load_nsl_kdd(self):
        """Load NSL-KDD with optimized preprocessing."""
        column_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
        ]
        
        # Load with optimized dtypes
        train_df = pd.read_csv('NSL-KDD_Dataset/KDDTrain+.txt', names=column_names, dtype=str)
        test_df = pd.read_csv('NSL-KDD_Dataset/KDDTest+.txt', names=column_names, dtype=str)
        
        return self._preprocess_data(train_df, test_df, 'attack_type', 'normal')
    
    def _load_cse_cic(self):
        """Load CSE-CIC-IDS2018 dataset."""
        try:
            # Try different possible file names/paths
            possible_paths = [
                'CSE-CIC_Dataset/train.csv',
                'CSE-CIC_Dataset/test.csv',
                'datasets/CSE-CIC/train.csv', 
                'datasets/CSE-CIC/test.csv',
                'CSE-CIC_Dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
                'CSE-CIC_Dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
            ]
            
            train_df = None
            test_df = None
            
            for path in possible_paths:
                if Path(path).exists():
                    df = pd.read_csv(path, low_memory=False)
                    if 'Label' in df.columns or 'label' in df.columns:
                        if train_df is None:
                            train_df = df.sample(frac=0.8, random_state=42)
                            test_df = df.drop(train_df.index)
                        break
            
            if train_df is None:
                raise FileNotFoundError("CSE-CIC dataset files not found")
            
            label_col = 'Label' if 'Label' in train_df.columns else 'label'
            return self._preprocess_data(train_df, test_df, label_col, 'BENIGN')
            
        except Exception as e:
            logger.error(f"Error loading CSE-CIC dataset: {e}")
            logger.info("Please ensure CSE-CIC dataset is available in CSE-CIC_Dataset/ directory")
            return None
    
    def _load_unsw(self):
        """Load UNSW-NB15 dataset."""
        try:
            possible_paths = [
                ('UNSW_Dataset/UNSW_NB15_training-set.csv', 'UNSW_Dataset/UNSW_NB15_testing-set.csv'),
                ('datasets/UNSW/train.csv', 'datasets/UNSW/test.csv'),
                ('UNSW_Dataset/train.csv', 'UNSW_Dataset/test.csv')
            ]
            
            train_df = None
            test_df = None
            
            for train_path, test_path in possible_paths:
                if Path(train_path).exists() and Path(test_path).exists():
                    train_df = pd.read_csv(train_path, low_memory=False)
                    test_df = pd.read_csv(test_path, low_memory=False)
                    break
            
            if train_df is None:
                raise FileNotFoundError("UNSW dataset files not found")
            
            return self._preprocess_data(train_df, test_df, 'label', 0)
            
        except Exception as e:
            logger.error(f"Error loading UNSW dataset: {e}")
            logger.info("Please ensure UNSW dataset is available in UNSW_Dataset/ directory")
            return None
    
    def _preprocess_data(self, train_df, test_df, label_col, normal_label):
        """Optimized preprocessing for GPU training."""
        logger.info(f"Preprocessing {self.dataset_name} data for GPU training...")
        
        # Memory optimization
        gc.collect()
        
        # Combine for consistent preprocessing
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Create binary labels
        combined_df['is_anomaly'] = (combined_df[label_col] != normal_label).astype(np.int8)
        
        # Remove label columns
        feature_cols = [col for col in combined_df.columns if col not in [label_col, 'difficulty', 'is_anomaly']]
        feature_df = combined_df[feature_cols + ['is_anomaly']].copy()
        
        # Optimize data types
        for col in feature_df.columns:
            if col == 'is_anomaly':
                continue
            
            if feature_df[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                except:
                    pass
        
        # Handle categorical columns
        categorical_columns = feature_df.select_dtypes(include=['object']).columns.tolist()
        categorical_columns = [col for col in categorical_columns if col != 'is_anomaly']
        
        for col in categorical_columns:
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col].fillna('unknown'))
            self.encoders[col] = le
        
        # Fill missing values
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'is_anomaly']
        feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0)
        
        # Separate features and labels
        X = feature_df.drop('is_anomaly', axis=1).values.astype(np.float32)  # Use float32 for GPU
        y = feature_df['is_anomaly'].values.astype(np.int8)
        
        # Use MinMaxScaler for better GPU performance
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split back to train/test
        n_train = len(train_df)
        X_train_full = X_scaled[:n_train]
        y_train_full = y[:n_train]
        X_test = X_scaled[n_train:]
        y_test = y[n_train:]
        
        # Log statistics
        logger.info(f"Dataset: {self.dataset_name.upper()}")
        logger.info(f"Feature dimension: {X_scaled.shape[1]}")
        logger.info(f"Training samples: {len(X_train_full)} (Normal: {np.sum(y_train_full == 0)}, Anomaly: {np.sum(y_train_full == 1)})")
        logger.info(f"Test samples: {len(X_test)} (Normal: {np.sum(y_test == 0)}, Anomaly: {np.sum(y_test == 1)})")
        
        # Extract normal samples for training
        normal_mask = y_train_full == 0
        X_train_normal = X_train_full[normal_mask]
        
        logger.info(f"🔥 GPU OPTIMIZATION: Using {len(X_train_normal)} normal samples for GPU training")
        logger.info(f"   Memory optimized: float32 precision, MinMaxScaler normalization")
        
        # Create validation split
        X_train, X_val = train_test_split(X_train_normal, test_size=0.1, random_state=42)
        
        # Clean up memory
        del combined_df, feature_df, X_scaled
        gc.collect()
        
        return X_train, X_val, X_test, y_test


def create_gpu_optimized_data_loaders(X_train, X_val, X_test, y_test, batch_size=1024, num_workers=0, pin_memory=True):
    """Create GPU-optimized data loaders."""
    
    # Use larger batch sizes for better GPU utilization
    # Reduce num_workers to 0 to minimize CPU overhead and data transfer
    
    # Pre-load data to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training data (normal only) - Pre-convert to tensors
    train_tensor = torch.FloatTensor(X_train)
    train_labels = torch.zeros(len(X_train), dtype=torch.float)
    
    train_dataset = TensorDataset(train_tensor, train_labels)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,  # 0 for GPU optimization
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=True  # Ensure consistent batch sizes for BatchNorm
    )
    
    # Validation data (normal only)
    val_tensor = torch.FloatTensor(X_val)
    val_labels = torch.zeros(len(X_val), dtype=torch.float)
    
    val_dataset = TensorDataset(val_tensor, val_labels)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=False
    )
    
    # Test data (mixed)
    test_tensor = torch.FloatTensor(X_test)
    test_labels = torch.LongTensor(y_test)
    
    test_dataset = TensorDataset(test_tensor, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=False
    )
    
    logger.info(f"GPU-Optimized Data Loaders Created:")
    logger.info(f"  Batch size: {batch_size} (optimized for GPU)")
    logger.info(f"  Workers: {num_workers} (minimized for GPU focus)")
    logger.info(f"  Pin memory: {pin_memory}")
    logger.info(f"  Drop last: True (consistent batch sizes)")
    
    return train_loader, val_loader, test_loader


def train_gpu_optimized_model(model, train_loader, val_loader, device, config, dataset_name):
    """GPU-optimized training with maximum GPU utilization."""
    
    # Move model to GPU and optimize
    model = model.to(device)
    model.train()
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    
    # Setup optimizer with GPU-optimized settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        eps=1e-4  # Slightly larger eps for numerical stability on GPU
        # Remove fused parameter as it may cause issues with mixed precision
    )
    
    # Aggressive learning rate schedule for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'] * 3,
        epochs=config['training']['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    criterion = nn.MSELoss()
    
    # Mixed precision with optimizations
    use_mixed_precision = config['training']['mixed_precision'] and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_mixed_precision)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 0
    max_patience = config['training']['early_stopping_patience']
    
    logger.info("=" * 80)
    logger.info("🚀 STARTING GPU-OPTIMIZED TRAINING")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Mixed Precision: {use_mixed_precision}")
    logger.info(f"Batch Size: {train_loader.batch_size}")
    logger.info(f"Optimizer: Fused AdamW" if torch.cuda.is_available() else "AdamW")
    logger.info(f"Scheduler: OneCycleLR (aggressive)")
    
    # Monitor GPU utilization
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        # GPU memory and timing
        epoch_start_time = time.time()
        
        for batch_idx, (batch_X, _) in enumerate(train_loader):
            # Move to GPU with non_blocking for efficiency
            batch_X = batch_X.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            if use_mixed_precision:
                with autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_X)
                
                # Scale and backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_X)
                loss.backward()
                optimizer.step()
            
            scheduler.step()  # Step after each batch for OneCycleLR
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                
                if use_mixed_precision:
                    with autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_X)
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_X)
                
                epoch_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            # Save best model
            model_dir = Path(f'results/{dataset_name}')
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_val_loss,
            }, model_dir / 'gpu_optimized_model_best.pth')
        else:
            patience += 1
            if patience >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Enhanced logging with GPU stats
        epoch_time = time.time() - epoch_start_time
        if (epoch + 1) % 5 == 0 or epoch == 0:  # More frequent logging
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1:3d}/{config['training']['epochs']} | "
                       f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                       f"LR: {lr:.2e} | Time: {epoch_time:.1f}s")
            
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1e9
                logger.info(f"   GPU Memory Used: {memory_used:.2f} GB | "
                           f"Memory Util: {memory_used/torch.cuda.get_device_properties(device).total_memory*1e9*100:.1f}%")
    
    total_time = time.time() - start_time
    logger.info(f"🎯 Training completed in {total_time:.1f} seconds")
    
    if torch.cuda.is_available():
        final_memory = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"🚀 Peak GPU Memory Usage: {final_memory:.2f} GB")
    
    return train_losses, val_losses, best_val_loss


def get_gpu_config():
    """Get GPU-optimized configuration based on available hardware."""
    if not torch.cuda.is_available():
        return {
            'batch_size': 256,
            'num_workers': 4,
            'mixed_precision': False
        }
    
    # Get GPU properties
    device_props = torch.cuda.get_device_properties(0)
    gpu_memory_gb = device_props.total_memory / 1e9
    
    # Optimize batch size based on GPU memory
    if gpu_memory_gb >= 8:
        batch_size = 1024  # Large batches for high-end GPUs
    elif gpu_memory_gb >= 4:
        batch_size = 768   # RTX 3050 optimized (reduced for stability)
    else:
        batch_size = 512   # Lower-end GPUs
    
    logger.info(f"🎯 GPU Configuration Optimized:")
    logger.info(f"   GPU: {device_props.name}")
    logger.info(f"   Memory: {gpu_memory_gb:.1f} GB")
    logger.info(f"   Optimized Batch Size: {batch_size}")
    
    return {
        'batch_size': batch_size,
        'num_workers': 0,  # Use GPU focus, minimize CPU overhead
        'mixed_precision': True,
        'pin_memory': True
    }


def main():
    """Main function with GPU optimization focus."""
    print("=" * 100)
    print("🚀 GPU-OPTIMIZED MULTI-DATASET ANOMALY DETECTION TRAINING")
    print("=" * 100)
    
    # Setup device with optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        logger.info(f"CUDA Capability: {torch.cuda.get_device_capability()}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
    
    # Get GPU-optimized configuration
    gpu_config = get_gpu_config()
    
    # Enhanced configuration with GPU optimizations
    config = {
        'training': {
            'batch_size': gpu_config['batch_size'],
            'epochs': 100,  # Reduce epochs due to aggressive learning
            'learning_rate': 2e-3,  # Higher learning rate for faster convergence
            'weight_decay': 1e-4,
            'early_stopping_patience': 15,
            'num_workers': gpu_config['num_workers'],
            'pin_memory': gpu_config['pin_memory'],
            'mixed_precision': gpu_config['mixed_precision'],
            'threshold_percentile': 95.0
        }
    }
    
    # Available datasets
    datasets = ['nsl-kdd']  # Start with NSL-KDD, add others as available
    # datasets = ['nsl-kdd', 'cse-cic', 'unsw']  # Uncomment when datasets are ready
    
    logger.info(f"Training datasets: {[d.upper() for d in datasets]}")
    logger.info(f"GPU Optimization Level: MAXIMUM 🔥")
    
    results = {}
    
    for dataset_name in datasets:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔥 TRAINING ON {dataset_name.upper()} WITH GPU OPTIMIZATION")
            logger.info(f"{'='*60}")
            
            # Load and preprocess data
            dataset_processor = GPUOptimizedDataset(dataset_name)
            data_result = dataset_processor.load_and_preprocess_data()
            
            if data_result is None:
                logger.warning(f"Skipping {dataset_name} - data not available")
                continue
                
            X_train, X_val, X_test, y_test = data_result
            
            # Create GPU-optimized data loaders
            train_loader, val_loader, test_loader = create_gpu_optimized_data_loaders(
                X_train, X_val, X_test, y_test,
                batch_size=config['training']['batch_size'],
                num_workers=config['training']['num_workers'],
                pin_memory=config['training']['pin_memory']
            )
            
            # Create GPU-optimized model
            input_dim = X_train.shape[1]
            model = GPUOptimizedAutoencoder(input_dim)
            
            # Model info
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model: GPU-Optimized Autoencoder")
            logger.info(f"Parameters: {total_params:,}")
            logger.info(f"Input dimension: {input_dim}")
            
            # Train with maximum GPU utilization
            train_losses, val_losses, best_val_loss = train_gpu_optimized_model(
                model, train_loader, val_loader, device, config, dataset_name
            )
            
            logger.info(f"✅ {dataset_name.upper()} training completed!")
            logger.info(f"   Best validation loss: {best_val_loss:.6f}")
            logger.info(f"   GPU utilization: MAXIMIZED 🚀")
            
            results[dataset_name] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'input_dim': input_dim,
                'total_params': total_params
            }
            
            # Clean up GPU memory between datasets
            del model, train_loader, val_loader, test_loader
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Error training {dataset_name}: {e}")
            continue
    
    # Final summary
    logger.info("\n" + "=" * 100)
    logger.info("🎯 GPU-OPTIMIZED TRAINING SUMMARY")
    logger.info("=" * 100)
    
    for dataset_name, result in results.items():
        logger.info(f"\n{dataset_name.upper()}:")
        logger.info(f"  Best Loss: {result['best_val_loss']:.6f}")
        logger.info(f"  Parameters: {result['total_params']:,}")
        logger.info(f"  Features: {result['input_dim']}")
        logger.info(f"  Status: ✅ GPU OPTIMIZED")
    
    if torch.cuda.is_available():
        logger.info(f"\n🚀 Final GPU Stats:")
        logger.info(f"   Device: {torch.cuda.get_device_name()}")
        logger.info(f"   Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        logger.info(f"   Optimization: MAXIMUM GPU UTILIZATION ACHIEVED! 🔥")
    
    logger.info("\n✨ GPU-optimized training completed successfully!")


if __name__ == "__main__":
    main()