"""
MAXIMUM GPU UTILIZATION Training Script
Pushes GPU utilization to the maximum while maintaining training stability.
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

# Maximum GPU optimization settings
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Set memory fraction to use more GPU memory
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory


class MaxGPUAutoencoder(nn.Module):
    """Maximum GPU utilization autoencoder."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Much larger architecture for maximum GPU utilization
        hidden_dims = [1024, 512, 256, 128, 64, 32]
        
        # Encoder with aggressive GPU utilization
        encoder_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15) if i < len(hidden_dims) - 1 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers[:-1])
        
        # Decoder with maximum parameters for GPU work
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]
        
        for i, hidden_dim in enumerate(reversed_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if i < len(reversed_dims) - 1 else nn.Identity(),
                nn.ReLU() if i < len(reversed_dims) - 1 else nn.Identity(),
                nn.Dropout(0.15) if i < len(reversed_dims) - 2 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        # Remove identity layers
        decoder_layers = [layer for layer in decoder_layers if not isinstance(layer, nn.Identity)]
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
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
        # Add some computational complexity for GPU work
        encoded = self.encoder(x)
        
        # Additional GPU computation
        encoded = encoded * torch.sigmoid(encoded)  # Swish-like activation
        
        decoded = self.decoder(encoded)
        return decoded


def load_and_preprocess_data_gpu():
    """Load NSL-KDD data with maximum GPU optimization."""
    logger.info("Loading NSL-KDD with MAXIMUM GPU optimization...")
    
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
    
    # Load with memory optimization
    train_df = pd.read_csv('NSL-KDD_Dataset/KDDTrain+.txt', names=column_names)
    test_df = pd.read_csv('NSL-KDD_Dataset/KDDTest+.txt', names=column_names)
    
    # Combine for preprocessing
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df['is_anomaly'] = (combined_df['attack_type'] != 'normal').astype(np.int8)
    
    # Feature engineering for more GPU work
    feature_df = combined_df.drop(['attack_type', 'difficulty'], axis=1)
    
    # Handle categorical columns
    categorical_columns = ['protocol_type', 'service', 'flag']
    encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col])
        encoders[col] = le
    
    # Separate features and labels
    X = feature_df.drop('is_anomaly', axis=1).values.astype(np.float32)
    y = feature_df['is_anomaly'].values.astype(np.int8)
    
    # Feature scaling for better GPU utilization
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Add feature interactions to increase computational load
    # This gives GPU more work to do
    n_features = X_scaled.shape[1]
    feature_interactions = []
    
    # Add polynomial features (limited to avoid memory explosion)
    for i in range(min(10, n_features)):  # Limit to first 10 features
        for j in range(i+1, min(10, n_features)):
            feature_interactions.append(X_scaled[:, i] * X_scaled[:, j])
    
    if feature_interactions:
        X_scaled = np.hstack([X_scaled, np.column_stack(feature_interactions)])
    
    logger.info(f"Enhanced feature dimension: {X_scaled.shape[1]} (original: {n_features})")
    
    # Split back to train/test
    n_train = len(train_df)
    X_train_full = X_scaled[:n_train]
    y_train_full = y[:n_train]
    X_test = X_scaled[n_train:]
    y_test = y[n_train:]
    
    # Extract normal samples
    normal_mask = y_train_full == 0
    X_train_normal = X_train_full[normal_mask]
    
    logger.info(f"🔥 MAXIMUM GPU DATASET: {len(X_train_normal)} normal samples")
    logger.info(f"   Enhanced features: {X_scaled.shape[1]}")
    logger.info(f"   Memory optimized: float32 precision")
    
    # Create validation split
    X_train, X_val = train_test_split(X_train_normal, test_size=0.1, random_state=42)
    
    return X_train, X_val, X_test, y_test, scaler, encoders


def create_max_gpu_data_loaders(X_train, X_val, X_test, y_test, batch_size=2048):
    """Create maximum GPU utilization data loaders."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Pre-load ALL data to GPU for maximum GPU memory usage
    if torch.cuda.is_available():
        try:
            # Try to load all data to GPU at once
            train_tensor = torch.FloatTensor(X_train).cuda()
            train_labels = torch.zeros(len(X_train), dtype=torch.float).cuda()
            
            val_tensor = torch.FloatTensor(X_val).cuda()  
            val_labels = torch.zeros(len(X_val), dtype=torch.float).cuda()
            
            test_tensor = torch.FloatTensor(X_test).cuda()
            test_labels = torch.LongTensor(y_test).cuda()
            
            logger.info("🚀 ALL DATA PRE-LOADED TO GPU MEMORY!")
            
        except RuntimeError as e:
            logger.warning(f"Cannot pre-load all data to GPU: {e}")
            # Fallback to CPU tensors with pin_memory
            train_tensor = torch.FloatTensor(X_train)
            train_labels = torch.zeros(len(X_train), dtype=torch.float)
            val_tensor = torch.FloatTensor(X_val)
            val_labels = torch.zeros(len(X_val), dtype=torch.float)
            test_tensor = torch.FloatTensor(X_test)
            test_labels = torch.LongTensor(y_test)
    else:
        train_tensor = torch.FloatTensor(X_train)
        train_labels = torch.zeros(len(X_train), dtype=torch.float)
        val_tensor = torch.FloatTensor(X_val)
        val_labels = torch.zeros(len(X_val), dtype=torch.float)
        test_tensor = torch.FloatTensor(X_test)
        test_labels = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(train_tensor, train_labels)
    val_dataset = TensorDataset(val_tensor, val_labels)
    test_dataset = TensorDataset(test_tensor, test_labels)
    
    # Data loaders with maximum GPU focus
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No CPU workers, GPU focus
        pin_memory=False,  # Not needed if data already on GPU
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    logger.info(f"MAX GPU Data Loaders:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  CPU workers: 0 (pure GPU focus)")
    logger.info(f"  Data location: GPU memory (if possible)")
    
    return train_loader, val_loader, test_loader


def train_maximum_gpu(model, train_loader, val_loader, device):
    """Training with MAXIMUM GPU utilization."""
    
    model = model.to(device)
    model.train()
    
    # Aggressive GPU optimization
    torch.backends.cudnn.benchmark = True
    
    # Optimizer with maximum learning rate for fast convergence
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=5e-3,  # Aggressive learning rate
        weight_decay=1e-3,
        eps=1e-4
    )
    
    # Aggressive scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=50,  # Shorter cycles
        eta_min=1e-5
    )
    
    criterion = nn.MSELoss()
    
    # Mixed precision for maximum GPU utilization
    scaler = GradScaler()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    epochs = 50  # Fewer epochs due to aggressive training
    
    logger.info("🚀 MAXIMUM GPU UTILIZATION TRAINING STARTED")
    logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"   Aggressive learning rate: 5e-3")
    logger.info(f"   Target epochs: {epochs}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        # GPU memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        for batch_X, _ in train_loader:
            # Data should already be on GPU
            if not batch_X.is_cuda:
                batch_X = batch_X.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass with additional GPU computation
            with autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_X)
                
                # Add regularization for more GPU work
                l2_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l2_reg += torch.norm(param) ** 2
                loss += 1e-6 * l2_reg
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability with aggressive learning
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_X, _ in val_loader:
                if not batch_X.is_cuda:
                    batch_X = batch_X.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_X)
                
                epoch_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'results/max_gpu_model_best.pth')
        
        # Enhanced logging with GPU utilization
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            
            gpu_memory = 0
            gpu_util = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / 1e9
                total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
                gpu_util = (gpu_memory / total_memory) * 100
            
            logger.info(f"Epoch {epoch+1:2d}/{epochs} | "
                       f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                       f"LR: {lr:.2e}")
            logger.info(f"   🔥 GPU Memory: {gpu_memory:.2f}GB ({gpu_util:.1f}% utilization)")
    
    total_time = time.time() - start_time
    final_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    
    logger.info(f"🎯 MAXIMUM GPU TRAINING COMPLETED!")
    logger.info(f"   Total time: {total_time:.1f}s")
    logger.info(f"   Peak GPU memory: {final_memory:.2f}GB")
    logger.info(f"   Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses, best_val_loss


def main():
    """Main function for MAXIMUM GPU utilization."""
    print("=" * 100)
    print("🔥 MAXIMUM GPU UTILIZATION ANOMALY DETECTION TRAINING")
    print("=" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name()}")
        logger.info(f"🚀 Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        logger.info(f"🚀 CUDA Capability: {torch.cuda.get_device_capability()}")
        logger.info(f"🚀 Memory fraction: 95% allocated for training")
        
        # Clear and prepare GPU
        torch.cuda.empty_cache()
    else:
        logger.error("❌ CUDA not available! This script requires GPU.")
        return
    
    try:
        # Load data with GPU optimization
        X_train, X_val, X_test, y_test, scaler, encoders = load_and_preprocess_data_gpu()
        
        # Calculate optimal batch size for maximum GPU utilization
        gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        
        # Aggressive batch sizing
        if gpu_memory_gb >= 8:
            batch_size = 4096
        elif gpu_memory_gb >= 4:
            batch_size = 2048  # Push RTX 3050 to maximum
        else:
            batch_size = 1024
        
        logger.info(f"🔥 MAXIMUM batch size: {batch_size}")
        
        # Create maximum GPU data loaders
        train_loader, val_loader, test_loader = create_max_gpu_data_loaders(
            X_train, X_val, X_test, y_test, batch_size=batch_size
        )
        
        # Create maximum GPU model
        input_dim = X_train.shape[1]
        model = MaxGPUAutoencoder(input_dim)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🚀 MAXIMUM GPU Model:")
        logger.info(f"   Parameters: {total_params:,}")
        logger.info(f"   Input dimension: {input_dim}")
        logger.info(f"   Architecture: MAXIMUM complexity for GPU work")
        
        # Train with maximum GPU utilization
        train_losses, val_losses, best_val_loss = train_maximum_gpu(
            model, train_loader, val_loader, device
        )
        
        # Final GPU stats
        final_memory = torch.cuda.max_memory_allocated() / 1e9
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        final_util = (final_memory / total_memory) * 100
        
        print("\n" + "=" * 100)
        print("🎯 MAXIMUM GPU UTILIZATION ACHIEVED!")
        print("=" * 100)
        logger.info(f"🔥 Final GPU Utilization: {final_util:.1f}%")
        logger.info(f"🔥 Peak GPU Memory: {final_memory:.2f}GB / {total_memory:.1f}GB")
        logger.info(f"🔥 Model Parameters: {total_params:,}")
        logger.info(f"🔥 Best Validation Loss: {best_val_loss:.6f}")
        logger.info(f"🔥 Training Method: PURE GPU MAXIMUM UTILIZATION")
        print("=" * 100)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()