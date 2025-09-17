"""
EXTREME GPU UTILIZATION Training Script
Pushes RTX 3050 to absolute maximum GPU memory and compute utilization.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# EXTREME GPU optimization settings
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.set_per_process_memory_fraction(0.98)  # Use 98% of GPU memory


class ExtremeGPUAutoencoder(nn.Module):
    """Extreme GPU utilization autoencoder - designed to max out RTX 3050."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # MASSIVE architecture to consume maximum GPU memory
        # Each layer designed to use significant GPU resources
        hidden_dims = [2048, 1024, 512, 256, 128, 64, 32, 16]  # Deep and wide
        
        # Build encoder with extreme GPU utilization
        encoder_modules = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Multiple parallel paths for more GPU work
            linear1 = nn.Linear(prev_dim, hidden_dim)
            linear2 = nn.Linear(prev_dim, hidden_dim) if prev_dim >= hidden_dim else nn.Linear(prev_dim, prev_dim)
            
            encoder_modules.append(nn.ModuleDict({
                'linear1': linear1,
                'linear2': linear2 if prev_dim >= hidden_dim else nn.Identity(),
                'bn1': nn.BatchNorm1d(hidden_dim),
                'bn2': nn.BatchNorm1d(hidden_dim) if prev_dim >= hidden_dim else nn.Identity(),
                'dropout': nn.Dropout(0.2),
                'activation': nn.ReLU()
            }))
            prev_dim = hidden_dim
        
        self.encoder_modules = encoder_modules
        
        # Build massive decoder
        decoder_modules = nn.ModuleList()
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]
        
        for i, hidden_dim in enumerate(reversed_dims):
            # Multiple parallel paths in decoder too
            linear1 = nn.Linear(prev_dim, hidden_dim)
            linear2 = nn.Linear(prev_dim, hidden_dim)
            
            decoder_modules.append(nn.ModuleDict({
                'linear1': linear1,
                'linear2': linear2,
                'bn1': nn.BatchNorm1d(hidden_dim) if i < len(reversed_dims) - 1 else nn.Identity(),
                'bn2': nn.BatchNorm1d(hidden_dim) if i < len(reversed_dims) - 1 else nn.Identity(),
                'dropout': nn.Dropout(0.2) if i < len(reversed_dims) - 2 else nn.Identity(),
                'activation': nn.ReLU() if i < len(reversed_dims) - 1 else nn.Identity()
            }))
            prev_dim = hidden_dim
        
        self.decoder_modules = decoder_modules
        
        # Additional GPU-intensive layers for maximum utilization
        self.attention = nn.MultiheadAttention(hidden_dims[0], num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dims[0])
        
        # Initialize for maximum GPU work
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder with extreme GPU utilization
        h = x
        
        for i, module in enumerate(self.encoder_modules):
            # Dual-path processing for more GPU work
            h1 = module['linear1'](h)
            
            if not isinstance(module['linear2'], nn.Identity):
                h2 = module['linear2'](h)
                h = (h1 + h2) / 2  # Combine paths
            else:
                h = h1
            
            if not isinstance(module['bn1'], nn.Identity):
                h = module['bn1'](h)
            
            h = module['activation'](h)
            h = module['dropout'](h)
            
            # Add attention mechanism at the first layer for extra GPU work
            if i == 0:
                h_att = h.unsqueeze(1)  # Add sequence dimension
                h_att, _ = self.attention(h_att, h_att, h_att)
                h_att = self.layer_norm(h_att)
                h = h + h_att.squeeze(1)  # Residual connection
        
        # Store encoded representation
        encoded = h
        
        # Decoder with extreme GPU utilization
        for i, module in enumerate(self.decoder_modules):
            # Dual-path processing in decoder
            h1 = module['linear1'](h)
            h2 = module['linear2'](h)
            h = (h1 + h2) / 2  # Combine paths
            
            if not isinstance(module['bn1'], nn.Identity):
                h = module['bn1'](h)
            
            if not isinstance(module['activation'], nn.Identity):
                h = module['activation'](h)
            
            if not isinstance(module['dropout'], nn.Identity):
                h = module['dropout'](h)
        
        # Additional GPU computation for maximum utilization
        reconstructed = h
        
        # Apply some complex operations to increase GPU workload
        reconstructed = reconstructed * torch.sigmoid(reconstructed)  # Swish-like
        reconstructed = reconstructed + 0.1 * torch.tanh(reconstructed)  # Mixed activations
        
        return reconstructed


def create_extreme_gpu_data():
    """Create data with extreme GPU memory utilization."""
    logger.info("Creating EXTREME GPU dataset...")
    
    # Load NSL-KDD
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
    
    train_df = pd.read_csv('NSL-KDD_Dataset/KDDTrain+.txt', names=column_names)
    test_df = pd.read_csv('NSL-KDD_Dataset/KDDTest+.txt', names=column_names)
    
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df['is_anomaly'] = (combined_df['attack_type'] != 'normal').astype(np.int8)
    
    feature_df = combined_df.drop(['attack_type', 'difficulty'], axis=1)
    
    # Handle categorical
    categorical_columns = ['protocol_type', 'service', 'flag']
    for col in categorical_columns:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col])
    
    X = feature_df.drop('is_anomaly', axis=1).values.astype(np.float32)
    y = feature_df['is_anomaly'].values.astype(np.int8)
    
    # Extreme feature engineering for maximum GPU memory usage
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create MASSIVE feature space for maximum GPU memory utilization
    logger.info("Creating extreme feature space for maximum GPU memory usage...")
    
    original_features = X_scaled.shape[1]
    
    # Polynomial features (more aggressive)
    poly_features = []
    for i in range(min(20, original_features)):
        for j in range(i+1, min(20, original_features)):
            poly_features.append(X_scaled[:, i] * X_scaled[:, j])
            # Add cubic interactions for even more features
            for k in range(j+1, min(10, original_features)):
                poly_features.append(X_scaled[:, i] * X_scaled[:, j] * X_scaled[:, k])
    
    # Trigonometric features for more complexity
    trig_features = []
    for i in range(min(15, original_features)):
        trig_features.append(np.sin(X_scaled[:, i] * np.pi))
        trig_features.append(np.cos(X_scaled[:, i] * np.pi))
        trig_features.append(np.tanh(X_scaled[:, i]))
    
    # Exponential features
    exp_features = []
    for i in range(min(10, original_features)):
        exp_features.append(np.exp(-X_scaled[:, i]))
        exp_features.append(np.log(X_scaled[:, i] + 1e-8))
    
    # Combine all features
    all_features = [X_scaled]
    if poly_features:
        all_features.append(np.column_stack(poly_features))
    if trig_features:
        all_features.append(np.column_stack(trig_features))
    if exp_features:
        all_features.append(np.column_stack(exp_features))
    
    X_extreme = np.hstack(all_features)
    
    logger.info(f"EXTREME feature expansion: {original_features} → {X_extreme.shape[1]} features")
    logger.info(f"Memory increase: {X_extreme.nbytes / X_scaled.nbytes:.1f}x")
    
    # Split data
    n_train = len(train_df)
    X_train_full = X_extreme[:n_train]
    y_train_full = y[:n_train]
    X_test = X_extreme[n_train:]
    y_test = y[n_train:]
    
    # Normal samples only
    normal_mask = y_train_full == 0
    X_train_normal = X_train_full[normal_mask]
    
    # Create validation split
    X_train, X_val = train_test_split(X_train_normal, test_size=0.1, random_state=42)
    
    logger.info(f"🔥 EXTREME GPU Dataset Ready:")
    logger.info(f"   Features: {X_extreme.shape[1]:,}")
    logger.info(f"   Training samples: {len(X_train):,}")
    logger.info(f"   Memory per sample: {X_extreme.shape[1] * 4} bytes")
    
    return X_train, X_val, X_test, y_test


def train_extreme_gpu():
    """Training with EXTREME GPU utilization."""
    device = torch.device('cuda')
    
    logger.info("🚀 EXTREME GPU TRAINING INITIATED")
    logger.info(f"   GPU: {torch.cuda.get_device_name()}")
    logger.info(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}GB")
    
    # Create extreme dataset
    X_train, X_val, X_test, y_test = create_extreme_gpu_data()
    
    # Calculate extreme batch size to max out GPU memory
    feature_size = X_train.shape[1]
    sample_memory = feature_size * 4  # float32
    
    # Target using 90% of GPU memory for data
    gpu_memory = torch.cuda.get_device_properties(device).total_memory * 0.9
    model_memory_estimate = 1e9  # Estimate 1GB for model
    available_for_data = gpu_memory - model_memory_estimate
    
    max_batch_size = int(available_for_data / sample_memory)
    batch_size = min(max_batch_size, 8192)  # Cap at reasonable maximum
    
    logger.info(f"🔥 EXTREME batch size calculation:")
    logger.info(f"   Feature dimension: {feature_size:,}")
    logger.info(f"   Memory per sample: {sample_memory:,} bytes")
    logger.info(f"   Calculated max batch size: {max_batch_size:,}")
    logger.info(f"   Using batch size: {batch_size:,}")
    
    try:
        # Pre-load as much data to GPU as possible
        device_train = torch.FloatTensor(X_train).cuda()
        device_val = torch.FloatTensor(X_val).cuda()
        
        logger.info("🚀 Training data pre-loaded to GPU memory!")
        
        train_dataset = TensorDataset(device_train, torch.zeros(len(device_train), device=device))
        val_dataset = TensorDataset(device_val, torch.zeros(len(device_val), device=device))
        
    except RuntimeError as e:
        logger.warning(f"Cannot pre-load all data to GPU: {e}")
        logger.info("Falling back to CPU tensors with pin_memory")
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.zeros(len(X_train)))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.zeros(len(X_val)))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    # Create EXTREME model
    model = ExtremeGPUAutoencoder(feature_size).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"🔥 EXTREME GPU Model:")
    logger.info(f"   Parameters: {total_params:,}")
    logger.info(f"   Model size estimate: {total_params * 4 / 1e6:.1f}MB")
    
    # Extreme optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    # Training loop
    epochs = 30
    best_val_loss = float('inf')
    
    logger.info("🚀 Starting EXTREME GPU training...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Monitor GPU memory at start of epoch
        torch.cuda.reset_peak_memory_stats()
        
        for batch_idx, (batch_x, _) in enumerate(train_loader):
            if not batch_x.is_cuda:
                batch_x = batch_x.cuda(non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                output = model(batch_x)
                loss = criterion(output, batch_x)
                
                # Add extra GPU computation
                reg_loss = sum(torch.norm(p)**2 for p in model.parameters()) * 1e-6
                loss += reg_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Report GPU usage mid-training
            if batch_idx == 0:
                current_memory = torch.cuda.memory_allocated() / 1e9
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
                total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
                utilization = peak_memory / total_memory * 100
                
                logger.info(f"   Epoch {epoch+1} - Current GPU: {current_memory:.2f}GB, Peak: {peak_memory:.2f}GB ({utilization:.1f}%)")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                if not batch_x.is_cuda:
                    batch_x = batch_x.cuda(non_blocking=True)
                
                with autocast():
                    output = model(batch_x)
                    loss = criterion(output, batch_x)
                val_loss += loss.item()
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'results/extreme_gpu_model.pth')
        
        # Comprehensive GPU stats every 5 epochs
        if (epoch + 1) % 5 == 0:
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
            utilization = peak_memory / total_memory * 100
            
            logger.info(f"Epoch {epoch+1:2d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
            logger.info(f"   🔥🔥🔥 EXTREME GPU STATS:")
            logger.info(f"   Memory: {peak_memory:.2f}GB / {total_memory:.1f}GB")
            logger.info(f"   Utilization: {utilization:.1f}%")
            logger.info(f"   Batch size: {batch_size:,}")
            logger.info(f"   Model params: {total_params:,}")
    
    # Final extreme GPU stats
    final_memory = torch.cuda.max_memory_allocated() / 1e9
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
    final_utilization = final_memory / total_memory * 100
    
    print("\n" + "🔥" * 100)
    print("🔥🔥🔥 EXTREME GPU UTILIZATION ACHIEVED! 🔥🔥🔥")
    print("🔥" * 100)
    logger.info(f"🔥 Final GPU Memory: {final_memory:.2f}GB / {total_memory:.1f}GB")
    logger.info(f"🔥 Final GPU Utilization: {final_utilization:.1f}%")
    logger.info(f"🔥 Model Parameters: {total_params:,}")
    logger.info(f"🔥 Feature Dimension: {feature_size:,}")
    logger.info(f"🔥 Batch Size: {batch_size:,}")
    logger.info(f"🔥 Best Validation Loss: {best_val_loss:.6f}")
    print("🔥" * 100)


def main():
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! This script requires GPU.")
        return
    
    try:
        train_extreme_gpu()
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()