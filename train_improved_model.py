"""
Train improved autoencoder with proper anomaly detection methodology.
This script addresses all issues identified in the comprehensive analysis.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedAutoencoder(nn.Module):
    """Improved autoencoder with proper architecture."""
    
    def __init__(self, input_dim: int, hidden_dims: list = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32, 16]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Remove last dropout
        encoder_layers = encoder_layers[:-1]
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]
        
        for i, hidden_dim in enumerate(reversed_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(reversed_dims) - 1:  # Don't add ReLU and Dropout to last layer
                decoder_layers.extend([
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VariationalAutoencoder(nn.Module):
    """Variational autoencoder implementation."""
    
    def __init__(self, input_dim: int, hidden_dims: list = None, latent_dim: int = 8):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims)) + [input_dim]
        prev_dim = latent_dim
        
        for i, hidden_dim in enumerate(reversed_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(reversed_dims) - 1:
                decoder_layers.extend([
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def load_and_preprocess_data():
    """Load and preprocess NSL-KDD data with proper methodology."""
    logger.info("Loading NSL-KDD dataset...")
    
    # Column names for NSL-KDD
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
    
    # Load data
    train_df = pd.read_csv('NSL-KDD_Dataset/KDDTrain+.txt', names=column_names)
    test_df = pd.read_csv('NSL-KDD_Dataset/KDDTest+.txt', names=column_names)
    
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Combine for preprocessing
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Create binary labels (normal vs anomaly)
    combined_df['is_anomaly'] = (combined_df['attack_type'] != 'normal').astype(int)
    
    # Remove unnecessary columns
    feature_df = combined_df.drop(['attack_type', 'difficulty'], axis=1)
    
    # Encode categorical variables
    categorical_columns = ['protocol_type', 'service', 'flag']
    encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col])
        encoders[col] = le
    
    # Separate features and labels
    X = feature_df.drop('is_anomaly', axis=1).values
    y = feature_df['is_anomaly'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split back to train/test
    n_train = len(train_df)
    X_train_full = X_scaled[:n_train]
    y_train_full = y[:n_train]
    X_test = X_scaled[n_train:]
    y_test = y[n_train:]
    
    logger.info(f"Feature dimension: {X_scaled.shape[1]}")
    logger.info(f"Training samples: {len(X_train_full)} (Normal: {np.sum(y_train_full == 0)}, Anomaly: {np.sum(y_train_full == 1)})")
    logger.info(f"Test samples: {len(X_test)} (Normal: {np.sum(y_test == 0)}, Anomaly: {np.sum(y_test == 1)})")
    
    # CRITICAL: Extract only normal samples for training
    normal_mask = y_train_full == 0
    X_train_normal = X_train_full[normal_mask]
    
    logger.info(f"🔥 CRITICAL IMPROVEMENT: Using ONLY {len(X_train_normal)} normal samples for training")
    logger.info(f"   Excluded {np.sum(~normal_mask)} anomaly samples from training (proper methodology)")
    
    # Create validation split from normal training data
    X_train, X_val = train_test_split(X_train_normal, test_size=0.1, random_state=42)
    
    return (X_train, X_val, X_test, y_test, scaler, encoders, X_scaled.shape[1])


def create_data_loaders(X_train, X_val, X_test, y_test, batch_size=512, num_workers=4, pin_memory=True):
    """Create data loaders with proper configurations."""
    # Training data (normal only)
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.zeros(len(X_train))  # Dummy labels for normal data
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    # Validation data (normal only)
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.zeros(len(X_val))  # Dummy labels for normal data
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    # Test data (mixed normal and anomaly)
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def create_model(model_type: str, input_dim: int, config: dict) -> nn.Module:
    """Create model based on configuration."""
    if model_type.lower() == 'vae':
        model = VariationalAutoencoder(
            input_dim=input_dim,
            hidden_dims=config['model']['hidden_dims'],
            latent_dim=config['model']['latent_dim']
        )
    else:
        model = ImprovedAutoencoder(
            input_dim=input_dim,
            hidden_dims=config['model']['hidden_dims']
        )
    
    return model


def train_model(model, train_loader, val_loader, device, config):
    """Train the improved model."""
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs'], 
        eta_min=config['training']['learning_rate'] * 0.01
    )
    criterion = nn.MSELoss()
    
    # Mixed precision training
    use_mixed_precision = config['training']['mixed_precision']
    if use_mixed_precision:
        scaler = GradScaler()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 0
    max_patience = config['training']['early_stopping_patience']
    
    logger.info("=" * 60)
    logger.info("STARTING IMPROVED TRAINING")
    logger.info("Using ONLY normal samples for training")
    logger.info("=" * 60)
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        for batch_X, _ in train_loader:  # Ignore labels, use only normal data
            batch_X = batch_X.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if use_mixed_precision:
                with autocast():
                    if isinstance(model, VariationalAutoencoder):
                        outputs, mu, logvar = model(batch_X)
                        recon_loss = criterion(outputs, batch_X)
                        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + 0.001 * kld_loss  # Beta-VAE with small beta
                    else:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_X)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if isinstance(model, VariationalAutoencoder):
                    outputs, mu, logvar = model(batch_X)
                    recon_loss = criterion(outputs, batch_X)
                    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + 0.001 * kld_loss
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_X)
                
                loss.backward()
                optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_X, _ in val_loader:  # Ignore labels, use only normal data
                batch_X = batch_X.to(device, non_blocking=True)
                
                if use_mixed_precision:
                    with autocast():
                        if isinstance(model, VariationalAutoencoder):
                            outputs, mu, logvar = model(batch_X)
                            recon_loss = criterion(outputs, batch_X)
                            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                            loss = recon_loss + 0.001 * kld_loss
                        else:
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_X)
                else:
                    if isinstance(model, VariationalAutoencoder):
                        outputs, mu, logvar = model(batch_X)
                        recon_loss = criterion(outputs, batch_X)
                        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + 0.001 * kld_loss
                    else:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_X)
                
                epoch_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        scheduler.step()
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            # Save best model
            Path('results').mkdir(exist_ok=True)
            torch.save(model.state_dict(), 'results/improved_model_best.pth')
        else:
            patience += 1
            if patience >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1:3d}/{config['training']['epochs']} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {lr:.2e}")
    
    return train_losses, val_losses, best_val_loss


def calibrate_threshold(model, train_loader, device, percentile=95):
    """Calibrate threshold using normal training data."""
    logger.info("=" * 60)
    logger.info("CALIBRATING THRESHOLD USING NORMAL DATA")
    logger.info("=" * 60)
    
    model.eval()
    normal_errors = []
    
    with torch.no_grad():
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            
            with autocast():
                if isinstance(model, VariationalAutoencoder):
                    outputs, _, _ = model(batch_X)
                else:
                    outputs = model(batch_X)
                errors = torch.mean((outputs - batch_X) ** 2, dim=1)
            
            normal_errors.extend(errors.cpu().numpy())
    
    normal_errors = np.array(normal_errors)
    threshold = np.percentile(normal_errors, percentile)
    
    # Statistics
    stats = {
        'mean': np.mean(normal_errors),
        'std': np.std(normal_errors),
        'median': np.median(normal_errors),
        'p95': np.percentile(normal_errors, 95),
        'p99': np.percentile(normal_errors, 99),
        'threshold': threshold
    }
    
    logger.info(f"Normal error statistics:")
    logger.info(f"  Count: {len(normal_errors):,}")
    logger.info(f"  Mean: {stats['mean']:.6f}")
    logger.info(f"  Std: {stats['std']:.6f}")
    logger.info(f"  Median: {stats['median']:.6f}")
    logger.info(f"  95th percentile: {stats['p95']:.6f}")
    logger.info(f"  99th percentile: {stats['p99']:.6f}")
    logger.info(f"Optimal threshold ({percentile}th percentile): {threshold:.6f}")
    
    return threshold, normal_errors, stats


def evaluate_model(model, test_loader, device, threshold):
    """Comprehensive evaluation."""
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE EVALUATION")
    logger.info("=" * 60)
    
    model.eval()
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            
            with autocast():
                if isinstance(model, VariationalAutoencoder):
                    outputs, _, _ = model(batch_X)
                else:
                    outputs = model(batch_X)
                errors = torch.mean((outputs - batch_X) ** 2, dim=1)
            
            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    
    # Predictions
    predictions = (all_errors > threshold).astype(int)
    
    # Metrics
    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    f1 = f1_score(all_labels, predictions, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_errors)
    avg_precision = average_precision_score(all_labels, all_errors)
    cm = confusion_matrix(all_labels, predictions)
    
    # Statistics
    normal_count = np.sum(all_labels == 0)
    anomaly_count = np.sum(all_labels == 1)
    
    results = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'confusion_matrix': cm,
        'normal_samples': normal_count,
        'anomaly_samples': anomaly_count,
        'errors': all_errors,
        'labels': all_labels
    }
    
    # Logging
    logger.info(f"Test Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  Average Precision: {avg_precision:.4f}")
    logger.info(f"  Normal samples: {normal_count:,}")
    logger.info(f"  Anomaly samples: {anomaly_count:,}")
    
    return results


def main():
    """Main training function with improved methodology."""
    print("=" * 80)
    print("IMPROVED ANOMALY DETECTION TRAINING")
    print("Implementing proper methodology to fix identified issues")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Create improved GPU-optimized configuration
    config = {
        'data': {
            'train_path': 'NSL-KDD_Dataset/KDDTrain+.txt',
            'test_path': 'NSL-KDD_Dataset/KDDTest+.txt'
        },
        'model': {
            'type': 'autoencoder',  # or 'vae'
            'hidden_dims': [256, 128, 64, 32, 16],
            'latent_dim': 8,
            'dropout_rate': 0.1
        },
        'training': {
            'batch_size': 512,  # Optimized for RTX 3050
            'epochs': 150,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'validation_split': 0.1,
            'early_stopping_patience': 20,
            'num_workers': 4,
            'pin_memory': True,
            'mixed_precision': True,
            'threshold_percentile': 95.0  # For threshold calibration
        },
        'evaluation': {
            'save_plots': True,
            'detailed_analysis': True
        }
    }
    
    print("Configuration:")
    print(f"  Model type: {config['model']['type']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Mixed precision: {config['training']['mixed_precision']}")
    print(f"  Threshold percentile: {config['training']['threshold_percentile']}")
    
    try:
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        X_train, X_val, X_test, y_test, scaler, encoders, input_dim = load_and_preprocess_data()
        print(f"Input dimension: {input_dim}")
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, X_val, X_test, y_test, 
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
        
        # Create model
        print(f"Creating {config['model']['type']} model...")
        model = create_model(config['model']['type'], input_dim, config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Train the model (uses ONLY normal samples)
        print("\nStarting training with improved methodology...")
        print("- Using ONLY normal samples for training")
        print("- Proper threshold calibration")
        print("- Advanced evaluation metrics")
        
        train_losses, val_losses, best_val_loss = train_model(
            model, train_loader, val_loader, device, config
        )
        
        print("\nTraining completed!")
        print(f"Final training loss: {train_losses[-1]:.6f}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Epochs trained: {len(train_losses)}")
        
        # Load best model
        model.load_state_dict(torch.load('results/improved_model_best.pth'))
        
        # Calibrate threshold using normal training data
        print("\nCalibrating optimal threshold...")
        threshold, normal_errors, stats = calibrate_threshold(
            model, train_loader, device, 
            percentile=config['training']['threshold_percentile']
        )
        print(f"Optimal threshold: {threshold:.6f}")
        
        # Comprehensive evaluation
        print("\nPerforming comprehensive evaluation...")
        evaluation_results = evaluate_model(model, test_loader, device, threshold)
        
        # Save results
        Path('results').mkdir(exist_ok=True)
        with open('results/improved_training_results.pkl', 'wb') as f:
            pickle.dump({
                'config': config,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'threshold': threshold,
                'stats': stats,
                'evaluation_results': evaluation_results,
                'scaler': scaler,
                'encoders': encoders
            }, f)
        
        # Final summary
        print("\n" + "=" * 80)
        print("IMPROVED TRAINING SUMMARY")
        print("=" * 80)
        print(f"Model: {config['model']['type'].upper()}")
        print(f"Device: {device}")
        print(f"Mixed Precision: {config['training']['mixed_precision']}")
        print(f"Training Method: NORMAL SAMPLES ONLY ✓")
        print(f"Threshold Method: CALIBRATED ON NORMAL DATA ✓")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Accuracy:         {evaluation_results['accuracy']:.4f}")
        print(f"  Precision:        {evaluation_results['precision']:.4f}")
        print(f"  Recall:           {evaluation_results['recall']:.4f}")
        print(f"  F1-Score:         {evaluation_results['f1_score']:.4f}")
        print(f"  ROC-AUC:          {evaluation_results['roc_auc']:.4f}")
        print(f"  Average Precision: {evaluation_results['average_precision']:.4f}")
        print()
        print("DATA DISTRIBUTION:")
        print(f"  Normal test samples:   {evaluation_results['normal_samples']:,}")
        print(f"  Anomaly test samples:  {evaluation_results['anomaly_samples']:,}")
        print(f"  Total test samples:    {evaluation_results['normal_samples'] + evaluation_results['anomaly_samples']:,}")
        print()
        print(f"Results saved to: results/")
        print("✅ Improved training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        logging.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()