"""
Standalone improved anomaly detection training script.
This demonstrates the correct methodology identified in the comprehensive analysis.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
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
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # Bottleneck
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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


def create_data_loaders(X_train, X_val, X_test, y_test, batch_size=512):
    """Create data loaders."""
    # Training data (normal only)
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.zeros(len(X_train))  # Dummy labels for normal data
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    # Validation data (normal only)
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.zeros(len(X_val))  # Dummy labels for normal data
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Test data (mixed normal and anomaly)
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_improved_model(model, train_loader, val_loader, device, epochs=100):
    """Train model using only normal samples."""
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    
    # Mixed precision training
    scaler = GradScaler()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 0
    max_patience = 20
    
    logger.info("=" * 60)
    logger.info("STARTING IMPROVED TRAINING")
    logger.info("Using ONLY normal samples for training")
    logger.info("=" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        for batch_X, _ in train_loader:  # Ignore labels, use only normal data
            batch_X = batch_X.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_X)  # Reconstruct normal data
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
                
                with autocast():
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
            torch.save(model.state_dict(), 'results/improved_model_best.pth')
        else:
            patience += 1
            if patience >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {lr:.2e}")
    
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


def plot_results(train_losses, val_losses, normal_errors, results):
    """Generate comprehensive plots."""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training curves
    axes[0, 0].plot(train_losses, label='Training Loss', alpha=0.8)
    axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training Curves (Normal Data Only)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Error distributions
    normal_test_errors = results['errors'][results['labels'] == 0]
    anomaly_test_errors = results['errors'][results['labels'] == 1]
    
    axes[0, 1].hist(normal_errors[:5000], bins=50, alpha=0.6, label='Normal (Train)', density=True, color='blue')
    axes[0, 1].hist(normal_test_errors, bins=50, alpha=0.6, label='Normal (Test)', density=True, color='green')
    axes[0, 1].hist(anomaly_test_errors, bins=50, alpha=0.6, label='Anomaly (Test)', density=True, color='red')
    axes[0, 1].axvline(results['threshold'], color='black', linestyle='--', label='Threshold')
    axes[0, 1].set_xlabel('Reconstruction Error')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Error Distributions')
    axes[0, 1].legend()
    axes[0, 1].set_xscale('log')
    
    # 3. Confusion Matrix
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
               ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')
    
    # 4. Metrics summary
    metrics_text = f"""
    IMPROVED METHODOLOGY RESULTS:
    
    ✅ Training: Normal samples only
    ✅ Threshold: Calibrated on normal data
    ✅ Evaluation: Proper metrics
    
    Performance:
    Accuracy:    {results['accuracy']:.4f}
    Precision:   {results['precision']:.4f}
    Recall:      {results['recall']:.4f}
    F1-Score:    {results['f1_score']:.4f}
    ROC-AUC:     {results['roc_auc']:.4f}
    Avg Precision: {results['average_precision']:.4f}
    
    Dataset:
    Normal:      {results['normal_samples']:,}
    Anomaly:     {results['anomaly_samples']:,}
    
    Threshold:   {results['threshold']:.6f}
    """
    
    axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=9)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Improved Results Summary')
    
    plt.tight_layout()
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/improved_anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Results plot saved: results/improved_anomaly_detection_results.png")


def main():
    """Main function implementing improved anomaly detection methodology."""
    print("=" * 80)
    print("🔥 IMPROVED ANOMALY DETECTION WITH CORRECTED METHODOLOGY")
    print("=" * 80)
    print("Fixes implemented:")
    print("✅ 1. Using ONLY normal samples for training")
    print("✅ 2. Proper threshold calibration on normal data")
    print("✅ 3. Comprehensive evaluation metrics")
    print("✅ 4. GPU optimization with mixed precision")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    try:
        # Load and preprocess data
        X_train, X_val, X_test, y_test, scaler, encoders, input_dim = load_and_preprocess_data()
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, X_val, X_test, y_test, batch_size=512
        )
        
        # Create model
        model = ImprovedAutoencoder(input_dim)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        
        # Train model
        train_losses, val_losses, best_val_loss = train_improved_model(
            model, train_loader, val_loader, device, epochs=100
        )
        
        logger.info(f"Training completed! Best validation loss: {best_val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(torch.load('results/improved_model_best.pth'))
        
        # Calibrate threshold
        threshold, normal_errors, stats = calibrate_threshold(model, train_loader, device)
        
        # Evaluate model
        results = evaluate_model(model, test_loader, device, threshold)
        
        # Generate plots
        plot_results(train_losses, val_losses, normal_errors, results)
        
        # Save results
        with open('results/improved_results.pkl', 'wb') as f:
            pickle.dump({
                'results': results,
                'stats': stats,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'scaler': scaler,
                'encoders': encoders
            }, f)
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 IMPROVED TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"   Accuracy:         {results['accuracy']:.4f}")
        print(f"   Precision:        {results['precision']:.4f}")
        print(f"   Recall:           {results['recall']:.4f}")
        print(f"   F1-Score:         {results['f1_score']:.4f}")
        print(f"   ROC-AUC:          {results['roc_auc']:.4f}")
        print(f"   Average Precision: {results['average_precision']:.4f}")
        print(f"📈 METHODOLOGY IMPROVEMENTS:")
        print(f"   ✅ Training: {len(X_train):,} normal samples only")
        print(f"   ✅ Threshold: {threshold:.6f} (calibrated on normal data)")
        print(f"   ✅ Evaluation: Comprehensive metrics for imbalanced data")
        print(f"📁 Results saved to: results/")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()