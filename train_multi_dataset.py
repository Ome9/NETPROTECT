"""
Multi-Dataset Anomaly Detection Training Script
Supports NSL-KDD, CSE-CIC-IDS2017, and UNSW-NB15 datasets
with proper anomaly detection methodology.
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
import os
from typing import Tuple, Dict, List, Optional

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


class MultiDatasetLoader:
    """
    Universal data loader for multiple network intrusion detection datasets.
    Supports NSL-KDD, CSE-CIC-IDS2017, and UNSW-NB15.
    """
    
    def __init__(self, dataset_name: str, data_paths: Dict[str, str]):
        self.dataset_name = dataset_name.lower()
        self.data_paths = data_paths
        self.scaler = None
        self.encoders = {}
        self.feature_columns = []
        self.label_column = None
        
    def load_nsl_kdd_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load NSL-KDD dataset."""
        logger.info("Loading NSL-KDD dataset...")
        
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
        
        train_df = pd.read_csv(self.data_paths['train'], names=column_names)
        test_df = pd.read_csv(self.data_paths['test'], names=column_names)
        
        # Create binary labels (normal vs anomaly)
        train_df['is_anomaly'] = (train_df['attack_type'] != 'normal').astype(int)
        test_df['is_anomaly'] = (test_df['attack_type'] != 'normal').astype(int)
        
        # Remove unnecessary columns
        train_df = train_df.drop(['attack_type', 'difficulty'], axis=1)
        test_df = test_df.drop(['attack_type', 'difficulty'], axis=1)
        
        self.label_column = 'is_anomaly'
        self.feature_columns = [col for col in train_df.columns if col != 'is_anomaly']
        
        return train_df, test_df
    
    def load_cse_cic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load CSE-CIC-IDS2017 dataset."""
        logger.info("Loading CSE-CIC-IDS2017 dataset...")
        
        # CSE-CIC-IDS2017 can have multiple files, handle both .csv and .parquet
        if os.path.isdir(self.data_paths['train']):
            # Load all data files from directory (CSV or Parquet)
            train_files = [f for f in os.listdir(self.data_paths['train']) 
                          if f.endswith('.csv') or f.endswith('.parquet')]
            train_dfs = []
            
            logger.info(f"Found {len(train_files)} data files in {self.data_paths['train']}")
            
            for file in train_files[:5]:  # Limit to first 5 files for manageable training
                file_path = os.path.join(self.data_paths['train'], file)
                logger.info(f"Loading {file}...")
                
                if file.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                logger.info(f"  Loaded {len(df):,} samples with {len(df.columns)} features")
                train_dfs.append(df)
            
            if train_dfs:
                train_df = pd.concat(train_dfs, ignore_index=True)
                logger.info(f"Combined training data: {len(train_df):,} samples")
            else:
                raise ValueError("No valid data files found in training directory")
        else:
            # Single file
            if self.data_paths['train'].endswith('.parquet'):
                train_df = pd.read_parquet(self.data_paths['train'])
            else:
                train_df = pd.read_csv(self.data_paths['train'])
        
        # For CSE-CIC, we'll use 80% for training, 20% for testing
        logger.info("Splitting CSE-CIC dataset into train/test (80/20)...")
        test_df = train_df.sample(frac=0.2, random_state=42)
        train_df = train_df.drop(test_df.index)
        
        # Clean column names (remove spaces and special characters)
        train_df.columns = train_df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        test_df.columns = test_df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        
        # Find label column (usually 'Label' or similar)
        possible_label_cols = ['Label', 'label', 'Label_', 'Attack', 'attack', 'Class', 'class']
        label_col = None
        for col in possible_label_cols:
            if col in train_df.columns:
                label_col = col
                break
        
        if label_col is None:
            raise ValueError(f"Could not find label column. Available columns: {train_df.columns.tolist()}")
        
        logger.info(f"Found label column: {label_col}")
        logger.info(f"Label values: {train_df[label_col].value_counts().to_dict()}")
        
        # Create binary labels (BENIGN vs others)
        train_df['is_anomaly'] = (train_df[label_col].str.upper() != 'BENIGN').astype(int)
        test_df['is_anomaly'] = (test_df[label_col].str.upper() != 'BENIGN').astype(int)
        
        # Remove original label column
        train_df = train_df.drop([label_col], axis=1)
        test_df = test_df.drop([label_col], axis=1)
        
        self.label_column = 'is_anomaly'
        self.feature_columns = [col for col in train_df.columns if col != 'is_anomaly']
        
        return train_df, test_df
    
    def load_unsw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load UNSW-NB15 dataset."""
        logger.info("Loading UNSW-NB15 dataset...")
        
        train_df = pd.read_csv(self.data_paths['train'])
        test_df = pd.read_csv(self.data_paths['test'])
        
        # Clean column names
        train_df.columns = train_df.columns.str.strip().str.replace(' ', '_')
        test_df.columns = test_df.columns.str.strip().str.replace(' ', '_')
        
        # Find label column (usually 'Label' or 'attack_cat')
        possible_label_cols = ['Label', 'label', 'attack_cat', 'Attack_cat', 'class', 'Class']
        label_col = None
        for col in possible_label_cols:
            if col in train_df.columns:
                label_col = col
                break
        
        if label_col is None:
            # If no label column found, check if there's a binary column (0/1)
            binary_cols = [col for col in train_df.columns if train_df[col].nunique() == 2]
            if binary_cols:
                label_col = binary_cols[-1]  # Take the last binary column
                logger.info(f"Using binary column '{label_col}' as label")
        
        if label_col is None:
            raise ValueError("Could not find label column in UNSW dataset")
        
        # Create binary labels
        if train_df[label_col].dtype == 'object':
            # String labels (normal vs attack names)
            train_df['is_anomaly'] = (train_df[label_col].str.lower() != 'normal').astype(int)
            test_df['is_anomaly'] = (test_df[label_col].str.lower() != 'normal').astype(int)
        else:
            # Numeric labels (0 = normal, 1 = anomaly)
            train_df['is_anomaly'] = train_df[label_col].astype(int)
            test_df['is_anomaly'] = test_df[label_col].astype(int)
        
        # Remove original label column
        train_df = train_df.drop([label_col], axis=1)
        test_df = test_df.drop([label_col], axis=1)
        
        self.label_column = 'is_anomaly'
        self.feature_columns = [col for col in train_df.columns if col != 'is_anomaly']
        
        return train_df, test_df
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset based on dataset name."""
        if self.dataset_name == 'nsl-kdd':
            return self.load_nsl_kdd_data()
        elif self.dataset_name == 'cse-cic' or self.dataset_name == 'cic':
            return self.load_cse_cic_data()
        elif self.dataset_name == 'unsw' or self.dataset_name == 'unsw-nb15':
            return self.load_unsw_data()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess the loaded data."""
        logger.info("Preprocessing data...")
        
        # Combine for consistent preprocessing
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Handle missing values
        logger.info(f"Missing values before cleaning: {combined_df.isnull().sum().sum()}")
        
        # Fill missing values
        for col in combined_df.columns:
            if combined_df[col].dtype == 'object':
                combined_df[col] = combined_df[col].fillna('unknown')
            else:
                combined_df[col] = combined_df[col].fillna(combined_df[col].median())
        
        # Handle infinite values
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.fillna(0)
        
        logger.info(f"Missing values after cleaning: {combined_df.isnull().sum().sum()}")
        
        # Separate features and labels
        y = combined_df[self.label_column].values
        feature_df = combined_df[self.feature_columns]
        
        # Encode categorical variables
        categorical_columns = feature_df.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Categorical columns: {categorical_columns}")
        
        for col in categorical_columns:
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col].astype(str))
            self.encoders[col] = le
        
        # Convert to numpy
        X = feature_df.values.astype(np.float32)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split back to train/test
        n_train = len(train_df)
        X_train_full = X_scaled[:n_train]
        y_train_full = y[:n_train]
        X_test = X_scaled[n_train:]
        y_test = y[n_train:]
        
        logger.info(f"Dataset: {self.dataset_name.upper()}")
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
        
        return X_train, X_val, X_test, y_test


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


def train_model(model, train_loader, val_loader, device, config, dataset_name):
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
    logger.info(f"STARTING IMPROVED TRAINING - {dataset_name.upper()}")
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
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_X)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_X)
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
            model_dir = Path(f'results/{dataset_name}')
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / 'improved_model_best.pth')
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


def evaluate_model(model, test_loader, device, threshold, dataset_name):
    """Comprehensive evaluation."""
    logger.info("=" * 60)
    logger.info(f"COMPREHENSIVE EVALUATION - {dataset_name.upper()}")
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
        'dataset': dataset_name,
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
    logger.info(f"Test Results for {dataset_name.upper()}:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  Average Precision: {avg_precision:.4f}")
    logger.info(f"  Normal samples: {normal_count:,}")
    logger.info(f"  Anomaly samples: {anomaly_count:,}")
    
    return results


def train_dataset(dataset_name: str, data_paths: Dict[str, str], config: dict, device: torch.device):
    """Train a model on a specific dataset."""
    logger.info("=" * 80)
    logger.info(f"TRAINING ON {dataset_name.upper()} DATASET")
    logger.info("=" * 80)
    
    try:
        # Load and preprocess data
        data_loader = MultiDatasetLoader(dataset_name, data_paths)
        train_df, test_df = data_loader.load_data()
        X_train, X_val, X_test, y_test = data_loader.preprocess_data(train_df, test_df)
        
        input_dim = X_train.shape[1]
        logger.info(f"Input dimension: {input_dim}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, X_val, X_test, y_test, 
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
        
        # Create model
        model = ImprovedAutoencoder(input_dim, config['model']['hidden_dims'])
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        
        # Train model
        train_losses, val_losses, best_val_loss = train_model(
            model, train_loader, val_loader, device, config, dataset_name
        )
        
        logger.info(f"Training completed! Best validation loss: {best_val_loss:.6f}")
        
        # Load best model
        model_path = Path(f'results/{dataset_name}/improved_model_best.pth')
        model.load_state_dict(torch.load(model_path))
        
        # Calibrate threshold
        threshold, normal_errors, stats = calibrate_threshold(
            model, train_loader, device, 
            percentile=config['training']['threshold_percentile']
        )
        
        # Evaluate model
        evaluation_results = evaluate_model(model, test_loader, device, threshold, dataset_name)
        
        # Save results
        results_dir = Path(f'results/{dataset_name}')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / 'training_results.pkl', 'wb') as f:
            pickle.dump({
                'dataset_name': dataset_name,
                'config': config,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'threshold': threshold,
                'stats': stats,
                'evaluation_results': evaluation_results,
                'scaler': data_loader.scaler,
                'encoders': data_loader.encoders,
                'feature_columns': data_loader.feature_columns
            }, f)
        
        logger.info(f"Results saved to: {results_dir}")
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error training on {dataset_name}: {e}", exc_info=True)
        return None


def main():
    """Main training function for multiple datasets."""
    print("=" * 80)
    print("🔥 MULTI-DATASET ANOMALY DETECTION TRAINING")
    print("Supporting NSL-KDD, CSE-CIC-IDS2017, and UNSW-NB15")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Configuration
    config = {
        'model': {
            'hidden_dims': [256, 128, 64, 32, 16]
        },
        'training': {
            'batch_size': 512,
            'epochs': 100,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'early_stopping_patience': 20,
            'num_workers': 4,
            'pin_memory': True,
            'mixed_precision': True,
            'threshold_percentile': 95.0
        }
    }
    
    # Dataset configurations
    # IMPORTANT: Update these paths to match your actual data locations
    datasets = {
        'nsl-kdd': {
            'train': 'NSL-KDD_Dataset/KDDTrain+.txt',
            'test': 'NSL-KDD_Dataset/KDDTest+.txt'
        },
        'cse-cic': {
            'train': 'CSE-CIC_Dataset',  # Directory or single file
            'test': 'CSE-CIC_Dataset'    # Directory or single file
        },
        'unsw': {
            'train': 'UNSW_Dataset/UNSW_NB15_training-set.csv',
            'test': 'UNSW_Dataset/UNSW_NB15_testing-set.csv'
        }
    }
    
    print("\nDataset Configurations:")
    for dataset, paths in datasets.items():
        print(f"  {dataset.upper()}:")
        print(f"    Train: {paths['train']}")
        print(f"    Test:  {paths['test']}")
    
    # Train on each dataset
    all_results = {}
    
    for dataset_name, data_paths in datasets.items():
        # Check if dataset files exist
        if not all(os.path.exists(path) for path in data_paths.values()):
            logger.warning(f"Dataset {dataset_name} not found, skipping...")
            continue
        
        logger.info(f"\nStarting training on {dataset_name.upper()} dataset...")
        results = train_dataset(dataset_name, data_paths, config, device)
        
        if results:
            all_results[dataset_name] = results
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 MULTI-DATASET TRAINING SUMMARY")
    print("=" * 80)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()} RESULTS:")
        print(f"  Accuracy:         {results['accuracy']:.4f}")
        print(f"  Precision:        {results['precision']:.4f}")
        print(f"  Recall:           {results['recall']:.4f}")
        print(f"  F1-Score:         {results['f1_score']:.4f}")
        print(f"  ROC-AUC:          {results['roc_auc']:.4f}")
        print(f"  Average Precision: {results['average_precision']:.4f}")
        print(f"  Normal samples:   {results['normal_samples']:,}")
        print(f"  Anomaly samples:  {results['anomaly_samples']:,}")
    
    print(f"\n✅ Training completed on {len(all_results)} datasets!")
    print("Results saved to individual directories in results/")


if __name__ == "__main__":
    main()