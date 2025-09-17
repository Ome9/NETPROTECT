"""
Data preprocessing module for NSL-KDD dataset.
Handles loading, cleaning, encoding, and preparation of the network traffic data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, Dict, List, Optional
import pickle
import os


class NSLKDDDataset(Dataset):
    """PyTorch Dataset for NSL-KDD data."""
    
    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Initialize dataset.
        
        Args:
            features: Feature array
            labels: Labels (optional, for unsupervised training)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict:
        item = {'features': self.features[idx]}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item


class DataProcessor:
    """Main data processing class for NSL-KDD dataset."""
    
    def __init__(self, data_path: str):
        """
        Initialize data processor.
        
        Args:
            data_path: Path to the NSL-KDD dataset directory
        """
        self.data_path = data_path
        self.feature_columns = self._get_feature_columns()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.attack_type_mapping = {}
        self.binary_mapping = {'normal': 0, 'attack': 1}
        
    def _get_feature_columns(self) -> List[str]:
        """Define the 41 feature columns based on KDD Cup 99 specification."""
        return [
            'duration', 'protocol_type', 'service', 'flag',
            'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
    
    def load_data(self, train_file: str = "KDDTrain+.txt", 
                  test_file: str = "KDDTest+.txt") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data.
        
        Args:
            train_file: Training data filename
            test_file: Test data filename
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_path = os.path.join(self.data_path, train_file)
        test_path = os.path.join(self.data_path, test_file)
        
        # Column names: features + label + difficulty
        columns = self.feature_columns + ['attack_type', 'difficulty']
        
        train_df = pd.read_csv(train_path, names=columns, header=None)
        test_df = pd.read_csv(test_path, names=columns, header=None)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        return train_df, test_df
    
    def preprocess_features(self, train_df: pd.DataFrame, 
                          test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features: encode categorical variables and normalize.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (train_features, test_features)
        """
        # Identify categorical columns
        categorical_columns = ['protocol_type', 'service', 'flag']
        
        # Make copies for preprocessing
        train_features = train_df[self.feature_columns].copy()
        test_features = test_df[self.feature_columns].copy()
        
        # Encode categorical features
        for col in categorical_columns:
            le = LabelEncoder()
            # Fit on combined data to ensure consistency
            combined_values = pd.concat([train_features[col], test_features[col]])
            le.fit(combined_values)
            
            train_features[col] = le.transform(train_features[col])
            test_features[col] = le.transform(test_features[col])
            
            self.label_encoders[col] = le
        
        # Convert to numpy arrays
        train_features = train_features.values.astype(np.float32)
        test_features = test_features.values.astype(np.float32)
        
        # Normalize features
        train_features = self.scaler.fit_transform(train_features)
        test_features = self.scaler.transform(test_features)
        
        print(f"Processed feature shape: {train_features.shape}")
        print(f"Feature ranges - Min: {train_features.min():.3f}, Max: {train_features.max():.3f}")
        
        return train_features, test_features
    
    def prepare_labels(self, train_df: pd.DataFrame, 
                      test_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Prepare both binary and multi-class labels.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (train_labels_dict, test_labels_dict)
        """
        # Get unique attack types
        all_attacks = pd.concat([train_df['attack_type'], test_df['attack_type']]).unique()
        
        # Create attack type to index mapping
        self.attack_type_mapping = {attack: idx for idx, attack in enumerate(sorted(all_attacks))}
        
        # Prepare labels for training set
        train_labels = {}
        train_labels['binary'] = (train_df['attack_type'] != 'normal').astype(int).values
        train_labels['multiclass'] = train_df['attack_type'].map(self.attack_type_mapping).values
        
        # Prepare labels for test set
        test_labels = {}
        test_labels['binary'] = (test_df['attack_type'] != 'normal').astype(int).values
        test_labels['multiclass'] = test_df['attack_type'].map(self.attack_type_mapping).values
        
        print(f"Number of classes: {len(self.attack_type_mapping)}")
        print(f"Binary distribution in training: Normal={np.sum(train_labels['binary'] == 0)}, Attack={np.sum(train_labels['binary'] == 1)}")
        
        return train_labels, test_labels
    
    def get_normal_data(self, features: np.ndarray, labels: Dict) -> np.ndarray:
        """
        Extract only normal traffic data for autoencoder training.
        
        Args:
            features: Feature array
            labels: Label dictionary
            
        Returns:
            Normal features only
        """
        normal_mask = labels['binary'] == 0
        normal_features = features[normal_mask]
        
        print(f"Normal data shape: {normal_features.shape}")
        return normal_features
    
    def create_dataloaders(self, features: np.ndarray, labels: Optional[Dict] = None,
                          batch_size: int = 256, shuffle: bool = True,
                          val_split: float = 0.2, num_workers: int = 4) -> Dict:
        """
        Create PyTorch DataLoaders with GPU optimizations.
        
        Args:
            features: Feature array
            labels: Optional labels dictionary
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            val_split: Validation split ratio
            num_workers: Number of worker processes for data loading
            
        Returns:
            Dictionary of DataLoaders
        """
        dataloaders = {}
        
        # GPU optimization settings
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,  # Speed up data transfer to GPU
            'persistent_workers': True if num_workers > 0 else False  # Keep workers alive
        }
        
        if labels is not None and val_split > 0:
            # Split into train/validation
            train_features, val_features, train_labels, val_labels = train_test_split(
                features, labels['binary'], test_size=val_split, random_state=42, stratify=labels['binary']
            )
            
            train_dataset = NSLKDDDataset(train_features, train_labels)
            val_dataset = NSLKDDDataset(val_features, val_labels)
            
            dataloaders['train'] = DataLoader(train_dataset, shuffle=shuffle, **dataloader_kwargs)
            dataloaders['val'] = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
            
        else:
            # No validation split (for normal data training)
            dataset = NSLKDDDataset(features)
            dataloaders['train'] = DataLoader(dataset, shuffle=shuffle, **dataloader_kwargs)
        
        return dataloaders
    
    def save_preprocessor(self, save_path: str):
        """Save preprocessing objects for later use."""
        preprocessing_objects = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'attack_type_mapping': self.attack_type_mapping,
            'feature_columns': self.feature_columns
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessing_objects, f)
        
        print(f"Preprocessing objects saved to {save_path}")
    
    def load_preprocessor(self, load_path: str):
        """Load preprocessing objects."""
        with open(load_path, 'rb') as f:
            preprocessing_objects = pickle.load(f)
        
        self.scaler = preprocessing_objects['scaler']
        self.label_encoders = preprocessing_objects['label_encoders']
        self.attack_type_mapping = preprocessing_objects['attack_type_mapping']
        self.feature_columns = preprocessing_objects['feature_columns']
        
        print(f"Preprocessing objects loaded from {load_path}")


def get_attack_categories() -> Dict[str, str]:
    """
    Map attack types to categories based on NSL-KDD documentation.
    
    Returns:
        Dictionary mapping attack name to category
    """
    return {
        'normal': 'normal',
        # DoS attacks
        'apache2': 'dos', 'back': 'dos', 'land': 'dos', 'mailbomb': 'dos',
        'neptune': 'dos', 'pod': 'dos', 'processtable': 'dos', 'smurf': 'dos',
        'teardrop': 'dos', 'snmpgetattack': 'dos', 'udpstorm': 'dos',
        # Probe attacks
        'ipsweep': 'probe', 'mscan': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
        'saint': 'probe', 'satan': 'probe',
        # R2L attacks
        'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l',
        'named': 'r2l', 'phf': 'r2l', 'sendmail': 'r2l', 'snmpguess': 'r2l',
        'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l', 'xlock': 'r2l',
        'xsnoop': 'r2l',
        # U2R attacks
        'buffer_overflow': 'u2r', 'httptunnel': 'u2r', 'loadmodule': 'u2r',
        'perl': 'u2r', 'ps': 'u2r', 'rootkit': 'u2r', 'sqlattack': 'u2r',
        'xterm': 'u2r'
    }