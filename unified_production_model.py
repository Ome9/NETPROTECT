"""
Unified Production Model for Network Anomaly Detection
Supports NSL-KDD, CSE-CIC-IDS2017, and UNSW-NB15 datasets with automatic detection.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedAutoencoder(nn.Module):
    """Production-ready autoencoder model."""
    
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


class UnifiedAnomalyDetector:
    """
    Unified anomaly detection model supporting multiple datasets.
    Automatically detects dataset type and applies appropriate preprocessing.
    """
    
    DATASET_CONFIGS = {
        'nsl-kdd': {
            'model_path': 'results/nsl-kdd/improved_model_best.pth',
            'results_path': 'results/nsl-kdd/training_results.pkl',
            'input_dim': 41,
            'feature_names': [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
            ],
            'categorical_features': ['protocol_type', 'service', 'flag']
        },
        'cse-cic': {
            'model_path': 'results/cse-cic/improved_model_best.pth',
            'results_path': 'results/cse-cic/training_results.pkl',
            'input_dim': 77,
            'feature_names': None,  # Will be loaded from training results
            'categorical_features': []  # CSE-CIC has mostly numerical features
        },
        'unsw': {
            'model_path': 'results/unsw/improved_model_best.pth',
            'results_path': 'results/unsw/training_results.pkl',
            'input_dim': 44,
            'feature_names': None,  # Will be loaded from training results
            'categorical_features': ['proto', 'service', 'state', 'attack_cat']
        }
    }
    
    def __init__(self, device: str = None):
        """
        Initialize unified anomaly detector.
        
        Args:
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.thresholds = {}
        self.feature_columns = {}
        
        logger.info(f"Initializing Unified Anomaly Detector on {self.device}")
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all available trained models."""
        for dataset_name, config in self.DATASET_CONFIGS.items():
            try:
                self._load_single_model(dataset_name, config)
                logger.info(f"✅ Loaded {dataset_name.upper()} model")
            except Exception as e:
                logger.warning(f"❌ Could not load {dataset_name.upper()} model: {e}")
    
    def _load_single_model(self, dataset_name: str, config: Dict):
        """Load a single model with its preprocessing components."""
        model_path = Path(config['model_path'])
        results_path = Path(config['results_path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        model = ImprovedAutoencoder(config['input_dim'])
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.models[dataset_name] = model
        
        # Load training results (contains scaler, threshold, etc.)
        if results_path.exists():
            with open(results_path, 'rb') as f:
                training_results = pickle.load(f)
            
            self.scalers[dataset_name] = training_results.get('scaler')
            self.thresholds[dataset_name] = training_results.get('threshold')
            self.encoders[dataset_name] = training_results.get('encoders', {})
            
            # Store feature information
            if 'feature_columns' in training_results:
                self.feature_columns[dataset_name] = training_results['feature_columns']
            elif config['feature_names']:
                self.feature_columns[dataset_name] = config['feature_names']
        
        logger.info(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"   Threshold: {self.thresholds.get(dataset_name, 'N/A')}")
    
    def detect_dataset_type(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        Automatically detect the dataset type based on data characteristics.
        
        Args:
            data: Input data
            
        Returns:
            Dataset name ('nsl-kdd', 'cse-cic', or 'unsw')
        """
        if isinstance(data, np.ndarray):
            num_features = data.shape[1]
            column_names = []
        else:
            num_features = data.shape[1] 
            column_names = data.columns.tolist()
        
        # Check by number of features first
        if num_features == 41 or num_features == 42:  # NSL-KDD (with/without label)
            return 'nsl-kdd'
        elif num_features == 77 or num_features == 78:  # CSE-CIC (with/without label)
            return 'cse-cic'
        elif num_features == 44 or num_features == 45:  # UNSW (with/without label)
            return 'unsw'
        
        # Check by column names if available
        if column_names:
            nsl_indicators = ['protocol_type', 'service', 'flag', 'duration']
            cse_indicators = ['Flow_Duration', 'Total_Fwd_Packets', 'Protocol']
            unsw_indicators = ['proto', 'service', 'state', 'spkts']
            
            nsl_score = sum(1 for col in column_names if col in nsl_indicators)
            cse_score = sum(1 for col in column_names if col in cse_indicators)
            unsw_score = sum(1 for col in column_names if col in unsw_indicators)
            
            scores = {'nsl-kdd': nsl_score, 'cse-cic': cse_score, 'unsw': unsw_score}
            return max(scores, key=scores.get)
        
        # Default fallback
        logger.warning(f"Could not determine dataset type for {num_features} features")
        return 'nsl-kdd'  # Default to NSL-KDD
    
    def preprocess_data(self, data: Union[pd.DataFrame, np.ndarray], dataset_type: str = None) -> np.ndarray:
        """
        Preprocess data for the specified dataset type.
        
        Args:
            data: Input data
            dataset_type: Dataset type (auto-detected if None)
            
        Returns:
            Preprocessed data ready for model input
        """
        if dataset_type is None:
            dataset_type = self.detect_dataset_type(data)
        
        logger.info(f"Preprocessing data for {dataset_type.upper()} dataset")
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            if dataset_type in self.feature_columns:
                columns = self.feature_columns[dataset_type]
                if len(columns) == data.shape[1]:
                    data = pd.DataFrame(data, columns=columns)
                else:
                    data = pd.DataFrame(data)
            else:
                data = pd.DataFrame(data)
        
        # Remove label column if present
        label_columns = ['is_anomaly', 'attack_type', 'Label', 'label']
        for col in label_columns:
            if col in data.columns:
                data = data.drop(col, axis=1)
        
        # Handle categorical features
        categorical_features = self.DATASET_CONFIGS[dataset_type]['categorical_features']
        for col in categorical_features:
            if col in data.columns and col in self.encoders.get(dataset_type, {}):
                # Use stored encoder
                encoder = self.encoders[dataset_type][col]
                # Handle unseen categories
                data[col] = data[col].astype(str)
                known_categories = set(encoder.classes_)
                data[col] = data[col].apply(lambda x: x if x in known_categories else 'unknown')
                
                # Add 'unknown' category if needed
                if 'unknown' not in known_categories and 'unknown' in data[col].values:
                    encoder.classes_ = np.append(encoder.classes_, 'unknown')
                
                try:
                    data[col] = encoder.transform(data[col])
                except ValueError:
                    # Fallback: encode as numeric
                    data[col] = pd.Categorical(data[col]).codes
        
        # Apply scaling
        if dataset_type in self.scalers and self.scalers[dataset_type] is not None:
            scaler = self.scalers[dataset_type]
            data_array = scaler.transform(data.values)
        else:
            logger.warning(f"No scaler found for {dataset_type}, using raw data")
            data_array = data.values.astype(np.float32)
        
        return data_array
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray], dataset_type: str = None, 
                return_details: bool = False) -> Dict:
        """
        Predict anomalies in the input data.
        
        Args:
            data: Input data
            dataset_type: Dataset type (auto-detected if None)
            return_details: Whether to return detailed information
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        if dataset_type is None:
            dataset_type = self.detect_dataset_type(data)
        
        if dataset_type not in self.models:
            raise ValueError(f"Model for {dataset_type} not available")
        
        # Preprocess data
        processed_data = self.preprocess_data(data, dataset_type)
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(processed_data).to(self.device)
        
        # Get model and threshold
        model = self.models[dataset_type]
        threshold = self.thresholds.get(dataset_type, 0.1)  # Default threshold
        
        with torch.no_grad():
            # Get reconstructions
            reconstructions = model(data_tensor)
            
            # Calculate reconstruction errors
            errors = torch.mean((reconstructions - data_tensor) ** 2, dim=1)
            errors_np = errors.cpu().numpy()
            
            # Make predictions
            predictions = (errors_np > threshold).astype(int)
            
            # Calculate confidence scores
            confidence = np.clip(errors_np / (threshold * 2), 0, 1)
        
        results = {
            'predictions': predictions,
            'confidence': confidence,
            'dataset_type': dataset_type,
            'threshold': threshold
        }
        
        if return_details:
            results.update({
                'reconstruction_errors': errors_np,
                'num_samples': len(predictions),
                'num_anomalies': np.sum(predictions),
                'anomaly_percentage': np.mean(predictions) * 100
            })
        
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available trained models."""
        return list(self.models.keys())
    
    def get_model_info(self, dataset_type: str = None) -> Dict:
        """Get information about loaded models."""
        if dataset_type:
            if dataset_type not in self.models:
                return {}
            
            model = self.models[dataset_type]
            return {
                'dataset': dataset_type,
                'parameters': sum(p.numel() for p in model.parameters()),
                'input_dim': self.DATASET_CONFIGS[dataset_type]['input_dim'],
                'threshold': self.thresholds.get(dataset_type),
                'device': str(next(model.parameters()).device)
            }
        else:
            return {name: self.get_model_info(name) for name in self.models.keys()}


def main():
    """Demonstration of the unified anomaly detector."""
    print("🔥 UNIFIED NETWORK ANOMALY DETECTOR")
    print("=" * 60)
    
    try:
        # Initialize detector
        detector = UnifiedAnomalyDetector()
        
        # Show available models
        available_models = detector.get_available_models()
        print(f"\n📊 Available Models: {', '.join(available_models)}")
        
        # Show model information
        print("\n🔧 Model Information:")
        model_info = detector.get_model_info()
        for dataset, info in model_info.items():
            print(f"  {dataset.upper()}:")
            print(f"    Parameters: {info['parameters']:,}")
            print(f"    Input dimension: {info['input_dim']}")
            print(f"    Threshold: {info['threshold']:.6f}")
            print(f"    Device: {info['device']}")
        
        print("\n✅ Unified Production Model is ready for deployment!")
        print("\nUsage:")
        print("  detector.predict(your_data)  # Auto-detects dataset type")
        print("  detector.predict(your_data, dataset_type='nsl-kdd')  # Explicit type")
        
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()