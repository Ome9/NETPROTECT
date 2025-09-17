"""
Configuration management utilities.
Handles loading and validation of experiment configurations.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import argparse
from dataclasses import dataclass, field
import torch


@dataclass
class DataConfig:
    """Data configuration parameters."""
    dataset_path: str
    train_file: str = "KDDTrain+.txt"
    test_file: str = "KDDTest+.txt"
    validation_split: float = 0.2
    batch_size: int = 256
    num_workers: int = 4
    shuffle_train: bool = True


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    type: str = "autoencoder"
    input_dim: int = 41
    hidden_dims: list = field(default_factory=lambda: [32, 24, 16])
    latent_dim: int = 8
    activation: str = "leaky_relu"
    use_batch_norm: bool = True
    dropout_rate: float = 0.2
    use_skip_connections: bool = False


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    num_epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    scheduler: Optional[str] = "reduce_plateau"
    loss_type: str = "mse"
    early_stopping_patience: int = 15
    min_delta: float = 1e-6
    gradient_clip_value: Optional[float] = 1.0
    weight_decay: float = 1e-5
    log_interval: int = 100
    save_interval: int = 10


@dataclass
class DeviceConfig:
    """Device configuration parameters."""
    use_cuda: bool = True
    cuda_device: int = 0
    memory_efficient: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    threshold_methods: list = field(default_factory=lambda: ["percentile_95", "optimal_f1"])
    error_type: str = "mse"
    save_plots: bool = True
    plot_format: str = "png"


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    name: str = "default_experiment"
    description: str = ""
    save_model: bool = True
    save_results: bool = True
    save_plots: bool = True
    results_dir: str = "../results"
    models_dir: str = "../models"
    plots_dir: str = "../plots"


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    device: DeviceConfig
    evaluation: EvaluationConfig
    experiment: ExperimentConfig
    seed: int = 42
    vae: Optional[Dict] = None


class ConfigManager:
    """Configuration manager for loading and validating configurations."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self.config = None
    
    def load_config(self, config_path: Union[str, Path]) -> Config:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to dataclass objects
        data_config = DataConfig(**config_dict['data'])
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        device_config = DeviceConfig(**config_dict['device'])
        evaluation_config = EvaluationConfig(**config_dict['evaluation'])
        experiment_config = ExperimentConfig(**config_dict['experiment'])
        
        # Optional VAE configuration
        vae_config = config_dict.get('vae', None)
        
        self.config = Config(
            data=data_config,
            model=model_config,
            training=training_config,
            device=device_config,
            evaluation=evaluation_config,
            experiment=experiment_config,
            seed=config_dict.get('seed', 42),
            vae=vae_config
        )
        
        # Validate configuration
        self._validate_config()
        
        return self.config
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config is None:
            raise ValueError("Configuration not loaded")
            
        config = self.config
        
        # Validate data configuration
        if config.data.validation_split < 0 or config.data.validation_split >= 1:
            raise ValueError("validation_split must be between 0 and 1")
        
        if config.data.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # Validate model configuration
        valid_model_types = ["autoencoder", "vae", "deep"]
        if config.model.type not in valid_model_types:
            raise ValueError(f"model.type must be one of {valid_model_types}")
        
        if config.model.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        
        if config.model.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        
        valid_activations = ["relu", "leaky_relu", "elu"]
        if config.model.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")
        
        # Validate training configuration
        if config.training.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        if config.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        valid_optimizers = ["adam", "rmsprop", "sgd", "adamw"]
        if config.training.optimizer not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}")
        
        valid_loss_types = ["mse", "mae", "rmse", "vae"]
        if config.training.loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}")
        
        # VAE specific validation
        if config.model.type == "vae":
            if config.vae is None:
                raise ValueError("VAE configuration required for VAE model")
            if config.training.loss_type != "vae":
                print("Warning: Using VAE model but loss_type is not 'vae'")
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.config is None:
            raise ValueError("Configuration not loaded")
            
        if self.config.device.use_cuda and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.device.cuda_device}")
            print(f"Using device: {device}")
            if self.config.device.memory_efficient:
                print("Memory efficient mode enabled")
        else:
            device = torch.device("cpu")
            print("Using device: CPU")
        
        return device
    
    def setup_directories(self):
        """Create necessary directories for the experiment."""
        if self.config is None:
            raise ValueError("Configuration not loaded")
        
        exp_config = self.config.experiment
        
        # Create base directories
        results_dir = Path(exp_config.results_dir)
        models_dir = Path(exp_config.models_dir)
        plots_dir = Path(exp_config.plots_dir)
        
        # Create experiment-specific subdirectories
        exp_name = exp_config.name
        
        exp_results_dir = results_dir / exp_name
        exp_models_dir = models_dir / exp_name
        exp_plots_dir = plots_dir / exp_name
        
        # Create directories
        for directory in [exp_results_dir, exp_models_dir, exp_plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Created experiment directories for: {exp_name}")
        
        return {
            'results_dir': exp_results_dir,
            'models_dir': exp_models_dir,
            'plots_dir': exp_plots_dir
        }
    
    def save_config(self, save_path: Union[str, Path]):
        """Save current configuration to file."""
        if self.config is None:
            raise ValueError("No configuration to save")
        
        # Convert config to dictionary
        config_dict = self._config_to_dict()
        
        save_path = Path(save_path)
        
        if save_path.suffix == '.yaml' or save_path.suffix == '.yml':
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif save_path.suffix == '.json':
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError("Config file must have .yaml, .yml, or .json extension")
        
        print(f"Configuration saved to: {save_path}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        if self.config is None:
            raise ValueError("No configuration to convert")
            
        config_dict = {
            'data': self.config.data.__dict__,
            'model': self.config.model.__dict__,
            'training': self.config.training.__dict__,
            'device': self.config.device.__dict__,
            'evaluation': self.config.evaluation.__dict__,
            'experiment': self.config.experiment.__dict__,
            'seed': self.config.seed
        }
        
        if self.config.vae is not None:
            config_dict['vae'] = self.config.vae
        
        return config_dict
    
    def print_config(self):
        """Print configuration in a readable format."""
        if self.config is None:
            print("No configuration loaded")
            return
        
        print("=" * 50)
        print(f"Configuration: {self.config.experiment.name}")
        print("=" * 50)
        
        print(f"\nData Configuration:")
        for key, value in self.config.data.__dict__.items():
            print(f"  {key}: {value}")
        
        print(f"\nModel Configuration:")
        for key, value in self.config.model.__dict__.items():
            print(f"  {key}: {value}")
        
        print(f"\nTraining Configuration:")
        for key, value in self.config.training.__dict__.items():
            print(f"  {key}: {value}")
        
        print(f"\nDevice Configuration:")
        for key, value in self.config.device.__dict__.items():
            print(f"  {key}: {value}")
        
        print(f"\nEvaluation Configuration:")
        for key, value in self.config.evaluation.__dict__.items():
            print(f"  {key}: {value}")
        
        if self.config.vae:
            print(f"\nVAE Configuration:")
            for key, value in self.config.vae.items():
                print(f"  {key}: {value}")
        
        print(f"\nSeed: {self.config.seed}")
        print("=" * 50)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Network Anomaly Detection with Autoencoders')
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Override dataset path'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        help='Override device setting'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Override random seed'
    )
    
    return parser.parse_args()


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply command line argument overrides to configuration."""
    
    if args.data_path:
        config.data.dataset_path = args.data_path
    
    if args.device:
        config.device.use_cuda = args.device == 'cuda'
    
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    if args.epochs:
        config.training.num_epochs = args.epochs
    
    if args.lr:
        config.training.learning_rate = args.lr
    
    if args.seed:
        config.seed = args.seed
    
    return config


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seeds set to: {seed}")