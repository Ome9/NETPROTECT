"""
Autoencoder models for network anomaly detection.
Implements various autoencoder architectures optimized for NSL-KDD dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class AutoEncoder(nn.Module):
    """
    Fully connected autoencoder with configurable architecture.
    Designed for memory efficiency and GPU compatibility.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout_rate: float = 0.2,
        use_skip_connections: bool = False
    ):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
            activation: Activation function ('relu', 'leaky_relu', 'elu')
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            use_skip_connections: Whether to use skip connections
        """
        super(AutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        
        # Set activation function
        self.activation = self._get_activation(activation)
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            encoder_layers.append(self.activation)
            
            # Dropout
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final encoder layer to latent space
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        
        # Reverse hidden dimensions for decoder
        decoder_hidden_dims = hidden_dims[::-1]
        
        for i, hidden_dim in enumerate(decoder_hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            
            decoder_layers.append(self.activation)
            
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final decoder layer to output
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation.lower() == 'elu':
            return nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed_output, latent_representation)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
    
    def compute_reconstruction_error(
        self, 
        x: torch.Tensor, 
        reduction: str = 'mean',
        error_type: str = 'mse'
    ) -> torch.Tensor:
        """
        Compute reconstruction error.
        
        Args:
            x: Input tensor
            reduction: Reduction method ('mean', 'sum', 'none')
            error_type: Error type ('mse', 'mae', 'rmse')
            
        Returns:
            Reconstruction error
        """
        reconstructed, _ = self.forward(x)
        
        if error_type == 'mse':
            error = F.mse_loss(reconstructed, x, reduction=reduction)
        elif error_type == 'mae':
            error = F.l1_loss(reconstructed, x, reduction=reduction)
        elif error_type == 'rmse':
            error = torch.sqrt(F.mse_loss(reconstructed, x, reduction=reduction))
        else:
            raise ValueError(f"Unsupported error type: {error_type}")
        
        return error


class VariationalAutoEncoder(nn.Module):
    """
    Variational Autoencoder for network anomaly detection.
    Uses variational inference to learn a probabilistic latent representation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout_rate: float = 0.2
    ):
        """
        Initialize VAE.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
        """
        super(VariationalAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Set activation function
        self.activation = self._get_activation(activation)
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*[layer for layer in encoder_layers if not isinstance(layer, nn.Identity)])
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        decoder_hidden_dims = hidden_dims[::-1]
        
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*[layer for layer in decoder_layers if not isinstance(layer, nn.Identity)])
        
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation.lower() == 'elu':
            return nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space parameters.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean, log_variance)
        """
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed_output, mean, log_variance)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def compute_loss(self, x: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            x: Input tensor
            beta: Weight for KL divergence term
            
        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_loss)
        """
        reconstructed, mu, logvar = self.forward(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Average over batch
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class DeepAutoEncoder(nn.Module):
    """
    Deep autoencoder with more layers for complex pattern learning.
    Optimized for memory efficiency.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        activation: str = 'leaky_relu',
        use_batch_norm: bool = True,
        dropout_rate: float = 0.3
    ):
        """
        Initialize deep autoencoder with predefined architecture.
        
        Args:
            input_dim: Number of input features (should be 41 for NSL-KDD)
            latent_dim: Latent space dimension
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
        """
        super(DeepAutoEncoder, self).__init__()
        
        # Optimized architecture for 4GB VRAM
        hidden_dims = [32, 24, 16]  # Smaller dimensions for memory efficiency
        
        self.autoencoder = AutoEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        return self.autoencoder(x)
    
    def compute_reconstruction_error(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute reconstruction error."""
        return self.autoencoder.compute_reconstruction_error(x, **kwargs)


def create_model(model_type: str, input_dim: int, **kwargs) -> nn.Module:
    """
    Factory function to create autoencoder models.
    
    Args:
        model_type: Type of model ('autoencoder', 'vae', 'deep')
        input_dim: Input dimension
        **kwargs: Additional model parameters
        
    Returns:
        Initialized model
    """
    if model_type == 'autoencoder':
        return AutoEncoder(input_dim, **kwargs)
    elif model_type == 'vae':
        return VariationalAutoEncoder(input_dim, **kwargs)
    elif model_type == 'deep':
        return DeepAutoEncoder(input_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_memory_usage(model: nn.Module, input_size: Tuple[int, ...], batch_size: int = 1) -> float:
    """
    Estimate model memory usage in MB.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        batch_size: Batch size
        
    Returns:
        Estimated memory usage in MB
    """
    # Parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Input memory
    input_memory = batch_size * torch.prod(torch.tensor(input_size)) * 4  # float32
    
    # Rough estimate of intermediate activations (2x input for safety)
    activation_memory = input_memory * 2
    
    total_memory = param_memory + input_memory + activation_memory
    return float(total_memory / (1024 * 1024))  # Convert to MB