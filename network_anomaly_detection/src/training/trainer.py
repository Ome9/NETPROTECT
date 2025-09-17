"""
Training module for autoencoder models.
Implements training loops with various optimizers, schedulers, and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import logging
from pathlib import Path
import json


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
            verbose: Whether to print early stopping info
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Validation loss
            model: Model to potentially save weights
            
        Returns:
            Whether to stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping triggered after {self.patience} epochs without improvement")
            
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict({k: v.to(next(model.parameters()).device) 
                                     for k, v in self.best_weights.items()})
                if self.verbose:
                    print("Restored best weights")
        
        return self.early_stop


class AutoEncoderTrainer:
    """
    Trainer class for autoencoder models with comprehensive training features.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer_name: str = 'adam',
        learning_rate: float = 1e-3,
        scheduler_name: Optional[str] = 'reduce_plateau',
        early_stopping_patience: int = 15,
        gradient_clip_value: Optional[float] = 1.0,
        log_interval: int = 100
    ):
        """
        Initialize trainer.
        
        Args:
            model: Autoencoder model
            device: Training device
            optimizer_name: Optimizer type ('adam', 'rmsprop', 'sgd')
            learning_rate: Learning rate
            scheduler_name: Scheduler type ('reduce_plateau', 'step', 'cosine')
            early_stopping_patience: Early stopping patience
            gradient_clip_value: Gradient clipping value
            log_interval: Logging interval
        """
        self.model = model.to(device)
        self.device = device
        self.log_interval = log_interval
        self.gradient_clip_value = gradient_clip_value
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer_name, learning_rate)
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler(scheduler_name) if scheduler_name else None
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Initialize mixed precision training
        self.use_amp = device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _create_optimizer(self, optimizer_name: str, learning_rate: float) -> optim.Optimizer:
        """Create optimizer based on name."""
        optimizer_name = optimizer_name.lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-2)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_scheduler(self, scheduler_name: str) -> Optional[Any]:
        """Create learning rate scheduler."""
        scheduler_name = scheduler_name.lower()
        
        if scheduler_name == 'reduce_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        elif scheduler_name == 'exponential':
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    def _compute_loss(self, batch: Dict, loss_type: str = 'mse') -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            batch: Batch of data
            loss_type: Loss function type
            
        Returns:
            Computed loss
        """
        features = batch['features'].to(self.device)
        
        # Handle different model types
        if hasattr(self.model, 'compute_loss') and callable(getattr(self.model, 'compute_loss', None)):  # VAE
            loss, recon_loss, kl_loss = self.model.compute_loss(features)
            return loss
        else:  # Regular autoencoder
            return self.model.compute_reconstruction_error(features, error_type=loss_type)
    
    def train_epoch(self, dataloader: DataLoader, loss_type: str = 'mse') -> float:
        """
        Train for one epoch with mixed precision support.
        
        Args:
            dataloader: Training dataloader
            loss_type: Loss function type
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Zero gradients
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                # Mixed precision training
                with autocast():
                    loss = self._compute_loss(batch, loss_type)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_value is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                
                # Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training
                loss = self._compute_loss(batch, loss_type)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                
                # Optimizer step
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Logging
            if batch_idx % self.log_interval == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.6f}'
                )
        
        return total_loss / num_batches
    
    def validate_epoch(self, dataloader: DataLoader, loss_type: str = 'mse') -> float:
        """
        Validate for one epoch.
        
        Args:
            dataloader: Validation dataloader
            loss_type: Loss function type
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                loss = self._compute_loss(batch, loss_type)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        loss_type: str = 'mse',
        save_path: Optional[str] = None,
        save_interval: int = 10
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            loss_type: Loss function type
            save_path: Path to save model checkpoints
            save_interval: Interval for saving checkpoints
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_dataloader, loss_type)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            val_loss = None
            if val_dataloader is not None:
                val_loss = self.validate_epoch(val_dataloader, loss_type)
                self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    self.scheduler.step()
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Record epoch time
            epoch_time = time.time() - epoch_start_time
            self.history['epoch_time'].append(epoch_time)
            
            # Logging
            log_message = (
                f'Epoch {epoch+1}/{num_epochs}: '
                f'Train Loss: {train_loss:.6f}'
            )
            if val_loss is not None:
                log_message += f', Val Loss: {val_loss:.6f}'
            log_message += f', LR: {current_lr:.8f}, Time: {epoch_time:.2f}s'
            
            self.logger.info(log_message)
            
            # Early stopping
            if val_loss is not None:
                if self.early_stopping(val_loss, self.model):
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Save checkpoint
            if save_path and (epoch + 1) % save_interval == 0:
                checkpoint_path = f"{save_path}_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path, epoch, train_loss, val_loss)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model
        if save_path:
            final_path = f"{save_path}_final.pt"
            self.save_checkpoint(final_path, num_epochs-1, train_loss, val_loss)
        
        return self.history
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        self.logger.info(f"Checkpoint loaded from {path}")
        return checkpoint


class ModelEvaluator:
    """Evaluate trained autoencoder models."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Computation device
        """
        self.model = model.to(device)
        self.device = device
        
    def compute_reconstruction_errors(
        self,
        dataloader: DataLoader,
        error_type: str = 'mse'
    ) -> np.ndarray:
        """
        Compute reconstruction errors for all samples.
        
        Args:
            dataloader: Data loader
            error_type: Error type ('mse', 'mae', 'rmse')
            
        Returns:
            Array of reconstruction errors
        """
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                
                if error_type == 'sample_wise':
                    # Compute error for each sample separately
                    if hasattr(self.model, 'compute_reconstruction_error') and callable(getattr(self.model, 'compute_reconstruction_error', None)):
                        error = self.model.compute_reconstruction_error(
                            features, reduction='none', error_type='mse'
                        )
                    else:
                        # Fallback for models without this method
                        reconstructed, _ = self.model(features)
                        error = torch.mean((features - reconstructed) ** 2, dim=1)
                    # Mean across features for each sample
                    if len(error.shape) > 1:
                        error = error.mean(dim=1)
                else:
                    if hasattr(self.model, 'compute_reconstruction_error') and callable(getattr(self.model, 'compute_reconstruction_error', None)):
                        error = self.model.compute_reconstruction_error(
                            features, reduction='none', error_type=error_type
                        )
                    else:
                        # Handle different model output formats
                        model_output = self.model(features)
                        if isinstance(model_output, tuple) and len(model_output) >= 2:
                            reconstructed = model_output[0]  # First element is always reconstruction
                        else:
                            reconstructed = model_output
                        
                        if error_type == 'mse':
                            error = torch.mean((features - reconstructed) ** 2, dim=1)
                        elif error_type == 'mae':
                            error = torch.mean(torch.abs(features - reconstructed), dim=1)
                        else:
                            error = torch.mean((features - reconstructed) ** 2, dim=1)
                    
                    if len(error.shape) > 1:
                        error = error.mean(dim=1)
                
                errors.extend(error.cpu().numpy())
        
        return np.array(errors)
    
    def get_latent_representations(self, dataloader: DataLoader) -> np.ndarray:
        """Get latent representations for all samples."""
        self.model.eval()
        representations = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                
                if hasattr(self.model, 'encode') and callable(getattr(self.model, 'encode', None)):
                    if hasattr(self.model, 'reparameterize') and callable(getattr(self.model, 'reparameterize', None)):  # VAE
                        mu, logvar = self.model.encode(features)
                        z = self.model.reparameterize(mu, logvar)
                    else:  # Regular autoencoder
                        z = self.model.encode(features)
                else:
                    # If model doesn't have encode method, use forward and get latent
                    _, z = self.model(features)
                
                representations.extend(z.cpu().numpy())
        
        return np.array(representations)


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger