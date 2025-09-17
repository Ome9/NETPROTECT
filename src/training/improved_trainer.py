"""
Improved trainer that addresses all identified anomaly detection issues.
This trainer implements proper anomaly detection methodology.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pickle
import os
import logging
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)
from typing import Tuple, Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns


class ImprovedTrainer:
    """
    Improved trainer that implements proper anomaly detection methodology:
    - Uses ONLY normal samples for training
    - Proper threshold selection based on normal data statistics
    - Robust evaluation metrics for imbalanced data
    - Advanced GPU optimization with mixed precision
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_mixed_precision: bool = True,
        results_dir: str = "results",
        model_name: str = "improved_autoencoder"
    ):
        self.model = model
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision components
        if self.use_mixed_precision:
            self.scaler = GradScaler()
        
        # Setup logging
        self._setup_logging()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        # Threshold and evaluation
        self.optimal_threshold = None
        self.threshold_stats = {}
        
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.results_dir / f"{self.model_name}_training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def extract_normal_samples(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract only normal samples for training.
        This is CRITICAL for proper anomaly detection.
        """
        normal_mask = y == 0  # Assuming 0 = normal, 1 = anomaly
        normal_X = X[normal_mask]
        
        self.logger.info(f"Total samples: {len(X)}")
        self.logger.info(f"Normal samples extracted: {len(normal_X)}")
        self.logger.info(f"Anomaly samples excluded: {len(X) - len(normal_X)}")
        
        return normal_X
    
    def train(
        self,
        train_data: torch.utils.data.DataLoader,
        val_data: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 10,
        save_best_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train the autoencoder using ONLY normal samples.
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING IMPROVED ANOMALY DETECTION TRAINING")
        self.logger.info("=" * 60)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.01
        )
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Training phase
            for batch_X, batch_y in train_data:
                # CRITICAL: Extract only normal samples
                normal_X = self.extract_normal_samples(batch_X, batch_y)
                
                if len(normal_X) == 0:
                    continue  # Skip if no normal samples in batch
                
                normal_X = normal_X.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(normal_X)
                        loss = criterion(outputs, normal_X)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(normal_X)
                    loss = criterion(outputs, normal_X)
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            self.train_losses.append(avg_loss)
            
            # Validation phase
            val_loss = 0.0
            if val_data is not None:
                val_loss = self._validate(val_data, criterion)
                self.val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_best_model:
                        self._save_model("best")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                lr = optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Loss: {avg_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {lr:.2e}"
                )
        
        # Final model save
        if save_best_model:
            self._save_model("final")
        
        training_results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
        
        self.logger.info("Training completed successfully!")
        return training_results
    
    def _validate(
        self,
        val_data: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> float:
        """Validate using only normal samples."""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_data:
                # Extract only normal samples for validation
                normal_X = self.extract_normal_samples(batch_X, batch_y)
                
                if len(normal_X) == 0:
                    continue
                
                normal_X = normal_X.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(normal_X)
                        loss = criterion(outputs, normal_X)
                else:
                    outputs = self.model(normal_X)
                    loss = criterion(outputs, normal_X)
                
                val_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return val_loss / num_batches if num_batches > 0 else 0
    
    def calibrate_threshold(
        self,
        normal_data: torch.utils.data.DataLoader,
        percentile: float = 95.0
    ) -> float:
        """
        Calibrate threshold using ONLY normal training data.
        This is the CORRECT way to set threshold for anomaly detection.
        """
        self.logger.info("=" * 60)
        self.logger.info("CALIBRATING OPTIMAL THRESHOLD")
        self.logger.info("=" * 60)
        
        self.model.eval()
        normal_errors = []
        
        with torch.no_grad():
            for batch_X, batch_y in normal_data:
                # Extract only normal samples
                normal_X = self.extract_normal_samples(batch_X, batch_y)
                
                if len(normal_X) == 0:
                    continue
                
                normal_X = normal_X.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(normal_X)
                        errors = torch.mean((outputs - normal_X) ** 2, dim=1)
                else:
                    outputs = self.model(normal_X)
                    errors = torch.mean((outputs - normal_X) ** 2, dim=1)
                
                normal_errors.extend(errors.cpu().numpy())
        
        normal_errors = np.array(normal_errors)
        
        # Calculate threshold statistics
        self.threshold_stats = {
            'mean': np.mean(normal_errors),
            'std': np.std(normal_errors),
            'median': np.median(normal_errors),
            'p90': np.percentile(normal_errors, 90),
            'p95': np.percentile(normal_errors, 95),
            'p99': np.percentile(normal_errors, 99),
            'max': np.max(normal_errors)
        }
        
        # Set threshold based on percentile
        self.optimal_threshold = np.percentile(normal_errors, percentile)
        
        self.logger.info(f"Normal error statistics:")
        self.logger.info(f"  Count: {len(normal_errors)}")
        self.logger.info(f"  Mean: {self.threshold_stats['mean']:.6f}")
        self.logger.info(f"  Std: {self.threshold_stats['std']:.6f}")
        self.logger.info(f"  Median: {self.threshold_stats['median']:.6f}")
        self.logger.info(f"  95th percentile: {self.threshold_stats['p95']:.6f}")
        self.logger.info(f"  99th percentile: {self.threshold_stats['p99']:.6f}")
        self.logger.info(f"Optimal threshold ({percentile}th percentile): {self.optimal_threshold:.6f}")
        
        return self.optimal_threshold
    
    def evaluate_comprehensive(
        self,
        test_data: torch.utils.data.DataLoader,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation with proper anomaly detection metrics.
        """
        if threshold is None:
            threshold = self.optimal_threshold
            if threshold is None:
                raise ValueError("No threshold set. Run calibrate_threshold first.")
        
        self.logger.info("=" * 60)
        self.logger.info("COMPREHENSIVE EVALUATION")
        self.logger.info("=" * 60)
        
        self.model.eval()
        all_errors = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_data:
                batch_X = batch_X.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(batch_X)
                        errors = torch.mean((outputs - batch_X) ** 2, dim=1)
                else:
                    outputs = self.model(batch_X)
                    errors = torch.mean((outputs - batch_X) ** 2, dim=1)
                
                all_errors.extend(errors.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        all_errors = np.array(all_errors)
        all_labels = np.array(all_labels)
        
        # Generate predictions
        predictions = (all_errors > threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, predictions)
        precision = precision_score(all_labels, predictions, zero_division=0)
        recall = recall_score(all_labels, predictions, zero_division=0)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        roc_auc = roc_auc_score(all_labels, all_errors)
        avg_precision = average_precision_score(all_labels, all_errors)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, predictions)
        
        # Calculate class-specific metrics
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
            'total_samples': len(all_labels),
            'errors': all_errors,
            'labels': all_labels,
            'predictions': predictions
        }
        
        # Logging
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  Threshold: {threshold:.6f}")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Precision: {precision:.4f}")
        self.logger.info(f"  Recall: {recall:.4f}")
        self.logger.info(f"  F1-Score: {f1:.4f}")
        self.logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        self.logger.info(f"  Average Precision: {avg_precision:.4f}")
        self.logger.info(f"  Normal samples: {normal_count}")
        self.logger.info(f"  Anomaly samples: {anomaly_count}")
        
        # Save results
        self._save_evaluation_results(results)
        self._plot_evaluation_results(results)
        
        return results
    
    def _save_model(self, suffix: str) -> None:
        """Save model and training state."""
        model_path = self.results_dir / f"{self.model_name}_{suffix}.pth"
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'optimal_threshold': self.optimal_threshold,
            'threshold_stats': self.threshold_stats
        }
        
        torch.save(save_dict, model_path)
        self.logger.info(f"Model saved: {model_path}")
    
    def _save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results."""
        results_path = self.results_dir / f"{self.model_name}_evaluation.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        self.logger.info(f"Evaluation results saved: {results_path}")
    
    def _plot_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive evaluation plots."""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Training curves
        if self.train_losses and self.val_losses:
            axes[0, 0].plot(self.train_losses, label='Training Loss', alpha=0.8)
            axes[0, 0].plot(self.val_losses, label='Validation Loss', alpha=0.8)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Curves')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error distributions
        normal_errors = results['errors'][results['labels'] == 0]
        anomaly_errors = results['errors'][results['labels'] == 1]
        
        axes[0, 1].hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True)
        axes[0, 1].hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', density=True)
        axes[0, 1].axvline(results['threshold'], color='red', linestyle='--', label='Threshold')
        axes[0, 1].set_xlabel('Reconstruction Error')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Error Distributions')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # 3. Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
                   ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
        
        # 4. Metrics summary
        metrics_text = f"""
        Evaluation Metrics:
        
        Accuracy:    {results['accuracy']:.4f}
        Precision:   {results['precision']:.4f}
        Recall:      {results['recall']:.4f}
        F1-Score:    {results['f1_score']:.4f}
        ROC-AUC:     {results['roc_auc']:.4f}
        Avg Precision: {results['average_precision']:.4f}
        
        Dataset:
        Normal:      {results['normal_samples']:,}
        Anomaly:     {results['anomaly_samples']:,}
        Total:       {results['total_samples']:,}
        
        Threshold:   {results['threshold']:.6f}
        """
        
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Evaluation Summary')
        
        plt.tight_layout()
        plot_path = self.results_dir / f"{self.model_name}_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Evaluation plots saved: {plot_path}")

    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Load a saved model and training state."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore training history if available
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'optimal_threshold' in checkpoint:
            self.optimal_threshold = checkpoint['optimal_threshold']
        if 'threshold_stats' in checkpoint:
            self.threshold_stats = checkpoint['threshold_stats']
        
        self.logger.info(f"Model loaded from: {model_path}")
        return checkpoint