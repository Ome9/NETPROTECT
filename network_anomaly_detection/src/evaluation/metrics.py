"""
Evaluation module for anomaly detection using autoencoders.
Implements threshold determination and comprehensive evaluation metrics.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import pandas as pd
from pathlib import Path


class ThresholdDeterminer:
    """
    Determine optimal threshold for anomaly detection based on reconstruction error.
    """
    
    def __init__(self):
        """Initialize threshold determiner."""
        self.threshold = None
        self.threshold_method = None
        self.statistics = {}
    
    def fit_statistical_threshold(
        self,
        normal_errors: np.ndarray,
        method: str = 'percentile',
        percentile: float = 95.0,
        std_multiplier: float = 3.0
    ) -> float:
        """
        Determine threshold based on normal data statistics.
        
        Args:
            normal_errors: Reconstruction errors from normal data
            method: Method to use ('percentile', 'std', 'iqr')
            percentile: Percentile value for percentile method
            std_multiplier: Standard deviation multiplier
            
        Returns:
            Computed threshold value
        """
        if method == 'percentile':
            threshold = np.percentile(normal_errors, percentile)
        elif method == 'std':
            mean = np.mean(normal_errors)
            std = np.std(normal_errors)
            threshold = mean + std_multiplier * std
        elif method == 'iqr':
            q75, q25 = np.percentile(normal_errors, [75, 25])
            iqr = q75 - q25
            threshold = q75 + 1.5 * iqr
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        self.threshold = threshold
        self.threshold_method = f"{method}_{percentile if method == 'percentile' else std_multiplier}"
        
        # Store statistics
        self.statistics = {
            'mean': float(np.mean(normal_errors)),
            'std': float(np.std(normal_errors)),
            'min': float(np.min(normal_errors)),
            'max': float(np.max(normal_errors)),
            'q25': float(np.percentile(normal_errors, 25)),
            'q50': float(np.percentile(normal_errors, 50)),
            'q75': float(np.percentile(normal_errors, 75)),
            'q95': float(np.percentile(normal_errors, 95)),
            'q99': float(np.percentile(normal_errors, 99)),
        }
        
        return float(threshold)
    
    def fit_optimal_threshold(
        self,
        errors: np.ndarray,
        true_labels: np.ndarray,
        method: str = 'f1'
    ) -> float:
        """
        Find optimal threshold using supervised approach.
        
        Args:
            errors: All reconstruction errors
            true_labels: True binary labels (0=normal, 1=anomaly)
            method: Optimization metric ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            Optimal threshold value
        """
        # Try different thresholds
        thresholds = np.linspace(np.min(errors), np.max(errors), 1000)
        best_score = -1
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            predictions = (errors > threshold).astype(int)
            
            if method == 'f1':
                score = f1_score(true_labels, predictions, zero_division=0)
            elif method == 'precision':
                score = precision_score(true_labels, predictions, zero_division=0)
            elif method == 'recall':
                score = recall_score(true_labels, predictions, zero_division=0)
            elif method == 'accuracy':
                score = accuracy_score(true_labels, predictions)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.threshold = best_threshold
        self.threshold_method = f"optimal_{method}"
        
        return best_threshold
    
    def predict(self, errors: np.ndarray) -> np.ndarray:
        """
        Make predictions based on threshold.
        
        Args:
            errors: Reconstruction errors
            
        Returns:
            Binary predictions (0=normal, 1=anomaly)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_* method first.")
        
        return (errors > self.threshold).astype(int)


class AnomalyDetectionEvaluator:
    """
    Comprehensive evaluation for anomaly detection models.
    """
    
    def __init__(self, attack_type_mapping: Optional[Dict] = None):
        """
        Initialize evaluator.
        
        Args:
            attack_type_mapping: Mapping from attack names to indices
        """
        self.attack_type_mapping = attack_type_mapping
        self.results = {}
    
    def evaluate_binary_classification(
        self,
        errors: np.ndarray,
        true_labels: np.ndarray,
        threshold: float
    ) -> Dict:
        """
        Evaluate binary classification performance.
        
        Args:
            errors: Reconstruction errors
            true_labels: True binary labels (0=normal, 1=anomaly)
            threshold: Classification threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = (errors > threshold).astype(int)
        
        # Basic metrics
        results = {
            'threshold': threshold,
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1_score': f1_score(true_labels, predictions, zero_division=0),
        }
        
        # ROC-AUC
        try:
            results['roc_auc'] = roc_auc_score(true_labels, errors)
        except ValueError:
            results['roc_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        results['confusion_matrix'] = cm.tolist()
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            results['true_negatives'] = int(tn)
            results['false_positives'] = int(fp)
            results['false_negatives'] = int(fn)
            results['true_positives'] = int(tp)
            
            # Additional metrics
            results['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            results['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return results
    
    def evaluate_multiclass_performance(
        self,
        errors: np.ndarray,
        true_multiclass_labels: np.ndarray,
        threshold: float
    ) -> Dict:
        """
        Evaluate performance on different attack types.
        
        Args:
            errors: Reconstruction errors
            true_multiclass_labels: True multiclass labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of per-class metrics
        """
        if self.attack_type_mapping is None:
            return {}
        
        predictions = (errors > threshold).astype(int)
        
        # Create reverse mapping
        idx_to_attack = {idx: attack for attack, idx in self.attack_type_mapping.items()}
        
        results = {}
        
        # Get unique classes
        unique_classes = np.unique(true_multiclass_labels)
        
        for class_idx in unique_classes:
            class_name = idx_to_attack.get(class_idx, f"class_{class_idx}")
            class_mask = true_multiclass_labels == class_idx
            
            if class_name == 'normal':
                # For normal class, correct prediction is 0 (no anomaly)
                class_predictions = predictions[class_mask]
                class_true = np.zeros(len(class_predictions))
            else:
                # For attack classes, correct prediction is 1 (anomaly)
                class_predictions = predictions[class_mask]
                class_true = np.ones(len(class_predictions))
            
            if len(class_predictions) > 0:
                results[class_name] = {
                    'count': len(class_predictions),
                    'accuracy': accuracy_score(class_true, class_predictions),
                    'precision': precision_score(class_true, class_predictions, zero_division=0),
                    'recall': recall_score(class_true, class_predictions, zero_division=0),
                    'f1_score': f1_score(class_true, class_predictions, zero_division=0),
                }
        
        return results
    
    def compute_roc_curve(
        self,
        errors: np.ndarray,
        true_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve.
        
        Args:
            errors: Reconstruction errors
            true_labels: True binary labels
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        return roc_curve(true_labels, errors)
    
    def compute_precision_recall_curve(
        self,
        errors: np.ndarray,
        true_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve.
        
        Args:
            errors: Reconstruction errors
            true_labels: True binary labels
            
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        return precision_recall_curve(true_labels, errors)
    
    def comprehensive_evaluation(
        self,
        errors: np.ndarray,
        binary_labels: np.ndarray,
        multiclass_labels: Optional[np.ndarray] = None,
        threshold_methods: List[str] = ['percentile_95', 'percentile_99', 'optimal_f1']
    ) -> Dict:
        """
        Perform comprehensive evaluation with multiple thresholds.
        
        Args:
            errors: Reconstruction errors
            binary_labels: Binary labels (0=normal, 1=anomaly)
            multiclass_labels: Multiclass labels (optional)
            threshold_methods: Methods for threshold determination
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            'error_statistics': {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'min': float(np.min(errors)),
                'max': float(np.max(errors)),
                'median': float(np.median(errors)),
            },
            'threshold_results': {}
        }
        
        # Separate normal errors for statistical thresholds
        normal_mask = binary_labels == 0
        normal_errors = errors[normal_mask] if np.any(normal_mask) else errors
        
        for method in threshold_methods:
            threshold_det = ThresholdDeterminer()
            
            if method.startswith('percentile_'):
                percentile = float(method.split('_')[1])
                threshold = threshold_det.fit_statistical_threshold(
                    normal_errors, method='percentile', percentile=percentile
                )
            elif method.startswith('std_'):
                multiplier = float(method.split('_')[1])
                threshold = threshold_det.fit_statistical_threshold(
                    normal_errors, method='std', std_multiplier=multiplier
                )
            elif method.startswith('optimal_'):
                metric = method.split('_')[1]
                threshold = threshold_det.fit_optimal_threshold(
                    errors, binary_labels, method=metric
                )
            else:
                continue
            
            # Binary evaluation
            binary_results = self.evaluate_binary_classification(
                errors, binary_labels, threshold
            )
            
            # Multiclass evaluation
            multiclass_results = {}
            if multiclass_labels is not None:
                multiclass_results = self.evaluate_multiclass_performance(
                    errors, multiclass_labels, threshold
                )
            
            results['threshold_results'][method] = {
                'threshold': threshold,
                'binary_metrics': binary_results,
                'multiclass_metrics': multiclass_results,
                'statistics': threshold_det.statistics
            }
        
        # ROC and PR curves
        fpr, tpr, roc_thresholds = self.compute_roc_curve(errors, binary_labels)
        precision, recall, pr_thresholds = self.compute_precision_recall_curve(errors, binary_labels)
        
        results['curves'] = {
            'roc': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'precision_recall': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        }
        
        return results
    
    def plot_error_distribution(
        self,
        normal_errors: np.ndarray,
        anomaly_errors: np.ndarray,
        threshold: Optional[float] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot reconstruction error distribution.
        
        Args:
            normal_errors: Errors from normal samples
            anomaly_errors: Errors from anomalous samples
            threshold: Threshold to mark on plot
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot histograms
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        
        # Add threshold line
        if threshold is not None:
            plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Reconstruction Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: AUC score
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 8))
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = ['Normal', 'Anomaly'],
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Class names
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, results: Dict, save_path: str):
        """Save evaluation results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_json = convert_numpy(results)
        
        with open(save_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Results saved to {save_path}")


def create_evaluation_report(results: Dict, save_path: Optional[str] = None) -> str:
    """
    Create a comprehensive evaluation report.
    
    Args:
        results: Evaluation results dictionary
        save_path: Optional path to save the report
        
    Returns:
        Report string
    """
    report = "# Network Anomaly Detection Evaluation Report\n\n"
    
    # Error statistics
    report += "## Reconstruction Error Statistics\n\n"
    error_stats = results['error_statistics']
    report += f"- Mean: {error_stats['mean']:.6f}\n"
    report += f"- Standard Deviation: {error_stats['std']:.6f}\n"
    report += f"- Minimum: {error_stats['min']:.6f}\n"
    report += f"- Maximum: {error_stats['max']:.6f}\n"
    report += f"- Median: {error_stats['median']:.6f}\n\n"
    
    # Threshold results
    report += "## Threshold-based Evaluation Results\n\n"
    
    for method, method_results in results['threshold_results'].items():
        report += f"### {method.replace('_', ' ').title()}\n\n"
        
        threshold = method_results['threshold']
        binary_metrics = method_results['binary_metrics']
        
        report += f"- Threshold: {threshold:.6f}\n"
        report += f"- Accuracy: {binary_metrics['accuracy']:.4f}\n"
        report += f"- Precision: {binary_metrics['precision']:.4f}\n"
        report += f"- Recall: {binary_metrics['recall']:.4f}\n"
        report += f"- F1-Score: {binary_metrics['f1_score']:.4f}\n"
        report += f"- ROC-AUC: {binary_metrics['roc_auc']:.4f}\n"
        
        if 'true_positives' in binary_metrics:
            report += f"- True Positives: {binary_metrics['true_positives']}\n"
            report += f"- False Positives: {binary_metrics['false_positives']}\n"
            report += f"- True Negatives: {binary_metrics['true_negatives']}\n"
            report += f"- False Negatives: {binary_metrics['false_negatives']}\n"
        
        # Multiclass results
        if method_results['multiclass_metrics']:
            report += "\n#### Per-Attack Type Performance:\n\n"
            for attack_type, metrics in method_results['multiclass_metrics'].items():
                report += f"**{attack_type}** (n={metrics['count']})\n"
                report += f"- Accuracy: {metrics['accuracy']:.4f}\n"
                report += f"- Precision: {metrics['precision']:.4f}\n"
                report += f"- Recall: {metrics['recall']:.4f}\n"
                report += f"- F1-Score: {metrics['f1_score']:.4f}\n\n"
        
        report += "\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {save_path}")
    
    return report