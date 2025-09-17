#!/usr/bin/env python3
"""
Comprehensive anomaly detection analysis and debugging script.
This script addresses the critical issues identified in the current implementation.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import torch
import warnings
warnings.filterwarnings('ignore')

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir / 'src'
sys.path.insert(0, str(src_dir))

from utils.config import ConfigManager  # type: ignore
from data.preprocessing import DataProcessor  # type: ignore
from models.autoencoder import create_model  # type: ignore
from training.trainer import ModelEvaluator  # type: ignore

class AnomalyDetectionAnalyzer:
    """Comprehensive analyzer for anomaly detection model debugging."""
    
    def __init__(self, model_path, config_path, preprocessor_path, data_path):
        self.model_path = model_path
        self.config_path = config_path
        self.preprocessor_path = preprocessor_path
        self.data_path = data_path
        
        # Load components
        self._load_config()
        self._load_model()
        self._load_data()
        
    def _load_config(self):
        """Load configuration."""
        config_manager = ConfigManager()
        self.config = config_manager.load_config(self.config_path)
        self.device = config_manager.get_device()
        
    def _load_model(self):
        """Load trained model."""
        # Create model
        model_kwargs = {
            'latent_dim': self.config.model.latent_dim,
            'activation': self.config.model.activation,
            'use_batch_norm': self.config.model.use_batch_norm,
            'dropout_rate': self.config.model.dropout_rate
        }
        
        if self.config.model.type == 'autoencoder':
            if hasattr(self.config.model, 'hidden_dims'):
                model_kwargs['hidden_dims'] = self.config.model.hidden_dims
        
        self.model = create_model(
            self.config.model.type,
            self.config.model.input_dim,
            **model_kwargs
        )
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
    def _load_data(self):
        """Load and preprocess data."""
        # Load data processor
        self.data_processor = DataProcessor(self.data_path)
        self.data_processor.load_preprocessor(self.preprocessor_path)
        
        # Load raw data
        train_df, test_df = self.data_processor.load_data(
            self.config.data.train_file, 
            self.config.data.test_file
        )
        
        # Preprocess features
        train_features, test_features = self.data_processor.preprocess_features(train_df, test_df)
        train_labels, test_labels = self.data_processor.prepare_labels(train_df, test_df)
        
        # Extract normal training data (THIS IS CRITICAL!)
        self.normal_features = self.data_processor.get_normal_data(train_features, train_labels)
        
        self.train_features = train_features
        self.test_features = test_features
        self.train_labels = train_labels
        self.test_labels = test_labels
        
        print(f"Data loaded:")
        print(f"  Total training samples: {len(train_features)}")
        print(f"  Normal training samples: {len(self.normal_features)}")
        print(f"  Test samples: {len(test_features)}")
        print(f"  Normal test samples: {np.sum(test_labels['binary'] == 0)}")
        print(f"  Anomaly test samples: {np.sum(test_labels['binary'] == 1)}")
        
    def verify_training_data_purity(self):
        """CRITICAL: Verify that training data contains only normal samples."""
        print("\n" + "="*60)
        print("VERIFYING TRAINING DATA PURITY")
        print("="*60)
        
        # Check if any anomalies leaked into training
        normal_mask = self.train_labels['binary'] == 0
        anomaly_mask = self.train_labels['binary'] == 1
        
        total_train = len(self.train_features)
        normal_train = np.sum(normal_mask)
        anomaly_train = np.sum(anomaly_mask)
        
        print(f"Training data composition:")
        print(f"  Total samples: {total_train}")
        print(f"  Normal samples: {normal_train} ({normal_train/total_train*100:.1f}%)")
        print(f"  Anomaly samples: {anomaly_train} ({anomaly_train/total_train*100:.1f}%)")
        
        # Check what was actually used for training
        normal_used = len(self.normal_features)
        print(f"\\nActually used for training:")
        print(f"  Normal samples only: {normal_used}")
        print(f"  ✓ Training data is pure!" if normal_used == normal_train else "⚠️ Data leakage detected!")
        
        if anomaly_train > 0:
            print(f"\\n⚠️  WARNING: {anomaly_train} anomaly samples found in training set!")
            print("   This could cause the autoencoder to learn anomaly patterns.")
            print("   Recommendation: Remove all anomalies from training data.")
        
        return normal_used == normal_train and anomaly_train == 0
        
    def compute_reconstruction_errors(self):
        """Compute reconstruction errors for all samples."""
        print("\n" + "="*60)
        print("COMPUTING RECONSTRUCTION ERRORS")
        print("="*60)
        
        model_evaluator = ModelEvaluator(self.model, self.device)
        
        # Create dataloaders
        normal_dataloader = self.data_processor.create_dataloaders(
            self.normal_features, batch_size=512, shuffle=False, val_split=0.0, num_workers=2
        )['train']
        
        test_dataloader = self.data_processor.create_dataloaders(
            self.test_features, batch_size=512, shuffle=False, val_split=0.0, num_workers=2
        )['train']
        
        # Compute errors
        print("Computing normal training errors...")
        self.normal_train_errors = model_evaluator.compute_reconstruction_errors(
            normal_dataloader, 'mse'
        )
        
        print("Computing test errors...")
        self.test_errors = model_evaluator.compute_reconstruction_errors(
            test_dataloader, 'mse'
        )
        
        # Separate test errors by class
        normal_test_mask = self.test_labels['binary'] == 0
        anomaly_test_mask = self.test_labels['binary'] == 1
        
        self.normal_test_errors = self.test_errors[normal_test_mask]
        self.anomaly_test_errors = self.test_errors[anomaly_test_mask]
        
        print(f"Reconstruction errors computed:")
        print(f"  Normal training: {len(self.normal_train_errors)} samples")
        print(f"  Normal test: {len(self.normal_test_errors)} samples")
        print(f"  Anomaly test: {len(self.anomaly_test_errors)} samples")
        
        return {
            'normal_train': self.normal_train_errors,
            'normal_test': self.normal_test_errors,
            'anomaly_test': self.anomaly_test_errors,
            'all_test': self.test_errors
        }
        
    def analyze_error_distributions(self):
        """Analyze and visualize error distributions."""
        print("\n" + "="*60)
        print("ANALYZING ERROR DISTRIBUTIONS")
        print("="*60)
        
        # Statistical analysis
        stats = {}
        for name, errors in [
            ('Normal Training', self.normal_train_errors),
            ('Normal Test', self.normal_test_errors), 
            ('Anomaly Test', self.anomaly_test_errors)
        ]:
            stats[name] = {
                'count': len(errors),
                'mean': np.mean(errors),
                'std': np.std(errors),
                'min': np.min(errors),
                'max': np.max(errors),
                'median': np.median(errors),
                'p95': np.percentile(errors, 95),
                'p99': np.percentile(errors, 99)
            }
            
        # Print statistics
        for name, stat in stats.items():
            print(f"\\n{name}:")
            print(f"  Count: {stat['count']}")
            print(f"  Mean ± Std: {stat['mean']:.6f} ± {stat['std']:.6f}")
            print(f"  Range: [{stat['min']:.6f}, {stat['max']:.6f}]")
            print(f"  Median: {stat['median']:.6f}")
            print(f"  95th percentile: {stat['p95']:.6f}")
            print(f"  99th percentile: {stat['p99']:.6f}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reconstruction Error Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram comparison
        ax1 = axes[0, 0]
        ax1.hist(self.normal_test_errors, bins=50, alpha=0.7, label='Normal Test', color='blue', density=True)
        ax1.hist(self.anomaly_test_errors, bins=50, alpha=0.7, label='Anomaly Test', color='red', density=True)
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Density')
        ax1.set_title('Error Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Log scale histogram
        ax2 = axes[0, 1]
        ax2.hist(self.normal_test_errors, bins=50, alpha=0.7, label='Normal Test', color='blue', density=True)
        ax2.hist(self.anomaly_test_errors, bins=50, alpha=0.7, label='Anomaly Test', color='red', density=True)
        ax2.set_xlabel('Reconstruction Error (log scale)')
        ax2.set_ylabel('Density')
        ax2.set_title('Error Distribution (Log Scale)')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plots
        ax3 = axes[1, 0]
        data_to_plot = [self.normal_train_errors, self.normal_test_errors, self.anomaly_test_errors]
        box_plot = ax3.boxplot(data_to_plot, labels=['Normal\\nTrain', 'Normal\\nTest', 'Anomaly\\nTest'], patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        ax3.set_ylabel('Reconstruction Error')
        ax3.set_title('Error Distribution Box Plots')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative distribution
        ax4 = axes[1, 1]
        sorted_normal = np.sort(self.normal_test_errors)
        sorted_anomaly = np.sort(self.anomaly_test_errors)
        ax4.plot(sorted_normal, np.linspace(0, 1, len(sorted_normal)), label='Normal Test', color='blue')
        ax4.plot(sorted_anomaly, np.linspace(0, 1, len(sorted_anomaly)), label='Anomaly Test', color='red')
        ax4.set_xlabel('Reconstruction Error')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution Function')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reconstruction_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Separation analysis
        print(f"\\n" + "="*40)
        print("DISTRIBUTION SEPARATION ANALYSIS")
        print("="*40)
        
        # Check overlap
        normal_max = np.max(self.normal_test_errors)
        anomaly_min = np.min(self.anomaly_test_errors)
        
        print(f"Normal test max error: {normal_max:.6f}")
        print(f"Anomaly test min error: {anomaly_min:.6f}")
        
        if normal_max < anomaly_min:
            print("✓ Perfect separation: No overlap between distributions!")
            overlap_count = 0
        else:
            # Count overlap more carefully
            normal_above_anomaly_min = np.sum(self.normal_test_errors > anomaly_min)
            anomaly_below_normal_max = np.sum(self.anomaly_test_errors < normal_max)
            overlap_count = normal_above_anomaly_min + anomaly_below_normal_max
            print(f"⚠️ Overlap detected:")
            print(f"   Normal samples above anomaly min: {normal_above_anomaly_min}")
            print(f"   Anomaly samples below normal max: {anomaly_below_normal_max}")
            print(f"   Total overlap samples: {overlap_count}")
            print(f"   Overlap ratio: {overlap_count / len(self.test_errors) * 100:.2f}%")
        
        # Mean separation
        normal_mean = np.mean(self.normal_test_errors)
        anomaly_mean = np.mean(self.anomaly_test_errors)
        separation_ratio = anomaly_mean / normal_mean
        
        print(f"\\nMean separation:")
        print(f"  Normal mean: {normal_mean:.6f}")
        print(f"  Anomaly mean: {anomaly_mean:.6f}")
        print(f"  Separation ratio: {separation_ratio:.2f}x")
        
        if separation_ratio < 2.0:
            print("⚠️ Poor separation: Anomaly errors should be much higher than normal errors")
            print("   Recommendation: Reduce model capacity or adjust architecture")
        else:
            print("✓ Good separation: Anomalies have significantly higher errors")
        
        return stats
        
    def optimize_threshold_selection(self):
        """Implement proper threshold selection using validation data."""
        print("\n" + "="*60)
        print("OPTIMIZING THRESHOLD SELECTION")
        print("="*60)
        
        # Use normal training errors to determine thresholds
        print("Using normal training errors for threshold calibration...")
        
        # Calculate percentile-based thresholds from NORMAL training data
        thresholds = {
            'p90': np.percentile(self.normal_train_errors, 90),
            'p95': np.percentile(self.normal_train_errors, 95),
            'p99': np.percentile(self.normal_train_errors, 99),
            'mean_plus_2std': np.mean(self.normal_train_errors) + 2 * np.std(self.normal_train_errors),
            'mean_plus_3std': np.mean(self.normal_train_errors) + 3 * np.std(self.normal_train_errors),
        }
        
        print("\\nThreshold candidates (based on normal training data):")
        for name, thresh in thresholds.items():
            print(f"  {name}: {thresh:.6f}")
        
        # Evaluate each threshold
        test_binary_labels = self.test_labels['binary']
        results = {}
        
        for name, threshold in thresholds.items():
            predictions = (self.test_errors > threshold).astype(int)
            
            # Calculate metrics
            tp = np.sum((predictions == 1) & (test_binary_labels == 1))
            fp = np.sum((predictions == 1) & (test_binary_labels == 0))
            tn = np.sum((predictions == 0) & (test_binary_labels == 0))
            fn = np.sum((predictions == 0) & (test_binary_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            
            results[name] = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            }
        
        # Find optimal thresholds
        best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
        best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
        
        print("\\nThreshold evaluation results:")
        print("-" * 80)
        print(f"{'Threshold':<15} {'Value':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10}")
        print("-" * 80)
        
        for name, result in results.items():
            print(f"{name:<15} {result['threshold']:<12.6f} {result['precision']:<10.4f} "
                  f"{result['recall']:<10.4f} {result['f1']:<10.4f} {result['accuracy']:<10.4f}")
        
        print("-" * 80)
        print(f"Best F1: {best_f1[0]} (F1={best_f1[1]['f1']:.4f})")
        print(f"Best Accuracy: {best_accuracy[0]} (Acc={best_accuracy[1]['accuracy']:.4f})")
        
        return results, best_f1, best_accuracy
        
    def implement_proper_metrics(self):
        """Implement proper anomaly detection metrics."""
        print("\n" + "="*60)
        print("IMPLEMENTING PROPER ANOMALY DETECTION METRICS")
        print("="*60)
        
        test_binary_labels = self.test_labels['binary']
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(test_binary_labels, self.test_errors)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve (better for imbalanced data)
        precision, recall, pr_thresholds = precision_recall_curve(test_binary_labels, self.test_errors)
        avg_precision = average_precision_score(test_binary_labels, self.test_errors)
        
        print(f"Advanced Metrics:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Average Precision (AP): {avg_precision:.4f}")
        print(f"  Number of normal samples: {np.sum(test_binary_labels == 0)}")
        print(f"  Number of anomaly samples: {np.sum(test_binary_labels == 1)}")
        print(f"  Imbalance ratio: {np.sum(test_binary_labels == 0) / np.sum(test_binary_labels == 1):.1f}:1")
        
        # Plot ROC and PR curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='blue', linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        ax2.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.4f})', color='red', linewidth=2)
        ax2.axhline(y=np.sum(test_binary_labels == 1) / len(test_binary_labels), 
                   color='k', linestyle='--', alpha=0.5, label='Random Classifier')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall
        }
        
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        # Run all analyses
        purity_check = self.verify_training_data_purity()
        error_stats = self.analyze_error_distributions()
        threshold_results, best_f1, best_accuracy = self.optimize_threshold_selection()
        metrics = self.implement_proper_metrics()
        
        # Generate recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        if not purity_check:
            recommendations.append("🔴 CRITICAL: Remove all anomaly samples from training data")
        
        normal_mean = np.mean(self.normal_test_errors)
        anomaly_mean = np.mean(self.anomaly_test_errors)
        separation_ratio = anomaly_mean / normal_mean
        
        if separation_ratio < 2.0:
            recommendations.append("🟡 Model Architecture: Reduce bottleneck size for better anomaly detection")
            recommendations.append("🟡 Regularization: Add dropout or L1/L2 regularization")
        
        if metrics['average_precision'] < 0.8:
            recommendations.append("🟡 Model Training: Increase epochs or adjust learning rate")
        
        if best_f1[1]['f1'] < 0.7:
            recommendations.append("🟡 Threshold: Use validation-based threshold selection")
        
        if len(recommendations) == 0:
            recommendations.append("✅ Model performance looks good!")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Summary
        print(f"\\n" + "="*40)
        print("SUMMARY")
        print("="*40)
        print(f"Training Data Purity: {'✅ Clean' if purity_check else '❌ Contains anomalies'}")
        print(f"Error Separation Ratio: {separation_ratio:.2f}x")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        print(f"Best F1 Threshold: {best_f1[0]} ({best_f1[1]['f1']:.4f})")
        print(f"Recommended Threshold: {best_f1[1]['threshold']:.6f}")
        
        return {
            'purity_check': purity_check,
            'error_stats': error_stats,
            'threshold_results': threshold_results,
            'metrics': metrics,
            'recommendations': recommendations,
            'best_threshold': best_f1[1]['threshold']
        }

def main():
    """Run comprehensive anomaly detection analysis."""
    print("🔍 COMPREHENSIVE ANOMALY DETECTION ANALYSIS")
    print("=" * 80)
    
    # Use the most recent GPU-optimized model
    model_path = "../models/gpu_optimized_autoencoder/gpu_optimized_autoencoder_model_final.pt"
    config_path = "../results/gpu_optimized_autoencoder/gpu_optimized_autoencoder_config.yaml"
    preprocessor_path = "../models/gpu_optimized_autoencoder/gpu_optimized_autoencoder_preprocessor.pkl"
    data_path = "../NSL-KDD_Dataset"
    
    # Check if files exist
    for path, name in [(model_path, "Model"), (config_path, "Config"), (preprocessor_path, "Preprocessor")]:
        if not Path(path).exists():
            print(f"❌ {name} file not found: {path}")
            return
    
    try:
        # Initialize analyzer
        analyzer = AnomalyDetectionAnalyzer(model_path, config_path, preprocessor_path, data_path)
        
        # Compute reconstruction errors
        analyzer.compute_reconstruction_errors()
        
        # Run comprehensive analysis
        report = analyzer.generate_comprehensive_report()
        
        print(f"\\n✅ Analysis complete! Check the generated plots and recommendations.")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()