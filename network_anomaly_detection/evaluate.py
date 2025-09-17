#!/usr/bin/env python3
"""
Quick evaluation script for trained models.
Loads a trained model and evaluates it on test data.
"""

import sys
import os
from pathlib import Path
import argparse

# Add the src directory to Python path
script_dir = Path(__file__).parent
src_dir = script_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.config import ConfigManager, set_random_seeds  # type: ignore
from data.preprocessing import DataProcessor  # type: ignore
from models.autoencoder import create_model  # type: ignore
from training.trainer import ModelEvaluator  # type: ignore
from evaluation.metrics import AnomalyDetectionEvaluator, create_evaluation_report  # type: ignore

import torch
import numpy as np


def parse_eval_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate trained anomaly detection model')
    
    parser.add_argument(
        '--model-path', 
        type=str, 
        required=True,
        help='Path to trained model file (.pt)'
    )
    
    parser.add_argument(
        '--config-path',
        type=str,
        required=True,
        help='Path to configuration file used for training'
    )
    
    parser.add_argument(
        '--preprocessor-path',
        type=str,
        required=True,
        help='Path to saved preprocessor (.pkl)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Override dataset path'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./eval_results',
        help='Output directory for evaluation results'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_eval_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config_path)
    
    # Override data path if provided
    if args.data_path:
        config.data.dataset_path = args.data_path
    
    # Set random seeds
    set_random_seeds(config.seed)
    
    # Get device
    device = config_manager.get_device()
    
    print("Loading trained model and preprocessor...")
    
    # Load preprocessor
    data_processor = DataProcessor(config.data.dataset_path)
    data_processor.load_preprocessor(args.preprocessor_path)
    
    # Load test data
    _, test_df = data_processor.load_data(config.data.train_file, config.data.test_file)
    
    # Preprocess test features (we need train data for fitting, but we'll use the loaded preprocessor)
    train_df, _ = data_processor.load_data(config.data.train_file, config.data.test_file)
    train_features, test_features = data_processor.preprocess_features(train_df, test_df)
    
    # Prepare labels
    _, test_labels = data_processor.prepare_labels(train_df, test_df)
    
    # Create test dataloader
    test_dataloaders = data_processor.create_dataloaders(
        test_features, 
        test_labels,
        batch_size=config.data.batch_size,
        shuffle=False,
        val_split=0.0
    )
    
    # Create and load model
    model_kwargs = {
        'hidden_dims': config.model.hidden_dims,
        'latent_dim': config.model.latent_dim,
        'activation': config.model.activation,
        'use_batch_norm': config.model.use_batch_norm,
        'dropout_rate': config.model.dropout_rate
    }
    
    model = create_model(
        config.model.type,
        config.model.input_dim,
        **model_kwargs
    )
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("Computing reconstruction errors...")
    
    # Create evaluator and compute errors
    model_evaluator = ModelEvaluator(model, device)
    test_errors = model_evaluator.compute_reconstruction_errors(
        test_dataloaders['train'], config.evaluation.error_type
    )
    
    print("Performing evaluation...")
    
    # Comprehensive evaluation
    anomaly_evaluator = AnomalyDetectionEvaluator(data_processor.attack_type_mapping)
    
    evaluation_results = anomaly_evaluator.comprehensive_evaluation(
        errors=test_errors,
        binary_labels=test_labels['binary'],
        multiclass_labels=test_labels['multiclass'],
        threshold_methods=config.evaluation.threshold_methods
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    anomaly_evaluator.save_results(evaluation_results, str(results_path))
    
    # Create report
    report_path = output_dir / "evaluation_report.md"
    report = create_evaluation_report(evaluation_results, str(report_path))
    
    # Print summary
    print("\\nEvaluation Results Summary:")
    print("=" * 50)
    
    for method, results in evaluation_results['threshold_results'].items():
        binary_metrics = results['binary_metrics']
        print(f"\\n{method.replace('_', ' ').title()}:")
        print(f"  Threshold: {results['threshold']:.6f}")
        print(f"  Accuracy: {binary_metrics['accuracy']:.4f}")
        print(f"  Precision: {binary_metrics['precision']:.4f}")
        print(f"  Recall: {binary_metrics['recall']:.4f}")
        print(f"  F1-Score: {binary_metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {binary_metrics['roc_auc']:.4f}")
    
    print(f"\\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()