#!/usr/bin/env python3
"""
Main training script for Network Anomaly Detection using Autoencoders.
This script orchestrates the entire training pipeline.
"""

import sys
import os
from pathlib import Path
import time
import logging

# Add the src directory to Python path
script_dir = Path(__file__).parent
src_dir = script_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Now import our modules
from utils.config import ConfigManager, parse_args, apply_cli_overrides, set_random_seeds  # type: ignore
from data.preprocessing import DataProcessor  # type: ignore
from models.autoencoder import create_model, count_parameters  # type: ignore
from training.trainer import AutoEncoderTrainer, ModelEvaluator, setup_logging  # type: ignore
from evaluation.metrics import AnomalyDetectionEvaluator, ThresholdDeterminer, create_evaluation_report  # type: ignore

import torch
import numpy as np
from datetime import datetime
import json


def main():
    """Main training and evaluation pipeline."""
    
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Apply command line overrides
    config = apply_cli_overrides(config, args)
    
    # Set random seeds for reproducibility
    set_random_seeds(config.seed)
    
    # Setup experiment directories
    directories = config_manager.setup_directories()
    
    # Setup logging
    log_file = directories['results_dir'] / f"{config.experiment.name}_training.log"
    logger = setup_logging(str(log_file))
    
    # Print configuration
    config_manager.print_config()
    
    # Save configuration for this experiment
    config_path = directories['results_dir'] / f"{config.experiment.name}_config.yaml"
    config_manager.save_config(config_path)
    
    # Get device
    device = config_manager.get_device()
    
    print("\\n" + "="*60)
    print("STARTING NETWORK ANOMALY DETECTION TRAINING")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Step 1: Load and preprocess data
        print("\\n1. Loading and preprocessing data...")
        data_processor = DataProcessor(config.data.dataset_path)
        
        # Load raw data
        train_df, test_df = data_processor.load_data(
            config.data.train_file, 
            config.data.test_file
        )
        
        # Preprocess features
        train_features, test_features = data_processor.preprocess_features(train_df, test_df)
        
        # Prepare labels
        train_labels, test_labels = data_processor.prepare_labels(train_df, test_df)
        
        # Extract normal data for autoencoder training
        normal_features = data_processor.get_normal_data(train_features, train_labels)
        
        # Create data loaders
        print("Creating data loaders...")
        train_dataloaders = data_processor.create_dataloaders(
            normal_features, 
            batch_size=config.data.batch_size,
            val_split=config.data.validation_split,
            shuffle=config.data.shuffle_train,
            num_workers=config.data.num_workers
        )
        
        # Create test dataloader (with all data for evaluation)
        test_dataloaders = data_processor.create_dataloaders(
            test_features, 
            test_labels,
            batch_size=config.data.batch_size,
            shuffle=False,
            val_split=0.0,
            num_workers=config.data.num_workers
        )
        
        # Save preprocessing objects
        preprocessor_path = directories['models_dir'] / f"{config.experiment.name}_preprocessor.pkl"
        data_processor.save_preprocessor(str(preprocessor_path))
        
        print(f"Data preprocessing completed successfully!")
        print(f"Normal training samples: {len(normal_features)}")
        print(f"Test samples: {len(test_features)}")
        
        # Step 2: Create and initialize model
        print("\\n2. Creating and initializing model...")
        
        model_kwargs = {
            'latent_dim': config.model.latent_dim,
            'activation': config.model.activation,
            'use_batch_norm': config.model.use_batch_norm,
            'dropout_rate': config.model.dropout_rate
        }
        
        # Add model-specific parameters
        if config.model.type == 'autoencoder':
            if hasattr(config.model, 'hidden_dims'):
                model_kwargs['hidden_dims'] = config.model.hidden_dims
            if hasattr(config.model, 'use_skip_connections'):
                model_kwargs['use_skip_connections'] = config.model.use_skip_connections
        elif config.model.type in ['vae', 'variational']:
            if hasattr(config.model, 'hidden_dims'):
                model_kwargs['hidden_dims'] = config.model.hidden_dims
        
        model = create_model(
            config.model.type,
            config.model.input_dim,
            **model_kwargs
        )
        
        # Print model information
        param_count = count_parameters(model)
        print(f"Created {config.model.type} with {param_count:,} parameters")
        print(f"Model architecture: {config.model.input_dim} -> {config.model.hidden_dims} -> {config.model.latent_dim}")
        
        # Step 3: Setup trainer and train model
        print("\\n3. Setting up trainer and starting training...")
        
        trainer = AutoEncoderTrainer(
            model=model,
            device=device,
            optimizer_name=config.training.optimizer,
            learning_rate=config.training.learning_rate,
            scheduler_name=config.training.scheduler,
            early_stopping_patience=config.training.early_stopping_patience,
            gradient_clip_value=config.training.gradient_clip_value,
            log_interval=config.training.log_interval
        )
        
        # Train the model
        val_dataloader = train_dataloaders.get('val', None)
        model_save_path = directories['models_dir'] / f"{config.experiment.name}_model"
        
        training_history = trainer.train(
            train_dataloader=train_dataloaders['train'],
            val_dataloader=val_dataloader,
            num_epochs=config.training.num_epochs,
            loss_type=config.training.loss_type,
            save_path=str(model_save_path),
            save_interval=config.training.save_interval
        )
        
        print("\\nTraining completed successfully!")
        
        # Step 4: Evaluation
        print("\\n4. Starting evaluation...")
        
        # Create evaluator
        model_evaluator = ModelEvaluator(model, device)
        
        # Compute reconstruction errors for normal training data (for threshold determination)
        print("Computing reconstruction errors for threshold determination...")
        normal_dataloader = data_processor.create_dataloaders(
            normal_features, batch_size=config.data.batch_size, shuffle=False, val_split=0.0, num_workers=config.data.num_workers
        )['train']
        
        normal_errors = model_evaluator.compute_reconstruction_errors(
            normal_dataloader, config.evaluation.error_type
        )
        
        # Compute reconstruction errors for test data
        print("Computing reconstruction errors for test data...")
        test_errors = model_evaluator.compute_reconstruction_errors(
            test_dataloaders['train'], config.evaluation.error_type
        )
        
        # Comprehensive evaluation
        print("Performing comprehensive evaluation...")
        anomaly_evaluator = AnomalyDetectionEvaluator(data_processor.attack_type_mapping)
        
        evaluation_results = anomaly_evaluator.comprehensive_evaluation(
            errors=test_errors,
            binary_labels=test_labels['binary'],
            multiclass_labels=test_labels['multiclass'],
            threshold_methods=config.evaluation.threshold_methods
        )
        
        # Add training history to results
        evaluation_results['training_history'] = training_history
        evaluation_results['model_info'] = {
            'type': config.model.type,
            'parameters': param_count,
            'architecture': {
                'input_dim': config.model.input_dim,
                'hidden_dims': config.model.hidden_dims,
                'latent_dim': config.model.latent_dim
            }
        }
        
        # Save evaluation results
        results_path = directories['results_dir'] / f"{config.experiment.name}_results.json"
        anomaly_evaluator.save_results(evaluation_results, str(results_path))
        
        # Create and save evaluation report
        report_path = directories['results_dir'] / f"{config.experiment.name}_report.md"
        report = create_evaluation_report(evaluation_results, str(report_path))
        
        print("\\n5. Generating plots...")
        
        if config.evaluation.save_plots:
            # Plot error distributions
            normal_mask = test_labels['binary'] == 0
            anomaly_mask = test_labels['binary'] == 1
            
            normal_test_errors = test_errors[normal_mask]
            anomaly_test_errors = test_errors[anomaly_mask]
            
            # Get best threshold for plotting
            best_method = 'optimal_f1' if 'optimal_f1' in evaluation_results['threshold_results'] else list(evaluation_results['threshold_results'].keys())[0]
            best_threshold = evaluation_results['threshold_results'][best_method]['threshold']
            
            # Error distribution plot
            plot_path = directories['plots_dir'] / f"{config.experiment.name}_error_distribution.{config.evaluation.plot_format}"
            anomaly_evaluator.plot_error_distribution(
                normal_test_errors, anomaly_test_errors, best_threshold, str(plot_path)
            )
            
            # ROC curve
            roc_data = evaluation_results['curves']['roc']
            roc_auc = evaluation_results['threshold_results'][best_method]['binary_metrics']['roc_auc']
            roc_plot_path = directories['plots_dir'] / f"{config.experiment.name}_roc_curve.{config.evaluation.plot_format}"
            anomaly_evaluator.plot_roc_curve(
                np.array(roc_data['fpr']), np.array(roc_data['tpr']), roc_auc, str(roc_plot_path)
            )
            
            # Confusion matrix
            cm = np.array(evaluation_results['threshold_results'][best_method]['binary_metrics']['confusion_matrix'])
            cm_plot_path = directories['plots_dir'] / f"{config.experiment.name}_confusion_matrix.{config.evaluation.plot_format}"
            anomaly_evaluator.plot_confusion_matrix(cm, save_path=str(cm_plot_path))
        
        # Print summary
        print("\\n" + "="*60)
        print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        # Print best results
        print("\\nBest Results Summary:")
        for method, results in evaluation_results['threshold_results'].items():
            binary_metrics = results['binary_metrics']
            print(f"\\n{method.replace('_', ' ').title()}:")
            print(f"  Threshold: {results['threshold']:.6f}")
            print(f"  Accuracy: {binary_metrics['accuracy']:.4f}")
            print(f"  Precision: {binary_metrics['precision']:.4f}")
            print(f"  Recall: {binary_metrics['recall']:.4f}")
            print(f"  F1-Score: {binary_metrics['f1_score']:.4f}")
            print(f"  ROC-AUC: {binary_metrics['roc_auc']:.4f}")
        
        print(f"\\nResults saved to: {directories['results_dir']}")
        print(f"Model saved to: {directories['models_dir']}")
        if config.evaluation.save_plots:
            print(f"Plots saved to: {directories['plots_dir']}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        print(f"\\nERROR: Training failed with error: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\\nTraining pipeline completed.")


if __name__ == "__main__":
    main()