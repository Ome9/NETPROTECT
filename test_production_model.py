"""
Comprehensive Test Suite for Unified Production Model
Tests the model with sample data from all three datasets.
"""
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from unified_production_model import UnifiedAnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionModelTester:
    """Test suite for the unified production model."""
    
    def __init__(self):
        self.detector = UnifiedAnomalyDetector()
        self.test_results = {}
    
    def load_test_data(self) -> dict:
        """Load sample test data from all datasets."""
        test_data = {}
        
        # Load NSL-KDD test data
        try:
            logger.info("Loading NSL-KDD test data...")
            column_names = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
            ]
            
            nsl_df = pd.read_csv('NSL-KDD_Dataset/KDDTest+.txt', names=column_names, nrows=1000)
            nsl_df['is_anomaly'] = (nsl_df['attack_type'] != 'normal').astype(int)
            test_data['nsl-kdd'] = nsl_df.drop(['attack_type', 'difficulty'], axis=1)
            logger.info(f"  Loaded {len(nsl_df)} NSL-KDD samples")
            
        except Exception as e:
            logger.error(f"Failed to load NSL-KDD test data: {e}")
        
        # Load CSE-CIC test data
        try:
            logger.info("Loading CSE-CIC test data...")
            cse_df = pd.read_parquet('CSE-CIC_Dataset/Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet')
            cse_df = cse_df.head(1000)  # Take first 1000 rows
            # Clean column names
            cse_df.columns = cse_df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
            cse_df['is_anomaly'] = (cse_df['Label'].str.upper() != 'BENIGN').astype(int)
            test_data['cse-cic'] = cse_df.drop(['Label'], axis=1)
            logger.info(f"  Loaded {len(cse_df)} CSE-CIC samples")
            
        except Exception as e:
            logger.error(f"Failed to load CSE-CIC test data: {e}")
        
        # Load UNSW test data
        try:
            logger.info("Loading UNSW test data...")
            unsw_df = pd.read_csv('UNSW_Dataset/UNSW_NB15_testing-set.csv', nrows=1000)
            # Map labels to binary
            unsw_df['is_anomaly'] = (unsw_df['label'] == 1).astype(int)
            # Keep all features except 'label' to match training (id was kept during training)
            test_data['unsw'] = unsw_df.drop(['label'], axis=1, errors='ignore')
            logger.info(f"  Loaded {len(unsw_df)} UNSW samples")
            
        except Exception as e:
            logger.error(f"Failed to load UNSW test data: {e}")
        
        return test_data
    
    def test_dataset_detection(self, test_data: dict):
        """Test automatic dataset type detection."""
        logger.info("\n🔍 Testing Dataset Type Detection:")
        
        for dataset_name, data in test_data.items():
            # Remove label for detection test
            data_without_label = data.drop(['is_anomaly'], axis=1, errors='ignore')
            
            detected_type = self.detector.detect_dataset_type(data_without_label)
            is_correct = detected_type == dataset_name
            
            logger.info(f"  {dataset_name.upper()}: Detected as '{detected_type}' {'✅' if is_correct else '❌'}")
            logger.info(f"    Shape: {data_without_label.shape}")
            
            self.test_results[f"detection_{dataset_name}"] = is_correct
    
    def test_preprocessing(self, test_data: dict):
        """Test data preprocessing for each dataset."""
        logger.info("\n🔧 Testing Data Preprocessing:")
        
        for dataset_name, data in test_data.items():
            try:
                data_without_label = data.drop(['is_anomaly'], axis=1, errors='ignore')
                
                # Test preprocessing
                processed_data = self.detector.preprocess_data(data_without_label, dataset_name)
                
                expected_shape = self.detector.DATASET_CONFIGS[dataset_name]['input_dim']
                actual_shape = processed_data.shape[1]
                is_correct_shape = actual_shape == expected_shape
                
                logger.info(f"  {dataset_name.upper()}: Shape {processed_data.shape} {'✅' if is_correct_shape else '❌'}")
                logger.info(f"    Expected features: {expected_shape}, Got: {actual_shape}")
                
                # Check for NaN or infinite values
                has_nan = np.any(np.isnan(processed_data))
                has_inf = np.any(np.isinf(processed_data))
                
                if has_nan or has_inf:
                    logger.warning(f"    ⚠️  Found NaN: {has_nan}, Inf: {has_inf}")
                
                self.test_results[f"preprocessing_{dataset_name}"] = is_correct_shape and not has_nan and not has_inf
                
            except Exception as e:
                logger.error(f"  {dataset_name.upper()}: Failed - {e}")
                self.test_results[f"preprocessing_{dataset_name}"] = False
    
    def test_predictions(self, test_data: dict):
        """Test model predictions on each dataset."""
        logger.info("\n🔮 Testing Model Predictions:")
        
        for dataset_name, data in test_data.items():
            try:
                # Separate features and labels
                if 'is_anomaly' in data.columns:
                    true_labels = data['is_anomaly'].values
                    data_without_label = data.drop(['is_anomaly'], axis=1)
                else:
                    true_labels = None
                    data_without_label = data
                
                # Make predictions
                results = self.detector.predict(data_without_label, dataset_name, return_details=True)
                
                predictions = results['predictions']
                confidence = results['confidence']
                errors = results['reconstruction_errors']
                
                # Basic validation
                is_valid_predictions = len(predictions) == len(data_without_label)
                is_valid_confidence = np.all((confidence >= 0) & (confidence <= 1))
                is_valid_errors = np.all(errors >= 0)
                
                logger.info(f"  {dataset_name.upper()}:")
                logger.info(f"    Samples processed: {results['num_samples']}")
                logger.info(f"    Anomalies detected: {results['num_anomalies']} ({results['anomaly_percentage']:.1f}%)")
                logger.info(f"    Threshold: {results['threshold']:.6f}")
                logger.info(f"    Error range: [{errors.min():.6f}, {errors.max():.6f}]")
                logger.info(f"    Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
                
                # Calculate accuracy if true labels available
                if true_labels is not None:
                    accuracy = np.mean(predictions == true_labels)
                    precision = np.sum((predictions == 1) & (true_labels == 1)) / max(np.sum(predictions == 1), 1)
                    recall = np.sum((predictions == 1) & (true_labels == 1)) / max(np.sum(true_labels == 1), 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                    
                    logger.info(f"    Accuracy: {accuracy:.3f}")
                    logger.info(f"    Precision: {precision:.3f}")
                    logger.info(f"    Recall: {recall:.3f}")
                    logger.info(f"    F1-Score: {f1:.3f}")
                    
                    self.test_results[f"accuracy_{dataset_name}"] = accuracy
                
                # Validation checks
                validation_passed = is_valid_predictions and is_valid_confidence and is_valid_errors
                logger.info(f"    Validation: {'✅' if validation_passed else '❌'}")
                
                self.test_results[f"prediction_{dataset_name}"] = validation_passed
                
            except Exception as e:
                logger.error(f"  {dataset_name.upper()}: Failed - {e}")
                self.test_results[f"prediction_{dataset_name}"] = False
    
    def test_auto_detection_workflow(self, test_data: dict):
        """Test the complete auto-detection workflow."""
        logger.info("\n🤖 Testing Auto-Detection Workflow:")
        
        for dataset_name, data in test_data.items():
            try:
                # Remove labels and let model auto-detect
                data_without_label = data.drop(['is_anomaly'], axis=1, errors='ignore')
                
                # Auto-detect and predict
                results = self.detector.predict(data_without_label)  # No dataset_type specified
                
                detected_correctly = results['dataset_type'] == dataset_name
                
                logger.info(f"  {dataset_name.upper()}: Auto-detected as '{results['dataset_type']}' {'✅' if detected_correctly else '❌'}")
                logger.info(f"    Processed {len(results['predictions'])} samples")
                logger.info(f"    Detected {np.sum(results['predictions'])} anomalies")
                
                self.test_results[f"auto_workflow_{dataset_name}"] = detected_correctly
                
            except Exception as e:
                logger.error(f"  {dataset_name.upper()}: Failed - {e}")
                self.test_results[f"auto_workflow_{dataset_name}"] = False
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("\n📊 TEST REPORT SUMMARY:")
        logger.info("=" * 50)
        
        categories = {
            'Detection': [k for k in self.test_results.keys() if k.startswith('detection_')],
            'Preprocessing': [k for k in self.test_results.keys() if k.startswith('preprocessing_')],
            'Prediction': [k for k in self.test_results.keys() if k.startswith('prediction_')],
            'Auto-Workflow': [k for k in self.test_results.keys() if k.startswith('auto_workflow_')],
            'Accuracy': [k for k in self.test_results.keys() if k.startswith('accuracy_')]
        }
        
        overall_passed = 0
        overall_total = 0
        
        for category, tests in categories.items():
            if not tests:
                continue
                
            passed = sum(self.test_results[test] if isinstance(self.test_results[test], bool) else 
                        (self.test_results[test] > 0.7 if isinstance(self.test_results[test], float) else False) 
                        for test in tests)
            total = len(tests)
            
            logger.info(f"{category}: {passed}/{total} passed")
            
            for test in tests:
                result = self.test_results[test]
                if isinstance(result, bool):
                    status = "✅" if result else "❌"
                    logger.info(f"  {test}: {status}")
                elif isinstance(result, float):
                    status = "✅" if result > 0.7 else "❌" 
                    logger.info(f"  {test}: {result:.3f} {status}")
            
            overall_passed += passed
            overall_total += total
        
        success_rate = (overall_passed / overall_total) * 100 if overall_total > 0 else 0
        
        logger.info("\n" + "=" * 50)
        logger.info(f"🏆 OVERALL SUCCESS RATE: {overall_passed}/{overall_total} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            logger.info("🎉 PRODUCTION MODEL IS READY FOR DEPLOYMENT!")
        elif success_rate >= 60:
            logger.info("⚠️  PRODUCTION MODEL NEEDS MINOR IMPROVEMENTS")
        else:
            logger.info("❌ PRODUCTION MODEL NEEDS MAJOR FIXES")
        
        return success_rate
    
    def run_full_test_suite(self):
        """Run the complete test suite."""
        logger.info("🚀 STARTING COMPREHENSIVE PRODUCTION MODEL TESTING")
        logger.info("=" * 70)
        
        try:
            # Load test data
            test_data = self.load_test_data()
            
            if not test_data:
                logger.error("❌ No test data loaded, cannot proceed with testing")
                return False
            
            # Run all tests
            self.test_dataset_detection(test_data)
            self.test_preprocessing(test_data)
            self.test_predictions(test_data)
            self.test_auto_detection_workflow(test_data)
            
            # Generate report
            success_rate = self.generate_test_report()
            
            return success_rate >= 80
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            return False


def main():
    """Run the production model test suite."""
    tester = ProductionModelTester()
    success = tester.run_full_test_suite()
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)