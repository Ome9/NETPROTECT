"""
PRACTICAL DEMONSTRATION: HOW TO USE YOUR ANOMALY DETECTION MODEL
================================================================

This script demonstrates exactly how to use your trained model with real data.
"""

import pandas as pd
import numpy as np
from unified_production_model import UnifiedAnomalyDetector

def demonstrate_model_usage():
    """Show practical examples of using the anomaly detection model."""
    
    print("🔥 PRACTICAL ANOMALY DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize the detector
    detector = UnifiedAnomalyDetector()
    
    print("\n🎯 EXAMPLE 1: Using Real NSL-KDD Data")
    print("-" * 40)
    
    try:
        # Load real NSL-KDD test data
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
        
        # Load 50 samples from NSL-KDD
        data = pd.read_csv('NSL-KDD_Dataset/KDDTest+.txt', names=column_names, nrows=50)
        
        # Keep true labels for comparison (but don't give to model)
        true_labels = (data['attack_type'] != 'normal').astype(int)
        
        # Remove labels (model should not see these)
        network_features = data.drop(['attack_type', 'difficulty'], axis=1)
        
        print(f"📊 Loaded {len(network_features)} network samples")
        print(f"   Features: {network_features.shape[1]} (NSL-KDD format)")
        print(f"   True anomalies in sample: {sum(true_labels)}/{len(true_labels)}")
        
        # Predict anomalies
        results = detector.predict(network_features, return_details=True)
        
        print(f"\n🔮 PREDICTION RESULTS:")
        print(f"   Detected dataset type: {results['dataset_type']}")
        print(f"   Predicted anomalies: {results['num_anomalies']}/{results['num_samples']}")
        print(f"   Anomaly percentage: {results['anomaly_percentage']:.1f}%")
        print(f"   Confidence range: [{results['confidence'].min():.3f}, {results['confidence'].max():.3f}]")
        
        # Compare with true labels
        predictions = results['predictions']
        accuracy = np.mean(predictions == true_labels)
        print(f"   Accuracy vs true labels: {accuracy:.3f}")
        
        # Show individual predictions for first 10 samples
        print(f"\n📋 FIRST 10 SAMPLES:")
        for i in range(min(10, len(predictions))):
            true_label = "ANOMALY" if true_labels.iloc[i] else "NORMAL"
            pred_label = "ANOMALY" if predictions[i] else "NORMAL"
            confidence = results['confidence'][i]
            match = "✅" if predictions[i] == true_labels.iloc[i] else "❌"
            
            print(f"   Sample {i+1}: True={true_label:7} | Pred={pred_label:7} | Conf={confidence:.3f} {match}")
            
    except Exception as e:
        print(f"❌ Error with NSL-KDD example: {e}")
    
    print("\n" + "="*60)
    print("🎯 EXAMPLE 2: Using CSE-CIC Data")
    print("-" * 40)
    
    try:
        # Load real CSE-CIC data
        cse_data = pd.read_parquet('CSE-CIC_Dataset/Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet')
        cse_sample = cse_data.head(30)
        
        # Keep true labels for comparison
        true_labels = (cse_sample['Label'].str.upper() != 'BENIGN').astype(int)
        
        # Remove label
        network_features = cse_sample.drop(['Label'], axis=1)
        # Clean column names
        network_features.columns = network_features.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        
        print(f"📊 Loaded {len(network_features)} CSE-CIC samples")
        print(f"   Features: {network_features.shape[1]} (CSE-CIC format)")
        print(f"   True anomalies in sample: {sum(true_labels)}/{len(true_labels)}")
        
        # Predict anomalies
        results = detector.predict(network_features, return_details=True)
        
        print(f"\n🔮 PREDICTION RESULTS:")
        print(f"   Detected dataset type: {results['dataset_type']}")
        print(f"   Predicted anomalies: {results['num_anomalies']}/{results['num_samples']}")
        print(f"   Anomaly percentage: {results['anomaly_percentage']:.1f}%")
        
        # Compare with true labels
        predictions = results['predictions']
        accuracy = np.mean(predictions == true_labels)
        print(f"   Accuracy vs true labels: {accuracy:.3f}")
        
    except Exception as e:
        print(f"❌ Error with CSE-CIC example: {e}")
    
    print("\n" + "="*60)
    print("🎯 EXAMPLE 3: Your Own Data Format")
    print("-" * 40)
    
    print("""
    HERE'S HOW TO USE IT WITH YOUR OWN NETWORK DATA:
    
    # Step 1: Load your data
    your_data = pd.read_csv('your_network_data.csv')
    
    # Step 2: Remove any label/target columns
    features = your_data.drop(['label', 'target', 'class'], axis=1, errors='ignore')
    
    # Step 3: Make sure you have the right number of features
    print(f"Your data has {features.shape[1]} features")
    
    # Step 4: Predict anomalies
    results = detector.predict(features)
    
    # Step 5: Get results
    anomalies = results['predictions']        # 0=normal, 1=anomaly
    confidence = results['confidence']        # confidence scores
    dataset_type = results['dataset_type']    # detected type
    
    # Step 6: Analyze results
    print(f"Detected {np.sum(anomalies)} anomalies")
    print(f"Dataset type: {dataset_type}")
    
    # Step 7: Flag high-confidence anomalies
    high_confidence_anomalies = (anomalies == 1) & (confidence > 0.8)
    print(f"High-confidence anomalies: {np.sum(high_confidence_anomalies)}")
    """)
    
    print("\n" + "="*60)
    print("✨ SUMMARY: WHAT DATA TO PROVIDE")
    print("="*60)
    
    print("""
    🎯 YOUR MODEL EXPECTS:
    
    📊 DATA FORMAT:
       - pandas DataFrame (preferred) or numpy array
       - Network connection/flow/packet features
       - NO label columns
    
    🔢 FEATURE COUNTS:
       - 41 features → NSL-KDD format (connection records)
       - 77 features → CSE-CIC format (flow statistics)  
       - 44 features → UNSW format (packet analysis)
    
    📋 FEATURE TYPES:
       - Network duration, bytes, packets, rates
       - Protocol information (tcp, udp, etc.)
       - Service information (http, ftp, etc.)
       - Connection state flags
       - Statistical measures
    
    ❌ DON'T INCLUDE:
       - Labels ('normal', 'attack', 'anomaly')
       - Target columns ('label', 'class', 'target')
       - Non-network features (unless part of original training)
    
    ✅ THE MODEL HANDLES:
       - Dataset type detection automatically
       - Feature scaling and normalization
       - Categorical variable encoding
       - Missing value handling
       - GPU acceleration
    
    🚀 OUTPUT:
       - Binary predictions (0=normal, 1=anomaly)
       - Confidence scores (0.0 to 1.0)
       - Dataset type detected
       - Reconstruction errors (optional)
    """)
    
    print("\n🎉 You're ready to detect network anomalies!")

if __name__ == "__main__":
    demonstrate_model_usage()