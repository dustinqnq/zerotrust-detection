import os
from pathlib import Path
from processors.iot23_processor import IoT23Processor
from zero_trust_ids import ZeroTrustIDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def main():
    # Initialize processor
    data_dir = Path('data')
    processor = IoT23Processor(data_dir)
    
    # Process the smallest dataset
    file_path = data_dir / 'iot-23/CTU-IoT-Malware-Capture-3-1/bro/conn.log.labeled'
    print(f"Processing {file_path}...")
    
    # Get features and labels
    features, binary_labels, type_a_labels, type_b_labels = processor.process(file_path)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split data
    X_train, X_test, y_train, y_test, \
    y_type_a_train, y_type_a_test, \
    y_type_b_train, y_type_b_test = train_test_split(
        features_scaled, binary_labels, type_a_labels, type_b_labels,
        test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    print("\nTraining model...")
    model = ZeroTrustIDS(input_dim=features.shape[1])
    model.fit(
        X_train, y_train,
        y_type_a_train, y_type_b_train,
        epochs=10,
        batch_size=64,
        validation_split=0.15
    )
    
    # Make predictions
    print("\nMaking predictions...")
    type_a_binary, type_a_multi, type_b_binary, type_b_multi, dbscan_labels = model.predict(X_test)
    
    # Evaluate results
    print("\nEvaluation:")
    print("Type-A Binary Accuracy:", np.mean((type_a_binary > 0.5).astype(int) == y_test))
    
    if type_a_multi is not None:
        print("Type-A Multi-class Accuracy:", np.mean(np.argmax(type_a_multi, axis=1) == np.argmax(y_type_a_test, axis=1)))
    else:
        print("Type-A Multi-class: Not trained")
    
    print("Type-B Binary Accuracy:", np.mean((type_b_binary > 0.5).astype(int) == y_test))
    
    if type_b_multi is not None:
        print("Type-B Multi-class Accuracy:", np.mean(np.argmax(type_b_multi, axis=1) == np.argmax(y_type_b_test, axis=1)))
    else:
        print("Type-B Multi-class: Not trained")
        
    print("DBSCAN Clusters:", np.unique(dbscan_labels))
    
    # Print distribution analysis
    print("\nData Distribution Analysis:")
    print(f"Total samples: {len(y_test)}")
    print(f"Benign samples: {np.sum(y_test == 0)} ({np.mean(y_test == 0)*100:.1f}%)")
    print(f"Malicious samples: {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.1f}%)")
    
    # Save model
    print("\nSaving model...")
    os.makedirs('models', exist_ok=True)
    model.save_model('models/iot23_model')
    print("Model saved to models/iot23_model")

if __name__ == '__main__':
    main() 