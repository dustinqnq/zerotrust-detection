import argparse
from pathlib import Path
from data_processor import DataProcessor
from zero_trust_ids import ZeroTrustIDS
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json

def train_and_evaluate(dataset_name, model_path=None, save_path=None):
    """Train and evaluate the Zero Trust IDS on a specific dataset"""
    # Initialize processors
    processor = DataProcessor()
    model = ZeroTrustIDS()
    
    print(f"Processing dataset: {dataset_name}")
    
    # Prepare dataset
    (X_train, X_test, y_train, y_test,
     y_type_a_train, y_type_a_test,
     y_type_b_train, y_type_b_test) = processor.prepare_dataset(dataset_name)
     
    if model_path:
        print(f"Loading model from {model_path}")
        model.load_model(model_path)
    else:
        print("Training model...")
        model.fit(X_train, y_train, y_type_a_train, y_type_b_train)
        
        if save_path:
            print(f"Saving model to {save_path}")
            model.save_model(save_path)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Calculate metrics
    print("\nEvaluation Results:")
    print("Classification Report:")
    print(classification_report(y_test, predictions == 0, target_names=['Malicious', 'Benign']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions == 0)
    print(cm)
    
    # Calculate additional metrics
    unknown_attacks = (predictions == 2).sum()
    known_attacks = (predictions == 1).sum()
    benign = (predictions == 0).sum()
    
    metrics = {
        'total_samples': len(predictions),
        'unknown_attacks': int(unknown_attacks),
        'known_attacks': int(known_attacks),
        'benign': int(benign),
        'unknown_attack_ratio': float(unknown_attacks / len(predictions)),
        'detection_rate': float(((predictions != 0) == (y_test == 1)).mean())
    }
    
    print("\nDetailed Metrics:")
    print(json.dumps(metrics, indent=2))
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate Zero Trust IDS')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cic-ids-2017', 'cic-ids-2018', 'bot-iot', 'iot-23'],
                       help='Dataset to use')
    parser.add_argument('--model-path', type=str,
                       help='Path to load pre-trained model')
    parser.add_argument('--save-path', type=str,
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Train and evaluate
    metrics = train_and_evaluate(args.dataset, args.model_path, args.save_path)
    
    # Save metrics
    metrics_file = results_dir / f"{args.dataset}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_file}")

if __name__ == "__main__":
    main() 