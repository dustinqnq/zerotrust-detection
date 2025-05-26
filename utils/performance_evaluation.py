import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time

class PerformanceEvaluator:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir or Path(__file__).parent.parent / 'results'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def evaluate_model(self, y_true, y_pred, y_prob=None, model_name="model"):
        """Evaluate model performance with multiple metrics"""
        start_time = time.time()
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC curve and AUC (if probabilities are provided)
        roc_auc = None
        if y_prob is not None:
            if y_prob.shape[1] == 2:  # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
            
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'evaluation_time': time.time() - start_time
        }
        
        return self.results[model_name]
        
    def plot_confusion_matrix(self, model_name, labels=None):
        """Plot confusion matrix heatmap"""
        if model_name not in self.results:
            raise ValueError(f"No results found for model: {model_name}")
            
        cm = np.array(self.results[model_name]['confusion_matrix'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels or True,
                   yticklabels=labels or True)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plt.savefig(self.save_dir / f'confusion_matrix_{model_name}.png')
        plt.close()
        
    def plot_roc_curve(self, y_true, y_prob, model_name):
        """Plot ROC curve"""
        if y_prob.shape[1] != 2:
            raise ValueError("ROC curve plotting requires binary classification probabilities")
            
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        # Save plot
        plt.savefig(self.save_dir / f'roc_curve_{model_name}.png')
        plt.close()
        
    def generate_report(self, include_plots=True):
        """Generate comprehensive performance report"""
        report = {
            'summary': {},
            'detailed_results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Calculate summary statistics
        all_accuracies = [r['accuracy'] for r in self.results.values()]
        all_f1_scores = [r['f1_score'] for r in self.results.values()]
        
        report['summary'] = {
            'number_of_models': len(self.results),
            'average_accuracy': np.mean(all_accuracies),
            'average_f1_score': np.mean(all_f1_scores),
            'best_model': max(self.results.items(),
                            key=lambda x: x[1]['accuracy'])[0]
        }
        
        # Save report
        with open(self.save_dir / 'performance_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        # Generate plots if requested
        if include_plots:
            self.plot_summary_charts()
            
        return report
        
    def plot_summary_charts(self):
        """Plot summary charts comparing model performances"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(self.results.keys())
        
        # Prepare data for plotting
        data = {metric: [self.results[model][metric]
                        for model in model_names]
                for metric in metrics}
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, data[metric],
                   width, label=metric.capitalize())
            
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * 1.5, model_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.save_dir / 'model_comparison.png')
        plt.close()
        
    def save_results_to_csv(self):
        """Save results to CSV file"""
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df.to_csv(self.save_dir / 'performance_results.csv')

if __name__ == "__main__":
    # Example usage
    evaluator = PerformanceEvaluator()
    
    # Simulate some results
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 1])
    y_prob = np.array([[0.8, 0.2],
                       [0.3, 0.7],
                       [0.1, 0.9],
                       [0.4, 0.6],
                       [0.2, 0.8]])
                       
    # Evaluate model
    results = evaluator.evaluate_model(y_true, y_pred, y_prob, "example_model")
    
    # Generate report
    report = evaluator.generate_report()
    print("Evaluation complete. Results saved in 'results' directory.") 