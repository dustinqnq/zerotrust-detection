import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from tensorflow import keras
import joblib
import json
import os
from pathlib import Path

class EnhancedMultiStageDetector:
    def __init__(self, config_path=None):
        self.boundary_detector = IsolationForest(contamination=0.1, random_state=42)
        self.behavior_analyzer = RandomForestClassifier(n_estimators=100, random_state=42)
        self.anomaly_detector = OneClassSVM(kernel='rbf', nu=0.1)
        self.threat_analyzer = None  # Will be initialized during setup
        self.load_threat_intelligence()
        
    def load_threat_intelligence(self):
        """Load threat intelligence data from JSON database"""
        ti_path = Path(__file__).parent.parent / 'threat_intelligence' / 'threat_intelligence_db.json'
        try:
            with open(ti_path, 'r') as f:
                self.threat_intel = json.load(f)
        except FileNotFoundError:
            self.threat_intel = {}
            
    def setup_threat_analyzer(self, input_shape):
        """Initialize the neural network-based threat analyzer"""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.threat_analyzer = model
        
    def boundary_detection(self, X):
        """First stage: Boundary detection using Isolation Forest"""
        return self.boundary_detector.predict(X)
    
    def behavior_analysis(self, X):
        """Second stage: Behavior analysis using Random Forest"""
        return self.behavior_analyzer.predict_proba(X)
    
    def anomaly_detection(self, X):
        """Third stage: Anomaly detection using One-Class SVM"""
        return self.anomaly_detector.predict(X)
    
    def threat_analysis(self, X):
        """Fourth stage: Threat analysis using Neural Network"""
        if self.threat_analyzer is None:
            self.setup_threat_analyzer(X.shape[1])
        return self.threat_analyzer.predict(X)
    
    def fit(self, X, y=None):
        """Train all stages of the detector"""
        # Train boundary detector
        self.boundary_detector.fit(X)
        
        # Train behavior analyzer
        if y is not None:
            self.behavior_analyzer.fit(X, y)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X)
        
        # Train threat analyzer
        if y is not None and self.threat_analyzer is None:
            self.setup_threat_analyzer(X.shape[1])
            self.threat_analyzer.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
            
    def predict(self, X):
        """Full pipeline prediction"""
        # Stage 1: Boundary Detection
        boundary_results = self.boundary_detection(X)
        
        # Stage 2: Behavior Analysis
        behavior_results = self.behavior_analysis(X)
        
        # Stage 3: Anomaly Detection
        anomaly_results = self.anomaly_detection(X)
        
        # Stage 4: Threat Analysis
        threat_results = self.threat_analysis(X)
        
        # Combine results (you can customize this logic)
        final_predictions = np.where(
            (boundary_results == -1) |  # Isolation Forest outliers
            (behavior_results[:, 1] > 0.8) |  # High probability of malicious behavior
            (anomaly_results == -1) |  # SVM anomalies
            (threat_results > 0.7),  # High threat probability
            1,  # Malicious
            0   # Benign
        )
        
        return final_predictions
    
    def save_model(self, path):
        """Save the trained model"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.boundary_detector, os.path.join(path, 'boundary_detector.joblib'))
        joblib.dump(self.behavior_analyzer, os.path.join(path, 'behavior_analyzer.joblib'))
        joblib.dump(self.anomaly_detector, os.path.join(path, 'anomaly_detector.joblib'))
        if self.threat_analyzer:
            self.threat_analyzer.save(os.path.join(path, 'threat_analyzer'))
            
    def load_model(self, path):
        """Load a trained model"""
        self.boundary_detector = joblib.load(os.path.join(path, 'boundary_detector.joblib'))
        self.behavior_analyzer = joblib.load(os.path.join(path, 'behavior_analyzer.joblib'))
        self.anomaly_detector = joblib.load(os.path.join(path, 'anomaly_detector.joblib'))
        if os.path.exists(os.path.join(path, 'threat_analyzer')):
            self.threat_analyzer = keras.models.load_model(os.path.join(path, 'threat_analyzer'))

if __name__ == "__main__":
    # Example usage
    detector = EnhancedMultiStageDetector()
    # Add your training and testing code here 