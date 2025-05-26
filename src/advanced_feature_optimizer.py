import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

class AdvancedFeatureOptimizer:
    def __init__(self, n_features=37):
        self.n_features = n_features
        self.selected_features = None
        self.feature_importance = {}
        self.scaler = StandardScaler()
        
    def load_feature_patterns(self):
        """Load feature patterns from database"""
        patterns_path = Path(__file__).parent.parent / 'threat_intelligence' / 'feature_patterns_db.json'
        try:
            with open(patterns_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
            
    def handle_missing_values(self, X):
        """Handle NaN values in the dataset"""
        if isinstance(X, pd.DataFrame):
            # Fill numeric columns with median
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
            
            # Fill categorical columns with mode
            categorical_columns = X.select_dtypes(exclude=[np.number]).columns
            X[categorical_columns] = X[categorical_columns].fillna(X[categorical_columns].mode().iloc[0])
        return X
        
    def select_features(self, X, y):
        """Perform feature selection using multiple methods"""
        # F-score based selection
        f_selector = SelectKBest(score_func=f_classif, k=self.n_features)
        f_selector.fit(X, y)
        f_scores = pd.Series(f_selector.scores_, index=X.columns)
        
        # Mutual information based selection
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
        mi_selector.fit(X, y)
        mi_scores = pd.Series(mi_selector.scores_, index=X.columns)
        
        # Random Forest based selection
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_scores = pd.Series(rf.feature_importances_, index=X.columns)
        
        # Combine scores from all methods
        combined_scores = pd.DataFrame({
            'f_score': f_scores,
            'mi_score': mi_scores,
            'rf_score': rf_scores
        })
        
        # Normalize scores
        combined_scores = (combined_scores - combined_scores.min()) / (combined_scores.max() - combined_scores.min())
        
        # Calculate average score
        final_scores = combined_scores.mean(axis=1)
        
        # Select top features
        self.selected_features = final_scores.nlargest(self.n_features).index.tolist()
        
        # Store feature importance
        self.feature_importance = final_scores.to_dict()
        
        return self.selected_features
        
    def transform(self, X):
        """Transform data using selected features"""
        if self.selected_features is None:
            raise ValueError("Feature selection has not been performed yet. Call select_features first.")
            
        if isinstance(X, pd.DataFrame):
            X = X[self.selected_features]
        else:
            raise ValueError("Input must be a pandas DataFrame")
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.selected_features, index=X.index)
        
    def fit_transform(self, X, y):
        """Perform feature selection and transform data"""
        self.select_features(X, y)
        return self.transform(X)
        
    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.feature_importance:
            raise ValueError("Feature selection has not been performed yet.")
        return self.feature_importance
        
    def save_feature_config(self, path):
        """Save feature configuration"""
        config = {
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'scaler_params': {
                'scale_': self.scaler.scale_.tolist(),
                'mean_': self.scaler.mean_.tolist()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
            
    def load_feature_config(self, path):
        """Load feature configuration"""
        with open(path, 'r') as f:
            config = json.load(f)
            
        self.selected_features = config['selected_features']
        self.feature_importance = config['feature_importance']
        self.scaler.scale_ = np.array(config['scaler_params']['scale_'])
        self.scaler.mean_ = np.array(config['scaler_params']['mean_'])

if __name__ == "__main__":
    # Example usage
    optimizer = AdvancedFeatureOptimizer()
    # Add your feature optimization code here 