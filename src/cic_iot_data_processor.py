import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class CICIoTDataProcessor:
    def __init__(self, data_path=None):
        self.data_path = data_path or Path(__file__).parent.parent / 'data' / 'cic_iot_2023'
        self.label_encoder = LabelEncoder()
        self.attack_types = None
        
    def load_data(self, filename):
        """Load data from CSV file"""
        file_path = Path(self.data_path) / filename
        try:
            data = pd.read_csv(file_path)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
            
    def preprocess_data(self, data):
        """Preprocess the dataset"""
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(exclude=[np.number]).columns
        
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
        
        # Encode labels
        if 'label' in data.columns:
            data['label'] = self.label_encoder.fit_transform(data['label'])
            self.attack_types = dict(enumerate(self.label_encoder.classes_))
            
        return data
        
    def split_features_labels(self, data):
        """Split features and labels"""
        if 'label' not in data.columns:
            raise ValueError("Dataset does not contain 'label' column")
            
        X = data.drop('label', axis=1)
        y = data['label']
        return X, y
        
    def prepare_dataset(self, train_size=0.8, random_state=42):
        """Prepare the complete dataset"""
        # Load and combine all CSV files in the data directory
        data_files = list(Path(self.data_path).glob('*.csv'))
        if not data_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
            
        all_data = pd.concat([self.load_data(file.name) for file in data_files])
        
        # Preprocess data
        processed_data = self.preprocess_data(all_data)
        
        # Split features and labels
        X, y = self.split_features_labels(processed_data)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
        
    def get_attack_types(self):
        """Get mapping of encoded labels to attack types"""
        if self.attack_types is None:
            raise ValueError("Labels have not been encoded yet. Process the data first.")
        return self.attack_types
        
    def save_processed_data(self, X_train, X_test, y_train, y_test, output_dir):
        """Save processed datasets"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training data
        pd.concat([X_train, y_train], axis=1).to_csv(output_path / 'train.csv', index=False)
        
        # Save test data
        pd.concat([X_test, y_test], axis=1).to_csv(output_path / 'test.csv', index=False)
        
        # Save attack types mapping
        if self.attack_types:
            with open(output_path / 'attack_types.json', 'w') as f:
                import json
                json.dump(self.attack_types, f, indent=4)
                
    def load_processed_data(self, input_dir):
        """Load processed datasets"""
        input_path = Path(input_dir)
        
        # Load training data
        train_data = pd.read_csv(input_path / 'train.csv')
        X_train = train_data.drop('label', axis=1)
        y_train = train_data['label']
        
        # Load test data
        test_data = pd.read_csv(input_path / 'test.csv')
        X_test = test_data.drop('label', axis=1)
        y_test = test_data['label']
        
        # Load attack types mapping
        with open(input_path / 'attack_types.json', 'r') as f:
            import json
            self.attack_types = json.load(f)
            
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    processor = CICIoTDataProcessor()
    # Add your data processing code here 