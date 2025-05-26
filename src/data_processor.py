import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, base_path=None):
        self.base_path = base_path or Path(__file__).parent.parent / 'data'
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_dataset(self, dataset_name):
        """Load one of the four benchmark datasets
        
        Args:
            dataset_name: One of 'cic-ids-2017', 'cic-ids-2018', 'bot-iot', 'iot-23'
        """
        dataset_path = self.base_path / dataset_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")
            
        # Load all CSV files in the dataset directory
        data_files = list(dataset_path.glob('*.csv'))
        if not data_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")
            
        dfs = []
        for file in data_files:
            df = pd.read_csv(file)
            dfs.append(df)
            
        # Combine all dataframes
        data = pd.concat(dfs, ignore_index=True)
        return data
        
    def preprocess_data(self, data):
        """Preprocess the dataset"""
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
        
        # Encode categorical features
        for col in categorical_cols:
            if col != 'label':  # Don't encode the label yet
                data[col] = self.label_encoder.fit_transform(data[col])
                
        return data
        
    def prepare_type_ab_labels(self, data):
        """Prepare labels for type-A and type-B classification
        
        Returns:
            y: Binary labels (0 for benign, 1 for malicious)
            y_type_a: Labels for main attack types
            y_type_b: Labels for attack subtypes
        """
        # Encode main attack types (type-A)
        y_type_a = self.label_encoder.fit_transform(data['attack_type'])
        
        # Encode attack subtypes (type-B)
        y_type_b = self.label_encoder.fit_transform(data['attack_subtype'])
        
        # Create binary labels
        y = (data['attack_type'] != 'benign').astype(int)
        
        return y, y_type_a, y_type_b
        
    def prepare_dataset(self, dataset_name, test_size=0.2, random_state=42):
        """Prepare a complete dataset for training
        
        Args:
            dataset_name: Name of the dataset to prepare
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test: Training and test features
            y_train, y_test: Binary labels
            y_type_a_train, y_type_a_test: Type-A labels
            y_type_b_train, y_type_b_test: Type-B labels
        """
        # Load and preprocess data
        data = self.load_dataset(dataset_name)
        data = self.preprocess_data(data)
        
        # Prepare features and labels
        X = data.drop(['label', 'attack_type', 'attack_subtype'], axis=1)
        y, y_type_a, y_type_b = self.prepare_type_ab_labels(data)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test, y_type_a_train, y_type_a_test, y_type_b_train, y_type_b_test = \
            train_test_split(X, y, y_type_a, y_type_b, test_size=test_size, random_state=random_state)
            
        return (X_train, X_test, y_train, y_test, 
                y_type_a_train, y_type_a_test, 
                y_type_b_train, y_type_b_test)
                
    def save_processed_data(self, output_dir, **data):
        """Save processed datasets"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each array as a numpy file
        for name, array in data.items():
            np.save(output_path / f"{name}.npy", array)
            
    def load_processed_data(self, input_dir):
        """Load processed datasets"""
        input_path = Path(input_dir)
        
        # Load all numpy files in the directory
        data = {}
        for file in input_path.glob("*.npy"):
            name = file.stem
            data[name] = np.load(file)
            
        return data 