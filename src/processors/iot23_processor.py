import pandas as pd
import numpy as np
from pathlib import Path
from .base_processor import BaseDataProcessor

class IoT23Processor(BaseDataProcessor):
    """Processor for IoT-23 dataset"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        # Column names for the conn.log format
        self.columns = [
            'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
            'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
            'conn_state', 'local_orig', 'local_resp', 'missed_bytes',
            'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts',
            'resp_ip_bytes', 'tunnel_parents', 'label', 'detailed-label'
        ]
        
        # Define label mappings
        self.type_a_mapping = {
            'Benign': 0,
            'C&C': 1,
            'DDoS': 2,
            'FileDownload': 3,
            'Attack': 4,
            'Malicious': 4,  # Map generic malicious to Attack
            'unknown': 4     # Map unknown to Attack
        }
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from conn.log.labeled file"""
        file_path = Path(file_path)
        print(f"\nReading file: {file_path}")
        
        # Read the file, skipping comment lines and parsing Zeek log format
        data = []
        line_count = 0
        valid_count = 0
        separator = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line_count += 1
                # Parse separator from header
                if line.startswith('#separator'):
                    separator = bytes(line.strip().split(' ')[1], 'utf-8').decode('unicode_escape')
                    continue
                # Skip other header lines
                if line.startswith('#'):
                    continue
                    
                # Process data lines
                fields = line.strip().split(separator)
                
                # Handle missing fields
                if len(fields) < len(self.columns):
                    # Add empty values for missing fields
                    fields.extend([None] * (len(self.columns) - len(fields)))
                elif len(fields) > len(self.columns):
                    # Truncate extra fields
                    fields = fields[:len(self.columns)]
                
                # Replace '-' with None for missing values
                fields = [None if f == '-' or f == '(empty)' else f for f in fields]
                data.append(fields)
                valid_count += 1
        
        print(f"\nProcessed {line_count} lines")
        print(f"Found {valid_count} valid records")
        
        if not data:
            raise ValueError("No valid data records found in the file")
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=self.columns)
        print(f"\nDataFrame shape: {df.shape}")
        print("\nSample of data:")
        print(df.head())
        return df
        
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data"""
        print("\nPreprocessing data...")
        print(f"Initial shape: {data.shape}")
        
        # Convert numeric columns
        numeric_cols = ['id.orig_p', 'id.resp_p', 'duration', 'orig_bytes',
                       'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes',
                       'resp_pkts', 'resp_ip_bytes']
                       
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
        # Fill missing values
        data = data.fillna({
            'duration': 0,
            'orig_bytes': 0,
            'resp_bytes': 0,
            'missed_bytes': 0,
            'orig_pkts': 0,
            'orig_ip_bytes': 0,
            'resp_pkts': 0,
            'resp_ip_bytes': 0,
            'service': 'unknown',
            'proto': 'unknown',
            'conn_state': 'unknown',
            'label': 'unknown',
            'detailed-label': 'unknown'
        })
        
        # Convert timestamp
        data['ts'] = pd.to_datetime(data['ts'].astype(float), unit='s')
        
        print(f"Final shape: {data.shape}")
        return data
        
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from data"""
        print("\nExtracting features...")
        features = pd.DataFrame()
        
        # Basic features
        features['duration'] = data['duration']
        features['orig_bytes'] = data['orig_bytes']
        features['resp_bytes'] = data['resp_bytes']
        features['orig_pkts'] = data['orig_pkts']
        features['resp_pkts'] = data['resp_pkts']
        
        # Derived features
        features['bytes_per_pkt'] = (data['orig_bytes'] + data['resp_bytes']) / \
                                  (data['orig_pkts'] + data['resp_pkts']).clip(lower=1)
        features['pkts_per_sec'] = (data['orig_pkts'] + data['resp_pkts']) / \
                                 data['duration'].clip(lower=0.1)
        features['bytes_per_sec'] = (data['orig_bytes'] + data['resp_bytes']) / \
                                  data['duration'].clip(lower=0.1)
        
        # Protocol one-hot encoding
        features = pd.concat([
            features,
            pd.get_dummies(data['proto'], prefix='proto')
        ], axis=1)
        
        # Service one-hot encoding
        features = pd.concat([
            features,
            pd.get_dummies(data['service'], prefix='service')
        ], axis=1)
        
        # Connection state one-hot encoding
        features = pd.concat([
            features,
            pd.get_dummies(data['conn_state'], prefix='state')
        ], axis=1)
        
        print(f"Extracted features shape: {features.shape}")
        print("\nFeature columns:")
        print(features.columns.tolist())
        return features
        
    def prepare_labels(self, data: pd.DataFrame) -> tuple:
        """Prepare labels for type-A and type-B classification"""
        print("\nPreparing labels...")
        
        # Binary labels (benign vs malicious)
        binary_labels = (data['label'] != 'Benign').astype(int)
        
        # Type-A labels (main attack types)
        # Extract main attack type from label and convert to one-hot encoding
        main_types = data['label'].map(self.type_a_mapping).fillna(4)  # Map unknown to Attack
        type_a_labels = pd.get_dummies(main_types).values
        
        # If there are missing classes in the data, add zero columns for them
        if type_a_labels.shape[1] < len(self.type_a_mapping):
            missing_cols = len(self.type_a_mapping) - type_a_labels.shape[1]
            type_a_labels = np.hstack([
                type_a_labels,
                np.zeros((len(type_a_labels), missing_cols))
            ])
        
        # Type-B labels (attack subtypes)
        # Extract detailed attack types
        type_b_labels = pd.get_dummies(data['detailed-label']).values
        
        print(f"\nLabel counts:")
        print(f"Binary labels: {np.bincount(binary_labels)}")
        print(f"Type-A labels shape: {type_a_labels.shape}")
        print(f"Type-B labels shape: {type_b_labels.shape}")
        
        return binary_labels, type_a_labels, type_b_labels 