from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path

class BaseDataProcessor(ABC):
    """Base class for all dataset processors"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file"""
        pass
        
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data"""
        pass
        
    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from data"""
        pass
        
    @abstractmethod
    def prepare_labels(self, data: pd.DataFrame) -> tuple:
        """Prepare labels for type-A and type-B classification
        
        Returns:
            tuple: (binary_labels, type_a_labels, type_b_labels)
        """
        pass
        
    def process(self, file_path: str) -> tuple:
        """Process the data file
        
        Returns:
            tuple: (features, binary_labels, type_a_labels, type_b_labels)
        """
        # Load data
        data = self.load_data(file_path)
        
        # Preprocess
        data = self.preprocess(data)
        
        # Extract features
        features = self.extract_features(data)
        
        # Prepare labels
        binary_labels, type_a_labels, type_b_labels = self.prepare_labels(data)
        
        return features, binary_labels, type_a_labels, type_b_labels 