import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import DBSCAN
import joblib
import os

class ZeroTrustIDS:
    """Zero Trust Intrusion Detection System"""
    
    def __init__(self, input_dim=27, type_a_classes=5, type_b_classes=None):
        """Initialize the model
        
        Args:
            input_dim (int): Input dimension for the model
            type_a_classes (int): Number of Type-A classes
            type_b_classes (int): Number of Type-B classes (dynamic)
        """
        self.input_dim = input_dim
        self.type_a_classes = type_a_classes
        self.type_b_classes = type_b_classes
        
        # Create models
        self.type_a_binary = self._create_shallow_model(output_dim=1)
        self.type_a_multi = None  # Will be created when we know the actual number of classes
        self.type_b_binary = self._create_deep_model(output_dim=1)
        self.type_b_multi = None  # Will be created dynamically
        
        # Initialize DBSCAN for unknown attack detection
        self.dbscan = DBSCAN(eps=0.3, min_samples=5)
        
    def _create_shallow_model(self, output_dim):
        """Create a shallow neural network model
        
        Args:
            output_dim (int): Output dimension
        
        Returns:
            model: Keras Sequential model
        """
        model = Sequential([
            Input(shape=(self.input_dim,)),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(output_dim, activation='sigmoid' if output_dim == 1 else 'softmax')
        ])
        
        # Use lower learning rate to prevent gradient explosion
        optimizer = Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy' if output_dim == 1 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def _create_deep_model(self, output_dim):
        """Create a deep neural network model
        
        Args:
            output_dim (int): Output dimension
        
        Returns:
            model: Keras Sequential model
        """
        model = Sequential([
            Input(shape=(self.input_dim,)),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(output_dim if output_dim else 1, activation='sigmoid' if output_dim == 1 else 'softmax')
        ])
        
        # Use lower learning rate to prevent gradient explosion
        optimizer = Adam(learning_rate=0.0005)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy' if output_dim == 1 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def fit(self, X_train, y_train, y_type_a_train=None, y_type_b_train=None, epochs=10, batch_size=32, validation_split=0.1):
        """Train all models
        
        Args:
            X_train (array): Training features
            y_train (array): Binary labels
            y_type_a_train (array): Type-A labels (multi-class)
            y_type_b_train (array): Type-B labels (multi-class)
            epochs (int): Number of epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
        """
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        print("\nTraining Type-A Binary Classifier...")
        self.type_a_binary.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        if y_type_a_train is not None:
            # Create Type-A multi-class model with correct dimensions
            actual_type_a_classes = y_type_a_train.shape[1]
            print(f"\nCreating Type-A Multi-class model with {actual_type_a_classes} classes...")
            self.type_a_multi = self._create_shallow_model(output_dim=actual_type_a_classes)
            
            print("\nTraining Type-A Multi-class Classifier...")
            self.type_a_multi.fit(
                X_train, y_type_a_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=1
            )
        
        if y_type_b_train is not None:
            # Create Type-B multi-class model with correct dimensions
            actual_type_b_classes = y_type_b_train.shape[1]
            print(f"\nCreating Type-B Multi-class model with {actual_type_b_classes} classes...")
            self.type_b_multi = self._create_deep_model(output_dim=actual_type_b_classes)
            
            print("\nTraining Type-B Binary Classifier...")
            self.type_b_binary.fit(
                X_train, y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=1
            )
            
            print("\nTraining Type-B Multi-class Classifier...")
            self.type_b_multi.fit(
                X_train, y_type_b_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=1
            )
            
        # Train DBSCAN
        print("\nTraining DBSCAN Clustering...")
        self.dbscan.fit(X_train)
        
    def predict(self, X):
        """Make predictions using all models
        
        Args:
            X (array): Input features
            
        Returns:
            tuple: Predictions from all models
        """
        # Type-A predictions
        type_a_binary = self.type_a_binary.predict(X, verbose=0)
        type_a_multi = self.type_a_multi.predict(X, verbose=0) if self.type_a_multi else None
        
        # Type-B predictions
        type_b_binary = self.type_b_binary.predict(X, verbose=0)
        type_b_multi = self.type_b_multi.predict(X, verbose=0) if self.type_b_multi else None
        
        # DBSCAN predictions
        dbscan_labels = self.dbscan.fit_predict(X)
        
        return type_a_binary, type_a_multi, type_b_binary, type_b_multi, dbscan_labels
        
    def save_model(self, path):
        """Save the trained model"""
        os.makedirs(path, exist_ok=True)
        
        # Save neural network models
        self.type_a_binary.save(os.path.join(path, 'type_a_binary'))
        if self.type_a_multi:
            self.type_a_multi.save(os.path.join(path, 'type_a_multi'))
        self.type_b_binary.save(os.path.join(path, 'type_b_binary'))
        if self.type_b_multi:
            self.type_b_multi.save(os.path.join(path, 'type_b_multi'))
        
        # Save DBSCAN
        joblib.dump(self.dbscan, os.path.join(path, 'dbscan.joblib'))
        
    def load_model(self, path):
        """Load a trained model"""
        import tensorflow as tf
        
        # Load neural network models
        self.type_a_binary = tf.keras.models.load_model(os.path.join(path, 'type_a_binary'))
        
        type_a_multi_path = os.path.join(path, 'type_a_multi')
        if os.path.exists(type_a_multi_path):
            self.type_a_multi = tf.keras.models.load_model(type_a_multi_path)
            
        self.type_b_binary = tf.keras.models.load_model(os.path.join(path, 'type_b_binary'))
        
        type_b_multi_path = os.path.join(path, 'type_b_multi')
        if os.path.exists(type_b_multi_path):
            self.type_b_multi = tf.keras.models.load_model(type_b_multi_path)
        
        # Load DBSCAN
        self.dbscan = joblib.load(os.path.join(path, 'dbscan.joblib')) 