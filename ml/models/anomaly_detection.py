"""
Anomaly detection module for system metrics.

This module provides a reusable AnomalyDetector class for detecting
unusual patterns in system metrics data using Isolation Forest and PCA.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from typing import Union, Tuple
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class AnomalyDetector:
    """
    Anomaly detector using Isolation Forest with PCA for dimensionality reduction.
    
    Attributes:
        contamination: Expected proportion of anomalies in the dataset
        scaler: StandardScaler for data normalization
        model: IsolationForest model
        pca: PCA for dimensionality reduction
    """
    
    def __init__(self, contamination: float = 0.05, n_components: int = 3):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (default: 0.05)
            n_components: Number of PCA components (default: 3)
        """
        if not 0 < contamination < 0.5:
            raise ValueError("Contamination must be between 0 and 0.5")
        
        self.contamination = contamination
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
            warm_start=False
        )
        self.pca = PCA(n_components=n_components)
        self.is_trained = False
        
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the anomaly detection model on the given data.
        
        Args:
            data: pandas DataFrame containing the training data
            
        Raises:
            ValueError: If data is empty or has insufficient samples
        """
        if len(data) == 0:
            raise ValueError("Training data cannot be empty")
        
        if len(data) < 10:
            raise ValueError(f"Insufficient training samples. Need at least 10, got {len(data)}")
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Adjust n_components if necessary
        max_components = min(scaled_data.shape[0], scaled_data.shape[1])
        if self.n_components > max_components:
            self.n_components = max_components
            self.pca = PCA(n_components=self.n_components)
        
        # Apply PCA for dimensionality reduction
        pca_data = self.pca.fit_transform(scaled_data)
        
        # Train the Isolation Forest
        self.model.fit(pca_data)
        self.is_trained = True
        
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in the given data.
        
        Args:
            data: pandas DataFrame containing the data to check for anomalies
            
        Returns:
            pandas DataFrame with anomaly scores and predictions
            
        Raises:
            RuntimeError: If model hasn't been trained yet
            ValueError: If data is empty
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before detecting anomalies. Call train() first.")
        
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
        
        # Scale the data
        scaled_data = self.scaler.transform(data)
        
        # Apply PCA
        pca_data = self.pca.transform(scaled_data)
        
        # Predict anomalies
        anomaly_scores = self.model.decision_function(pca_data)
        anomaly_predictions = self.model.predict(pca_data)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'timestamp': data.index if hasattr(data, 'index') else range(len(data)),
            'anomaly_score': anomaly_scores,
            'anomaly': anomaly_predictions
        })
        
        # Convert anomaly predictions to binary (1 for anomaly, 0 for normal)
        results['anomaly'] = results['anomaly'].apply(lambda x: 1 if x == -1 else 0)
        
        return results
    
    def get_anomaly_threshold(self) -> float:
        """
        Get the decision threshold used by the model.
        
        Returns:
            Decision threshold value
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        return self.model.threshold_

def detect_system_anomalies(data: pd.DataFrame, contamination: float = 0.05, 
                           train_ratio: float = 0.8) -> Tuple[pd.DataFrame, AnomalyDetector]:
    """
    Main function to detect anomalies in system metrics.
    
    Args:
        data: pandas DataFrame containing system metrics
            Expected columns: cpu, gpu, memory_usage, data_in, data_out
        contamination: Expected proportion of anomalies (default: 0.05)
        train_ratio: Proportion of data to use for training (default: 0.8)
    
    Returns:
        Tuple of (results DataFrame, trained detector)
        
    Raises:
        ValueError: If data is insufficient or invalid
    """
    if len(data) < 20:
        raise ValueError(f"Insufficient data for anomaly detection. Need at least 20 samples, got {len(data)}")
    
    if not 0.5 <= train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0.5 and 1.0")
    
    # Initialize the detector
    detector = AnomalyDetector(contamination=contamination)
    
    # Train on historical data
    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size]
    
    print(f"Training anomaly detector on {train_size} samples...")
    detector.train(train_data)
    
    # Detect anomalies in the full dataset
    print(f"Detecting anomalies in {len(data)} samples...")
    results = detector.detect_anomalies(data)
    
    anomaly_count = results['anomaly'].sum()
    print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(data)*100:.2f}% of data)")
    
    return results, detector


if __name__ == "__main__":
    """Example usage of the anomaly detection module."""
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'cpu': np.random.normal(50, 15, n_samples),
        'gpu': np.random.normal(60, 20, n_samples),
        'memory_usage': np.random.normal(70, 10, n_samples),
        'data_in': np.random.normal(1000, 300, n_samples),
        'data_out': np.random.normal(500, 150, n_samples)
    })
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
    sample_data.loc[anomaly_indices, 'cpu'] *= 1.5
    sample_data.loc[anomaly_indices, 'memory_usage'] *= 1.3
    
    # Detect anomalies
    results, detector = detect_system_anomalies(sample_data, contamination=0.05)
    
    print("\nAnomaly Detection Results:")
    print(f"Total samples: {len(results)}")
    print(f"Anomalies detected: {results['anomaly'].sum()}")
    print(f"Anomaly rate: {results['anomaly'].mean()*100:.2f}%")
    print(f"\nFirst few results:")
    print(results.head(10))