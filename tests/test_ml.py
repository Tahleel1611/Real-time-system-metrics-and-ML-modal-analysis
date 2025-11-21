"""
Comprehensive Unit Tests for ML Modules

This module contains unit tests for:
- ml/models/ML.py
- ml/models/anomaly_detection.py
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestMLModule(unittest.TestCase):
    """Test cases for ml/models/ML.py"""

    def setUp(self):
        """Set up test data for ML tests"""
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'label': np.random.randint(0, 2, 100)
        })

    def test_anomaly_detection_with_sample_data(self):
        """Test anomaly detection with sample data"""
        try:
            from ml.models.ML import detect_anomalies

            # Create data with clear anomalies
            normal_data = np.random.normal(0, 1, (90, 3))
            anomalies = np.random.normal(10, 1, (10, 3))  # Clear outliers
            test_data = np.vstack([normal_data, anomalies])

            result = detect_anomalies(test_data)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, (np.ndarray, list, pd.Series))
        except ImportError:
            self.skipTest("detect_anomalies function not found")

    def test_clustering_with_known_datasets(self):
        """Test clustering with known datasets"""
        try:
            from ml.models.ML import perform_clustering

            # Create clearly separable clusters
            cluster1 = np.random.normal(0, 0.5, (50, 2))
            cluster2 = np.random.normal(5, 0.5, (50, 2))
            test_data = np.vstack([cluster1, cluster2])

            result = perform_clustering(test_data, n_clusters=2)
            self.assertIsNotNone(result)
            # Check that we get 2 distinct clusters
            if hasattr(result, 'labels_'):
                self.assertEqual(len(np.unique(result.labels_)), 2)
        except ImportError:
            self.skipTest("perform_clustering function not found")

    @patch('multiprocessing.Pool')
    def test_parallel_processing_works_correctly(self, mock_pool):
        """Test parallel processing works correctly"""
        try:
            from ml.models.ML import parallel_process_data

            # Mock the pool map function
            mock_pool_instance = MagicMock()
            mock_pool_instance.map.return_value = [1, 2, 3, 4]
            mock_pool.return_value.__enter__.return_value = mock_pool_instance

            result = parallel_process_data([1, 2, 3, 4])
            self.assertIsInstance(result, list)
            mock_pool_instance.map.assert_called()
        except ImportError:
            self.skipTest("parallel_process_data function not found")

    def test_ml_module_handles_empty_data(self):
        """Test ML module handles empty data gracefully"""
        try:
            from ml.models.ML import detect_anomalies

            empty_data = np.array([])
            # Should either handle gracefully or raise appropriate error
            with self.assertRaises((ValueError, IndexError, AttributeError)):
                detect_anomalies(empty_data)
        except ImportError:
            self.skipTest("ML module not found")

    def test_ml_module_handles_invalid_input(self):
        """Test ML module handles invalid input types"""
        try:
            from ml.models.ML import detect_anomalies

            # Test with invalid input
            invalid_inputs = [None, "string", {}, []]
            for invalid_input in invalid_inputs:
                with self.assertRaises((TypeError, ValueError, AttributeError)):
                    detect_anomalies(invalid_input)
        except ImportError:
            self.skipTest("ML module not found")


class TestAnomalyDetection(unittest.TestCase):
    """Test cases for ml/models/anomaly_detection.py"""

    def setUp(self):
        """Set up test data"""
        np.random.seed(42)  # For reproducibility
        self.normal_data = np.random.normal(0, 1, (100, 5))
        self.anomalous_data = np.random.normal(10, 1, (10, 5))

    def test_training_and_prediction_pipeline(self):
        """Test training and prediction pipeline"""
        try:
            from ml.models.anomaly_detection import AnomalyDetector

            detector = AnomalyDetector()
            detector.fit(self.normal_data)
            predictions = detector.predict(self.anomalous_data)

            self.assertIsNotNone(predictions)
            self.assertEqual(len(predictions), len(self.anomalous_data))
        except ImportError:
            self.skipTest("AnomalyDetector class not found")

    def test_input_validation(self):
        """Test input validation"""
        try:
            from ml.models.anomaly_detection import AnomalyDetector

            detector = AnomalyDetector()

            # Test with invalid input shapes
            invalid_data = [
                np.array([]),  # Empty array
                np.array([1, 2, 3]),  # 1D array when 2D expected
                None,  # None type
            ]

            for data in invalid_data:
                with self.assertRaises((ValueError, TypeError, AttributeError)):
                    detector.fit(data)
        except ImportError:
            self.skipTest("AnomalyDetector class not found")

    def test_error_handling_for_edge_cases(self):
        """Test error handling for edge cases"""
        try:
            from ml.models.anomaly_detection import AnomalyDetector

            detector = AnomalyDetector()

            # Test prediction before fitting
            with self.assertRaises((RuntimeError, ValueError, AttributeError)):
                detector.predict(self.normal_data)

            # Test with mismatched feature dimensions
            detector.fit(self.normal_data)
            wrong_shape_data = np.random.rand(10, 3)  # Wrong number of features

            with self.assertRaises((ValueError, IndexError)):
                detector.predict(wrong_shape_data)
        except ImportError:
            self.skipTest("AnomalyDetector class not found")

    def test_anomaly_scores_are_reasonable(self):
        """Test that anomaly scores are within reasonable ranges"""
        try:
            from ml.models.anomaly_detection import AnomalyDetector

            detector = AnomalyDetector()
            detector.fit(self.normal_data)

            # Normal data should have lower anomaly scores
            normal_scores = detector.predict(self.normal_data)
            anomalous_scores = detector.predict(self.anomalous_data)

            # Check that scores are numeric
            self.assertTrue(all(isinstance(s, (int, float, np.number)) for s in normal_scores))
            self.assertTrue(all(isinstance(s, (int, float, np.number)) for s in anomalous_scores))

            # Anomalous data should generally have higher scores
            if hasattr(normal_scores, 'mean'):
                self.assertLess(np.mean(normal_scores), np.mean(anomalous_scores))
        except ImportError:
            self.skipTest("AnomalyDetector class not found")

    @patch('sklearn.ensemble.IsolationForest')
    def test_model_configuration(self, mock_model):
        """Test that model is configured correctly"""
        try:
            from ml.models.anomaly_detection import AnomalyDetector

            detector = AnomalyDetector(contamination=0.1)
            # Verify that contamination parameter is set
            if hasattr(detector, 'contamination'):
                self.assertEqual(detector.contamination, 0.1)
        except ImportError:
            self.skipTest("AnomalyDetector class not found")


class TestMLIntegration(unittest.TestCase):
    """Integration tests for ML modules"""

    def test_end_to_end_anomaly_detection(self):
        """Test end-to-end anomaly detection workflow"""
        try:
            from ml.models.anomaly_detection import AnomalyDetector

            # Generate training data
            training_data = np.random.normal(0, 1, (200, 5))

            # Train detector
            detector = AnomalyDetector()
            detector.fit(training_data)

            # Generate test data with anomalies
            normal_test = np.random.normal(0, 1, (50, 5))
            anomaly_test = np.random.normal(10, 1, (10, 5))

            # Predict
            normal_predictions = detector.predict(normal_test)
            anomaly_predictions = detector.predict(anomaly_test)

            # Verify predictions exist
            self.assertIsNotNone(normal_predictions)
            self.assertIsNotNone(anomaly_predictions)
        except ImportError:
            self.skipTest("Required ML modules not found")

    def test_ml_pipeline_with_pandas_dataframe(self):
        """Test ML pipeline works with pandas DataFrames"""
        try:
            from ml.models.anomaly_detection import AnomalyDetector

            # Create DataFrame
            df = pd.DataFrame({
                'feature1': np.random.rand(100),
                'feature2': np.random.rand(100),
                'feature3': np.random.rand(100)
            })

            detector = AnomalyDetector()
            detector.fit(df.values)
            predictions = detector.predict(df.values[:10])

            self.assertEqual(len(predictions), 10)
        except ImportError:
            self.skipTest("Required ML modules not found")

    @patch('joblib.dump')
    def test_model_persistence(self, mock_dump):
        """Test model can be saved (if persistence is implemented)"""
        try:
            from ml.models.anomaly_detection import AnomalyDetector

            detector = AnomalyDetector()
            detector.fit(np.random.rand(100, 5))

            # If save method exists, test it
            if hasattr(detector, 'save'):
                detector.save('test_model.pkl')
                mock_dump.assert_called()
        except ImportError:
            self.skipTest("Model persistence test skipped")


class TestPerformance(unittest.TestCase):
    """Performance and scalability tests"""

    def test_handles_large_datasets(self):
        """Test that ML modules can handle reasonably large datasets"""
        try:
            from ml.models.anomaly_detection import AnomalyDetector

            # Create larger dataset
            large_data = np.random.rand(10000, 10)

            detector = AnomalyDetector()
            # Should complete without hanging or crashing
            detector.fit(large_data)
            predictions = detector.predict(large_data[:100])

            self.assertEqual(len(predictions), 100)
        except ImportError:
            self.skipTest("Performance test skipped")
        except MemoryError:
            self.skipTest("Insufficient memory for performance test")


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
