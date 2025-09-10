"""
Unit tests for drift detection components.
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch
import tempfile
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fed_drift.drift_detection import (
    ADWINDriftDetector, EvidentiallyDriftDetector, MMDDriftDetector,
    MultiLevelDriftDetector, DriftResult
)


class TestDriftDetectors(unittest.TestCase):
    """Test drift detection algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = np.random.randn(100, 10)
        self.sample_embeddings = np.random.randn(50, 128)
        self.performance_metrics = [0.8, 0.82, 0.81, 0.6, 0.58]  # Drift after index 2
    
    def test_adwin_drift_detector_initialization(self):
        """Test ADWIN detector initialization."""
        detector = ADWINDriftDetector(delta=0.002)
        self.assertEqual(detector.delta, 0.002)
        self.assertFalse(detector.drift_detected)
        self.assertEqual(len(detector.performance_history), 0)
    
    def test_adwin_drift_detector_update_and_detect(self):
        """Test ADWIN detector update and detection."""
        detector = ADWINDriftDetector(delta=0.002)
        
        # Feed stable performance
        for metric in self.performance_metrics[:3]:
            detector.update(metric)
        
        result1 = detector.detect()
        self.assertIsInstance(result1, DriftResult)
        self.assertEqual(result1.drift_type, "concept_drift")
        
        # Feed drifted performance
        for metric in self.performance_metrics[3:]:
            detector.update(metric)
        
        result2 = detector.detect()
        # Note: ADWIN might not detect drift with this small sample, but structure should be correct
        self.assertIsInstance(result2, DriftResult)
    
    def test_adwin_drift_detector_reset(self):
        """Test ADWIN detector reset."""
        detector = ADWINDriftDetector(delta=0.002)
        
        # Add some data
        detector.update(0.8)
        detector.update(0.6)
        
        # Reset
        detector.reset()
        
        self.assertEqual(len(detector.performance_history), 0)
        self.assertFalse(detector.drift_detected)
    
    @patch('fed_drift.drift_detection.Report')
    @patch('fed_drift.drift_detection.DataDriftPreset')
    def test_evidently_drift_detector(self, mock_preset, mock_report):
        """Test Evidently detector with mocked dependencies."""
        # Mock the report behavior
        mock_report_instance = Mock()
        mock_report.return_value = mock_report_instance
        mock_report_instance.as_dict.return_value = {
            'metrics': [{
                'result': {
                    'dataset_drift': True,
                    'drift_share': 0.3
                }
            }]
        }
        
        detector = EvidentiallyDriftDetector(feature_names=['feature_1', 'feature_2'])
        
        # Set reference data
        reference_data = np.random.randn(50, 2)
        detector.set_reference_data(reference_data)
        
        # Update with current data
        current_data = np.random.randn(30, 2)
        detector.update(current_data)
        
        # Detect drift
        result = detector.detect()
        
        self.assertIsInstance(result, DriftResult)
        self.assertEqual(result.drift_type, "data_drift")
        self.assertTrue(result.is_drift)  # Based on mocked response
        self.assertEqual(result.drift_score, 0.3)
    
    def test_evidently_drift_detector_insufficient_data(self):
        """Test Evidently detector with insufficient data."""
        detector = EvidentiallyDriftDetector()
        
        # Try to detect without reference data
        result = detector.detect()
        
        self.assertFalse(result.is_drift)
        self.assertIn("error", result.additional_info)
    
    @patch('fed_drift.drift_detection.MMDDrift')
    def test_mmd_drift_detector(self, mock_mmd_class):
        """Test MMD detector with mocked dependencies."""
        # Mock MMD behavior
        mock_mmd_instance = Mock()
        mock_mmd_class.return_value = mock_mmd_instance
        mock_mmd_instance.predict.return_value = {
            'data': {
                'is_drift': True,
                'p_val': 0.02,
                'distance': 0.15,
                'threshold': 0.1
            }
        }
        
        detector = MMDDriftDetector(p_val=0.05)
        
        # Set reference embeddings
        reference_embeddings = np.random.randn(40, 128)
        detector.set_reference_embeddings(reference_embeddings)
        
        # Update with current embeddings
        current_embeddings = np.random.randn(30, 128)
        detector.update(current_embeddings)
        
        # Detect drift
        result = detector.detect()
        
        self.assertIsInstance(result, DriftResult)
        self.assertEqual(result.drift_type, "embedding_drift")
        self.assertTrue(result.is_drift)
        self.assertEqual(result.p_value, 0.02)
    
    def test_mmd_drift_detector_no_detector(self):
        """Test MMD detector without initialized detector."""
        detector = MMDDriftDetector()
        
        # Try to detect without setting reference
        result = detector.detect()
        
        self.assertFalse(result.is_drift)
        self.assertIn("error", result.additional_info)
    
    def test_multi_level_drift_detector(self):
        """Test multi-level drift detector integration."""
        detector = MultiLevelDriftDetector()
        
        # Set reference data
        reference_data = np.random.randn(50, 10)
        reference_embeddings = np.random.randn(50, 128)
        detector.set_reference_data(reference_data, reference_embeddings)
        
        # Update with performance, data, and embeddings
        detector.update_performance(0.8)
        detector.update_data(np.random.randn(30, 10))
        detector.update_embeddings(np.random.randn(30, 128))
        
        # Detect all
        results = detector.detect_all()
        
        self.assertIn('concept_drift', results)
        self.assertIn('data_drift', results)
        self.assertIn('embedding_drift', results)
        
        for result in results.values():
            self.assertIsInstance(result, DriftResult)
        
        # Test aggregated signal
        aggregated = detector.get_aggregated_drift_signal(results)
        self.assertIsInstance(aggregated, DriftResult)
        self.assertEqual(aggregated.drift_type, "aggregated")
    
    def test_drift_result_creation(self):
        """Test DriftResult creation and properties."""
        result = DriftResult(
            is_drift=True,
            drift_score=0.75,
            p_value=0.03,
            drift_type="test_drift",
            additional_info={"test_key": "test_value"}
        )
        
        self.assertTrue(result.is_drift)
        self.assertEqual(result.drift_score, 0.75)
        self.assertEqual(result.p_value, 0.03)
        self.assertEqual(result.drift_type, "test_drift")
        self.assertEqual(result.additional_info["test_key"], "test_value")
    
    def test_drift_result_default_additional_info(self):
        """Test DriftResult with default additional_info."""
        result = DriftResult(is_drift=False, drift_score=0.1)
        
        self.assertIsInstance(result.additional_info, dict)
        self.assertEqual(len(result.additional_info), 0)


class TestDriftDetectionUtilities(unittest.TestCase):
    """Test drift detection utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock drift history
        self.drift_history = []
        for round_num in range(30):
            drift_results = {
                'concept_drift': DriftResult(
                    is_drift=(round_num > 15 and round_num % 3 == 0),
                    drift_score=0.3 + 0.4 * (round_num > 15)
                ),
                'data_drift': DriftResult(
                    is_drift=(round_num > 18),
                    drift_score=0.2 + 0.5 * (round_num > 18)
                ),
                'embedding_drift': DriftResult(
                    is_drift=(round_num > 20 and round_num % 2 == 0),
                    drift_score=0.1 + 0.6 * (round_num > 20)
                )
            }
            self.drift_history.append(drift_results)
    
    def test_calculate_drift_metrics(self):
        """Test drift metrics calculation."""
        from fed_drift.drift_detection import calculate_drift_metrics
        
        injection_round = 15
        metrics = calculate_drift_metrics(self.drift_history, injection_round)
        
        # Check that metrics are calculated for each detector type
        self.assertIn('concept_drift_detection_delay', metrics)
        self.assertIn('concept_drift_detection_rate', metrics)
        self.assertIn('data_drift_detection_delay', metrics)
        self.assertIn('data_drift_detection_rate', metrics)
        self.assertIn('embedding_drift_detection_delay', metrics)
        self.assertIn('embedding_drift_detection_rate', metrics)
        
        # Check that detection delays are reasonable
        self.assertGreaterEqual(metrics['concept_drift_detection_delay'], 0)
        self.assertGreaterEqual(metrics['data_drift_detection_delay'], 0)
        self.assertGreaterEqual(metrics['embedding_drift_detection_delay'], 0)
        
        # Check that detection rates are between 0 and 1
        for key in metrics:
            if 'detection_rate' in key:
                self.assertGreaterEqual(metrics[key], 0.0)
                self.assertLessEqual(metrics[key], 1.0)


if __name__ == '__main__':
    unittest.main()