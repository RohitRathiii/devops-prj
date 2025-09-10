"""
Unit tests for server strategies and mitigation mechanisms.
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fed_drift.server import DriftAwareFedAvg, FedTrimmedAvg, create_drift_aware_strategy
from fed_drift.drift_detection import DriftResult
from flwr.common import FitRes, EvaluateRes, Parameters, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy


class TestFedTrimmedAvg(unittest.TestCase):
    """Test FedTrimmedAvg robust aggregation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.beta = 0.2
        self.trimmed_avg = FedTrimmedAvg(beta=self.beta)
        
        # Create mock client results
        self.mock_results = []
        for i in range(5):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            
            # Create mock parameters (2 layers)
            params = [
                np.random.randn(10, 5).astype(np.float32),
                np.random.randn(5).astype(np.float32)
            ]
            
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(params)
            fit_res.num_examples = 100 + i * 10  # Varying client sizes
            
            self.mock_results.append((client_proxy, fit_res))
    
    def test_fed_trimmed_avg_initialization(self):
        """Test FedTrimmedAvg initialization."""
        trimmed_avg = FedTrimmedAvg(beta=0.1)
        self.assertEqual(trimmed_avg.beta, 0.1)
        
        # Test invalid beta values
        with self.assertRaises(ValueError):
            FedTrimmedAvg(beta=0.5)  # Too high
        with self.assertRaises(ValueError):
            FedTrimmedAvg(beta=0.0)  # Too low
    
    def test_fed_trimmed_avg_aggregation(self):
        """Test FedTrimmedAvg parameter aggregation."""
        aggregated_params = self.trimmed_avg.aggregate(self.mock_results)
        
        # Check that we get parameters back
        self.assertIsInstance(aggregated_params, Parameters)
        
        # Check that parameters have correct structure
        from flwr.common import parameters_to_ndarrays
        param_arrays = parameters_to_ndarrays(aggregated_params)
        
        self.assertEqual(len(param_arrays), 2)  # Should have 2 layers
        self.assertEqual(param_arrays[0].shape, (10, 5))
        self.assertEqual(param_arrays[1].shape, (5,))
    
    def test_fed_trimmed_avg_empty_results(self):
        """Test FedTrimmedAvg with empty results."""
        with self.assertRaises(ValueError):
            self.trimmed_avg.aggregate([])
    
    def test_fed_trimmed_avg_single_client(self):
        """Test FedTrimmedAvg with single client (no trimming)."""
        single_result = [self.mock_results[0]]
        aggregated_params = self.trimmed_avg.aggregate(single_result)
        
        # Should work even with single client
        self.assertIsInstance(aggregated_params, Parameters)


class TestDriftAwareFedAvg(unittest.TestCase):
    """Test drift-aware federated averaging strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.drift_config = {
            'mmd_p_val': 0.05,
            'mmd_permutations': 100,
            'trimmed_beta': 0.2
        }
        
        self.strategy = DriftAwareFedAvg(
            min_fit_clients=2,
            min_evaluate_clients=2,
            drift_detection_config=self.drift_config,
            mitigation_trigger_threshold=0.3
        )
        
        # Create mock client results
        self.mock_fit_results = []
        self.mock_eval_results = []
        
        for i in range(5):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            
            # Fit results
            params = [
                np.random.randn(10, 5).astype(np.float32),
                np.random.randn(5).astype(np.float32)
            ]
            
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(params)
            fit_res.num_examples = 100
            fit_res.metrics = {
                'train_accuracy': 0.8 + np.random.random() * 0.1,
                'embeddings': np.random.randn(10, 128).tolist(),
                'adwin_drift': i > 2  # Simulate drift in last 2 clients
            }
            
            self.mock_fit_results.append((client_proxy, fit_res))
            
            # Evaluation results
            eval_res = Mock(spec=EvaluateRes)
            eval_res.loss = 0.5 + np.random.random() * 0.3
            eval_res.num_examples = 50
            eval_res.metrics = {
                'accuracy': 0.75 + np.random.random() * 0.15
            }
            
            self.mock_eval_results.append((client_proxy, eval_res))
    
    def test_drift_aware_strategy_initialization(self):
        """Test DriftAwareFedAvg initialization."""
        self.assertFalse(self.strategy.mitigation_active)
        self.assertEqual(self.strategy.mitigation_trigger_threshold, 0.3)
        self.assertIsNotNone(self.strategy.global_drift_detector)
        self.assertIsNotNone(self.strategy.trimmed_aggregator)
    
    def test_aggregate_fit_first_round(self):
        """Test aggregate_fit for first round (reference setting)."""
        server_round = 1
        aggregated_params, metrics = self.strategy.aggregate_fit(
            server_round, self.mock_fit_results, []
        )
        
        # Check that we get aggregated parameters
        self.assertIsNotNone(aggregated_params)
        self.assertIsInstance(metrics, dict)
        
        # Check metrics structure
        self.assertIn('aggregation_method', metrics)
        self.assertIn('mitigation_active', metrics)
        self.assertIn('global_drift_detected', metrics)
        self.assertIn('num_clients', metrics)
        
        # First round should use FedAvg
        self.assertEqual(metrics['aggregation_method'], 'FedAvg')
        self.assertFalse(metrics['mitigation_active'])
    
    def test_aggregate_fit_with_drift_detection(self):
        """Test aggregate_fit with drift detection triggering mitigation."""
        # Set up strategy to have reference embeddings
        self.strategy.reference_embeddings = np.random.randn(50, 128)
        
        # Mock global drift detection to return drift
        with patch.object(self.strategy, '_detect_global_drift') as mock_detect:
            mock_detect.return_value = DriftResult(
                is_drift=True,
                drift_score=0.8,
                p_value=0.02,
                drift_type="embedding_drift"
            )
            
            server_round = 10
            aggregated_params, metrics = self.strategy.aggregate_fit(
                server_round, self.mock_fit_results, []
            )
            
            # Should trigger mitigation
            self.assertTrue(self.strategy.mitigation_active)
            self.assertEqual(metrics['aggregation_method'], 'FedTrimmedAvg')
            self.assertTrue(metrics['global_drift_detected'])
    
    def test_aggregate_evaluate(self):
        """Test aggregate_evaluate with performance tracking."""
        server_round = 5
        aggregated_loss, metrics = self.strategy.aggregate_evaluate(
            server_round, self.mock_eval_results, []
        )
        
        # Check that we get aggregated loss
        self.assertIsNotNone(aggregated_loss)
        self.assertIsInstance(metrics, dict)
        
        # Check metrics structure
        self.assertIn('global_accuracy', metrics)
        self.assertIn('fairness_gap', metrics)
        self.assertIn('num_clients_evaluated', metrics)
        
        # Check that performance history is updated
        self.assertEqual(len(self.strategy.performance_history), 1)
        self.assertEqual(self.strategy.performance_history[0]['round'], server_round)
    
    def test_should_trigger_mitigation_global_drift(self):
        """Test mitigation trigger based on global drift."""
        global_drift_result = DriftResult(
            is_drift=True,
            drift_score=0.9,
            p_value=0.01
        )
        
        should_trigger = self.strategy._should_trigger_mitigation(
            server_round=10,
            client_drift_metrics={},
            global_drift_result=global_drift_result
        )
        
        self.assertTrue(should_trigger)
    
    def test_should_trigger_mitigation_client_quorum(self):
        """Test mitigation trigger based on client quorum."""
        global_drift_result = DriftResult(is_drift=False, drift_score=0.1)
        
        # Create client metrics with high drift ratio
        client_drift_metrics = {
            'client_0_adwin_drift': True,
            'client_1_adwin_drift': True,
            'client_2_adwin_drift': False,
            'client_3_adwin_drift': True,
            'client_4_adwin_drift': True
        }
        
        should_trigger = self.strategy._should_trigger_mitigation(
            server_round=15,
            client_drift_metrics=client_drift_metrics,
            global_drift_result=global_drift_result
        )
        
        # 4/5 = 0.8 > 0.3 threshold
        self.assertTrue(should_trigger)
    
    def test_should_not_trigger_mitigation(self):
        """Test no mitigation trigger when drift is low."""
        global_drift_result = DriftResult(is_drift=False, drift_score=0.1)
        
        # Low drift ratio
        client_drift_metrics = {
            'client_0_adwin_drift': False,
            'client_1_adwin_drift': True,
            'client_2_adwin_drift': False,
            'client_3_adwin_drift': False,
            'client_4_adwin_drift': False
        }
        
        should_trigger = self.strategy._should_trigger_mitigation(
            server_round=15,
            client_drift_metrics=client_drift_metrics,
            global_drift_result=global_drift_result
        )
        
        # 1/5 = 0.2 < 0.3 threshold
        self.assertFalse(should_trigger)
    
    def test_get_drift_summary(self):
        """Test drift summary generation."""
        # Add some mock history
        self.strategy.drift_history = [
            {'round': 1, 'result': DriftResult(is_drift=False, drift_score=0.1)},
            {'round': 2, 'result': DriftResult(is_drift=False, drift_score=0.2)},
            {'round': 3, 'result': DriftResult(is_drift=True, drift_score=0.8)}
        ]
        
        self.strategy.performance_history = [
            {'round': 1, 'global_accuracy': 0.8, 'fairness_gap': 0.1},
            {'round': 2, 'global_accuracy': 0.82, 'fairness_gap': 0.12},
            {'round': 3, 'global_accuracy': 0.75, 'fairness_gap': 0.15}
        ]
        
        summary = self.strategy.get_drift_summary()
        
        # Check summary structure
        self.assertIn('total_rounds', summary)
        self.assertIn('drift_detected_rounds', summary)
        self.assertIn('drift_detection_rate', summary)
        self.assertIn('mitigation_activated', summary)
        self.assertIn('performance_summary', summary)
        
        # Check values
        self.assertEqual(summary['total_rounds'], 3)
        self.assertEqual(summary['drift_detected_rounds'], 1)
        self.assertAlmostEqual(summary['drift_detection_rate'], 1/3)


class TestStrategyFactory(unittest.TestCase):
    """Test strategy factory function."""
    
    def test_create_drift_aware_strategy(self):
        """Test drift-aware strategy creation."""
        config = {
            'fraction_fit': 0.8,
            'fraction_evaluate': 0.6,
            'min_fit_clients': 3,
            'mitigation_threshold': 0.4,
            'drift_detection': {
                'mmd_p_val': 0.01,
                'trimmed_beta': 0.15
            }
        }
        
        strategy = create_drift_aware_strategy(config)
        
        self.assertIsInstance(strategy, DriftAwareFedAvg)
        self.assertEqual(strategy.mitigation_trigger_threshold, 0.4)
    
    def test_create_drift_aware_strategy_defaults(self):
        """Test drift-aware strategy creation with defaults."""
        strategy = create_drift_aware_strategy({})
        
        self.assertIsInstance(strategy, DriftAwareFedAvg)
        # Should use default values
        self.assertEqual(strategy.mitigation_trigger_threshold, 0.3)


if __name__ == '__main__':
    unittest.main()