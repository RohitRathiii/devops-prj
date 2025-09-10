"""
Server-side federated learning strategies with drift detection and mitigation.

This module implements:
- Drift-aware FedAvg strategy with global drift detection
- FedTrimmedAvg for robust aggregation under drift
- Adaptive strategy switching based on drift signals
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
import logging

from .drift_detection import MMDDriftDetector, DriftResult, MultiLevelDriftDetector

logger = logging.getLogger(__name__)


class FedTrimmedAvg:
    """FedTrimmedAvg implementation for robust aggregation."""
    
    def __init__(self, beta: float = 0.2):
        """
        Initialize FedTrimmedAvg.
        
        Args:
            beta: Fraction of clients to trim (0 < beta < 0.5)
        """
        self.beta = beta
        if not (0 < beta < 0.5):
            raise ValueError("Beta must be between 0 and 0.5")
    
    def aggregate(self, results: List[Tuple[ClientProxy, FitRes]]) -> Parameters:
        """
        Aggregate model parameters using trimmed mean.
        
        Args:
            results: List of (ClientProxy, FitRes) tuples
            
        Returns:
            Aggregated parameters
        """
        if not results:
            raise ValueError("No results to aggregate")
        
        # Extract parameters and weights
        parameters_list = []
        weights = []
        
        for client_proxy, fit_res in results:
            parameters_list.append(parameters_to_ndarrays(fit_res.parameters))
            weights.append(fit_res.num_examples)
        
        # Stack parameters for each layer
        aggregated_params = []
        
        for layer_idx in range(len(parameters_list[0])):
            # Get parameters for this layer from all clients
            layer_params = np.array([params[layer_idx] for params in parameters_list])
            layer_weights = np.array(weights)
            
            # Calculate weighted parameters
            weighted_params = layer_params * layer_weights.reshape(-1, *([1] * (layer_params.ndim - 1)))
            
            # Sort by parameter values and trim
            num_clients = len(layer_params)
            num_to_trim = int(num_clients * self.beta)
            
            if num_to_trim > 0:
                # Flatten for sorting
                flat_weighted = weighted_params.reshape(num_clients, -1)
                flat_weights = layer_weights
                
                # Sort by parameter magnitude and trim extreme values
                param_norms = np.linalg.norm(flat_weighted, axis=1)
                sorted_indices = np.argsort(param_norms)
                
                # Keep middle (1-2*beta) fraction
                start_idx = num_to_trim
                end_idx = num_clients - num_to_trim
                trimmed_indices = sorted_indices[start_idx:end_idx]
                
                # Aggregate trimmed parameters
                trimmed_weighted = weighted_params[trimmed_indices]
                trimmed_weights = layer_weights[trimmed_indices]
                
                if len(trimmed_weights) > 0:
                    layer_aggregate = np.sum(trimmed_weighted, axis=0) / np.sum(trimmed_weights)
                else:
                    # Fallback to simple average if all trimmed
                    layer_aggregate = np.mean(layer_params, axis=0)
            else:
                # No trimming needed
                layer_aggregate = np.sum(weighted_params, axis=0) / np.sum(layer_weights)
            
            aggregated_params.append(layer_aggregate)
        
        return ndarrays_to_parameters(aggregated_params)


class DriftAwareFedAvg(FedAvg):
    """
    Federated Averaging strategy with drift detection and adaptive mitigation.
    """
    
    def __init__(self, 
                 fraction_fit: float = 1.0,
                 fraction_evaluate: float = 1.0,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2,
                 evaluate_fn: Optional[Any] = None,
                 on_fit_config_fn: Optional[Any] = None,
                 on_evaluate_config_fn: Optional[Any] = None,
                 accept_failures: bool = True,
                 initial_parameters: Optional[Parameters] = None,
                 fit_metrics_aggregation_fn: Optional[Any] = None,
                 evaluate_metrics_aggregation_fn: Optional[Any] = None,
                 drift_detection_config: Optional[Dict[str, Any]] = None,
                 mitigation_trigger_threshold: float = 0.3):
        """
        Initialize DriftAwareFedAvg strategy.
        
        Args:
            mitigation_trigger_threshold: Threshold for triggering mitigation
            drift_detection_config: Configuration for drift detectors
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
        )
        
        # Drift detection setup
        self.drift_detection_config = drift_detection_config or {}
        self.global_drift_detector = MMDDriftDetector(
            p_val=self.drift_detection_config.get('mmd_p_val', 0.05),
            n_permutations=self.drift_detection_config.get('mmd_permutations', 100)
        )
        
        # Mitigation setup
        self.mitigation_active = False
        self.mitigation_trigger_threshold = mitigation_trigger_threshold
        self.trimmed_aggregator = FedTrimmedAvg(
            beta=self.drift_detection_config.get('trimmed_beta', 0.2)
        )
        
        # Tracking
        self.reference_embeddings = None
        self.round_results = []
        self.drift_history = []
        self.performance_history = []
        
        logger.info("Initialized DriftAwareFedAvg strategy")
    
    def aggregate_fit(self, 
                     server_round: int,
                     results: List[Tuple[ClientProxy, FitRes]],
                     failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model parameters with drift-aware mitigation.
        
        Args:
            server_round: Current round number
            results: Successful client results
            failures: Failed client results
            
        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            return None, {}
        
        # Extract client embeddings and metrics for drift detection
        client_embeddings = []
        client_drift_metrics = {}
        
        for client_proxy, fit_res in results:
            # Extract embeddings if available
            if hasattr(fit_res, 'metrics') and 'embeddings' in fit_res.metrics:
                embeddings = fit_res.metrics['embeddings']
                if isinstance(embeddings, list):
                    client_embeddings.extend(embeddings)
                elif isinstance(embeddings, np.ndarray):
                    client_embeddings.extend(embeddings.tolist())
            
            # Extract drift metrics
            if hasattr(fit_res, 'metrics'):
                for key, value in fit_res.metrics.items():
                    if 'drift' in key.lower():
                        client_drift_metrics[f"{client_proxy.cid}_{key}"] = value
        
        # Global drift detection
        global_drift_result = self._detect_global_drift(server_round, client_embeddings)
        
        # Check mitigation trigger
        should_trigger_mitigation = self._should_trigger_mitigation(
            server_round, client_drift_metrics, global_drift_result
        )
        
        if should_trigger_mitigation and not self.mitigation_active:
            self.mitigation_active = True
            logger.info(f"Round {server_round}: Triggered drift mitigation")
        
        # Aggregate parameters
        if self.mitigation_active:
            # Use robust aggregation
            aggregated_parameters = self.trimmed_aggregator.aggregate(results)
            aggregation_method = "FedTrimmedAvg"
        else:
            # Use standard FedAvg
            aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)
            aggregation_method = "FedAvg"
        
        # Calculate aggregation metrics
        aggregation_metrics = {
            "aggregation_method": aggregation_method,
            "mitigation_active": self.mitigation_active,
            "global_drift_detected": global_drift_result.is_drift,
            "global_drift_score": global_drift_result.drift_score,
            "num_clients": len(results),
            "server_round": server_round
        }
        
        # Add client drift metrics
        aggregation_metrics.update(client_drift_metrics)
        
        # Store results for analysis
        self.round_results.append({
            'round': server_round,
            'aggregation_method': aggregation_method,
            'global_drift': global_drift_result,
            'client_metrics': client_drift_metrics,
            'num_clients': len(results)
        })
        
        return aggregated_parameters, aggregation_metrics
    
    def aggregate_evaluate(self,
                          server_round: int,
                          results: List[Tuple[ClientProxy, EvaluateRes]],
                          failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results with performance tracking.
        
        Args:
            server_round: Current round number
            results: Successful evaluation results
            failures: Failed evaluation results
            
        Returns:
            Aggregated loss and metrics
        """
        if not results:
            return None, {}
        
        # Standard aggregation
        aggregated_loss, base_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Extract performance metrics
        accuracies = []
        losses = []
        
        for client_proxy, evaluate_res in results:
            losses.append(evaluate_res.loss)
            if hasattr(evaluate_res, 'metrics') and 'accuracy' in evaluate_res.metrics:
                accuracies.append(evaluate_res.metrics['accuracy'])
        
        # Calculate fairness metrics
        global_accuracy = np.mean(accuracies) if accuracies else 0.0
        fairness_gap = np.max(accuracies) - np.min(accuracies) if len(accuracies) > 1 else 0.0
        
        # Update performance history
        self.performance_history.append({
            'round': server_round,
            'global_accuracy': global_accuracy,
            'global_loss': aggregated_loss or 0.0,
            'fairness_gap': fairness_gap,
            'client_accuracies': accuracies,
            'client_losses': losses
        })
        
        # Enhanced metrics
        enhanced_metrics = {
            "global_accuracy": global_accuracy,
            "fairness_gap": fairness_gap,
            "num_clients_evaluated": len(results),
            "mitigation_active": self.mitigation_active
        }
        
        if base_metrics:
            enhanced_metrics.update(base_metrics)
        
        return aggregated_loss, enhanced_metrics
    
    def _detect_global_drift(self, server_round: int, client_embeddings: List[Any]) -> DriftResult:
        """
        Detect global drift using aggregated client embeddings.
        
        Args:
            server_round: Current round number
            client_embeddings: Embeddings from all clients
            
        Returns:
            Global drift detection result
        """
        if not client_embeddings:
            return DriftResult(
                is_drift=False,
                drift_score=0.0,
                drift_type="embedding_drift",
                additional_info={"error": "No embeddings available"}
            )
        
        try:
            # Convert to numpy array
            embeddings_array = np.array(client_embeddings)
            
            # Set reference embeddings on first round
            if server_round == 1:
                self.reference_embeddings = embeddings_array
                self.global_drift_detector.set_reference_embeddings(embeddings_array)
                return DriftResult(
                    is_drift=False,
                    drift_score=0.0,
                    drift_type="embedding_drift",
                    additional_info={"message": "Reference embeddings set"}
                )
            
            # Detect drift
            self.global_drift_detector.update(embeddings_array)
            drift_result = self.global_drift_detector.detect()
            
            # Store in history
            self.drift_history.append({
                'round': server_round,
                'result': drift_result
            })
            
            return drift_result
            
        except Exception as e:
            logger.warning(f"Global drift detection failed at round {server_round}: {e}")
            return DriftResult(
                is_drift=False,
                drift_score=0.0,
                drift_type="embedding_drift",
                additional_info={"error": str(e)}
            )
    
    def _should_trigger_mitigation(self, server_round: int, 
                                  client_drift_metrics: Dict[str, Any],
                                  global_drift_result: DriftResult) -> bool:
        """
        Determine if mitigation should be triggered.
        
        Args:
            server_round: Current round number
            client_drift_metrics: Drift metrics from clients
            global_drift_result: Global drift detection result
            
        Returns:
            Whether to trigger mitigation
        """
        # Global drift detection trigger
        if global_drift_result.is_drift:
            logger.info(f"Round {server_round}: Global drift detected (p={global_drift_result.p_value})")
            return True
        
        # Client quorum trigger
        client_drift_signals = []
        for key, value in client_drift_metrics.items():
            if 'adwin_drift' in key.lower() and isinstance(value, bool):
                client_drift_signals.append(value)
        
        if client_drift_signals:
            drift_ratio = sum(client_drift_signals) / len(client_drift_signals)
            if drift_ratio > self.mitigation_trigger_threshold:
                logger.info(f"Round {server_round}: Client quorum trigger ({drift_ratio:.2%} > {self.mitigation_trigger_threshold:.2%})")
                return True
        
        return False
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection and mitigation results."""
        if not self.drift_history:
            return {"message": "No drift detection results available"}
        
        # Calculate detection metrics
        total_rounds = len(self.drift_history)
        drift_rounds = sum(1 for entry in self.drift_history if entry['result'].is_drift)
        detection_rate = drift_rounds / total_rounds if total_rounds > 0 else 0.0
        
        # Performance metrics
        performance_summary = {}
        if self.performance_history:
            accuracies = [p['global_accuracy'] for p in self.performance_history]
            fairness_gaps = [p['fairness_gap'] for p in self.performance_history]
            
            performance_summary = {
                'final_accuracy': accuracies[-1] if accuracies else 0.0,
                'max_accuracy': max(accuracies) if accuracies else 0.0,
                'avg_accuracy': np.mean(accuracies) if accuracies else 0.0,
                'avg_fairness_gap': np.mean(fairness_gaps) if fairness_gaps else 0.0,
                'max_fairness_gap': max(fairness_gaps) if fairness_gaps else 0.0
            }
        
        return {
            'total_rounds': total_rounds,
            'drift_detected_rounds': drift_rounds,
            'drift_detection_rate': detection_rate,
            'mitigation_activated': self.mitigation_active,
            'performance_summary': performance_summary,
            'drift_history_length': len(self.drift_history)
        }


def create_drift_aware_strategy(config: Dict[str, Any]) -> DriftAwareFedAvg:
    """
    Factory function to create drift-aware federated strategy.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DriftAwareFedAvg strategy
    """
    strategy = DriftAwareFedAvg(
        fraction_fit=config.get('fraction_fit', 1.0),
        fraction_evaluate=config.get('fraction_evaluate', 1.0),
        min_fit_clients=config.get('min_fit_clients', 2),
        min_evaluate_clients=config.get('min_evaluate_clients', 2),
        min_available_clients=config.get('min_available_clients', 2),
        drift_detection_config=config.get('drift_detection', {}),
        mitigation_trigger_threshold=config.get('mitigation_threshold', 0.3)
    )
    
    logger.info(f"Created drift-aware strategy with config: {config}")
    return strategy