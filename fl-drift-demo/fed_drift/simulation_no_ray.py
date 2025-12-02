"""
Ray-free simulation implementation to avoid protobuf conflicts.

This alternative implementation uses Flower's VirtualClientEngine
instead of Ray for process isolation, avoiding the mutex lock issues.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Callable
from pathlib import Path
import json
from datetime import datetime
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import flwr as fl
from flwr.common import Context, EvaluateIns, EvaluateRes, FitIns, FitRes

from .data import FederatedDataLoader, MODEL_CONFIG, DRIFT_CONFIG, FEDERATED_CONFIG
from .models import create_model, get_device
from .client import create_drift_detection_client
from .server import create_drift_aware_strategy
from .drift_detection import calculate_drift_metrics

logger = logging.getLogger(__name__)


class ThreadedFederatedSimulation:
    """Ray-free federated simulation using threading for client isolation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize simulation without Ray dependencies."""
        self.config = config or {}
        
        # Extract configuration parameters
        self.num_rounds = self.config.get('simulation', {}).get('num_rounds', 50)
        self.num_clients = self.config.get('federated', {}).get('num_clients', 10)
        self.drift_injection_round = self.config.get('drift', {}).get('injection_round', 25)
        
        # Create simulation ID
        self.simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize data loader
        self.data_loader = FederatedDataLoader(
            num_clients=self.num_clients,
            alpha=self.config.get('federated', {}).get('alpha', 0.5),
            batch_size=self.config.get('model', {}).get('batch_size', 16)
        )
        
        logger.info(f"Initialized threaded simulation: {self.simulation_id}")
        logger.info(f"Config: {self.num_rounds} rounds, {self.num_clients} clients")
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run federated learning simulation using threading instead of Ray."""
        logger.info("Starting threaded federated simulation...")
        
        # Prepare federated data
        logger.info("Loading and partitioning federated data...")
        client_datasets, test_dataset = self.data_loader.create_federated_splits()
        
        # Get data loaders for clients
        client_loaders, test_loader = self.data_loader.get_data_loaders(client_datasets, test_dataset)
        
        # Convert to expected format for client_fn
        client_data = {}
        for client_id in range(self.num_clients):
            if client_id in client_loaders:
                client_data[client_id] = {
                    'train': client_loaders[client_id],
                    'test': test_loader
                }
            else:
                logger.warning(f"No data for client {client_id}")
        
        logger.info(f"Created data for {len(client_data)} clients")
        
        # Create strategy
        strategy = create_drift_aware_strategy(self.config)
        
        # Create client factory function
        def client_fn(client_id: str):
            """Create client instance for given client ID."""
            client_config = {
                'client_id': client_id,
                'data': client_data[int(client_id)],
                'model_config': self.config.get('model', MODEL_CONFIG),
                'drift_config': self.config.get('drift_detection', {})
            }
            return create_drift_detection_client(client_config)
        
        try:
            # Run simulation rounds manually
            history = self._run_manual_simulation(strategy, client_fn)
            
            logger.info("Simulation completed successfully")
            
            # Analyze results
            results = self._analyze_results(history, strategy)
            
            # Save results
            self._save_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise e
    
    def _run_manual_simulation(self, strategy, client_fn) -> Dict[str, Any]:
        """Manually run FL simulation without Ray."""
        logger.info("Running manual federated learning simulation...")
        
        # Initialize history tracking
        history = {
            'losses_distributed': [],
            'metrics_distributed': {'fit': [], 'evaluate': []},
            'losses_centralized': [],
            'metrics_centralized': []
        }
        
        # Get initial parameters
        initial_parameters = strategy.initialize_parameters(client_manager=None)
        current_parameters = initial_parameters
        
        for round_num in range(1, self.num_rounds + 1):
            logger.info(f"Starting round {round_num}/{self.num_rounds}")
            
            # --- FIT PHASE ---
            fit_results = self._run_fit_round(
                round_num, current_parameters, strategy, client_fn
            )
            
            if fit_results:
                # Aggregate fit results
                aggregated_params, fit_metrics = strategy.aggregate_fit(
                    round_num, fit_results, []
                )
                current_parameters = aggregated_params
                
                # Store fit metrics
                history['metrics_distributed']['fit'].append((round_num, fit_metrics))
                
                # Calculate average loss
                if fit_results:
                    avg_loss = np.mean([res.num_examples * getattr(res, 'loss', 0) 
                                      for _, res in fit_results]) / sum([res.num_examples for _, res in fit_results])
                    history['losses_distributed'].append((round_num, avg_loss))
            
            # --- EVALUATE PHASE ---
            eval_results = self._run_evaluate_round(
                round_num, current_parameters, strategy, client_fn
            )
            
            if eval_results:
                # Aggregate evaluation results
                aggregated_loss, eval_metrics = strategy.aggregate_evaluate(
                    round_num, eval_results, []
                )
                
                # Store evaluation metrics
                history['metrics_distributed']['evaluate'].append((round_num, eval_metrics))
                
                logger.info(f"Round {round_num} completed - "
                          f"Loss: {aggregated_loss:.4f}, "
                          f"Accuracy: {eval_metrics.get('global_accuracy', 'N/A')}")
        
        return history
    
    def _run_fit_round(self, round_num, parameters, strategy, client_fn):
        """Run fit phase for a single round using threading."""
        logger.debug(f"Running fit phase for round {round_num}")
        
        # Create fit instructions
        config = strategy.on_fit_config_fn(round_num) if strategy.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, config)
        
        # Run clients in parallel using threads
        fit_results = []
        
        with ThreadPoolExecutor(max_workers=min(self.num_clients, 4)) as executor:
            # Submit all client fit tasks
            future_to_client = {}
            for client_id in range(self.num_clients):
                client = client_fn(str(client_id))
                future = executor.submit(self._safe_client_fit, client, fit_ins)
                future_to_client[future] = client_id
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_client):
                client_id = future_to_client[future]
                try:
                    fit_res = future.result(timeout=60)  # 60 second timeout
                    if fit_res:
                        # Create mock client proxy
                        class MockClientProxy:
                            def __init__(self, cid):
                                self.cid = str(cid)
                        
                        client_proxy = MockClientProxy(client_id)
                        fit_results.append((client_proxy, fit_res))
                        
                except Exception as e:
                    logger.warning(f"Client {client_id} fit failed: {e}")
        
        logger.debug(f"Fit phase completed: {len(fit_results)} successful clients")
        return fit_results
    
    def _run_evaluate_round(self, round_num, parameters, strategy, client_fn):
        """Run evaluation phase for a single round using threading."""
        logger.debug(f"Running evaluation phase for round {round_num}")
        
        # Create evaluation instructions
        config = strategy.on_evaluate_config_fn(round_num) if strategy.on_evaluate_config_fn else {}
        evaluate_ins = EvaluateIns(parameters, config)
        
        # Run clients in parallel using threads
        eval_results = []
        
        with ThreadPoolExecutor(max_workers=min(self.num_clients, 4)) as executor:
            # Submit all client evaluate tasks
            future_to_client = {}
            for client_id in range(self.num_clients):
                client = client_fn(str(client_id))
                future = executor.submit(self._safe_client_evaluate, client, evaluate_ins)
                future_to_client[future] = client_id
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_client):
                client_id = future_to_client[future]
                try:
                    eval_res = future.result(timeout=30)  # 30 second timeout
                    if eval_res:
                        # Create mock client proxy
                        class MockClientProxy:
                            def __init__(self, cid):
                                self.cid = str(cid)
                        
                        client_proxy = MockClientProxy(client_id)
                        eval_results.append((client_proxy, eval_res))
                        
                except Exception as e:
                    logger.warning(f"Client {client_id} evaluation failed: {e}")
        
        logger.debug(f"Evaluation phase completed: {len(eval_results)} successful clients")
        return eval_results
    
    def _safe_client_fit(self, client, fit_ins):
        """Safely execute client fit with error handling."""
        try:
            return client.fit(fit_ins)
        except Exception as e:
            logger.warning(f"Client fit error: {e}")
            return None
    
    def _safe_client_evaluate(self, client, evaluate_ins):
        """Safely execute client evaluation with error handling."""
        try:
            return client.evaluate(evaluate_ins)
        except Exception as e:
            logger.warning(f"Client evaluation error: {e}")
            return None
    
    def _analyze_results(self, history: Dict[str, Any], strategy: Any) -> Dict[str, Any]:
        """Analyze simulation results."""
        logger.info("Analyzing simulation results...")
        
        results = {
            'simulation_id': self.simulation_id,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'simulation_type': 'threaded_no_ray'
        }
        
        # Extract training history
        results['training_losses'] = history.get('losses_distributed', [])
        results['training_metrics'] = history.get('metrics_distributed', {}).get('fit', [])
        results['evaluation_metrics'] = history.get('metrics_distributed', {}).get('evaluate', [])
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(history)
        results['performance_metrics'] = performance_metrics
        
        # Get drift detection summary
        drift_summary = strategy.get_drift_summary()
        results['drift_summary'] = drift_summary
        
        logger.info(f"Analysis complete. Final accuracy: {performance_metrics.get('final_accuracy', 'N/A')}")
        
        return results
    
    def _calculate_performance_metrics(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key performance metrics from simulation history."""
        metrics = {}
        
        try:
            # Extract evaluation metrics
            eval_metrics = history.get('metrics_distributed', {}).get('evaluate', [])
            
            if eval_metrics:
                # Extract accuracy values
                accuracies = []
                fairness_gaps = []
                
                for round_num, round_metrics in eval_metrics:
                    if 'global_accuracy' in round_metrics:
                        accuracies.append(round_metrics['global_accuracy'])
                    if 'fairness_gap' in round_metrics:
                        fairness_gaps.append(round_metrics['fairness_gap'])
                
                if accuracies:
                    metrics['final_accuracy'] = accuracies[-1]
                    metrics['peak_accuracy'] = max(accuracies)
                    metrics['avg_accuracy'] = np.mean(accuracies)
                    
                    # Calculate recovery rate if drift was injected
                    if len(accuracies) > self.drift_injection_round:
                        pre_drift_acc = np.mean(accuracies[:self.drift_injection_round])
                        post_drift_acc = accuracies[-1]
                        drift_impact = accuracies[self.drift_injection_round:self.drift_injection_round+3]
                        
                        if drift_impact:
                            min_drift_acc = min(drift_impact)
                            recovery_rate = (post_drift_acc - min_drift_acc) / (pre_drift_acc - min_drift_acc)
                            metrics['accuracy_recovery_rate'] = max(0, min(1, recovery_rate))
                
                if fairness_gaps:
                    metrics['final_fairness_gap'] = fairness_gaps[-1]
                    metrics['avg_fairness_gap'] = np.mean(fairness_gaps)
                    metrics['max_fairness_gap'] = max(fairness_gaps)
            
            # Add training loss metrics
            training_losses = history.get('losses_distributed', [])
            if training_losses:
                losses = [loss for _, loss in training_losses]
                metrics['final_loss'] = losses[-1]
                metrics['initial_loss'] = losses[0]
                metrics['min_loss'] = min(losses)
        
        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {e}")
            metrics = {
                'final_accuracy': 'Error',
                'peak_accuracy': 'Error', 
                'final_fairness_gap': 'Error'
            }
        
        return metrics
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save simulation results to file."""
        # Create results directory
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        results_file = results_dir / f"simulation_{self.simulation_id}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")