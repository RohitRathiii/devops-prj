"""
Main simulation script for federated learning with drift detection.

This script orchestrates:
- Federated dataset preparation with drift injection
- Client and server initialization
- Simulation execution with drift monitoring
- Results collection and analysis
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Callable
from pathlib import Path
import json
import csv
from datetime import datetime
import atexit

import flwr as fl
from flwr.simulation import start_simulation
from flwr.common import Context

# Import ray for proper cleanup
try:
    import ray
    RAY_AVAILABLE = True
    
    # Register Ray cleanup handler
    def cleanup_ray():
        if ray.is_initialized():
            ray.shutdown()
    
    atexit.register(cleanup_ray)
    
except ImportError:
    RAY_AVAILABLE = False

from .data import FederatedDataLoader, MODEL_CONFIG, DRIFT_CONFIG, FEDERATED_CONFIG
from .models import create_model, get_device
from .client import create_drift_detection_client
from .server import create_drift_aware_strategy
from .drift_detection import calculate_drift_metrics, visualize_drift_timeline
from .metrics_utils import find_stabilization_point

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FederatedDriftSimulation:
    """Main simulation coordinator for federated learning with drift detection."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize simulation with configuration.
        
        Args:
            config: Simulation configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Simulation parameters
        self.num_clients = self.config['federated']['num_clients']
        self.num_rounds = self.config['simulation']['num_rounds']
        self.drift_injection_round = self.config['drift']['injection_round']
        
        # Data and model setup
        self.device = get_device()
        self.data_loader = None
        self.client_datasets = {}
        self.test_dataset = None
        
        # Results tracking
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized simulation {self.simulation_id} with {self.num_clients} clients, {self.num_rounds} rounds")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default simulation configuration."""
        return {
            'model': MODEL_CONFIG,
            'drift': DRIFT_CONFIG,
            'federated': FEDERATED_CONFIG,
            'simulation': {
                'num_rounds': 50,
                'num_gpus': 0.0,  # Use CPU for simulation
                'num_cpus': 2,
                'ray_init_args': {"include_dashboard": False}
            },
            'drift_detection': {
                'adwin_delta': 0.002,
                'mmd_p_val': 0.05,
                'mmd_permutations': 100,
                'trimmed_beta': 0.2
            },
            'strategy': {
                'fraction_fit': 1.0,
                'fraction_evaluate': 1.0,
                'min_fit_clients': 2,
                'min_evaluate_clients': 2,
                'mitigation_threshold': 0.3
            }
        }
    
    def prepare_data(self) -> None:
        """Prepare federated datasets with potential drift injection."""
        logger.info("Preparing federated datasets...")
        
        # Initialize data loader
        self.data_loader = FederatedDataLoader(
            num_clients=self.num_clients,
            alpha=self.config['federated']['alpha'],
            batch_size=self.config['model']['batch_size']
        )
        
        # Create federated splits
        self.client_datasets, self.test_dataset = self.data_loader.create_federated_splits()
        
        # Log dataset statistics
        stats = self.data_loader.get_dataset_statistics(self.client_datasets)
        logger.info(f"Dataset statistics: {stats}")
        
        # Apply drift to specified clients (will be done at appropriate round)
        logger.info(f"Drift will be injected at round {self.drift_injection_round} "
                   f"for clients {self.config['drift']['affected_clients']}")
    
    def inject_drift(self) -> None:
        """Inject synthetic drift into specified clients."""
        logger.info(f"Injecting drift at round {self.drift_injection_round}...")
        
        affected_clients = self.config['drift']['affected_clients']
        drift_types = self.config['drift']['drift_types']
        
        # Apply drift
        self.client_datasets = self.data_loader.apply_drift_to_clients(
            self.client_datasets,
            affected_clients,
            drift_types
        )
        
        logger.info(f"Applied drift types {drift_types} to clients {affected_clients}")
    
    def create_client_fn(self) -> Callable[[str], Any]:
        """Create client factory function for simulation."""
        # Extract values to avoid capturing 'self' in closure
        model_config = self.config['model'].copy()
        drift_config = self.config['drift_detection'].copy()
        device = self.device
        client_datasets = self.client_datasets
        test_dataset = self.test_dataset

        def client_fn(context: Context) -> Any:
            """Create a client for the given context."""
            # Extract client ID from context - Ray generates random node IDs
            ray_node_id = context.node_id

            # Map Ray's node_id to our sequential client indices
            # Convert the large Ray ID to a client index within our range
            client_idx = int(ray_node_id) % len(client_datasets)

            print(f"[CLIENT] Ray node_id: {ray_node_id} → mapped to client_idx: {client_idx}")

            # Create model for this client
            model, _ = create_model(model_config, device)

            # Get client's data using the mapped index
            if client_idx in client_datasets:
                train_dataset = client_datasets[client_idx]

                # Create data loaders
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=model_config['batch_size'],
                    shuffle=True,
                    drop_last=True
                )

                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=model_config['batch_size'],
                    shuffle=False
                )

                # Create drift detection client
                numpy_client = create_drift_detection_client(
                    client_id=str(client_idx),  # Use mapped index as client ID
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    drift_config=drift_config
                )

                # Convert NumPyClient to Client (recommended by Flower)
                return numpy_client.to_client()
            else:
                print(f"[ERROR] No dataset found for client_idx {client_idx}")
                return None

        return client_fn
    
    def create_strategy(self) -> Any:
        """Create server strategy with drift detection."""
        strategy_config = {
            **self.config['strategy'],
            'drift_detection': self.config['drift_detection']
        }
        
        return create_drift_aware_strategy(strategy_config)
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run the federated learning simulation."""
        logger.info(f"Starting simulation with {self.num_rounds} rounds...")
        
        # Prepare data
        self.prepare_data()
        
        # Create client function with drift injection logic
        original_client_fn = self.create_client_fn()

        # Extract drift injection parameters to avoid capturing 'self'
        drift_injection_round = self.drift_injection_round
        affected_clients = self.config['drift']['affected_clients']
        drift_types = self.config['drift']['drift_types']
        drift_intensity = self.config['drift']['drift_intensity']
        client_datasets = self.client_datasets

        # Use a mutable container to track drift injection state
        drift_state = {'injected': False}

        def client_fn_with_drift(context: Context) -> Any:
            # Check if we should inject drift
            current_round = getattr(context, 'round', 0)

            if (current_round == drift_injection_round and not drift_state['injected']):
                print(f"[DRIFT] Injecting drift at round {drift_injection_round}...")

                # Apply drift directly without using data_loader object
                for client_id in affected_clients:
                    if client_id not in client_datasets:
                        continue

                    original_dataset = client_datasets[client_id]
                    texts = original_dataset.texts.copy()
                    labels = original_dataset.labels.copy()

                    # Simple drift injection (label noise only to avoid NLTK dependency)
                    if 'label_noise' in drift_types:
                        import numpy as np
                        num_samples = len(labels)
                        num_to_flip = int(num_samples * 0.2)  # 20% noise rate
                        indices_to_flip = np.random.choice(num_samples, num_to_flip, replace=False)

                        for idx in indices_to_flip:
                            original_label = labels[idx]
                            possible_labels = [i for i in range(4) if i != original_label]
                            labels[idx] = np.random.choice(possible_labels)

                    # Create new drifted dataset
                    from fed_drift.data import AGNewsDataset
                    drifted_dataset = AGNewsDataset(
                        texts=texts,
                        labels=labels,
                        tokenizer=original_dataset.tokenizer
                    )
                    client_datasets[client_id] = drifted_dataset

                print(f"[DRIFT] Applied drift types {drift_types} to clients {affected_clients}")
                drift_state['injected'] = True

            return original_client_fn(context)
        
        # Create strategy
        strategy = self.create_strategy()
        
        # Configure simulation
        simulation_config = {
            'backend_config': {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
            'ray_init_args': self.config['simulation']['ray_init_args']
        }
        
        try:
            # Ensure Ray is properly cleaned up from any previous runs
            if RAY_AVAILABLE and ray.is_initialized():
                logger.info("Shutting down existing Ray session...")
                ray.shutdown()
            
            # Create proper Flower config with num_rounds
            config = fl.server.ServerConfig(num_rounds=self.num_rounds)

            # Run simulation
            history = start_simulation(
                client_fn=client_fn_with_drift,
                num_clients=self.num_clients,
                config=config,
                strategy=strategy,
                client_resources={"num_cpus": 1, "num_gpus": 0.0},
                ray_init_args=simulation_config['ray_init_args']
            )
            
            logger.info("Simulation completed successfully")
            
            # Collect and analyze results
            results = self._analyze_results(history, strategy)
            
            # Save results
            self._save_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise e
        finally:
            # Ensure Ray is properly shutdown after simulation
            if RAY_AVAILABLE and ray.is_initialized():
                logger.info("Cleaning up Ray resources...")
                ray.shutdown()
    
    def _analyze_results(self, history: Any, strategy: Any) -> Dict[str, Any]:
        """Analyze simulation results."""
        logger.info("Analyzing simulation results...")
        
        results = {
            'simulation_id': self.simulation_id,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract training history from multiple possible sources
        if hasattr(history, 'losses_distributed'):
            results['training_losses'] = history.losses_distributed
        
        if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
            # Handle both fit and evaluate metrics from distributed
            if 'fit' in history.metrics_distributed:
                results['training_metrics'] = history.metrics_distributed['fit']
            if 'evaluate' in history.metrics_distributed:
                results['evaluation_metrics'] = history.metrics_distributed['evaluate']
        
        # Fallback to older attribute names for compatibility
        if hasattr(history, 'metrics_distributed_fit'):
            results['training_metrics'] = history.metrics_distributed_fit
        
        if hasattr(history, 'losses_centralized'):
            results['evaluation_losses'] = history.losses_centralized
        
        if hasattr(history, 'metrics_centralized') and history.metrics_centralized:
            results['evaluation_metrics'] = history.metrics_centralized
        
        # Get drift detection summary from strategy
        if hasattr(strategy, 'get_drift_summary'):
            results['drift_summary'] = strategy.get_drift_summary()
        
        # Calculate drift detection metrics
        if hasattr(strategy, 'drift_history') and strategy.drift_history:
            drift_metrics = calculate_drift_metrics(
                [entry['result'] for entry in strategy.drift_history],
                self.drift_injection_round
            )
            results['drift_metrics'] = drift_metrics
        
        # Calculate performance metrics with fallback
        try:
            results['performance_metrics'] = self._calculate_performance_metrics(history)
        except Exception as e:
            logger.warning(f"Performance metrics calculation failed, using fallback: {e}")
            results['performance_metrics'] = self._calculate_fallback_metrics(strategy)
        
        logger.info("Results analysis completed")
        return results
    
    def _calculate_performance_metrics(self, history: Any) -> Dict[str, float]:
        """Calculate key performance metrics."""
        metrics = {}
        
        try:
            # Extract accuracy data from distributed evaluation metrics
            accuracies = []
            fairness_gaps = []
            
            # Check multiple possible locations for metrics
            evaluation_metrics = None
            
            # Try metrics_distributed first (this is where Flower actually stores them)
            if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
                if 'evaluate' in history.metrics_distributed:
                    evaluation_metrics = history.metrics_distributed['evaluate']
                    logger.info(f"Found distributed evaluation metrics: {len(evaluation_metrics)} rounds")
            
            # Fallback to metrics_centralized if available
            elif hasattr(history, 'metrics_centralized') and history.metrics_centralized:
                evaluation_metrics = history.metrics_centralized
                logger.info(f"Found centralized evaluation metrics: {len(evaluation_metrics)} rounds")
            
            # Extract metrics from either source
            if evaluation_metrics:
                for round_data in evaluation_metrics:
                    if isinstance(round_data, (list, tuple)) and len(round_data) >= 2:
                        round_num, round_metrics = round_data[0], round_data[1]
                        
                        if isinstance(round_metrics, dict):
                            if 'global_accuracy' in round_metrics:
                                accuracies.append(float(round_metrics['global_accuracy']))
                            if 'fairness_gap' in round_metrics:
                                fairness_gaps.append(float(round_metrics['fairness_gap']))
                
                logger.info(f"Extracted {len(accuracies)} accuracy values: {accuracies}")
                logger.info(f"Extracted {len(fairness_gaps)} fairness gap values: {fairness_gaps}")
                
                if accuracies:
                    metrics.update({
                        'final_accuracy': accuracies[-1],
                        'peak_accuracy': max(accuracies),
                        'avg_accuracy': np.mean(accuracies),
                        'accuracy_std': np.std(accuracies)
                    })
                
                if fairness_gaps:
                    metrics.update({
                        'final_fairness_gap': fairness_gaps[-1],
                        'max_fairness_gap': max(fairness_gaps),
                        'avg_fairness_gap': np.mean(fairness_gaps)
                    })
                
                # Calculate comprehensive recovery metrics if drift was injected
                if len(accuracies) > self.drift_injection_round:
                    recovery_metrics = self._calculate_recovery_metrics(accuracies)
                    metrics.update(recovery_metrics)
            else:
                logger.warning("No evaluation metrics found in history object")
                # Debug: log available attributes
                logger.info(f"Available history attributes: {[attr for attr in dir(history) if not attr.startswith('_')]}")
        
        except Exception as e:
            logger.warning(f"Failed to calculate some performance metrics: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        return metrics

    def _calculate_recovery_metrics(
        self,
        accuracies: List[float],
        mitigation_start_round: int = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive recovery metrics following academic standards.

        Analyzes the recovery process after drift detection and mitigation activation,
        measuring speed, completeness, and quality of recovery.

        Metrics Calculated:
            - pre_drift_accuracy: Baseline accuracy before drift injection
            - at_drift_accuracy: Accuracy at drift injection round
            - post_recovery_accuracy: Final accuracy after recovery
            - recovery_speed_rounds: Rounds until stabilization
            - recovery_completeness: % of lost performance restored
            - recovery_quality_score: Combined metric (completeness * (1 / speed))
            - overshoot: If recovery exceeds pre-drift baseline
            - undershoot: If recovery falls short of baseline
            - stability_post_recovery: Std of accuracy after stabilization
            - full_recovery_achieved: Boolean flag

        Args:
            accuracies: List of global accuracy values across all rounds
            mitigation_start_round: Round when mitigation started (auto-detected if None)

        Returns:
            Dictionary with comprehensive recovery metrics
        """
        metrics = {}

        # Validate inputs
        if not accuracies or len(accuracies) <= self.drift_injection_round:
            logger.warning("Insufficient data for recovery metrics calculation")
            return metrics

        # Step 1: Calculate pre-drift baseline (average before drift injection)
        pre_drift_window = accuracies[:self.drift_injection_round]
        pre_drift_accuracy = float(np.mean(pre_drift_window))
        pre_drift_std = float(np.std(pre_drift_window))

        # Step 2: Measure drift impact (accuracy at injection round)
        at_drift_accuracy = float(accuracies[self.drift_injection_round])

        # Step 3: Detect mitigation start (if not provided)
        if mitigation_start_round is None:
            # Auto-detect: look for first improvement after drift
            mitigation_start_round = self.drift_injection_round
            for i in range(self.drift_injection_round + 1, len(accuracies)):
                if accuracies[i] > at_drift_accuracy:
                    mitigation_start_round = i
                    logger.debug(f"Auto-detected mitigation start at round {i}")
                    break

        # Step 4: Find stabilization point using window-based approach
        # Look for point where accuracy changes become minimal
        stabilization_threshold = 0.01  # 1% change threshold
        stabilization_window = 3  # Must be stable for 3 consecutive rounds

        stabilization_round = find_stabilization_point(
            values=accuracies,
            start_index=mitigation_start_round,
            threshold=stabilization_threshold,
            window_size=stabilization_window
        )

        # Step 5: Calculate recovery metrics
        post_recovery_accuracy = float(accuracies[stabilization_round])
        recovery_speed_rounds = stabilization_round - self.drift_injection_round

        # Calculate completeness: % of lost performance restored
        # Formula: (recovered - at_drift) / (pre_drift - at_drift)
        performance_lost = pre_drift_accuracy - at_drift_accuracy
        performance_recovered = post_recovery_accuracy - at_drift_accuracy

        if performance_lost > 0:
            recovery_completeness = performance_recovered / performance_lost
        else:
            # No performance lost, or accuracy increased at drift
            recovery_completeness = 1.0 if post_recovery_accuracy >= pre_drift_accuracy else 0.0

        # Clamp completeness to [0, inf) - can exceed 1.0 if overshoot occurs
        recovery_completeness = max(0.0, recovery_completeness)

        # Calculate recovery quality score (combines completeness and speed)
        # Higher completeness and lower speed = better quality
        # Formula: completeness * (1 / normalized_speed)
        max_rounds = len(accuracies)
        normalized_speed = recovery_speed_rounds / max_rounds  # Normalize to [0, 1]
        speed_factor = 1.0 / (normalized_speed + 0.1)  # Add 0.1 to avoid division by zero

        recovery_quality_score = recovery_completeness * speed_factor

        # Calculate overshoot and undershoot
        overshoot = max(0.0, post_recovery_accuracy - pre_drift_accuracy)
        undershoot = max(0.0, pre_drift_accuracy - post_recovery_accuracy)

        # Calculate stability after recovery
        post_stabilization_window = accuracies[stabilization_round:]
        stability_post_recovery = (
            float(np.std(post_stabilization_window))
            if len(post_stabilization_window) > 1
            else 0.0
        )

        # Determine if full recovery was achieved
        recovery_tolerance = 0.02  # Within 2% of baseline
        full_recovery_achieved = abs(post_recovery_accuracy - pre_drift_accuracy) <= recovery_tolerance

        # Step 6: Package all metrics
        metrics.update({
            # Baseline measurements
            'pre_drift_accuracy': pre_drift_accuracy,
            'pre_drift_std': pre_drift_std,
            'at_drift_accuracy': at_drift_accuracy,
            'post_recovery_accuracy': post_recovery_accuracy,

            # Recovery measurements
            'recovery_speed_rounds': recovery_speed_rounds,
            'recovery_completeness': recovery_completeness,
            'recovery_quality_score': recovery_quality_score,

            # Additional metrics
            'overshoot': overshoot,
            'undershoot': undershoot,
            'stability_post_recovery': stability_post_recovery,

            # Analysis flags
            'full_recovery_achieved': full_recovery_achieved,
            'stabilization_round': stabilization_round,
            'mitigation_start_round': mitigation_start_round,

            # Performance changes
            'performance_lost': performance_lost,
            'performance_recovered': performance_recovered,

            # Legacy metric (for backward compatibility)
            'accuracy_recovery_rate': post_recovery_accuracy / pre_drift_accuracy if pre_drift_accuracy > 0 else 0.0
        })

        logger.info(
            f"Recovery analysis: "
            f"Completeness={recovery_completeness:.2%}, "
            f"Speed={recovery_speed_rounds} rounds, "
            f"Quality={recovery_quality_score:.3f}, "
            f"Full recovery={'Yes' if full_recovery_achieved else 'No'}"
        )

        return metrics

    def _calculate_fallback_metrics(self, strategy: Any) -> Dict[str, float]:
        """Calculate performance metrics using strategy's performance history as fallback."""
        metrics = {}
        
        try:
            # Try to extract from strategy's performance history
            if hasattr(strategy, 'performance_history') and strategy.performance_history:
                accuracies = [p['global_accuracy'] for p in strategy.performance_history]
                fairness_gaps = [p['fairness_gap'] for p in strategy.performance_history]
                
                if accuracies:
                    metrics.update({
                        'final_accuracy': accuracies[-1],
                        'peak_accuracy': max(accuracies),
                        'avg_accuracy': np.mean(accuracies),
                        'accuracy_std': np.std(accuracies)
                    })
                
                if fairness_gaps:
                    metrics.update({
                        'final_fairness_gap': fairness_gaps[-1],
                        'max_fairness_gap': max(fairness_gaps),
                        'avg_fairness_gap': np.mean(fairness_gaps)
                    })
                
                logger.info(f"Fallback metrics extracted: {len(accuracies)} accuracy values")
            else:
                logger.warning("No fallback performance history available")
                
        except Exception as e:
            logger.error(f"Fallback metrics calculation also failed: {e}")
        
        return metrics
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save simulation results to files."""
        logger.info("Saving simulation results...")
        
        # Save JSON results
        json_path = self.results_dir / f"simulation_{self.simulation_id}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV summary
        csv_path = self.results_dir / f"summary_{self.simulation_id}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write configuration
            writer.writerow(["Configuration"])
            for key, value in results['config'].items():
                writer.writerow([key, str(value)])
            
            writer.writerow([])  # Empty row
            
            # Write performance metrics
            writer.writerow(["Performance Metrics"])
            if 'performance_metrics' in results:
                for key, value in results['performance_metrics'].items():
                    writer.writerow([key, value])
            
            writer.writerow([])  # Empty row
            
            # Write drift metrics
            writer.writerow(["Drift Detection Metrics"])
            if 'drift_metrics' in results:
                for key, value in results['drift_metrics'].items():
                    writer.writerow([key, value])
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
    
    def create_visualizations(self, results: Dict[str, Any]) -> None:
        """Create visualizations from simulation results."""
        try:
            # Create drift detection timeline
            if 'drift_summary' in results and 'drift_history_length' in results['drift_summary']:
                output_path = self.results_dir / f"drift_timeline_{self.simulation_id}.png"
                # This would need the actual drift history data
                logger.info("Visualization creation would require additional data extraction")
                
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")


def main():
    """Main entry point for the simulation."""
    # Create and run simulation
    simulation = FederatedDriftSimulation()
    
    try:
        results = simulation.run_simulation()
        
        # Print enhanced summary with better error handling
        print("\n" + "="*60)
        print("🎯 SIMULATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Performance metrics with safe formatting
        if 'performance_metrics' in results and results['performance_metrics']:
            metrics = results['performance_metrics']
            final_acc = metrics.get('final_accuracy')
            peak_acc = metrics.get('peak_accuracy')
            fairness_gap = metrics.get('final_fairness_gap')
            recovery_rate = metrics.get('accuracy_recovery_rate')
            
            print(f"📊 Final Global Accuracy: {final_acc:.4f}" if final_acc is not None else "📊 Final Global Accuracy: N/A")
            print(f"📈 Peak Accuracy: {peak_acc:.4f}" if peak_acc is not None else "📈 Peak Accuracy: N/A")
            print(f"⚖️  Fairness Gap: {fairness_gap:.4f}" if fairness_gap is not None else "⚖️  Fairness Gap: N/A")
            
            if recovery_rate is not None:
                print(f"🔄 Recovery Rate: {recovery_rate:.4f}")
        else:
            print("📊 Final Global Accuracy: N/A")
            print("📈 Peak Accuracy: N/A")
            print("⚖️  Fairness Gap: N/A")
        
        # Drift detection metrics with safe formatting
        if 'drift_summary' in results and results['drift_summary']:
            drift_summary = results['drift_summary']
            detection_rate = drift_summary.get('drift_detection_rate')
            mitigation_active = drift_summary.get('mitigation_activated')
            
            print(f"🔍 Drift Detection Rate: {detection_rate:.4f}" if detection_rate is not None else "🔍 Drift Detection Rate: N/A")
            print(f"🛡️  Mitigation Activated: {mitigation_active}" if mitigation_active is not None else "🛡️  Mitigation Activated: N/A")
        else:
            print("🔍 Drift Detection Rate: N/A")
            print("🛡️  Mitigation Activated: N/A")
        
        # Results location
        json_file = f"results/simulation_{results['simulation_id']}.json"
        print(f"💾 Results saved to: {json_file}")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise e


if __name__ == "__main__":
    main()