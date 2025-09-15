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

import flwr as fl
from flwr.simulation import start_simulation
from flwr.common import Context

from .data import FederatedDataLoader, MODEL_CONFIG, DRIFT_CONFIG, FEDERATED_CONFIG
from .models import create_model, get_device
from .client import create_drift_detection_client
from .server import create_drift_aware_strategy
from .drift_detection import calculate_drift_metrics, visualize_drift_timeline

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
    
    def _analyze_results(self, history: Any, strategy: Any) -> Dict[str, Any]:
        """Analyze simulation results."""
        logger.info("Analyzing simulation results...")
        
        results = {
            'simulation_id': self.simulation_id,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract training history
        if hasattr(history, 'losses_distributed'):
            results['training_losses'] = history.losses_distributed
        
        if hasattr(history, 'metrics_distributed_fit'):
            results['training_metrics'] = history.metrics_distributed_fit
        
        if hasattr(history, 'losses_centralized'):
            results['evaluation_losses'] = history.losses_centralized
        
        if hasattr(history, 'metrics_centralized'):
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
        
        # Calculate performance metrics
        results['performance_metrics'] = self._calculate_performance_metrics(history)
        
        logger.info("Results analysis completed")
        return results
    
    def _calculate_performance_metrics(self, history: Any) -> Dict[str, float]:
        """Calculate key performance metrics."""
        metrics = {}
        
        try:
            # Extract accuracy data
            if hasattr(history, 'metrics_centralized'):
                accuracies = []
                fairness_gaps = []
                
                for round_metrics in history.metrics_centralized:
                    if 'global_accuracy' in round_metrics[1]:
                        accuracies.append(round_metrics[1]['global_accuracy'])
                    if 'fairness_gap' in round_metrics[1]:
                        fairness_gaps.append(round_metrics[1]['fairness_gap'])
                
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
                
                # Calculate recovery metrics if drift was injected
                if len(accuracies) > self.drift_injection_round:
                    pre_drift_acc = np.mean(accuracies[:self.drift_injection_round])
                    post_drift_acc = accuracies[-1]
                    
                    metrics['pre_drift_accuracy'] = pre_drift_acc
                    metrics['post_drift_accuracy'] = post_drift_acc
                    metrics['accuracy_recovery_rate'] = post_drift_acc / pre_drift_acc if pre_drift_acc > 0 else 0.0
        
        except Exception as e:
            logger.warning(f"Failed to calculate some performance metrics: {e}")
        
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
        
        # Print summary
        print("\n" + "="*50)
        print("SIMULATION SUMMARY")
        print("="*50)
        print(f"Simulation ID: {results['simulation_id']}")
        
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"Final Accuracy: {metrics.get('final_accuracy', 'N/A'):.4f}")
            print(f"Peak Accuracy: {metrics.get('peak_accuracy', 'N/A'):.4f}")
            print(f"Fairness Gap: {metrics.get('final_fairness_gap', 'N/A'):.4f}")
            
            if 'accuracy_recovery_rate' in metrics:
                print(f"Recovery Rate: {metrics['accuracy_recovery_rate']:.4f}")
        
        if 'drift_summary' in results:
            drift_summary = results['drift_summary']
            print(f"Drift Detection Rate: {drift_summary.get('drift_detection_rate', 'N/A'):.4f}")
            print(f"Mitigation Activated: {drift_summary.get('mitigation_activated', 'N/A')}")
        
        print("="*50)
        
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise e


if __name__ == "__main__":
    main()