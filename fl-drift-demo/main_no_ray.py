#!/usr/bin/env python3
"""
Ray-free main entry point for the Federated Learning Drift Detection System.

This version avoids the protobuf/Ray mutex lock conflicts by using 
threading-based simulation instead of Ray.

Usage:
    python main_no_ray.py [--config CONFIG_FILE] [--rounds ROUNDS]
    
Example:
    python main_no_ray.py --rounds 3 --clients 2 --drift-round 2
"""

import argparse
import sys
import logging
import os
from pathlib import Path

# Set threading environment variables to prevent conflicts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fed_drift.simulation_no_ray import ThreadedFederatedSimulation
from fed_drift.config import ConfigManager


def setup_logging(config: dict) -> None:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_file = log_config.get('file', 'logs/simulation.log')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point for Ray-free simulation."""
    parser = argparse.ArgumentParser(
        description='Federated Learning Drift Detection System (Ray-free)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_no_ray.py                               # Run with default configuration
  python main_no_ray.py --config custom.yaml         # Run with custom config
  python main_no_ray.py --rounds 3 --clients 2       # Quick test run
  python main_no_ray.py --rounds 10 --drift-round 5  # Override drift settings
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--rounds', '-r',
        type=int,
        help='Number of training rounds (overrides config)'
    )
    
    parser.add_argument(
        '--clients', '-n',
        type=int,
        help='Number of clients (overrides config)'
    )
    
    parser.add_argument(
        '--drift-round',
        type=int,
        help='Round to inject drift (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for results (overrides config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress most output'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    
    # Apply command line overrides
    if args.rounds:
        config_manager.set('simulation.num_rounds', args.rounds)
    
    if args.clients:
        config_manager.set('federated.num_clients', args.clients)
    
    if args.drift_round:
        config_manager.set('drift.injection_round', args.drift_round)
    
    if args.output_dir:
        config_manager.set('output.results_dir', args.output_dir)
    
    # Adjust logging level
    if args.verbose:
        config_manager.set('logging.level', 'DEBUG')
    elif args.quiet:
        config_manager.set('logging.level', 'WARNING')
    
    # Setup logging
    setup_logging(config_manager.config)
    logger = logging.getLogger(__name__)
    
    # Run simulation
    try:
        logger.info("Starting Ray-free Federated Learning Drift Detection Simulation")
        logger.info(f"Configuration: {config_manager.config}")
        
        # Create and run simulation
        simulation = ThreadedFederatedSimulation(config_manager.config)
        results = simulation.run_simulation()
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 RAY-FREE SIMULATION COMPLETED SUCCESSFULLY")
        print("="*60)

        # Helper function to format numeric values
        def format_metric(value):
            if isinstance(value, (int, float)):
                return f"{value:.4f}"
            return str(value)

        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"📊 Final Global Accuracy: {format_metric(metrics.get('final_accuracy', 'N/A'))}")
            print(f"📈 Peak Accuracy: {format_metric(metrics.get('peak_accuracy', 'N/A'))}")
            print(f"⚖️  Fairness Gap: {format_metric(metrics.get('final_fairness_gap', 'N/A'))}")

            if 'accuracy_recovery_rate' in metrics:
                print(f"🔄 Recovery Rate: {format_metric(metrics['accuracy_recovery_rate'])}")

        if 'drift_summary' in results:
            drift_summary = results['drift_summary']
            print(f"🔍 Drift Detection Rate: {format_metric(drift_summary.get('drift_detection_rate', 'N/A'))}")
            print(f"🛡️  Mitigation Activated: {drift_summary.get('mitigation_activated', 'N/A')}")
        
        print(f"💾 Results saved to: results/simulation_{results['simulation_id']}.json")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)