#!/usr/bin/env python3
"""
Main entry point for the Federated LLM Drift Detection and Recovery System.

Usage:
    python main.py [--config CONFIG_FILE] [--mode MODE] [--rounds ROUNDS]
    
Example:
    python main.py --config custom_config.yaml --rounds 30
"""

import argparse
import sys
import logging
import os
from pathlib import Path

# Set threading environment variables to prevent mutex locks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fed_drift.simulation import FederatedDriftSimulation
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
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Federated LLM Drift Detection and Recovery System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with default configuration
  python main.py --config custom.yaml              # Run with custom config
  python main.py --rounds 30                       # Override number of rounds
  python main.py --clients 5 --drift-round 15      # Override client and drift settings
  python main.py --mode validate                   # Validate configuration only
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['run', 'validate', 'test'],
        default='run',
        help='Operation mode: run simulation, validate config, or run tests'
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
    
    # Validate configuration
    if args.mode in ['validate', 'run']:
        issues = config_manager.validate_config()
        if issues:
            logger.error("Configuration validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            
            if args.mode == 'validate':
                return 1
            else:
                logger.error("Cannot run simulation with invalid configuration")
                return 1
        else:
            logger.info("Configuration validation passed")
            
            if args.mode == 'validate':
                print("✓ Configuration is valid")
                return 0
    
    # Run simulation
    if args.mode == 'run':
        try:
            logger.info("Starting Federated LLM Drift Detection Simulation")
            logger.info(f"Configuration: {config_manager.config}")
            
            # Create and run simulation
            simulation = FederatedDriftSimulation(config_manager.config)
            results = simulation.run_simulation()
            
            # Print summary
            print("\n" + "="*60)
            print("🎯 SIMULATION COMPLETED SUCCESSFULLY")
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
    
    # Run tests
    elif args.mode == 'test':
        try:
            import pytest
            logger.info("Running unit tests...")
            
            # Run tests
            test_args = ['-v', 'tests/']
            if args.verbose:
                test_args.append('-s')
            
            exit_code = pytest.main(test_args)
            return exit_code
            
        except ImportError:
            logger.error("pytest not installed. Install with: pip install pytest")
            return 1
        except Exception as e:
            logger.error(f"Tests failed: {e}")
            return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)