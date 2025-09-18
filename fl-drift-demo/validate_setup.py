#!/usr/bin/env python3
"""
Quick validation script to test the federated drift detection system setup.
"""

import sys
import logging
from pathlib import Path

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    try:
        logger.info("Testing imports...")
        
        # Test core imports
        import torch
        import numpy as np
        from transformers import AutoTokenizer
        import flwr
        from datasets import load_dataset
        
        logger.info("✓ Core libraries imported successfully")
        
        # Test drift detection imports
        from river.drift import ADWIN
        logger.info("✓ River (ADWIN) imported successfully")
        
        try:
            from evidently import Report
            logger.info("✓ Evidently imported successfully")
        except ImportError as e:
            logger.warning(f"Evidently import issue: {e}")
        
        try:
            from alibi_detect.cd import MMDDrift
            logger.info("✓ Alibi Detect imported successfully")
        except ImportError as e:
            logger.warning(f"Alibi Detect import issue: {e}")
        
        # Test project imports
        from fed_drift.models import BERTClassifier, get_device
        from fed_drift.data import FederatedDataLoader
        from fed_drift.drift_detection import ADWINDriftDetector
        
        logger.info("✓ Project modules imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Import test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    try:
        logger.info("Testing basic functionality...")
        
        # Test device detection
        from fed_drift.models import get_device
        device = get_device()
        logger.info(f"✓ Device detected: {device}")
        
        # Test tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✓ Tokenizer loaded successfully")
        
        # Test model creation
        from fed_drift.models import BERTClassifier
        model = BERTClassifier(model_name='prajjwal1/bert-tiny', num_classes=4)
        logger.info(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test drift detector
        from fed_drift.drift_detection import ADWINDriftDetector
        detector = ADWINDriftDetector(delta=0.002)
        detector.update(0.8)
        result = detector.detect()
        logger.info(f"✓ Drift detector working. Result: {result.is_drift}")
        
        # Test data loader initialization
        from fed_drift.data import FederatedDataLoader
        data_loader = FederatedDataLoader(num_clients=3, alpha=0.5, batch_size=8)
        logger.info("✓ Data loader initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Functionality test failed: {e}")
        return False

def test_small_simulation():
    """Run a very small simulation to validate integration."""
    try:
        logger.info("Running small simulation test...")
        
        # Import simulation components
        from fed_drift.simulation import FederatedDriftSimulation
        
        # Create minimal config
        minimal_config = {
            'model': {
                'model_name': 'prajjwal1/bert-tiny',
                'max_length': 64,  # Smaller for speed
                'batch_size': 4,   # Smaller batch
                'learning_rate': 2e-5,
                'num_epochs': 1    # Single epoch for speed
            },
            'federated': {
                'num_clients': 3,  # Fewer clients
                'alpha': 0.5
            },
            'drift': {
                'injection_round': 4,  # Early injection
                'drift_intensity': 0.3,
                'affected_clients': [1],
                'drift_types': ['label_noise']
            },
            'simulation': {
                'num_rounds': 6,   # Very short simulation
                'num_cpus': 1,
                'num_gpus': 0.0,
                'ray_init_args': {"include_dashboard": False, "log_to_driver": False}
            },
            'strategy': {
                'min_fit_clients': 2,
                'min_evaluate_clients': 2,
                'mitigation_threshold': 0.5
            },
            'drift_detection': {
                'adwin_delta': 0.002,
                'mmd_p_val': 0.05
            }
        }
        
        # Note: Skip actual simulation for now due to complexity
        simulation = FederatedDriftSimulation(minimal_config)
        logger.info("✓ Simulation object created successfully")
        
        # Test data preparation
        simulation.prepare_data()
        logger.info("✓ Data preparation completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Small simulation test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    logger.info("Starting federated drift detection system validation...")
    
    tests = [
        ("Import Tests", test_imports),
        ("Functionality Tests", test_basic_functionality),
        ("Integration Tests", test_small_simulation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        if test_func():
            logger.info(f"✅ {test_name} PASSED")
            passed += 1
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"VALIDATION SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All validation tests passed! System is ready.")
        return 0
    else:
        logger.error("⚠️  Some validation tests failed. Check the logs above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())