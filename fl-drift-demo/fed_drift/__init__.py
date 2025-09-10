"""
Federated LLM Drift Detection and Recovery System

A comprehensive package for detecting and mitigating data drift 
in federated learning deployments with Large Language Models.
"""

from .data import FederatedDataLoader, DriftInjector, MODEL_CONFIG, DRIFT_CONFIG, FEDERATED_CONFIG
from .models import BERTClassifier, ModelTrainer, ModelUtils, create_model, get_device
from .drift_detection import (
    DriftDetector, ADWINDriftDetector, EvidentiallyDriftDetector, 
    MMDDriftDetector, MultiLevelDriftDetector, DriftResult
)
from .client import DriftDetectionClient, create_drift_detection_client
from .server import DriftAwareFedAvg, FedTrimmedAvg, create_drift_aware_strategy
from .simulation import FederatedDriftSimulation

__version__ = "0.1.0"

__all__ = [
    # Data handling
    "FederatedDataLoader",
    "DriftInjector", 
    "MODEL_CONFIG",
    "DRIFT_CONFIG",
    "FEDERATED_CONFIG",
    
    # Models
    "BERTClassifier",
    "ModelTrainer", 
    "ModelUtils",
    "create_model",
    "get_device",
    
    # Drift detection
    "DriftDetector",
    "ADWINDriftDetector",
    "EvidentiallyDriftDetector", 
    "MMDDriftDetector",
    "MultiLevelDriftDetector",
    "DriftResult",
    
    # Client
    "DriftDetectionClient",
    "create_drift_detection_client",
    
    # Server
    "DriftAwareFedAvg",
    "FedTrimmedAvg", 
    "create_drift_aware_strategy",
    
    # Simulation
    "FederatedDriftSimulation"
]