"""
Client-side federated learning implementation with integrated drift detection.

This module implements:
- DriftDetectionClient that integrates with Flower
- Local model training with drift monitoring
- Client-side metric collection and reporting
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging

from flwr.client import NumPyClient
from flwr.common import (
    Config, EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns, GetParametersRes,
    Status, Code, parameters_to_ndarrays, ndarrays_to_parameters
)

from .models import BERTClassifier, ModelTrainer, ModelUtils, get_device
from .data import AGNewsDataset
from .drift_detection import MultiLevelDriftDetector, DriftResult

logger = logging.getLogger(__name__)


class DriftDetectionClient(NumPyClient):
    """
    Flower client with integrated drift detection capabilities.
    """
    
    def __init__(self,
                 client_id: str,
                 model: BERTClassifier,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 drift_config: Dict[str, Any] = None):
        """
        Initialize drift detection client.
        
        Args:
            client_id: Unique client identifier
            model: BERT classifier model
            train_loader: Training data loader
            test_loader: Testing data loader  
            device: Training device
            drift_config: Drift detection configuration
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.drift_config = drift_config or {}
        
        # Initialize model trainer
        self.trainer = ModelTrainer(model, device)
        
        # Initialize drift detector
        self.drift_detector = MultiLevelDriftDetector(drift_config)
        
        # Training state
        self.round_number = 0
        self.reference_data_set = False
        self.training_history = []
        
        # Performance tracking
        self.performance_history = []
        self.drift_detection_history = []
        
        logger.info(f"Initialized drift detection client {client_id}")
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Get model parameters."""
        try:
            parameters = [param.cpu().numpy() for param in self.model.parameters()]
            return GetParametersRes(
                status=Status(code=Code.OK, message="Parameters retrieved successfully"),
                parameters=ndarrays_to_parameters(parameters)
            )
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to get parameters: {e}")
            return GetParametersRes(
                status=Status(code=Code.GET_PARAMETERS_NOT_IMPLEMENTED, message=str(e)),
                parameters=ndarrays_to_parameters([])
            )
    
    def set_parameters(self, parameters):
        """Set model parameters."""
        try:
            params_dict = zip(self.model.state_dict().keys(), parameters_to_ndarrays(parameters))
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to set parameters: {e}")
            raise e
    
    def fit(self, ins: FitIns) -> FitRes:
        """
        Train model locally with drift detection.
        
        Args:
            ins: Fit instructions from server
            
        Returns:
            Fit results with drift metrics
        """
        self.round_number += 1
        
        try:
            # Set parameters from server
            self.set_parameters(ins.parameters)
            
            # Configure training
            config = ins.config
            epochs = int(config.get("epochs", 3))
            learning_rate = float(config.get("learning_rate", 2e-5))
            
            # Train model
            train_metrics = self._train_model(epochs, learning_rate)
            
            # Collect embeddings for drift detection
            embeddings = self._collect_embeddings()
            
            # Perform drift detection
            drift_results = self._perform_drift_detection(train_metrics, embeddings)
            
            # Prepare response metrics
            response_metrics = {
                **train_metrics,
                **drift_results,
                "client_id": self.client_id,
                "round": self.round_number,
                "num_examples": len(self.train_loader.dataset)
            }
            
            # Add embeddings to response (sample for efficiency)
            if len(embeddings) > 0:
                # Sample embeddings to avoid large payloads
                sample_size = min(100, len(embeddings))
                sampled_embeddings = embeddings[:sample_size].tolist()
                response_metrics["embeddings"] = sampled_embeddings
            
            # Get updated parameters
            parameters = [param.cpu().numpy() for param in self.model.parameters()]
            
            return FitRes(
                status=Status(code=Code.OK, message="Training completed successfully"),
                parameters=ndarrays_to_parameters(parameters),
                num_examples=len(self.train_loader.dataset),
                metrics=response_metrics
            )
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Training failed: {e}")
            return FitRes(
                status=Status(code=Code.FIT_CLIENT_INTERRUPTED, message=str(e)),
                parameters=ins.parameters,
                num_examples=0,
                metrics={"error": str(e)}
            )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate model locally.
        
        Args:
            ins: Evaluation instructions from server
            
        Returns:
            Evaluation results with performance metrics
        """
        try:
            # Set parameters from server
            self.set_parameters(ins.parameters)
            
            # Evaluate model
            eval_metrics = self._evaluate_model()
            
            # Update drift detector with performance
            if "accuracy" in eval_metrics:
                self.drift_detector.update_performance(eval_metrics["accuracy"])
            
            # Prepare response metrics
            response_metrics = {
                **eval_metrics,
                "client_id": self.client_id,
                "round": self.round_number
            }
            
            return EvaluateRes(
                status=Status(code=Code.OK, message="Evaluation completed successfully"),
                loss=eval_metrics.get("loss", 0.0),
                num_examples=len(self.test_loader.dataset),
                metrics=response_metrics
            )
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Evaluation failed: {e}")
            return EvaluateRes(
                status=Status(code=Code.EVALUATE_CLIENT_INTERRUPTED, message=str(e)),
                loss=float('inf'),
                num_examples=0,
                metrics={"error": str(e)}
            )
    
    def _train_model(self, epochs: int, learning_rate: float) -> Dict[str, float]:
        """Train the model for specified epochs."""
        # Create optimizer
        optimizer = ModelUtils.create_optimizer(self.model, learning_rate)
        
        # Training loop
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            epoch_batches = 0
            
            for batch in self.train_loader:
                metrics = self.trainer.train_step(batch, optimizer)
                
                epoch_loss += metrics['loss']
                epoch_accuracy += metrics['accuracy']
                epoch_batches += 1
            
            if epoch_batches > 0:
                epoch_loss /= epoch_batches
                epoch_accuracy /= epoch_batches
                
                total_loss += epoch_loss
                total_accuracy += epoch_accuracy
                num_batches += 1
                
                logger.debug(f"Client {self.client_id}, Epoch {epoch + 1}: "
                           f"Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}")
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
        
        train_metrics = {
            "train_loss": avg_loss,
            "train_accuracy": avg_accuracy,
            "epochs_completed": epochs
        }
        
        # Store in history
        self.training_history.append({
            'round': self.round_number,
            'metrics': train_metrics
        })
        
        return train_metrics
    
    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model on test data."""
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        
        for batch in self.test_loader:
            metrics = self.trainer.evaluate_step(batch)
            
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1
            
            all_predictions.extend(metrics['predictions'])
            all_labels.extend(metrics['labels'])
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
        
        eval_metrics = {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "num_samples": len(all_predictions)
        }
        
        # Store in performance history
        self.performance_history.append({
            'round': self.round_number,
            'metrics': eval_metrics
        })
        
        return eval_metrics
    
    def _collect_embeddings(self) -> np.ndarray:
        """Collect model embeddings for drift detection."""
        embeddings = []
        
        self.model.eval()
        with torch.no_grad():
            # Sample a subset of training data for efficiency
            sample_size = min(100, len(self.train_loader.dataset))
            indices = np.random.choice(len(self.train_loader.dataset), sample_size, replace=False)
            
            for idx in indices:
                try:
                    sample = self.train_loader.dataset[idx]
                    
                    # Prepare batch
                    input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                    attention_mask = sample['attention_mask'].unsqueeze(0).to(self.device)
                    
                    # Get embeddings
                    embedding = self.model.get_embeddings(input_ids, attention_mask)
                    embeddings.append(embedding.cpu().numpy().flatten())
                    
                except Exception as e:
                    logger.warning(f"Failed to collect embedding for sample {idx}: {e}")
                    continue
        
        return np.array(embeddings) if embeddings else np.array([])
    
    def _perform_drift_detection(self, train_metrics: Dict[str, float], 
                                embeddings: np.ndarray) -> Dict[str, Any]:
        """Perform local drift detection."""
        drift_results = {}
        
        try:
            # Set reference data on first round
            if self.round_number == 1 and len(embeddings) > 0:
                # Use first round data as reference
                reference_embeddings = embeddings
                
                # Create reference data for Evidently (simplified feature representation)
                reference_data = np.mean(embeddings, axis=1).reshape(-1, 1) if embeddings.ndim > 1 else embeddings.reshape(-1, 1)
                
                self.drift_detector.set_reference_data(reference_data, reference_embeddings)
                self.reference_data_set = True
                
                logger.info(f"Client {self.client_id}: Set reference data with {len(embeddings)} samples")
                
                return {
                    "reference_data_set": True,
                    "reference_samples": len(embeddings)
                }
            
            # Perform drift detection if reference data is set
            if self.reference_data_set:
                # Update drift detectors
                if "train_loss" in train_metrics:
                    self.drift_detector.update_performance(train_metrics["train_loss"])
                
                if len(embeddings) > 0:
                    # Update data drift detector
                    current_data = np.mean(embeddings, axis=1).reshape(-1, 1) if embeddings.ndim > 1 else embeddings.reshape(-1, 1)
                    self.drift_detector.update_data(current_data)
                    
                    # Update embedding drift detector
                    self.drift_detector.update_embeddings(embeddings)
                
                # Detect drift
                all_drift_results = self.drift_detector.detect_all()
                aggregated_result = self.drift_detector.get_aggregated_drift_signal(all_drift_results)
                
                # Convert results to serializable format
                drift_results = {
                    "adwin_drift": all_drift_results.get('concept_drift', DriftResult(False, 0.0)).is_drift,
                    "adwin_score": all_drift_results.get('concept_drift', DriftResult(False, 0.0)).drift_score,
                    "data_drift": all_drift_results.get('data_drift', DriftResult(False, 0.0)).is_drift,
                    "data_drift_score": all_drift_results.get('data_drift', DriftResult(False, 0.0)).drift_score,
                    "embedding_drift": all_drift_results.get('embedding_drift', DriftResult(False, 0.0)).is_drift,
                    "embedding_drift_score": all_drift_results.get('embedding_drift', DriftResult(False, 0.0)).drift_score,
                    "aggregated_drift": aggregated_result.is_drift,
                    "aggregated_drift_score": aggregated_result.drift_score
                }
                
                # Store in detection history
                self.drift_detection_history.append({
                    'round': self.round_number,
                    'results': all_drift_results,
                    'aggregated': aggregated_result
                })
                
                logger.debug(f"Client {self.client_id}: Drift detection completed. "
                           f"Aggregated drift: {aggregated_result.is_drift} "
                           f"(score: {aggregated_result.drift_score:.3f})")
            
        except Exception as e:
            logger.warning(f"Client {self.client_id}: Drift detection failed: {e}")
            drift_results["drift_detection_error"] = str(e)
        
        return drift_results
    
    def get_client_summary(self) -> Dict[str, Any]:
        """Get summary of client training and drift detection."""
        summary = {
            "client_id": self.client_id,
            "total_rounds": self.round_number,
            "training_history_length": len(self.training_history),
            "performance_history_length": len(self.performance_history),
            "drift_detection_history_length": len(self.drift_detection_history),
            "reference_data_set": self.reference_data_set
        }
        
        # Add performance summary
        if self.performance_history:
            accuracies = [h['metrics']['accuracy'] for h in self.performance_history]
            losses = [h['metrics']['loss'] for h in self.performance_history]
            
            summary.update({
                "final_accuracy": accuracies[-1] if accuracies else 0.0,
                "best_accuracy": max(accuracies) if accuracies else 0.0,
                "avg_accuracy": np.mean(accuracies) if accuracies else 0.0,
                "final_loss": losses[-1] if losses else 0.0,
                "best_loss": min(losses) if losses else 0.0
            })
        
        # Add drift detection summary
        if self.drift_detection_history:
            drift_detections = [h['aggregated'].is_drift for h in self.drift_detection_history]
            drift_scores = [h['aggregated'].drift_score for h in self.drift_detection_history]
            
            summary.update({
                "total_drift_detections": sum(drift_detections),
                "drift_detection_rate": sum(drift_detections) / len(drift_detections),
                "avg_drift_score": np.mean(drift_scores),
                "max_drift_score": max(drift_scores) if drift_scores else 0.0
            })
        
        return summary


def create_drift_detection_client(client_id: str,
                                 model: BERTClassifier,
                                 train_loader: DataLoader,
                                 test_loader: DataLoader,
                                 device: torch.device,
                                 drift_config: Dict[str, Any] = None) -> DriftDetectionClient:
    """
    Factory function to create a drift detection client.
    
    Args:
        client_id: Unique client identifier
        model: BERT classifier model
        train_loader: Training data loader
        test_loader: Testing data loader
        device: Training device
        drift_config: Drift detection configuration
        
    Returns:
        Configured DriftDetectionClient
    """
    return DriftDetectionClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        drift_config=drift_config
    )