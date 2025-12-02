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
        
        # Embeddings storage for server retrieval
        self.current_embeddings = np.array([])
        
        logger.info(f"Initialized drift detection client {client_id}")
    
    def get_parameters(self, config):
        """Get model parameters."""
        try:
            parameters = [param.detach().cpu().numpy() for param in self.model.parameters()]
            return parameters
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to get parameters: {e}")
            return []
    
    def set_parameters(self, parameters):
        """Set model parameters with validation."""
        try:
            if not parameters:  # Handle empty parameters (first round)
                logger.info(f"Client {self.client_id}: Received empty parameters (initial round)")
                return

            # Calculate parameter checksum before setting
            old_checksum = sum(torch.sum(p).item() for p in self.model.parameters())
            
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=False)  # Use strict=False to handle mismatches
            
            # Calculate parameter checksum after setting
            new_checksum = sum(torch.sum(p).item() for p in self.model.parameters())
            
            if abs(old_checksum - new_checksum) < 1e-6:
                logger.warning(f"Client {self.client_id}: Model parameters unchanged after setting (checksum: {old_checksum:.6f})")
            else:
                logger.info(f"Client {self.client_id}: Model parameters updated (checksum: {old_checksum:.6f} → {new_checksum:.6f})")
                
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to set parameters: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            # Don't raise - continue with current parameters
    
    def fit(self, parameters, config):
        """
        Train model locally with drift detection.

        Args:
            parameters: Model parameters from server
            config: Training configuration

        Returns:
            Tuple of (parameters, num_examples, metrics)
        """
        self.round_number += 1

        try:
            # Set parameters from server
            self.set_parameters(parameters)

            # Configure training
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
            
            # Add embeddings transmission fix for Flower ConfigRecord TypeError
            if len(embeddings) > 0:
                # Fix: Use flattened embeddings or alternative transmission method
                # Flower ConfigRecord can't handle nested lists - use simple statistics instead
                embedding_stats = {
                    "embedding_count": len(embeddings),
                    "embedding_dim": embeddings.shape[1] if embeddings.ndim > 1 else 1,
                    "embedding_mean": float(np.mean(embeddings)),
                    "embedding_std": float(np.std(embeddings)),
                    "embedding_sample": embeddings[0].tolist() if len(embeddings) > 0 else []  # Send first embedding as sample
                }
                response_metrics.update(embedding_stats)
                
                # Store embeddings in client for later server retrieval (alternative approach)
                self.current_embeddings = embeddings
            else:
                response_metrics["embedding_count"] = 0
                self.current_embeddings = np.array([])
            
            # Get updated parameters
            updated_parameters = [param.detach().cpu().numpy() for param in self.model.parameters()]

            return updated_parameters, len(self.train_loader.dataset), response_metrics

        except Exception as e:
            logger.error(f"Client {self.client_id}: Training failed: {e}")
            # Return at least 1 example to avoid division by zero in aggregation
            num_examples = max(1, len(self.train_loader.dataset))
            return parameters, num_examples, {"error": str(e)}
    
    def evaluate(self, parameters, config):
        """
        Evaluate model locally.

        Args:
            parameters: Model parameters from server
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        try:
            # Set parameters from server
            self.set_parameters(parameters)

            # Evaluate model
            eval_metrics = self._evaluate_model()

            # Update drift detector with performance (fix ADWIN API)
            if "accuracy" in eval_metrics and hasattr(self.drift_detector, 'adwin_detector'):
                # ADWIN expects individual performance values, not a method call
                try:
                    self.drift_detector.adwin_detector.update(eval_metrics["accuracy"])
                except Exception as e:
                    logger.debug(f"Client {self.client_id}: ADWIN update failed: {e}")

            # Prepare response metrics
            response_metrics = {
                **eval_metrics,
                "client_id": self.client_id,
                "round": self.round_number
            }

            return eval_metrics.get("loss", 0.0), len(self.test_loader.dataset), response_metrics

        except Exception as e:
            logger.error(f"Client {self.client_id}: Evaluation failed: {e}")
            # Return at least 1 example to avoid division by zero in aggregation
            num_examples = max(1, len(self.test_loader.dataset))
            return float('inf'), num_examples, {"error": str(e)}
    
    def _train_model(self, epochs: int, learning_rate: float) -> Dict[str, float]:
        """Train the model for specified epochs with enhanced debugging."""
        # Calculate initial model checksum
        initial_checksum = sum(torch.sum(p).item() for p in self.model.parameters())
        logger.info(f"Client {self.client_id}: Starting training with {len(self.train_loader)} batches, LR={learning_rate}, initial checksum={initial_checksum:.6f}")
        
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
            
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    metrics = self.trainer.train_step(batch, optimizer)
                    
                    epoch_loss += metrics['loss']
                    epoch_accuracy += metrics['accuracy']
                    epoch_batches += 1
                    
                    # Log first few batches for debugging
                    if batch_idx < 3:
                        logger.debug(f"Client {self.client_id}, Epoch {epoch + 1}, Batch {batch_idx + 1}: "
                                   f"Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}")
                        
                except Exception as e:
                    logger.error(f"Client {self.client_id}: Training step failed on batch {batch_idx}: {e}")
                    continue
            
            if epoch_batches > 0:
                epoch_loss /= epoch_batches
                epoch_accuracy /= epoch_batches
                
                total_loss += epoch_loss
                total_accuracy += epoch_accuracy
                num_batches += 1
                
                logger.info(f"Client {self.client_id}, Epoch {epoch + 1}/{epochs}: "
                           f"Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}, Batches={epoch_batches}")
            else:
                logger.warning(f"Client {self.client_id}, Epoch {epoch + 1}: No batches processed successfully")
        
        # Calculate final model checksum
        final_checksum = sum(torch.sum(p).item() for p in self.model.parameters())
        logger.info(f"Client {self.client_id}: Training completed, final checksum={final_checksum:.6f}, change={final_checksum - initial_checksum:.6f}")
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
        
        train_metrics = {
            "train_loss": avg_loss,
            "train_accuracy": avg_accuracy,
            "epochs_completed": epochs,
            "batches_processed": num_batches,
            "parameter_change": float(final_checksum - initial_checksum)
        }
        
        # Store in history
        self.training_history.append({
            'round': self.round_number,
            'metrics': train_metrics
        })
        
        logger.info(f"Client {self.client_id}: Training metrics: {train_metrics}")
        return train_metrics
    
    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model on test data."""
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in self.test_loader:
                try:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    logits = self.model(input_ids, attention_mask)
                    loss = self.model.criterion(logits, labels)
                    
                    # Calculate accuracy
                    predictions = torch.argmax(logits, dim=-1)
                    correct = (predictions == labels).float().mean()
                    
                    total_loss += loss.item()
                    total_accuracy += correct.item()
                    num_batches += 1
                    
                    all_predictions.extend(predictions.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                    
                except Exception as batch_error:
                    logger.warning(f"Client {self.client_id}: Batch evaluation failed: {batch_error}")
                    continue
        
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