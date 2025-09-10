"""
Model definitions for federated learning with BERT-tiny.

This module contains:
- BERT model wrapper for text classification
- Model utilities and helper functions
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class BERTClassifier(nn.Module):
    """BERT-based text classifier for AG News dataset."""
    
    def __init__(self, model_name: str = 'prajjwal1/bert-tiny', num_classes: int = 4, dropout: float = 0.1):
        super(BERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get embeddings for drift detection."""
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Return [CLS] token embeddings
            return outputs.pooler_output


class ModelUtils:
    """Utility functions for model management."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    @staticmethod
    def freeze_bert_layers(model: BERTClassifier, num_layers_to_freeze: int = 2) -> BERTClassifier:
        """Freeze the first N layers of BERT for faster training."""
        layers_to_freeze = list(model.bert.encoder.layer[:num_layers_to_freeze])
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        
        logger.info(f"Frozen {num_layers_to_freeze} BERT layers")
        return model
    
    @staticmethod
    def create_optimizer(model: nn.Module, learning_rate: float = 2e-5, 
                        weight_decay: float = 0.01) -> torch.optim.AdamW:
        """Create AdamW optimizer with layer-wise learning rates."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        return optimizer
    
    @staticmethod
    def create_scheduler(optimizer: torch.optim.Optimizer, num_training_steps: int,
                        num_warmup_steps: int = 100) -> torch.optim.lr_scheduler.LambdaLR:
        """Create linear learning rate scheduler with warmup."""
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler


class ModelTrainer:
    """Training utilities for BERT classifier."""
    
    def __init__(self, model: BERTClassifier, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Perform a single training step."""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = self.model(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single evaluation step."""
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'predictions': predictions.cpu().numpy(),
            'labels': labels.cpu().numpy()
        }
    
    def get_embeddings_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get embeddings for a batch of data."""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            embeddings = self.model.get_embeddings(input_ids, attention_mask)
            return embeddings.cpu()


def create_model(model_config: Dict[str, Any], device: torch.device) -> Tuple[BERTClassifier, ModelTrainer]:
    """Factory function to create model and trainer."""
    model = BERTClassifier(
        model_name=model_config['model_name'],
        num_classes=4,  # AG News has 4 classes
        dropout=0.1
    )
    
    trainer = ModelTrainer(model, device)
    
    logger.info(f"Created model with {ModelUtils.count_parameters(model):,} parameters")
    logger.info(f"Model size: {ModelUtils.get_model_size_mb(model):.2f} MB")
    
    return model, trainer


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS device")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device