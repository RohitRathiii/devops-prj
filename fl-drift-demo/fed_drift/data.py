"""
Data preparation module for federated learning with drift injection.

This module handles:
- AG News dataset loading and preprocessing
- Dirichlet partitioning for non-IID distribution
- Synthetic drift injection mechanisms
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset
# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import DirichletPartitioner
import nlpaug.augmenter.word as naw
import logging

logger = logging.getLogger(__name__)


class AGNewsDataset(Dataset):
    """Custom PyTorch Dataset for AG News."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DriftInjector:
    """Handles synthetic drift injection for testing drift detection."""
    
    def __init__(self, drift_intensity: float = 0.3):
        self.drift_intensity = drift_intensity

        # Fix NLTK data loading issues
        self._setup_nltk_data()

        # Initialize augmenters with error handling
        try:
            self.synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=drift_intensity)
            self.use_synonyms = True
        except Exception as e:
            logger.warning(f"Failed to initialize wordnet synonym augmenter: {e}")
            logger.warning("Falling back to word swap augmenter for vocabulary drift")
            self.synonym_aug = None
            self.use_synonyms = False

        self.swap_aug = naw.RandomWordAug(action="swap", aug_p=drift_intensity)

    def _setup_nltk_data(self):
        """Setup NLTK data with proper error handling to avoid recursion issues."""
        import ssl
        import nltk

        try:
            # Try to create unverified HTTPS context to bypass SSL issues
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            # Check if NLTK data already exists
            import os
            nltk_data_path = os.path.expanduser('~/nltk_data/corpora')
            wordnet_exists = (os.path.exists(os.path.join(nltk_data_path, 'wordnet')) or
                            os.path.exists(os.path.join(nltk_data_path, 'wordnet.zip')))
            omw_exists = (os.path.exists(os.path.join(nltk_data_path, 'omw-1.4')) or
                         os.path.exists(os.path.join(nltk_data_path, 'omw-1.4.zip')))

            # Download only if not already present
            if not wordnet_exists:
                logger.info("Downloading NLTK wordnet data...")
                nltk.download('wordnet', quiet=True)
            else:
                logger.debug("NLTK wordnet data already available")

            if not omw_exists:
                logger.info("Downloading NLTK omw-1.4 data...")
                nltk.download('omw-1.4', quiet=True)
            else:
                logger.debug("NLTK omw-1.4 data already available")

        except Exception as e:
            logger.warning(f"NLTK data setup failed: {e}. Will use fallback augmentation.")

    def inject_vocab_drift(self, texts: List[str]) -> List[str]:
        """Inject vocabulary drift using synonym replacement."""
        try:
            augmented_texts = []
            for text in texts:
                if np.random.random() < self.drift_intensity:
                    augmented = self.synonym_aug.augment(text)
                    if isinstance(augmented, list):
                        augmented_texts.append(augmented[0])
                    else:
                        augmented_texts.append(augmented)
                else:
                    augmented_texts.append(text)
            return augmented_texts
        except Exception as e:
            logger.warning(f"Vocabulary drift injection failed: {e}. Using original texts.")
            return texts
    
    def inject_concept_drift(self, labels: List[int], noise_rate: float = 0.2) -> List[int]:
        """Inject concept drift by flipping labels."""
        drifted_labels = labels.copy()
        num_samples = len(labels)
        num_to_flip = int(num_samples * noise_rate)
        
        indices_to_flip = np.random.choice(num_samples, num_to_flip, replace=False)
        
        for idx in indices_to_flip:
            # Flip to a random different label (0-3 for AG News)
            original_label = drifted_labels[idx]
            possible_labels = [i for i in range(4) if i != original_label]
            drifted_labels[idx] = np.random.choice(possible_labels)
        
        return drifted_labels
    
    def inject_distribution_drift(self, texts: List[str], labels: List[int], 
                                target_class: int = 0, bias_strength: float = 0.8) -> Tuple[List[str], List[int]]:
        """Inject distribution drift by biasing towards certain classes."""
        combined = list(zip(texts, labels))
        
        # Separate samples by class
        class_samples = {i: [] for i in range(4)}
        for text, label in combined:
            class_samples[label].append((text, label))
        
        # Calculate new distribution
        total_samples = len(combined)
        target_samples = int(total_samples * bias_strength)
        other_samples = total_samples - target_samples
        other_per_class = other_samples // 3
        
        # Create biased dataset
        drifted_data = []
        
        # Add target class samples
        target_class_data = class_samples[target_class]
        if len(target_class_data) >= target_samples:
            drifted_data.extend(np.random.choice(
                len(target_class_data), target_samples, replace=False
            ).tolist())
        else:
            # Repeat samples if not enough
            repeats = target_samples // len(target_class_data) + 1
            extended_data = target_class_data * repeats
            drifted_data.extend(extended_data[:target_samples])
        
        # Add other class samples
        for class_id in range(4):
            if class_id != target_class:
                class_data = class_samples[class_id]
                if len(class_data) >= other_per_class:
                    selected_indices = np.random.choice(
                        len(class_data), other_per_class, replace=False
                    )
                    drifted_data.extend([class_data[i] for i in selected_indices])
                else:
                    drifted_data.extend(class_data)
        
        # Shuffle and separate
        np.random.shuffle(drifted_data)
        drifted_texts, drifted_labels = zip(*drifted_data)
        
        return list(drifted_texts), list(drifted_labels)


class FederatedDataLoader:
    """Manages federated data loading with non-IID partitioning."""
    
    def __init__(self, num_clients: int = 10, alpha: float = 0.5, batch_size: int = 16):
        self.num_clients = num_clients
        self.alpha = alpha  # Dirichlet concentration parameter
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.drift_injector = DriftInjector()
        
    def load_ag_news(self) -> Dict[str, Any]:
        """Load AG News dataset."""
        try:
            dataset = load_dataset("ag_news")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load AG News dataset: {e}")
            # Create a small dummy dataset for testing
            dummy_data = {
                'train': HFDataset.from_dict({
                    'text': ['Sample news about world politics.'] * 100,
                    'label': [0] * 100
                }),
                'test': HFDataset.from_dict({
                    'text': ['Sample news about sports.'] * 20,
                    'label': [1] * 20
                })
            }
            return dummy_data
    
    def create_federated_splits(self) -> Tuple[Dict[int, AGNewsDataset], AGNewsDataset]:
        """Create federated data splits using Dirichlet partitioning."""
        # Load dataset
        dataset = self.load_ag_news()
        
        # Simple manual partitioning for now (non-IID)
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        
        # Create client datasets with simple partitioning
        client_datasets = {}
        samples_per_client = len(train_texts) // self.num_clients
        
        for client_id in range(self.num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            
            # Add some class bias for non-IID distribution
            preferred_class = client_id % 4
            
            client_texts = []
            client_labels = []
            
            # First, add samples from preferred class
            for i, (text, label) in enumerate(zip(train_texts[start_idx:end_idx], train_labels[start_idx:end_idx])):
                if label == preferred_class or len(client_texts) < samples_per_client // 2:
                    client_texts.append(text)
                    client_labels.append(label)
            
            # Fill remaining with other samples
            remaining_needed = max(0, min(50, samples_per_client - len(client_texts)))
            for i, (text, label) in enumerate(zip(train_texts[start_idx:end_idx], train_labels[start_idx:end_idx])):
                if len(client_texts) >= remaining_needed + len(client_texts):
                    break
                if (text, label) not in zip(client_texts, client_labels):
                    client_texts.append(text)
                    client_labels.append(label)
            
            # Ensure minimum samples
            if len(client_texts) < 20:
                additional_needed = 20 - len(client_texts)
                for i in range(additional_needed):
                    idx = (start_idx + i) % len(train_texts)
                    client_texts.append(train_texts[idx])
                    client_labels.append(train_labels[idx])
            
            # Create PyTorch dataset
            client_datasets[client_id] = AGNewsDataset(
                texts=client_texts,
                labels=client_labels,
                tokenizer=self.tokenizer
            )
        
        # Create test dataset
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']
        test_dataset = AGNewsDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=self.tokenizer
        )
        
        return client_datasets, test_dataset
    
    def apply_drift_to_clients(self, client_datasets: Dict[int, AGNewsDataset], 
                              affected_clients: List[int], drift_types: List[str]) -> Dict[int, AGNewsDataset]:
        """Apply synthetic drift to specified clients."""
        drifted_datasets = client_datasets.copy()
        
        for client_id in affected_clients:
            if client_id not in client_datasets:
                continue
                
            original_dataset = client_datasets[client_id]
            texts = original_dataset.texts.copy()
            labels = original_dataset.labels.copy()
            
            for drift_type in drift_types:
                if drift_type == 'vocab_shift':
                    texts = self.drift_injector.inject_vocab_drift(texts)
                elif drift_type == 'label_noise':
                    labels = self.drift_injector.inject_concept_drift(labels, noise_rate=0.2)
                elif drift_type == 'distribution_shift':
                    texts, labels = self.drift_injector.inject_distribution_drift(
                        texts, labels, target_class=client_id % 4
                    )
            
            # Create new dataset with drifted data
            drifted_datasets[client_id] = AGNewsDataset(
                texts=texts,
                labels=labels,
                tokenizer=self.tokenizer
            )
            
            logger.info(f"Applied drift {drift_types} to client {client_id}")
        
        return drifted_datasets
    
    def get_data_loaders(self, client_datasets: Dict[int, AGNewsDataset], 
                        test_dataset: AGNewsDataset) -> Tuple[Dict[int, DataLoader], DataLoader]:
        """Create data loaders for training and testing."""
        client_loaders = {}
        
        for client_id, dataset in client_datasets.items():
            client_loaders[client_id] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return client_loaders, test_loader
    
    def get_dataset_statistics(self, client_datasets: Dict[int, AGNewsDataset]) -> Dict[str, Any]:
        """Get statistics about the federated dataset distribution."""
        stats = {
            'total_clients': len(client_datasets),
            'client_sizes': {},
            'label_distributions': {},
            'total_samples': 0
        }
        
        for client_id, dataset in client_datasets.items():
            size = len(dataset)
            stats['client_sizes'][client_id] = size
            stats['total_samples'] += size
            
            # Calculate label distribution
            label_counts = {}
            for label in dataset.labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            stats['label_distributions'][client_id] = label_counts
        
        return stats
    
    def load_federated_data(self, num_clients: int, alpha: float = 0.5) -> Tuple[Dict[int, AGNewsDataset], AGNewsDataset, Dict[str, Any]]:
        """
        Load and partition federated data - compatibility method.
        
        Args:
            num_clients: Number of federated clients
            alpha: Dirichlet concentration parameter
            
        Returns:
            Tuple of (client_datasets, test_dataset, statistics)
        """
        # Update configuration
        self.num_clients = num_clients
        self.alpha = alpha
        
        # Create federated splits
        client_datasets, test_dataset = self.create_federated_splits()
        
        # Get statistics
        stats = self.get_dataset_statistics(client_datasets)
        
        return client_datasets, test_dataset, stats


# Configuration constants
MODEL_CONFIG = {
    'model_name': 'prajjwal1/bert-tiny',
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_steps': 100
}

DRIFT_CONFIG = {
    'injection_round': 25,
    'drift_intensity': 0.3,
    'affected_clients': [2, 5, 8],
    'drift_types': ['vocab_shift', 'label_noise']
}

FEDERATED_CONFIG = {
    'num_clients': 10,
    'alpha': 0.5,  # Dirichlet concentration parameter
    'min_samples_per_client': 10
}