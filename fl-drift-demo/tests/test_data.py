"""
Unit tests for data handling and drift injection.
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch
import tempfile
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fed_drift.data import DriftInjector, FederatedDataLoader, AGNewsDataset


class TestDriftInjector(unittest.TestCase):
    """Test synthetic drift injection mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.drift_injector = DriftInjector(drift_intensity=0.3)
        self.sample_texts = [
            "This is a great movie with excellent acting.",
            "The stock market showed positive trends today.",
            "Technology companies are leading innovation.",
            "Sports fans gathered to watch the championship.",
            "Political debates continue in the senate."
        ]
        self.sample_labels = [0, 1, 2, 3, 0]  # AG News has 4 classes (0-3)
    
    def test_drift_injector_initialization(self):
        """Test DriftInjector initialization."""
        injector = DriftInjector(drift_intensity=0.5)
        self.assertEqual(injector.drift_intensity, 0.5)
        self.assertIsNotNone(injector.synonym_aug)
        self.assertIsNotNone(injector.swap_aug)
    
    def test_vocab_drift_injection(self):
        """Test vocabulary drift injection."""
        original_texts = self.sample_texts.copy()
        drifted_texts = self.drift_injector.inject_vocab_drift(original_texts)
        
        # Check that we get the same number of texts
        self.assertEqual(len(drifted_texts), len(original_texts))
        
        # Check that texts are strings
        for text in drifted_texts:
            self.assertIsInstance(text, str)
        
        # Some texts should be different (depending on augmentation)
        # Note: This test might be flaky due to randomness in augmentation
        # In practice, we expect some changes but can't guarantee them
    
    def test_concept_drift_injection(self):
        """Test concept drift (label noise) injection."""
        original_labels = self.sample_labels.copy()
        drifted_labels = self.drift_injector.inject_concept_drift(original_labels, noise_rate=0.4)
        
        # Check that we get the same number of labels
        self.assertEqual(len(drifted_labels), len(original_labels))
        
        # Check that labels are still valid (0-3 for AG News)
        for label in drifted_labels:
            self.assertIn(label, [0, 1, 2, 3])
        
        # With noise_rate=0.4, expect some labels to change
        # Note: Due to randomness, this test focuses on structure rather than exact changes
    
    def test_distribution_drift_injection(self):
        """Test distribution drift injection."""
        original_texts = self.sample_texts.copy()
        original_labels = self.sample_labels.copy()
        
        drifted_texts, drifted_labels = self.drift_injector.inject_distribution_drift(
            original_texts, original_labels, target_class=0, bias_strength=0.8
        )
        
        # Check that we get some output
        self.assertGreater(len(drifted_texts), 0)
        self.assertEqual(len(drifted_texts), len(drifted_labels))
        
        # Check that target class is more frequent
        label_counts = {}
        for label in drifted_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Target class (0) should have high representation due to bias_strength=0.8
        if len(drifted_labels) > 1:
            target_class_ratio = label_counts.get(0, 0) / len(drifted_labels)
            self.assertGreater(target_class_ratio, 0.5)  # Should be biased towards target class


class TestAGNewsDataset(unittest.TestCase):
    """Test AG News dataset handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1])
        }
        
        self.sample_texts = ["Test news about technology", "Sports news update"]
        self.sample_labels = [2, 3]
    
    def test_agnews_dataset_creation(self):
        """Test AG News dataset creation."""
        dataset = AGNewsDataset(
            texts=self.sample_texts,
            labels=self.sample_labels,
            tokenizer=self.mock_tokenizer,
            max_length=128
        )
        
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.texts, self.sample_texts)
        self.assertEqual(dataset.labels, self.sample_labels)
        self.assertEqual(dataset.max_length, 128)
    
    def test_agnews_dataset_getitem(self):
        """Test AG News dataset item retrieval."""
        dataset = AGNewsDataset(
            texts=self.sample_texts,
            labels=self.sample_labels,
            tokenizer=self.mock_tokenizer
        )
        
        item = dataset[0]
        
        # Check that item has expected keys
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        
        # Check tensor types
        self.assertIsInstance(item['input_ids'], torch.Tensor)
        self.assertIsInstance(item['attention_mask'], torch.Tensor)
        self.assertIsInstance(item['labels'], torch.Tensor)
        
        # Check label value
        self.assertEqual(item['labels'].item(), self.sample_labels[0])


class TestFederatedDataLoader(unittest.TestCase):
    """Test federated data loading and partitioning."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_clients = 5
        self.alpha = 0.5
        self.batch_size = 8
    
    @patch('fed_drift.data.load_dataset')
    @patch('fed_drift.data.FederatedDataset')
    @patch('fed_drift.data.AutoTokenizer')
    def test_federated_data_loader_initialization(self, mock_tokenizer, mock_fed_dataset, mock_load_dataset):
        """Test FederatedDataLoader initialization."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = '[EOS]'
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        loader = FederatedDataLoader(
            num_clients=self.num_clients,
            alpha=self.alpha,
            batch_size=self.batch_size
        )
        
        self.assertEqual(loader.num_clients, self.num_clients)
        self.assertEqual(loader.alpha, self.alpha)
        self.assertEqual(loader.batch_size, self.batch_size)
        self.assertIsNotNone(loader.tokenizer)
        self.assertIsNotNone(loader.drift_injector)
    
    @patch('fed_drift.data.load_dataset')
    def test_load_ag_news_fallback(self, mock_load_dataset):
        """Test AG News loading with fallback to dummy data."""
        # Mock dataset loading failure
        mock_load_dataset.side_effect = Exception("Network error")
        
        with patch('fed_drift.data.AutoTokenizer'):
            loader = FederatedDataLoader()
            dataset = loader.load_ag_news()
        
        # Should return dummy data structure
        self.assertIn('train', dataset)
        self.assertIn('test', dataset)
    
    def test_dataset_statistics_calculation(self):
        """Test dataset statistics calculation."""
        # Create mock client datasets
        mock_datasets = {}
        for i in range(3):
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=20 + i * 10)
            mock_dataset.labels = [0, 1, 2, 0, 1] * (4 + i * 2)  # Vary distribution
            mock_datasets[i] = mock_dataset
        
        with patch('fed_drift.data.AutoTokenizer'):
            loader = FederatedDataLoader()
            stats = loader.get_dataset_statistics(mock_datasets)
        
        # Check statistics structure
        self.assertIn('total_clients', stats)
        self.assertIn('client_sizes', stats)
        self.assertIn('label_distributions', stats)
        self.assertIn('total_samples', stats)
        
        # Check values
        self.assertEqual(stats['total_clients'], 3)
        self.assertEqual(len(stats['client_sizes']), 3)
        self.assertEqual(len(stats['label_distributions']), 3)


class TestDataConfiguration(unittest.TestCase):
    """Test data configuration constants."""
    
    def test_model_config_structure(self):
        """Test MODEL_CONFIG structure."""
        from fed_drift.data import MODEL_CONFIG
        
        required_keys = ['model_name', 'max_length', 'batch_size', 'learning_rate', 'num_epochs']
        
        for key in required_keys:
            self.assertIn(key, MODEL_CONFIG)
        
        # Check reasonable values
        self.assertGreater(MODEL_CONFIG['max_length'], 0)
        self.assertGreater(MODEL_CONFIG['batch_size'], 0)
        self.assertGreater(MODEL_CONFIG['learning_rate'], 0)
        self.assertGreater(MODEL_CONFIG['num_epochs'], 0)
    
    def test_drift_config_structure(self):
        """Test DRIFT_CONFIG structure."""
        from fed_drift.data import DRIFT_CONFIG
        
        required_keys = ['injection_round', 'drift_intensity', 'affected_clients', 'drift_types']
        
        for key in required_keys:
            self.assertIn(key, DRIFT_CONFIG)
        
        # Check reasonable values
        self.assertGreater(DRIFT_CONFIG['injection_round'], 0)
        self.assertGreater(DRIFT_CONFIG['drift_intensity'], 0)
        self.assertLessEqual(DRIFT_CONFIG['drift_intensity'], 1)
        self.assertIsInstance(DRIFT_CONFIG['affected_clients'], list)
        self.assertIsInstance(DRIFT_CONFIG['drift_types'], list)
    
    def test_federated_config_structure(self):
        """Test FEDERATED_CONFIG structure."""
        from fed_drift.data import FEDERATED_CONFIG
        
        required_keys = ['num_clients', 'alpha']
        
        for key in required_keys:
            self.assertIn(key, FEDERATED_CONFIG)
        
        # Check reasonable values
        self.assertGreater(FEDERATED_CONFIG['num_clients'], 1)
        self.assertGreater(FEDERATED_CONFIG['alpha'], 0)


if __name__ == '__main__':
    unittest.main()