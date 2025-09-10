"""
Configuration management for federated drift detection system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration for the federated drift detection system."""
    
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else Path("config.yaml")
        self.config = self._load_default_config()
        
        if self.config_path.exists():
            self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            # Model configuration
            'model': {
                'model_name': 'prajjwal1/bert-tiny',
                'max_length': 128,
                'batch_size': 16,
                'learning_rate': 2e-5,
                'num_epochs': 3,
                'warmup_steps': 100,
                'dropout': 0.1
            },
            
            # Federated learning configuration
            'federated': {
                'num_clients': 10,
                'alpha': 0.5,  # Dirichlet concentration
                'min_samples_per_client': 10
            },
            
            # Drift configuration
            'drift': {
                'injection_round': 25,
                'drift_intensity': 0.3,
                'affected_clients': [2, 5, 8],
                'drift_types': ['vocab_shift', 'label_noise']
            },
            
            # Drift detection configuration
            'drift_detection': {
                'adwin_delta': 0.002,
                'mmd_p_val': 0.05,
                'mmd_permutations': 100,
                'evidently_threshold': 0.25,
                'trimmed_beta': 0.2,
                'feature_names': [f'embedding_dim_{i}' for i in range(128)]
            },
            
            # Server strategy configuration
            'strategy': {
                'fraction_fit': 1.0,
                'fraction_evaluate': 1.0,
                'min_fit_clients': 2,
                'min_evaluate_clients': 2,
                'min_available_clients': 2,
                'mitigation_threshold': 0.3
            },
            
            # Simulation configuration
            'simulation': {
                'num_rounds': 50,
                'num_cpus': 4,
                'num_gpus': 0.0,  # Use CPU for simulation
                'ray_init_args': {
                    'include_dashboard': False,
                    'log_to_driver': False
                }
            },
            
            # Logging configuration
            'logging': {
                'level': 'INFO',
                'file': 'logs/simulation.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            
            # Output configuration
            'output': {
                'results_dir': 'results',
                'logs_dir': 'logs', 
                'save_checkpoints': True,
                'checkpoint_rounds': [10, 25, 50],
                'create_visualizations': True
            }
        }
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
            
            # Merge with default config
            self.config = self._deep_merge(self.config, file_config)
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
    
    def save_config(self, path: str = None) -> None:
        """Save current configuration to file."""
        save_path = Path(path) if path else self.config_path
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.suffix.lower() == '.yaml':
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved configuration to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {save_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dots)."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports nested keys with dots)."""
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate required sections
        required_sections = ['model', 'federated', 'drift', 'simulation']
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required section: {section}")
        
        # Validate federated config
        if 'federated' in self.config:
            fed_config = self.config['federated']
            if fed_config.get('num_clients', 0) < 2:
                issues.append("num_clients must be at least 2")
            if not (0 < fed_config.get('alpha', 0) <= 1):
                issues.append("alpha must be between 0 and 1")
        
        # Validate drift config
        if 'drift' in self.config:
            drift_config = self.config['drift']
            injection_round = drift_config.get('injection_round', 0)
            total_rounds = self.config.get('simulation', {}).get('num_rounds', 0)
            
            if injection_round >= total_rounds:
                issues.append("injection_round must be less than num_rounds")
            
            if not (0 < drift_config.get('drift_intensity', 0) <= 1):
                issues.append("drift_intensity must be between 0 and 1")
        
        # Validate drift detection config
        if 'drift_detection' in self.config:
            dd_config = self.config['drift_detection']
            if not (0 < dd_config.get('adwin_delta', 0) < 1):
                issues.append("adwin_delta must be between 0 and 1")
            if not (0 < dd_config.get('mmd_p_val', 0) < 1):
                issues.append("mmd_p_val must be between 0 and 1")
            if not (0 < dd_config.get('trimmed_beta', 0) < 0.5):
                issues.append("trimmed_beta must be between 0 and 0.5")
        
        return issues


# Global config instance
config = ConfigManager()

# Export configuration getters
def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return config.get('model', {})

def get_federated_config() -> Dict[str, Any]:
    """Get federated learning configuration."""
    return config.get('federated', {})

def get_drift_config() -> Dict[str, Any]:
    """Get drift configuration."""
    return config.get('drift', {})

def get_drift_detection_config() -> Dict[str, Any]:
    """Get drift detection configuration."""
    return config.get('drift_detection', {})

def get_simulation_config() -> Dict[str, Any]:
    """Get simulation configuration."""
    return config.get('simulation', {})

def get_strategy_config() -> Dict[str, Any]:
    """Get strategy configuration."""
    return config.get('strategy', {})