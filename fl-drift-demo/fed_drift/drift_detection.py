"""
Drift detection algorithms for federated learning.

This module implements:
- Local client-side drift detection (ADWIN, Evidently)
- Global server-side drift detection (MMD)
- Drift metrics and analysis utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import drift detection libraries
try:
    from river.drift import ADWIN
except ImportError:
    ADWIN = None
    
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently import ColumnMapping
except ImportError:
    Report = None
    DataDriftPreset = None
    ColumnMapping = None

try:
    from alibi_detect.cd import MMDDrift
except ImportError:
    MMDDrift = None

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Container for drift detection results."""
    is_drift: bool
    drift_score: float
    p_value: Optional[float] = None
    drift_type: str = "unknown"
    additional_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


class DriftDetector(ABC):
    """Abstract base class for drift detectors."""
    
    @abstractmethod
    def update(self, data: Any) -> None:
        """Update detector with new data."""
        pass
    
    @abstractmethod
    def detect(self) -> DriftResult:
        """Detect drift and return results."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass


class ADWINDriftDetector(DriftDetector):
    """ADWIN-based concept drift detector for performance monitoring."""
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.performance_history = []
        
        if ADWIN is None:
            raise ImportError("River library not available. Install with: pip install river")
        
        self.adwin = ADWIN(delta=delta)
        self.drift_detected = False
        
    def update(self, performance_metric: float) -> None:
        """Update with new performance metric (loss or accuracy)."""
        self.performance_history.append(performance_metric)
        self.adwin.update(performance_metric)
        self.drift_detected = self.adwin.drift_detected
        
    def detect(self) -> DriftResult:
        """Detect concept drift based on performance changes."""
        drift_score = len(self.performance_history) / 1000.0  # Normalize by history length
        
        return DriftResult(
            is_drift=self.drift_detected,
            drift_score=drift_score,
            drift_type="concept_drift",
            additional_info={
                "performance_history_length": len(self.performance_history),
                "last_performance": self.performance_history[-1] if self.performance_history else 0.0,
                "delta": self.delta
            }
        )
    
    def reset(self) -> None:
        """Reset ADWIN detector."""
        self.adwin = ADWIN(delta=self.delta)
        self.performance_history = []
        self.drift_detected = False


class EvidentiallyDriftDetector(DriftDetector):
    """Evidently-based data drift detector for batch analysis."""
    
    def __init__(self, feature_names: List[str] = None):
        if Report is None or DataDriftPreset is None:
            raise ImportError("Evidently library not available. Install with: pip install evidently")
        
        self.feature_names = feature_names or ['feature']
        self.reference_data = None
        self.current_data = []
        self.last_drift_result = None
        
    def set_reference_data(self, reference_data: np.ndarray) -> None:
        """Set reference data for drift detection."""
        self.reference_data = pd.DataFrame(
            reference_data, 
            columns=self.feature_names[:reference_data.shape[1]]
        )
        
    def update(self, data: np.ndarray) -> None:
        """Update with new data batch."""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        self.current_data.extend(data.tolist())
        
    def detect(self) -> DriftResult:
        """Detect data drift using Evidently."""
        if self.reference_data is None or len(self.current_data) == 0:
            return DriftResult(
                is_drift=False,
                drift_score=0.0,
                drift_type="data_drift",
                additional_info={"error": "Insufficient data for drift detection"}
            )
        
        try:
            # Convert current data to DataFrame
            current_df = pd.DataFrame(
                self.current_data,
                columns=self.feature_names[:len(self.current_data[0])]
            )
            
            # Create drift report
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=self.reference_data, current_data=current_df)
            
            # Extract results
            result_dict = report.as_dict()
            dataset_drift = result_dict['metrics'][0]['result']['dataset_drift']
            drift_share = result_dict['metrics'][0]['result']['drift_share']
            
            # Calculate drift score (0-1)
            drift_score = drift_share if drift_share is not None else 0.0
            
            self.last_drift_result = DriftResult(
                is_drift=dataset_drift,
                drift_score=drift_score,
                drift_type="data_drift",
                additional_info={
                    "drift_share": drift_share,
                    "reference_size": len(self.reference_data),
                    "current_size": len(current_df)
                }
            )
            
            return self.last_drift_result
            
        except Exception as e:
            logger.warning(f"Evidently drift detection failed: {e}")
            return DriftResult(
                is_drift=False,
                drift_score=0.0,
                drift_type="data_drift",
                additional_info={"error": str(e)}
            )
    
    def reset(self) -> None:
        """Reset current data buffer."""
        self.current_data = []
        self.last_drift_result = None


class MMDDriftDetector(DriftDetector):
    """MMD-based drift detector for high-dimensional embeddings."""
    
    def __init__(self, p_val: float = 0.05, n_permutations: int = 100):
        if MMDDrift is None:
            raise ImportError("Alibi-detect library not available. Install with: pip install alibi-detect")
        
        self.p_val = p_val
        self.n_permutations = n_permutations
        self.reference_embeddings = None
        self.detector = None
        self.current_embeddings = []
        
    def set_reference_embeddings(self, reference_embeddings: np.ndarray) -> None:
        """Set reference embeddings for drift detection."""
        self.reference_embeddings = reference_embeddings
        
        try:
            self.detector = MMDDrift(
                X_ref=reference_embeddings,
                p_val=self.p_val,
                n_permutations=self.n_permutations
            )
        except Exception as e:
            logger.warning(f"Failed to initialize MMD detector: {e}")
            self.detector = None
    
    def update(self, embeddings: np.ndarray) -> None:
        """Update with new embedding data."""
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        self.current_embeddings.extend(embeddings.tolist())
    
    def detect(self) -> DriftResult:
        """Detect drift in embedding space using MMD test."""
        if self.detector is None or len(self.current_embeddings) == 0:
            return DriftResult(
                is_drift=False,
                drift_score=0.0,
                drift_type="embedding_drift",
                additional_info={"error": "Detector not initialized or no data"}
            )
        
        try:
            current_array = np.array(self.current_embeddings)
            
            # Perform MMD test
            result = self.detector.predict(current_array)
            
            is_drift = result['data']['is_drift']
            p_value = result['data']['p_val']
            distance = result['data']['distance']
            
            # Normalize distance as drift score (0-1)
            drift_score = min(1.0, distance / 10.0) if distance is not None else 0.0
            
            return DriftResult(
                is_drift=is_drift,
                drift_score=drift_score,
                p_value=p_value,
                drift_type="embedding_drift",
                additional_info={
                    "mmd_distance": distance,
                    "threshold": result['data']['threshold'],
                    "reference_size": len(self.reference_embeddings),
                    "current_size": len(current_array)
                }
            )
            
        except Exception as e:
            logger.warning(f"MMD drift detection failed: {e}")
            return DriftResult(
                is_drift=False,
                drift_score=0.0,
                drift_type="embedding_drift",
                additional_info={"error": str(e)}
            )
    
    def reset(self) -> None:
        """Reset current embeddings buffer."""
        self.current_embeddings = []


class MultiLevelDriftDetector:
    """Combines multiple drift detection methods."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize detectors
        self.adwin_detector = ADWINDriftDetector(
            delta=self.config.get('adwin_delta', 0.002)
        )
        
        self.evidently_detector = EvidentiallyDriftDetector(
            feature_names=self.config.get('feature_names', ['embedding_dim_' + str(i) for i in range(128)])
        )
        
        self.mmd_detector = MMDDriftDetector(
            p_val=self.config.get('mmd_p_val', 0.05),
            n_permutations=self.config.get('mmd_permutations', 100)
        )
        
        self.drift_history = []
        
    def update_performance(self, performance_metric: float) -> None:
        """Update ADWIN with performance metric."""
        self.adwin_detector.update(performance_metric)
    
    def update_data(self, data: np.ndarray) -> None:
        """Update Evidently with data batch."""
        self.evidently_detector.update(data)
    
    def update_embeddings(self, embeddings: np.ndarray) -> None:
        """Update MMD with embeddings."""
        self.mmd_detector.update(embeddings)
    
    def set_reference_data(self, reference_data: np.ndarray, reference_embeddings: np.ndarray) -> None:
        """Set reference data for drift detectors."""
        self.evidently_detector.set_reference_data(reference_data)
        self.mmd_detector.set_reference_embeddings(reference_embeddings)
    
    def detect_all(self) -> Dict[str, DriftResult]:
        """Run all drift detection methods."""
        results = {
            'concept_drift': self.adwin_detector.detect(),
            'data_drift': self.evidently_detector.detect(),
            'embedding_drift': self.mmd_detector.detect()
        }
        
        # Store in history
        self.drift_history.append(results)
        
        return results
    
    def get_aggregated_drift_signal(self, results: Dict[str, DriftResult] = None) -> DriftResult:
        """Aggregate drift signals from multiple detectors."""
        if results is None:
            results = self.detect_all()
        
        # Count drift detections
        drift_count = sum(1 for result in results.values() if result.is_drift)
        total_detectors = len(results)
        
        # Calculate weighted drift score
        weighted_score = 0.0
        weights = {'concept_drift': 0.4, 'data_drift': 0.3, 'embedding_drift': 0.3}
        
        for detector_type, result in results.items():
            weight = weights.get(detector_type, 1.0 / total_detectors)
            weighted_score += weight * result.drift_score
        
        # Determine overall drift
        is_drift = drift_count >= 2 or results['embedding_drift'].is_drift
        
        return DriftResult(
            is_drift=is_drift,
            drift_score=weighted_score,
            drift_type="aggregated",
            additional_info={
                "individual_results": results,
                "drift_count": drift_count,
                "total_detectors": total_detectors
            }
        )
    
    def reset_all(self) -> None:
        """Reset all detectors."""
        self.adwin_detector.reset()
        self.evidently_detector.reset()
        self.mmd_detector.reset()
        self.drift_history = []


# Utility functions for drift analysis
def calculate_drift_metrics(drift_history: List[Dict[str, DriftResult]], 
                          injection_round: int) -> Dict[str, float]:
    """Calculate drift detection metrics."""
    if not drift_history:
        return {}
    
    metrics = {}
    
    for detector_type in drift_history[0].keys():
        # Extract drift signals for this detector
        drift_signals = [round_results[detector_type].is_drift for round_results in drift_history]
        
        # Detection delay (rounds until first detection after injection)
        detection_round = None
        for round_idx, is_drift in enumerate(drift_signals):
            if is_drift and round_idx >= injection_round:
                detection_round = round_idx
                break
        
        detection_delay = detection_round - injection_round if detection_round else len(drift_signals)
        
        # Detection rate (percentage of rounds with drift detected after injection)
        post_injection_signals = drift_signals[injection_round:]
        detection_rate = sum(post_injection_signals) / len(post_injection_signals) if post_injection_signals else 0.0
        
        metrics[f"{detector_type}_detection_delay"] = detection_delay
        metrics[f"{detector_type}_detection_rate"] = detection_rate
    
    return metrics


def visualize_drift_timeline(drift_history: List[Dict[str, DriftResult]], 
                           injection_round: int, output_path: str = None) -> None:
    """Create visualization of drift detection timeline."""
    try:
        import matplotlib.pyplot as plt
        
        rounds = list(range(len(drift_history)))
        detector_types = list(drift_history[0].keys()) if drift_history else []
        
        fig, axes = plt.subplots(len(detector_types), 1, figsize=(12, 8), sharex=True)
        if len(detector_types) == 1:
            axes = [axes]
        
        for idx, detector_type in enumerate(detector_types):
            drift_signals = [round_results[detector_type].is_drift for round_results in drift_history]
            drift_scores = [round_results[detector_type].drift_score for round_results in drift_history]
            
            # Plot drift scores
            axes[idx].plot(rounds, drift_scores, label=f'{detector_type} score', alpha=0.7)
            
            # Mark drift detections
            drift_rounds = [r for r, d in zip(rounds, drift_signals) if d]
            drift_round_scores = [drift_scores[r] for r in drift_rounds]
            axes[idx].scatter(drift_rounds, drift_round_scores, color='red', s=50, 
                            label=f'{detector_type} detections', zorder=5)
            
            # Mark injection point
            axes[idx].axvline(x=injection_round, color='orange', linestyle='--', 
                            label='Drift injection', alpha=0.8)
            
            axes[idx].set_ylabel('Drift Score')
            axes[idx].set_title(f'{detector_type.replace("_", " ").title()} Detection')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.xlabel('Training Round')
        plt.title('Drift Detection Timeline')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    except ImportError:
        logger.warning("Matplotlib not available. Skipping visualization.")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")


# Configuration for drift detection
DRIFT_DETECTION_CONFIG = {
    'adwin_delta': 0.002,
    'mmd_p_val': 0.05,
    'mmd_permutations': 100,
    'evidently_threshold': 0.25,
    'aggregation_weights': {
        'concept_drift': 0.4,
        'data_drift': 0.3,
        'embedding_drift': 0.3
    }
}