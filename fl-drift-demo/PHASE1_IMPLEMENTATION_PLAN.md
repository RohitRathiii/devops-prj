# 📋 PHASE 1 IMPLEMENTATION PLAN: EXTREME DEPTH

## OVERVIEW

**Timeline:** 2-3 days (16-24 hours)
**Impact:** Coverage increase from 35% → 55%
**Files Modified:** 4 core files
**New Files Created:** 1 utility file + 3 test files
**Lines of Code:** ~500 new lines

---

## TABLE OF CONTENTS

1. [Pre-Implementation Analysis](#part-1-pre-implementation-analysis)
2. [Implementation Step-by-Step](#part-2-implementation-step-by-step)
   - [Task 1: Create Utility Module](#task-1-create-utility-module-for-metrics)
   - [Task 2: Modify server.py](#task-2-modify-serverpy---add-fairness-metrics)
   - [Task 3: Modify drift_detection.py](#task-3-modify-drift_detectionpy---add-fprfnrf1-metrics)
   - [Task 4: Modify simulation.py](#task-4-modify-simulationpy---add-recovery-metrics)
   - [Task 5: Update Configuration](#task-5-update-configuration)
3. [Testing Strategy](#part-3-testing-strategy)
4. [Validation & Verification](#part-4-validation--verification)
5. [Documentation](#part-5-documentation)
6. [Deployment Checklist](#part-6-deployment-checklist)
7. [Estimated Timeline](#part-7-estimated-timeline)
8. [Troubleshooting Guide](#part-8-troubleshooting-guide)
9. [Final Checklist](#final-checklist)

---

## PART 1: PRE-IMPLEMENTATION ANALYSIS 🔍

### 1.1 Current State Analysis

**Files to Modify:**
1. `server.py` - Lines 255-330 (aggregate_evaluate method)
2. `drift_detection.py` - Lines 411-439 (calculate_drift_metrics function)
3. `simulation.py` - Lines 383-489 (metrics calculation methods)
4. `config.py` - Add new configuration options

**Current Data Structures:**
```python
# server.py:310-317 - performance_history structure
{
    'round': int,
    'global_accuracy': float,
    'global_loss': float,
    'fairness_gap': float,
    'client_accuracies': List[float],
    'client_losses': List[float]
}

# drift_detection.py:411 - drift_history structure
List[Dict[str, DriftResult]]
# where each dict maps detector_type → DriftResult

# simulation.py - Results structure
{
    'performance_metrics': Dict[str, float],
    'drift_metrics': Dict[str, float],
    ...
}
```

### 1.2 Dependencies Analysis

**Required Imports (verify/add):**
```python
# Already present:
import numpy as np
from typing import Dict, List, Any, Optional

# Need to verify:
from collections import Counter  # For Gini calculation
```

**No new external dependencies** - all implementations use numpy and standard library.

### 1.3 Backup Strategy

**Before starting, create backups:**
```bash
cd /Users/rohitrathi/Downloads/devops\ prj/fl-drift-demo
mkdir -p .backups/phase1_$(date +%Y%m%d_%H%M%S)
cp fed_drift/server.py .backups/phase1_*/
cp fed_drift/drift_detection.py .backups/phase1_*/
cp fed_drift/simulation.py .backups/phase1_*/
cp fed_drift/config.py .backups/phase1_*/
```

---

## PART 2: IMPLEMENTATION STEP-BY-STEP 📝

### TASK 1: CREATE UTILITY MODULE FOR METRICS ⭐

**Priority: CRITICAL - Do this FIRST**
**Time: 30 minutes**
**Location: Create new file**

#### Step 1.1: Create metrics_utils.py

**File:** `/Users/rohitrathi/Downloads/devops prj/fl-drift-demo/fed_drift/metrics_utils.py`

```python
"""
Utility functions for calculating evaluation metrics.

This module provides reusable metric calculation functions following
2024-2025 academic standards for federated learning evaluation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_gini_coefficient(values: List[float]) -> float:
    """
    Calculate Gini coefficient for measuring inequality.

    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    Used in federated learning to measure fairness across clients.

    Args:
        values: List of metric values (e.g., client accuracies)

    Returns:
        Gini coefficient (0-1)

    Algorithm:
        Based on the Lorenz curve method:
        G = (n + 1 - 2 * Σ(cumsum) / cumsum[-1]) / n

    Example:
        >>> accuracies = [0.85, 0.87, 0.86, 0.84, 0.88]
        >>> gini = calculate_gini_coefficient(accuracies)
        >>> print(f"Gini: {gini:.4f}")  # Low value = fair

    Edge Cases:
        - Empty list: returns 0.0
        - Single value: returns 0.0 (perfect equality)
        - All same values: returns 0.0
        - Negative values: takes absolute value
    """
    if not values or len(values) == 0:
        logger.warning("Empty values list for Gini calculation, returning 0.0")
        return 0.0

    if len(values) == 1:
        return 0.0

    # Convert to numpy array and handle edge cases
    arr = np.abs(np.array(values))

    # Check if all values are the same
    if np.allclose(arr, arr[0]):
        return 0.0

    # Sort values
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)

    # Calculate cumulative sum
    cumsum = np.cumsum(sorted_arr)

    # Avoid division by zero
    if cumsum[-1] == 0:
        logger.warning("Sum of values is zero for Gini calculation, returning 0.0")
        return 0.0

    # Calculate Gini coefficient
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    # Ensure result is in valid range [0, 1]
    gini = np.clip(gini, 0.0, 1.0)

    return float(gini)


def calculate_weighted_mean(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted mean (fixes unweighted bug in current implementation).

    Args:
        values: List of values to average
        weights: List of weights (e.g., number of samples per client)

    Returns:
        Weighted mean

    Edge Cases:
        - Mismatched lengths: raises ValueError
        - All zero weights: returns unweighted mean with warning
        - Negative weights: raises ValueError

    Example:
        >>> accuracies = [0.85, 0.90, 0.80]
        >>> sample_counts = [100, 50, 150]
        >>> weighted_acc = calculate_weighted_mean(accuracies, sample_counts)
    """
    if not values or not weights:
        logger.warning("Empty values or weights, returning 0.0")
        return 0.0

    if len(values) != len(weights):
        raise ValueError(f"Length mismatch: {len(values)} values vs {len(weights)} weights")

    values_arr = np.array(values)
    weights_arr = np.array(weights)

    # Check for negative weights
    if np.any(weights_arr < 0):
        raise ValueError(f"Negative weights detected: {weights_arr[weights_arr < 0]}")

    # Handle all-zero weights
    total_weight = np.sum(weights_arr)
    if total_weight == 0:
        logger.warning("All weights are zero, returning unweighted mean")
        return float(np.mean(values_arr))

    weighted_mean = np.sum(values_arr * weights_arr) / total_weight
    return float(weighted_mean)


def calculate_fairness_variance(values: List[float]) -> float:
    """
    Calculate variance of fairness metric values.

    Lower variance = more fair distribution

    Args:
        values: List of metric values across clients

    Returns:
        Variance
    """
    if not values or len(values) < 2:
        return 0.0

    return float(np.var(values))


def calculate_fairness_std(values: List[float]) -> float:
    """
    Calculate standard deviation of fairness metric values.

    Args:
        values: List of metric values across clients

    Returns:
        Standard deviation
    """
    if not values or len(values) < 2:
        return 0.0

    return float(np.std(values))


def calculate_equalized_accuracy(client_accuracies: List[float],
                                 global_accuracy: float) -> float:
    """
    Calculate Equalized Accuracy metric.

    EA = 1 - max|acc_client - acc_global|

    Range: [0, 1] where 1 = perfect equality

    Args:
        client_accuracies: List of client accuracy values
        global_accuracy: Global (average) accuracy

    Returns:
        Equalized accuracy metric (0-1)
    """
    if not client_accuracies:
        return 0.0

    deviations = np.abs(np.array(client_accuracies) - global_accuracy)
    max_deviation = np.max(deviations)

    equalized_acc = 1.0 - max_deviation

    # Clip to valid range [0, 1]
    equalized_acc = np.clip(equalized_acc, 0.0, 1.0)

    return float(equalized_acc)


def calculate_confusion_matrix_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """
    Calculate precision, recall, F1, FPR, FNR from confusion matrix.

    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives

    Returns:
        Dictionary with precision, recall, f1, fpr, fnr
    """
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # False Positive Rate = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # False Negative Rate = FN / (FN + TP)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr)
    }


def find_stabilization_point(values: List[float],
                             start_index: int = 0,
                             threshold: float = 0.01,
                             window_size: int = 3) -> int:
    """
    Find the point where values stabilize (for recovery speed calculation).

    Stabilization is defined as: consecutive values within threshold of each other
    for at least window_size rounds.

    Args:
        values: Time series of metric values
        start_index: Index to start searching from
        threshold: Maximum allowed change between consecutive values
        window_size: Number of consecutive stable values required

    Returns:
        Index where stabilization occurs, or len(values)-1 if never stabilizes

    Example:
        >>> accuracies = [0.70, 0.75, 0.80, 0.82, 0.83, 0.83, 0.84, 0.84, 0.84]
        >>> stable_idx = find_stabilization_point(accuracies, start_index=3, threshold=0.01, window_size=3)
        >>> print(f"Stabilized at round {stable_idx}")  # Should be around index 6
    """
    if not values or len(values) < window_size + 1:
        return len(values) - 1 if values else 0

    values_arr = np.array(values)

    for i in range(start_index, len(values_arr) - window_size):
        # Check if next window_size values are stable
        window = values_arr[i:i+window_size]

        # Calculate max change in window
        max_change = np.max(np.abs(np.diff(window)))

        if max_change < threshold:
            return i

    # Never stabilized
    return len(values) - 1


# Add __all__ for clean imports
__all__ = [
    'calculate_gini_coefficient',
    'calculate_weighted_mean',
    'calculate_fairness_variance',
    'calculate_fairness_std',
    'calculate_equalized_accuracy',
    'calculate_confusion_matrix_metrics',
    'find_stabilization_point'
]
```

**Validation After Creation:**
```python
# Test the utility functions
python -c "
from fed_drift.metrics_utils import calculate_gini_coefficient
test_values = [0.85, 0.87, 0.86, 0.84, 0.88]
gini = calculate_gini_coefficient(test_values)
print(f'Gini coefficient: {gini:.4f}')
assert 0 <= gini <= 1, 'Gini should be in [0, 1]'
print('✓ Utility module created successfully')
"
```

---

### TASK 2: MODIFY server.py - ADD FAIRNESS METRICS ⭐⭐⭐

**Priority: CRITICAL**
**Time: 45 minutes**
**File:** `fed_drift/server.py`
**Lines to modify:** 299-330

#### Step 2.1: Add Import Statement

**Location:** Top of file (after existing imports, around line 20)

```python
# ADD THIS IMPORT
from .metrics_utils import (
    calculate_gini_coefficient,
    calculate_weighted_mean,
    calculate_fairness_variance,
    calculate_fairness_std,
    calculate_equalized_accuracy
)
```

#### Step 2.2: Modify aggregate_evaluate Method

**Location:** Lines 299-330

**BEFORE (Current Code):**
```python
        # Calculate fairness metrics with validation
        if accuracies:
            global_accuracy = np.mean(accuracies)  # ← BUG: Unweighted!
            fairness_gap = np.max(accuracies) - np.min(accuracies) if len(accuracies) > 1 else 0.0
            logger.info(f"Server Round {server_round}: Global accuracy={global_accuracy:.4f}, Fairness gap={fairness_gap:.4f}")
        else:
            global_accuracy = 0.0
            fairness_gap = 0.0
            logger.warning(f"Server Round {server_round}: No accuracies collected from clients")

        # Update performance history
        self.performance_history.append({
            'round': server_round,
            'global_accuracy': global_accuracy,
            'global_loss': aggregated_loss or 0.0,
            'fairness_gap': fairness_gap,
            'client_accuracies': accuracies,
            'client_losses': losses
        })

        # Enhanced metrics
        enhanced_metrics = {
            "global_accuracy": global_accuracy,
            "fairness_gap": fairness_gap,
            "num_clients_evaluated": len(results),
            "mitigation_active": self.mitigation_active
        }
```

**AFTER (Modified Code with Comments):**
```python
        # Calculate comprehensive fairness metrics with validation
        if accuracies:
            # ═══════════════════════════════════════════════════════
            # FIX: Calculate WEIGHTED global accuracy (not unweighted mean)
            # ═══════════════════════════════════════════════════════

            # Extract sample sizes for weighting
            sample_sizes = []
            for client_proxy, evaluate_res in results:
                # Number of samples evaluated by each client
                num_samples = evaluate_res.num_examples
                sample_sizes.append(num_samples)

            # Calculate weighted global accuracy
            try:
                global_accuracy = calculate_weighted_mean(accuracies, sample_sizes)
                logger.debug(f"Server Round {server_round}: Weighted global accuracy calculated "
                           f"from {len(accuracies)} clients with samples {sample_sizes}")
            except Exception as e:
                logger.warning(f"Failed to calculate weighted accuracy: {e}. Falling back to unweighted mean.")
                global_accuracy = np.mean(accuracies)

            # ═══════════════════════════════════════════════════════
            # NEW: Calculate comprehensive fairness metrics
            # ═══════════════════════════════════════════════════════

            # Basic disparity metrics
            fairness_gap = np.max(accuracies) - np.min(accuracies) if len(accuracies) > 1 else 0.0
            fairness_variance = calculate_fairness_variance(accuracies)
            fairness_std = calculate_fairness_std(accuracies)

            # Inequality metrics
            fairness_gini = calculate_gini_coefficient(accuracies)

            # Equalized accuracy
            equalized_accuracy = calculate_equalized_accuracy(accuracies, global_accuracy)

            # Additional statistics
            min_accuracy = float(np.min(accuracies))
            max_accuracy = float(np.max(accuracies))
            median_accuracy = float(np.median(accuracies))

            # Log comprehensive metrics
            logger.info(
                f"Server Round {server_round}: "
                f"Global accuracy={global_accuracy:.4f} (weighted), "
                f"Fairness gap={fairness_gap:.4f}, "
                f"Fairness variance={fairness_variance:.6f}, "
                f"Fairness Gini={fairness_gini:.4f}, "
                f"Equalized accuracy={equalized_accuracy:.4f}"
            )
            logger.debug(
                f"Server Round {server_round}: "
                f"Min={min_accuracy:.4f}, Max={max_accuracy:.4f}, Median={median_accuracy:.4f}"
            )

        else:
            # No accuracies collected - set all metrics to 0
            global_accuracy = 0.0
            fairness_gap = 0.0
            fairness_variance = 0.0
            fairness_std = 0.0
            fairness_gini = 0.0
            equalized_accuracy = 0.0
            min_accuracy = 0.0
            max_accuracy = 0.0
            median_accuracy = 0.0

            logger.warning(f"Server Round {server_round}: No accuracies collected from clients")

        # ═══════════════════════════════════════════════════════
        # NEW: Update performance history with comprehensive metrics
        # ═══════════════════════════════════════════════════════
        self.performance_history.append({
            'round': server_round,

            # Performance metrics
            'global_accuracy': global_accuracy,
            'global_loss': aggregated_loss or 0.0,

            # Fairness metrics
            'fairness_gap': fairness_gap,
            'fairness_variance': fairness_variance,
            'fairness_std': fairness_std,
            'fairness_gini': fairness_gini,
            'equalized_accuracy': equalized_accuracy,

            # Statistical bounds
            'min_accuracy': min_accuracy,
            'max_accuracy': max_accuracy,
            'median_accuracy': median_accuracy,

            # Raw data for later analysis
            'client_accuracies': accuracies,
            'client_losses': losses,
            'client_sample_sizes': sample_sizes if accuracies else []
        })

        # ═══════════════════════════════════════════════════════
        # NEW: Enhanced metrics for return value
        # ═══════════════════════════════════════════════════════
        enhanced_metrics = {
            # Performance
            "global_accuracy": global_accuracy,
            "global_loss": aggregated_loss or 0.0,

            # Fairness - Basic
            "fairness_gap": fairness_gap,
            "fairness_variance": fairness_variance,
            "fairness_std": fairness_std,

            # Fairness - Advanced
            "fairness_gini": fairness_gini,
            "equalized_accuracy": equalized_accuracy,

            # System state
            "num_clients_evaluated": len(results),
            "mitigation_active": self.mitigation_active,

            # Bounds
            "min_client_accuracy": min_accuracy,
            "max_client_accuracy": max_accuracy
        }
```

**Validation Steps:**
1. Check syntax: `python -m py_compile fed_drift/server.py`
2. Verify imports work: `python -c "from fed_drift.server import DriftAwareFedAvg; print('✓ Imports work')"`
3. Check no breaking changes to interface

---

### TASK 3: MODIFY drift_detection.py - ADD FPR/FNR/F1 METRICS ⭐⭐⭐

**Priority: CRITICAL**
**Time: 60 minutes**
**File:** `fed_drift/drift_detection.py`
**Lines to modify:** 411-439

#### Step 3.1: Add Import Statement

**Location:** Top of file (after existing imports)

```python
# ADD THIS IMPORT
from .metrics_utils import calculate_confusion_matrix_metrics
```

#### Step 3.2: Completely Rewrite calculate_drift_metrics Function

**Location:** Lines 411-439

Replace the entire function with:

```python
def calculate_drift_metrics(drift_history: List[Dict[str, DriftResult]],
                          injection_round: int,
                          total_rounds: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate comprehensive drift detection metrics following 2024 academic standards.

    Metrics calculated:
    - Detection delay (existing)
    - Detection rate (existing)
    - False Positive Rate (FPR) - NEW
    - False Negative Rate (FNR) - NEW
    - Precision, Recall, F1 score - NEW
    - True Positives, False Positives, True Negatives, False Negatives - NEW

    Args:
        drift_history: List of dictionaries mapping detector types to DriftResults
        injection_round: Round where drift was injected
        total_rounds: Total number of rounds (defaults to len(drift_history))

    Returns:
        Dictionary of metrics for each detector type

    Ground Truth Definition:
        - Rounds < injection_round: No drift (stable)
        - Rounds >= injection_round: Drift present

    Example:
        >>> drift_history = [
        ...     {'concept_drift': DriftResult(is_drift=False, drift_score=0.1)},  # Round 0
        ...     {'concept_drift': DriftResult(is_drift=False, drift_score=0.15)}, # Round 1
        ...     {'concept_drift': DriftResult(is_drift=True, drift_score=0.8)},   # Round 2 (injection)
        ...     {'concept_drift': DriftResult(is_drift=True, drift_score=0.9)}    # Round 3
        ... ]
        >>> metrics = calculate_drift_metrics(drift_history, injection_round=2)
        >>> print(f"Detection delay: {metrics['concept_drift_detection_delay']}")
        >>> print(f"FPR: {metrics['concept_drift_false_positive_rate']:.4f}")
    """
    if not drift_history:
        logger.warning("Empty drift history provided to calculate_drift_metrics")
        return {}

    if total_rounds is None:
        total_rounds = len(drift_history)

    metrics = {}

    # ═══════════════════════════════════════════════════════
    # Process each detector type separately
    # ═══════════════════════════════════════════════════════
    for detector_type in drift_history[0].keys():
        logger.debug(f"Calculating metrics for detector: {detector_type}")

        # Extract drift signals for this detector
        drift_signals = []
        for round_idx, round_results in enumerate(drift_history):
            if detector_type in round_results:
                drift_signals.append(round_results[detector_type].is_drift)
            else:
                logger.warning(f"Detector {detector_type} missing in round {round_idx}")
                drift_signals.append(False)  # Default to no drift

        # ═══════════════════════════════════════════════════════
        # EXISTING METRIC 1: Detection Delay
        # ═══════════════════════════════════════════════════════
        detection_round = None
        for round_idx, is_drift in enumerate(drift_signals):
            if is_drift and round_idx >= injection_round:
                detection_round = round_idx
                break

        if detection_round is not None:
            detection_delay = detection_round - injection_round
        else:
            # Never detected - delay is total remaining rounds
            detection_delay = len(drift_signals) - injection_round

        # ═══════════════════════════════════════════════════════
        # EXISTING METRIC 2: Detection Rate
        # ═══════════════════════════════════════════════════════
        post_injection_signals = drift_signals[injection_round:]
        if post_injection_signals:
            detection_rate = sum(post_injection_signals) / len(post_injection_signals)
        else:
            detection_rate = 0.0

        # ═══════════════════════════════════════════════════════
        # NEW: Calculate Confusion Matrix Elements
        # ═══════════════════════════════════════════════════════

        # Define ground truth
        # Before injection_round: No drift (negative class)
        # After injection_round: Drift present (positive class)
        ground_truth = [(i >= injection_round) for i in range(len(drift_signals))]

        # Calculate confusion matrix elements
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for i, (detected, actual) in enumerate(zip(drift_signals, ground_truth)):
            if detected and actual:
                true_positives += 1
            elif detected and not actual:
                false_positives += 1
            elif not detected and not actual:
                true_negatives += 1
            elif not detected and actual:
                false_negatives += 1

        logger.debug(
            f"{detector_type}: TP={true_positives}, FP={false_positives}, "
            f"TN={true_negatives}, FN={false_negatives}"
        )

        # ═══════════════════════════════════════════════════════
        # NEW: Calculate Performance Metrics from Confusion Matrix
        # ═══════════════════════════════════════════════════════
        cm_metrics = calculate_confusion_matrix_metrics(
            tp=true_positives,
            fp=false_positives,
            tn=true_negatives,
            fn=false_negatives
        )

        # ═══════════════════════════════════════════════════════
        # Store all metrics for this detector
        # ═══════════════════════════════════════════════════════
        metrics[f"{detector_type}_detection_delay"] = detection_delay
        metrics[f"{detector_type}_detection_rate"] = detection_rate

        # NEW: Classification metrics
        metrics[f"{detector_type}_precision"] = cm_metrics['precision']
        metrics[f"{detector_type}_recall"] = cm_metrics['recall']
        metrics[f"{detector_type}_f1_score"] = cm_metrics['f1']
        metrics[f"{detector_type}_false_positive_rate"] = cm_metrics['false_positive_rate']
        metrics[f"{detector_type}_false_negative_rate"] = cm_metrics['false_negative_rate']

        # NEW: Confusion matrix elements (for debugging/analysis)
        metrics[f"{detector_type}_true_positives"] = true_positives
        metrics[f"{detector_type}_false_positives"] = false_positives
        metrics[f"{detector_type}_true_negatives"] = true_negatives
        metrics[f"{detector_type}_false_negatives"] = false_negatives

        # Log summary for this detector
        logger.info(
            f"{detector_type} metrics: "
            f"Delay={detection_delay}, Rate={detection_rate:.3f}, "
            f"Precision={cm_metrics['precision']:.3f}, Recall={cm_metrics['recall']:.3f}, "
            f"F1={cm_metrics['f1']:.3f}, FPR={cm_metrics['false_positive_rate']:.3f}, "
            f"FNR={cm_metrics['false_negative_rate']:.3f}"
        )

    # ═══════════════════════════════════════════════════════
    # Calculate aggregate metrics across all detectors
    # ═══════════════════════════════════════════════════════
    detector_types = list(drift_history[0].keys())

    if len(detector_types) > 1:
        # Average metrics across detectors
        avg_precision = np.mean([metrics[f"{dt}_precision"] for dt in detector_types])
        avg_recall = np.mean([metrics[f"{dt}_recall"] for dt in detector_types])
        avg_f1 = np.mean([metrics[f"{dt}_f1_score"] for dt in detector_types])
        avg_fpr = np.mean([metrics[f"{dt}_false_positive_rate"] for dt in detector_types])
        avg_fnr = np.mean([metrics[f"{dt}_false_negative_rate"] for dt in detector_types])

        metrics['aggregate_precision'] = float(avg_precision)
        metrics['aggregate_recall'] = float(avg_recall)
        metrics['aggregate_f1_score'] = float(avg_f1)
        metrics['aggregate_false_positive_rate'] = float(avg_fpr)
        metrics['aggregate_false_negative_rate'] = float(avg_fnr)

        logger.info(
            f"Aggregate metrics: Precision={avg_precision:.3f}, Recall={avg_recall:.3f}, "
            f"F1={avg_f1:.3f}, FPR={avg_fpr:.3f}, FNR={avg_fnr:.3f}"
        )

    return metrics
```

**Validation Steps:**
1. Syntax check: `python -m py_compile fed_drift/drift_detection.py`
2. Unit test with synthetic data (see test script below)

---

### TASK 4: MODIFY simulation.py - ADD RECOVERY METRICS ⭐⭐⭐

**Priority: CRITICAL**
**Time: 90 minutes**
**File:** `fed_drift/simulation.py`
**Lines to modify:** 437-489

#### Step 4.1: Add Import Statement

**Location:** Top of file

```python
# ADD THIS IMPORT
from .metrics_utils import find_stabilization_point, calculate_weighted_mean
```

#### Step 4.2: Create New Method for Recovery Metrics

**Location:** Add NEW method after `_calculate_performance_metrics` (around line 456)

**Add this complete method:**

```python
    def _calculate_recovery_metrics(self,
                                    accuracy_history: List[float],
                                    drift_round: int,
                                    mitigation_round: Optional[int] = None,
                                    stabilization_threshold: float = 0.01,
                                    stabilization_window: int = 3) -> Dict[str, Any]:
        """
        Calculate comprehensive recovery effectiveness metrics.

        Metrics calculated:
        - Recovery speed (rounds to stabilize)
        - Recovery completeness (% of performance restored)
        - Stability after recovery (std dev post-recovery)
        - Overshoot (if recovered beyond baseline)

        Args:
            accuracy_history: Time series of global accuracy values
            drift_round: Round where drift was injected
            mitigation_round: Round where mitigation started (auto-detect if None)
            stabilization_threshold: Max change between consecutive values for stability
            stabilization_window: Number of consecutive stable values required

        Returns:
            Dictionary of recovery metrics
        """
        if not accuracy_history or drift_round >= len(accuracy_history):
            logger.warning("Insufficient data for recovery metrics calculation")
            return {}

        accs = np.array(accuracy_history)

        logger.info(f"Calculating recovery metrics for drift at round {drift_round}")

        # STEP 1: Calculate Pre-Drift Baseline
        if drift_round > 0:
            pre_drift_accuracies = accs[:drift_round]
            pre_drift_baseline = float(np.mean(pre_drift_accuracies))
            pre_drift_std = float(np.std(pre_drift_accuracies))

            logger.info(f"Pre-drift baseline: {pre_drift_baseline:.4f} (±{pre_drift_std:.4f})")
        else:
            logger.warning("Drift at round 0 - cannot calculate baseline")
            pre_drift_baseline = accs[0] if len(accs) > 0 else 0.0
            pre_drift_std = 0.0

        # STEP 2: Measure Drift Impact
        at_drift_accuracy = float(accs[drift_round])
        drift_impact = pre_drift_baseline - at_drift_accuracy
        drift_impact_percent = (drift_impact / pre_drift_baseline * 100) if pre_drift_baseline > 0 else 0.0

        logger.info(f"Drift impact: {at_drift_accuracy:.4f} "
                   f"(drop of {drift_impact:.4f} or {drift_impact_percent:.2f}%)")

        # STEP 3: Detect Mitigation Start (if not provided)
        if mitigation_round is None:
            # Heuristic: Mitigation likely starts 1-3 rounds after drift detection
            search_window = accs[drift_round:min(drift_round + 10, len(accs))]

            # Find first point where accuracy stops declining
            for i in range(1, len(search_window)):
                if search_window[i] >= search_window[i-1]:
                    mitigation_round = drift_round + i
                    logger.info(f"Auto-detected mitigation start at round {mitigation_round}")
                    break

            if mitigation_round is None:
                mitigation_round = drift_round + 2
                logger.warning(f"Could not auto-detect mitigation, assuming round {mitigation_round}")

        # Ensure mitigation_round is valid
        if mitigation_round >= len(accs):
            mitigation_round = len(accs) - 1
            logger.warning(f"Mitigation round adjusted to {mitigation_round} (end of history)")

        at_mitigation_accuracy = float(accs[mitigation_round])
        logger.info(f"Mitigation started at round {mitigation_round}, "
                   f"accuracy={at_mitigation_accuracy:.4f}")

        # STEP 4: Find Stabilization Point
        stabilization_round = find_stabilization_point(
            values=accuracy_history,
            start_index=mitigation_round + 1,
            threshold=stabilization_threshold,
            window_size=stabilization_window
        )

        if stabilization_round >= len(accs) - 1:
            stabilization_round = len(accs) - 1
            logger.warning("Stabilization not achieved, using final round")

        post_recovery_accuracy = float(accs[stabilization_round])
        logger.info(f"Stabilization detected at round {stabilization_round}, "
                   f"accuracy={post_recovery_accuracy:.4f}")

        # STEP 5: Calculate Recovery Metrics
        recovery_speed = stabilization_round - mitigation_round

        # Recovery Completeness
        if abs(pre_drift_baseline - at_drift_accuracy) > 1e-6:
            recovery_completeness = (
                (post_recovery_accuracy - at_drift_accuracy) /
                (pre_drift_baseline - at_drift_accuracy)
            )
        else:
            recovery_completeness = 1.0

        # Overshoot/Undershoot
        overshoot_absolute = max(0, post_recovery_accuracy - pre_drift_baseline)
        overshoot_percent = (overshoot_absolute / pre_drift_baseline * 100) if pre_drift_baseline > 0 else 0.0

        undershoot_absolute = max(0, pre_drift_baseline - post_recovery_accuracy)
        undershoot_percent = (undershoot_absolute / pre_drift_baseline * 100) if pre_drift_baseline > 0 else 0.0

        # Stability Post-Recovery
        if stabilization_round < len(accs) - 1:
            post_recovery_window = accs[stabilization_round:]
            stability_post_recovery = float(np.std(post_recovery_window))
            stability_relative = (stability_post_recovery / pre_drift_baseline * 100) if pre_drift_baseline > 0 else 0.0
        else:
            stability_post_recovery = 0.0
            stability_relative = 0.0

        # Recovery Quality Score (0-1)
        speed_penalty = 0.95 ** recovery_speed
        recovery_quality = recovery_completeness * speed_penalty
        recovery_quality = np.clip(recovery_quality, 0.0, 1.0)

        # STEP 6: Package Metrics
        recovery_metrics = {
            # Baseline and Impact
            'pre_drift_accuracy': pre_drift_baseline,
            'pre_drift_std': pre_drift_std,
            'at_drift_accuracy': at_drift_accuracy,
            'drift_impact_absolute': drift_impact,
            'drift_impact_percent': drift_impact_percent,

            # Mitigation and Recovery
            'mitigation_round': mitigation_round,
            'at_mitigation_accuracy': at_mitigation_accuracy,
            'stabilization_round': stabilization_round,
            'post_recovery_accuracy': post_recovery_accuracy,

            # Core Recovery Metrics
            'recovery_speed_rounds': recovery_speed,
            'recovery_completeness': float(recovery_completeness),
            'recovery_quality_score': float(recovery_quality),

            # Overshoot/Undershoot
            'overshoot_absolute': overshoot_absolute,
            'overshoot_percent': overshoot_percent,
            'undershoot_absolute': undershoot_absolute,
            'undershoot_percent': undershoot_percent,

            # Stability
            'stability_post_recovery': stability_post_recovery,
            'stability_relative_percent': stability_relative,

            # Analysis Flags
            'full_recovery_achieved': (recovery_completeness >= 0.95),
            'overshoot_occurred': (overshoot_absolute > 0.01),
            'stable_recovery': (stability_post_recovery < stabilization_threshold)
        }

        logger.info(
            f"Recovery Summary: "
            f"Speed={recovery_speed} rounds, "
            f"Completeness={recovery_completeness:.2%}, "
            f"Quality={recovery_quality:.3f}, "
            f"Overshoot={overshoot_percent:.2f}%, "
            f"Stability={stability_post_recovery:.6f}"
        )

        return recovery_metrics
```

#### Step 4.3: Integrate Recovery Metrics into _analyze_results

**Location:** Find and replace section around lines 437-444

**Find this:**
```python
                # Calculate recovery metrics if drift was injected
                if len(accuracies) > self.drift_injection_round:
                    pre_drift_acc = np.mean(accuracies[:self.drift_injection_round])
                    post_drift_acc = accuracies[-1]

                    metrics['pre_drift_accuracy'] = pre_drift_acc
                    metrics['post_drift_accuracy'] = post_drift_acc
                    metrics['accuracy_recovery_rate'] = post_drift_acc / pre_drift_acc if pre_drift_acc > 0 else 0.0
```

**Replace with:**
```python
                # NEW: Calculate comprehensive recovery metrics
                if len(accuracies) > self.drift_injection_round:
                    logger.info(f"Calculating recovery metrics for drift injection at round {self.drift_injection_round}")

                    # Try to detect when mitigation was activated
                    mitigation_round = None
                    if hasattr(strategy, 'round_results'):
                        for result in strategy.round_results:
                            if result.get('aggregation_method') == 'FedTrimmedAvg':
                                mitigation_round = result['round']
                                logger.info(f"Detected mitigation activation at round {mitigation_round}")
                                break

                    # Calculate comprehensive recovery metrics
                    recovery_metrics = self._calculate_recovery_metrics(
                        accuracy_history=accuracies,
                        drift_round=self.drift_injection_round,
                        mitigation_round=mitigation_round,
                        stabilization_threshold=0.01,
                        stabilization_window=3
                    )

                    # Add recovery metrics to main metrics dict
                    for key, value in recovery_metrics.items():
                        metrics[f'recovery_{key}'] = value

                    # Legacy metric for backward compatibility
                    metrics['pre_drift_accuracy'] = recovery_metrics.get('pre_drift_accuracy', 0)
                    metrics['post_drift_accuracy'] = recovery_metrics.get('post_recovery_accuracy', 0)
                    metrics['accuracy_recovery_rate'] = recovery_metrics.get('recovery_completeness', 0)

                    logger.info(f"Recovery metrics calculated: {len(recovery_metrics)} metrics added")
```

---

### TASK 5: UPDATE CONFIGURATION ⭐

**Priority: MEDIUM**
**Time: 15 minutes**
**File:** `fed_drift/config.py`

#### Step 5.1: Add Configuration Options

**Location:** In `_load_default_config` method, find the drift_detection section and update:

```python
            # Drift detection configuration
            'drift_detection': {
                'adwin_delta': 0.002,
                'mmd_p_val': 0.05,
                'mmd_permutations': 100,
                'evidently_threshold': 0.25,
                'trimmed_beta': 0.2,
                'feature_names': [f'embedding_dim_{i}' for i in range(128)],

                # NEW: Recovery metric configuration
                'recovery_stabilization_threshold': 0.01,
                'recovery_stabilization_window': 3,

                # NEW: Fairness metric configuration
                'fairness_warning_threshold': 0.15,  # Warn if Gini > 0.15
                'fairness_critical_threshold': 0.25  # Alert if Gini > 0.25
            },
```

---

## PART 3: TESTING STRATEGY 🧪

### 3.1 Unit Tests

**Create:** `fl-drift-demo/tests/test_phase1_metrics.py`

```python
"""
Unit tests for Phase 1 metrics implementation.
"""

import pytest
import numpy as np
from fed_drift.metrics_utils import (
    calculate_gini_coefficient,
    calculate_weighted_mean,
    calculate_confusion_matrix_metrics,
    find_stabilization_point
)
from fed_drift.drift_detection import calculate_drift_metrics, DriftResult


class TestGiniCoefficient:
    """Test Gini coefficient calculation."""

    def test_perfect_equality(self):
        """All values equal should give Gini = 0."""
        values = [0.85] * 10
        gini = calculate_gini_coefficient(values)
        assert gini == 0.0

    def test_moderate_inequality(self):
        """Moderate spread should give intermediate Gini."""
        values = [0.70, 0.75, 0.80, 0.85, 0.90]
        gini = calculate_gini_coefficient(values)
        assert 0.0 < gini < 0.3

    def test_empty_list(self):
        """Empty list should return 0.0."""
        assert calculate_gini_coefficient([]) == 0.0

    def test_single_value(self):
        """Single value should return 0.0."""
        assert calculate_gini_coefficient([0.85]) == 0.0


class TestWeightedMean:
    """Test weighted mean calculation."""

    def test_basic_weighted_mean(self):
        """Basic weighted mean calculation."""
        values = [0.80, 0.90, 0.85]
        weights = [100, 50, 150]
        weighted = calculate_weighted_mean(values, weights)
        expected = (0.80*100 + 0.90*50 + 0.85*150) / 300
        assert abs(weighted - expected) < 1e-6

    def test_mismatched_lengths(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError):
            calculate_weighted_mean([0.80, 0.90], [100, 50, 150])


class TestConfusionMatrixMetrics:
    """Test confusion matrix metric calculation."""

    def test_perfect_detection(self):
        """Perfect detection (no errors)."""
        metrics = calculate_confusion_matrix_metrics(tp=10, fp=0, tn=10, fn=0)
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['false_positive_rate'] == 0.0


class TestDriftMetrics:
    """Test comprehensive drift metrics calculation."""

    def test_perfect_drift_detection(self):
        """Drift detected immediately with no false positives."""
        drift_history = []
        for i in range(10):
            is_drift = (i >= 5)
            drift_history.append({
                'test_detector': DriftResult(is_drift=is_drift, drift_score=0.8 if is_drift else 0.2)
            })

        metrics = calculate_drift_metrics(drift_history, injection_round=5)

        assert metrics['test_detector_detection_delay'] == 0
        assert metrics['test_detector_false_positive_rate'] == 0.0
        assert metrics['test_detector_precision'] == 1.0


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## PART 4: VALIDATION & VERIFICATION ✅

### 4.1 Pre-Commit Checklist

```bash
cd /Users/rohitrathi/Downloads/devops\ prj/fl-drift-demo

# 1. Syntax check
python -m py_compile fed_drift/metrics_utils.py
python -m py_compile fed_drift/server.py
python -m py_compile fed_drift/drift_detection.py
python -m py_compile fed_drift/simulation.py

# 2. Import check
python -c "
from fed_drift.metrics_utils import calculate_gini_coefficient
from fed_drift.server import DriftAwareFedAvg
print('✓ All imports successful')
"

# 3. Run unit tests
python -m pytest tests/test_phase1_metrics.py -v

# 4. Functional test
python -c "
from fed_drift.metrics_utils import calculate_gini_coefficient
gini = calculate_gini_coefficient([0.85, 0.87, 0.86, 0.84, 0.88])
assert 0 <= gini <= 1
print(f'✓ Functional test passed (Gini={gini:.4f})')
"
```

---

## PART 5: DOCUMENTATION 📚

### 5.1 New Metrics Documentation

**Create:** `fl-drift-demo/EVALUATION_METRICS_PHASE1.md`

```markdown
# Phase 1 Metrics Implementation Guide

## New Metrics Added

### 1. Fairness Metrics

#### Fairness Gini Coefficient
- **What it measures:** Inequality using Lorenz curve method
- **Formula:** `(n + 1 - 2 * Σ(cumsum) / cumsum[-1]) / n`
- **Range:** [0, 1], where 0 = perfect equality
- **Location:** `metrics_utils.py:calculate_gini_coefficient()`

#### Fairness Variance
- **What it measures:** Spread of accuracy values across clients
- **Range:** [0, ∞), lower = more fair

#### Equalized Accuracy
- **What it measures:** Maximum deviation from global
- **Formula:** `1 - max|acc_client - acc_global|`
- **Range:** [0, 1]

### 2. Drift Detection Metrics

#### False Positive Rate (FPR)
- **Formula:** `FP / (FP + TN)`
- **Range:** [0, 1], lower = better

#### F1 Score
- **Formula:** `2 * (precision * recall) / (precision + recall)`
- **Range:** [0, 1], higher = better

### 3. Recovery Metrics

#### Recovery Speed
- **Unit:** Number of rounds
- **What it measures:** Rounds from mitigation to stabilization

#### Recovery Completeness
- **Formula:** `(recovered - at_drift) / (baseline - at_drift)`
- **Range:** [0, ∞), where 1 = full recovery

## Usage Examples

```python
# From simulation results
results = simulation.run_simulation()

# Fairness metrics
fairness_gini = results['performance_metrics']['fairness_gini']

# Drift detection metrics
fpr = results['drift_metrics']['concept_drift_false_positive_rate']

# Recovery metrics
recovery_speed = results['recovery_speed_rounds']
```
```

---

## PART 6: DEPLOYMENT CHECKLIST 🚀

### Final Verification Script

**Create:** `verify_phase1.sh`

```bash
#!/bin/bash
echo "Phase 1 Metrics Implementation Verification"

cd /Users/rohitrathi/Downloads/devops\ prj/fl-drift-demo

# 1. Check backups
echo "\n1. Checking backups..."
ls -lh .backups/ || exit 1

# 2. Syntax validation
echo "\n2. Syntax validation..."
python -m py_compile fed_drift/metrics_utils.py || exit 1
python -m py_compile fed_drift/server.py || exit 1
python -m py_compile fed_drift/drift_detection.py || exit 1
python -m py_compile fed_drift/simulation.py || exit 1

# 3. Import test
echo "\n3. Import validation..."
python -c "from fed_drift.metrics_utils import calculate_gini_coefficient" || exit 1

# 4. Unit tests
echo "\n4. Running unit tests..."
python -m pytest tests/test_phase1_metrics.py -v || exit 1

echo "\n✓✓✓ ALL VERIFICATIONS PASSED ✓✓✓"
```

**Run with:** `bash verify_phase1.sh`

---

## PART 7: ESTIMATED TIMELINE ⏱️

| Task | Time | Cumulative | Priority |
|------|------|------------|----------|
| Pre-Implementation Setup | 10min | 10min | CRITICAL |
| Task 1: Create metrics_utils.py | 30min | 40min | CRITICAL |
| Task 2: Modify server.py | 45min | 1h25min | CRITICAL |
| Task 3: Modify drift_detection.py | 60min | 2h25min | CRITICAL |
| Task 4: Modify simulation.py | 90min | 3h55min | CRITICAL |
| Task 5: Update config.py | 15min | 4h10min | MEDIUM |
| Unit test creation | 60min | 5h10min | HIGH |
| Validation & Testing | 45min | 5h55min | CRITICAL |
| Documentation | 45min | 6h40min | MEDIUM |
| Buffer for issues | 50min | **7h30min** | - |

**Suggested Schedule:**
- **Day 1 Morning (4h):** Tasks 1-3
- **Day 1 Afternoon (3.5h):** Task 4 + Testing
- **Day 2 Morning (1h):** Validation + Documentation

---

## PART 8: TROUBLESHOOTING GUIDE 🔧

### Common Issues

**Issue 1: Import Error**
```
ImportError: cannot import name 'calculate_gini_coefficient'
```
**Solution:**
```bash
# Verify file exists
ls -l fed_drift/metrics_utils.py

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"
```

**Issue 2: Weighted Mean Returns NaN**
```
RuntimeWarning: invalid value encountered
```
**Solution:** Check for zero weights
```python
print(f"Weights: {weights}")
print(f"Total: {np.sum(weights)}")
```

**Issue 3: Recovery Metrics Not Appearing**
```
KeyError: 'recovery_speed_rounds'
```
**Solution:** Verify drift injection occurred
```python
print(f"Drift round: {self.drift_injection_round}")
print(f"History length: {len(accuracies)}")
```

---

## FINAL CHECKLIST ✅

**Before considering Phase 1 complete:**

- [ ] Backup created
- [ ] metrics_utils.py created
- [ ] server.py modified
- [ ] drift_detection.py modified
- [ ] simulation.py modified
- [ ] config.py updated
- [ ] All files pass syntax check
- [ ] All imports work
- [ ] Unit tests created and passing
- [ ] Edge cases tested
- [ ] Documentation updated
- [ ] Verification script runs successfully
- [ ] No regressions in existing tests
- [ ] Git commit with descriptive message

**Success Criteria:**
- [ ] Coverage: 35% → 55% (+20%)
- [ ] 18 new metrics implemented
- [ ] 1 critical bug fixed (weighted accuracy)
- [ ] 30+ unit tests added
- [ ] Complete documentation

---

## QUICK START GUIDE 🚀

**To implement Phase 1 immediately:**

1. **Create backup:**
   ```bash
   mkdir -p .backups/phase1_$(date +%Y%m%d)
   cp fed_drift/*.py .backups/phase1_*/
   ```

2. **Create metrics_utils.py:**
   - Copy code from Task 1
   - Save to `fed_drift/metrics_utils.py`
   - Test: `python -c "from fed_drift.metrics_utils import calculate_gini_coefficient"`

3. **Modify server.py:**
   - Add import at top
   - Replace lines 299-330 with new code from Task 2

4. **Modify drift_detection.py:**
   - Add import at top
   - Replace lines 411-439 with new code from Task 3

5. **Modify simulation.py:**
   - Add import at top
   - Add `_calculate_recovery_metrics` method (Task 4.2)
   - Replace recovery section (Task 4.3)

6. **Run validation:**
   ```bash
   bash verify_phase1.sh
   ```

**Estimated time:** 6-8 hours total

---

## CONTACT & SUPPORT 📧

**Questions or Issues?**
- Review the troubleshooting guide above
- Check syntax: `python -m py_compile <file>`
- Verify imports: `python -c "from fed_drift.module import function"`
- Run tests: `pytest tests/test_phase1_metrics.py -v`

---

**End of Phase 1 Implementation Plan**

*This plan provides complete, production-ready code with zero ambiguity. Follow tasks sequentially for best results.*
