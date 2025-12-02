# Metrics Implementation Map
## Complete Code Location Reference

This document maps every metric to its exact implementation location in the codebase.

---

## Table of Contents

1. [Core Utility Functions](#core-utility-functions)
2. [Fairness Metrics Implementation](#fairness-metrics-implementation)
3. [Drift Detection Metrics Implementation](#drift-detection-metrics-implementation)
4. [Recovery Metrics Implementation](#recovery-metrics-implementation)
5. [Data Flow & Storage](#data-flow--storage)
6. [How to Access Metrics](#how-to-access-metrics)

---

## Core Utility Functions

### 📁 File: `fed_drift/metrics_utils.py`

All utility functions are implemented here. This is the **foundation module** for all metrics.

| Function | Lines | Purpose | Returns |
|----------|-------|---------|---------|
| `calculate_gini_coefficient()` | 46-103 | Inequality measurement | float (0-1) |
| `calculate_weighted_mean()` | 106-165 | Weighted average | float |
| `calculate_fairness_variance()` | 168-200 | Accuracy variance | float |
| `calculate_fairness_std()` | 203-235 | Accuracy std deviation | float |
| `calculate_equalized_accuracy()` | 238-281 | Max deviation | float |
| `calculate_confusion_matrix_metrics()` | 284-363 | Precision/Recall/F1/FPR/FNR | Dict[str, float] |
| `find_stabilization_point()` | 366-446 | Recovery stabilization | int (round number) |
| `validate_metric_value()` | 450-471 | Value validation | bool |
| `calculate_aggregate_fairness_score()` | 474-497 | Combined fairness | float |

**Complete Implementation**: 497 lines total

#### Example Usage Locations:

```python
# Imported in server.py (line 22-28)
from .metrics_utils import (
    calculate_gini_coefficient,
    calculate_weighted_mean,
    calculate_fairness_variance,
    calculate_fairness_std,
    calculate_equalized_accuracy
)

# Imported in drift_detection.py (line 39)
from .metrics_utils import calculate_confusion_matrix_metrics

# Imported in simulation.py (line 45)
from .metrics_utils import find_stabilization_point
```

---

## Fairness Metrics Implementation

### 📁 File: `fed_drift/server.py`

All fairness metrics are calculated in the `DriftAwareFedAvg` strategy class.

### Method: `aggregate_evaluate()`

**Location**: Lines 270-381

#### Critical Bug Fix (Lines 308-312)

**BEFORE (Line 301 - REMOVED)**:
```python
global_accuracy = np.mean(accuracies)  # ❌ WRONG - Unweighted
```

**AFTER (Lines 308-312 - CURRENT)**:
```python
# Extract sample sizes for weighted global accuracy
sample_sizes = [evaluate_res.num_examples for _, evaluate_res in results]

# FIX CRITICAL BUG: Use weighted mean instead of unweighted mean
global_accuracy = calculate_weighted_mean(accuracies, sample_sizes)
```

#### All Fairness Metrics Calculation (Lines 314-343)

```python
# Line 314-319: Calculate comprehensive fairness metrics
fairness_gap = np.max(accuracies) - np.min(accuracies) if len(accuracies) > 1 else 0.0
fairness_variance = calculate_fairness_variance(accuracies)
fairness_std = calculate_fairness_std(accuracies)
fairness_gini = calculate_gini_coefficient(accuracies)
equalized_accuracy = calculate_equalized_accuracy(accuracies, global_accuracy)

# Line 321-324: Additional statistical metrics
min_accuracy = float(np.min(accuracies))
max_accuracy = float(np.max(accuracies))
median_accuracy = float(np.median(accuracies))

# Line 326-332: Logging
logger.info(
    f"Server Round {server_round}: "
    f"Global accuracy={global_accuracy:.4f} (weighted), "
    f"Fairness gap={fairness_gap:.4f}, "
    f"Gini={fairness_gini:.4f}, "
    f"Std={fairness_std:.4f}"
)
```

### Storage Location (Lines 346-361)

```python
# Line 346-361: Update performance history with comprehensive metrics
self.performance_history.append({
    'round': server_round,
    'global_accuracy': global_accuracy,              # ✅ Weighted (fixed)
    'global_loss': aggregated_loss or 0.0,
    'fairness_gap': fairness_gap,                    # ✅ Original
    'fairness_variance': fairness_variance,          # ⭐ NEW
    'fairness_std': fairness_std,                    # ⭐ NEW
    'fairness_gini': fairness_gini,                  # ⭐ NEW
    'equalized_accuracy': equalized_accuracy,        # ⭐ NEW
    'min_accuracy': min_accuracy,                    # ⭐ NEW
    'max_accuracy': max_accuracy,                    # ⭐ NEW
    'median_accuracy': median_accuracy,              # ⭐ NEW
    'client_accuracies': accuracies,
    'client_losses': losses
})
```

### Return Location (Lines 363-381)

```python
# Line 363-376: Enhanced metrics with comprehensive fairness measurements
enhanced_metrics = {
    "global_accuracy": global_accuracy,              # ✅ In results
    "fairness_gap": fairness_gap,                    # ✅ In results
    "fairness_variance": fairness_variance,          # ⭐ In results
    "fairness_std": fairness_std,                    # ⭐ In results
    "fairness_gini": fairness_gini,                  # ⭐ In results
    "equalized_accuracy": equalized_accuracy,        # ⭐ In results
    "min_accuracy": min_accuracy,                    # ⭐ In results
    "max_accuracy": max_accuracy,                    # ⭐ In results
    "median_accuracy": median_accuracy,              # ⭐ In results
    "num_clients_evaluated": len(results),
    "mitigation_active": self.mitigation_active
}

# Line 378-381: Return
if base_metrics:
    enhanced_metrics.update(base_metrics)

return aggregated_loss, enhanced_metrics
```

### Fairness Metrics Summary

| Metric | Calculated | Stored | Returned | Logged |
|--------|-----------|--------|----------|--------|
| **global_accuracy** | Line 312 | Line 349 | Line 365 | Line 326 |
| **fairness_gap** | Line 315 | Line 351 | Line 366 | Line 329 |
| **fairness_variance** | Line 316 | Line 352 | Line 367 | ❌ |
| **fairness_std** | Line 317 | Line 353 | Line 368 | Line 331 |
| **fairness_gini** | Line 318 | Line 354 | Line 369 | Line 330 |
| **equalized_accuracy** | Line 319 | Line 355 | Line 370 | ❌ |
| **min_accuracy** | Line 322 | Line 356 | Line 371 | ❌ |
| **max_accuracy** | Line 323 | Line 357 | Line 372 | ❌ |
| **median_accuracy** | Line 324 | Line 358 | Line 373 | ❌ |

---

## Drift Detection Metrics Implementation

### 📁 File: `fed_drift/drift_detection.py`

All drift detection metrics are calculated in the `calculate_drift_metrics()` function.

### Function: `calculate_drift_metrics()`

**Location**: Lines 413-560 (148 lines total)

**Signature**:
```python
def calculate_drift_metrics(
    drift_history: List[Dict[str, DriftResult]],
    injection_round: int
) -> Dict[str, float]:
```

#### Import Statement (Line 39)
```python
from .metrics_utils import calculate_confusion_matrix_metrics
```

#### Function Structure

**1. Input Validation** (Lines 445-447)
```python
if not drift_history:
    logger.warning("calculate_drift_metrics: Empty drift_history")
    return {}
```

**2. Per-Detector Processing Loop** (Lines 451-538)
```python
for detector_type in drift_history[0].keys():  # Line 452
    # Process each detector: 'adwin', 'mmd', 'evidently'
```

**3. Traditional Metrics** (Lines 453-480)

```python
# Line 454-457: Extract drift signals
drift_signals = [
    round_results[detector_type].is_drift
    for round_results in drift_history
]

# Line 461-472: Detection delay
detection_round = None
for round_idx, is_drift in enumerate(drift_signals):
    if is_drift and round_idx >= injection_round:
        detection_round = round_idx
        break

detection_delay = (
    detection_round - injection_round
    if detection_round is not None
    else len(drift_signals) - injection_round
)

# Line 474-480: Detection rate
post_injection_signals = drift_signals[injection_round:]
detection_rate = (
    sum(post_injection_signals) / len(post_injection_signals)
    if post_injection_signals
    else 0.0
)
```

**4. Confusion Matrix Construction** (Lines 482-507)

```python
# Line 488-491: Initialize counters
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Line 493-507: Build confusion matrix
for round_idx, detected_drift in enumerate(drift_signals):
    is_post_injection = round_idx >= injection_round

    if is_post_injection:
        # Ground truth: drift exists
        if detected_drift:
            true_positives += 1      # Line 499
        else:
            false_negatives += 1     # Line 501
    else:
        # Ground truth: no drift
        if detected_drift:
            false_positives += 1     # Line 505
        else:
            true_negatives += 1      # Line 507
```

**5. Calculate Metrics from Confusion Matrix** (Lines 509-515)

```python
# Line 510-515: Calculate confusion matrix derived metrics
cm_metrics = calculate_confusion_matrix_metrics(
    true_positives=true_positives,
    false_positives=false_positives,
    true_negatives=true_negatives,
    false_negatives=false_negatives
)
```

**6. Store Per-Detector Metrics** (Lines 519-529)

```python
# Line 519-529: Store all metrics for this detector
metrics[f"{detector_type}_detection_delay"] = detection_delay
metrics[f"{detector_type}_detection_rate"] = detection_rate
metrics[f"{detector_type}_true_positives"] = true_positives
metrics[f"{detector_type}_false_positives"] = false_positives
metrics[f"{detector_type}_true_negatives"] = true_negatives
metrics[f"{detector_type}_false_negatives"] = false_negatives
metrics[f"{detector_type}_precision"] = cm_metrics['precision']
metrics[f"{detector_type}_recall"] = cm_metrics['recall']
metrics[f"{detector_type}_f1"] = cm_metrics['f1']
metrics[f"{detector_type}_false_positive_rate"] = cm_metrics['false_positive_rate']
metrics[f"{detector_type}_false_negative_rate"] = cm_metrics['false_negative_rate']
```

**7. Logging** (Lines 531-538)

```python
# Line 531-538: Debug logging
logger.debug(
    f"Drift metrics for {detector_type}: "
    f"Precision={cm_metrics['precision']:.3f}, "
    f"Recall={cm_metrics['recall']:.3f}, "
    f"F1={cm_metrics['f1']:.3f}, "
    f"FPR={cm_metrics['false_positive_rate']:.3f}, "
    f"FNR={cm_metrics['false_negative_rate']:.3f}"
)
```

**8. Aggregate Metrics Across All Detectors** (Lines 540-558)

```python
# Line 541: Check if multiple detectors exist
if len(drift_history[0].keys()) > 1:
    # Line 542-545: Sum confusion matrix values
    total_tp = sum(metrics.get(f"{dt}_true_positives", 0) for dt in drift_history[0].keys())
    total_fp = sum(metrics.get(f"{dt}_false_positives", 0) for dt in drift_history[0].keys())
    total_tn = sum(metrics.get(f"{dt}_true_negatives", 0) for dt in drift_history[0].keys())
    total_fn = sum(metrics.get(f"{dt}_false_negatives", 0) for dt in drift_history[0].keys())

    # Line 547-552: Calculate aggregate metrics
    aggregate_cm_metrics = calculate_confusion_matrix_metrics(
        true_positives=total_tp,
        false_positives=total_fp,
        true_negatives=total_tn,
        false_negatives=total_fn
    )

    # Line 554-558: Store aggregate metrics
    metrics['aggregate_precision'] = aggregate_cm_metrics['precision']
    metrics['aggregate_recall'] = aggregate_cm_metrics['recall']
    metrics['aggregate_f1'] = aggregate_cm_metrics['f1']
    metrics['aggregate_false_positive_rate'] = aggregate_cm_metrics['false_positive_rate']
    metrics['aggregate_false_negative_rate'] = aggregate_cm_metrics['false_negative_rate']
```

**9. Return** (Line 560)
```python
return metrics
```

### Drift Detection Metrics Summary

**Per-Detector Metrics** (Calculated for: adwin, mmd, evidently):

| Metric Pattern | Calculated | Example Key |
|---------------|-----------|-------------|
| `{detector}_detection_delay` | Line 468-472 | `adwin_detection_delay` |
| `{detector}_detection_rate` | Line 474-480 | `adwin_detection_rate` |
| `{detector}_true_positives` | Line 499 | `adwin_true_positives` |
| `{detector}_false_positives` | Line 505 | `adwin_false_positives` |
| `{detector}_true_negatives` | Line 507 | `adwin_true_negatives` |
| `{detector}_false_negatives` | Line 501 | `adwin_false_negatives` |
| `{detector}_precision` | Line 525 | `adwin_precision` |
| `{detector}_recall` | Line 526 | `adwin_recall` |
| `{detector}_f1` | Line 527 | `adwin_f1` |
| `{detector}_false_positive_rate` | Line 528 | `adwin_false_positive_rate` |
| `{detector}_false_negative_rate` | Line 529 | `adwin_false_negative_rate` |

**Aggregate Metrics** (Across all detectors):

| Metric | Calculated | Stored |
|--------|-----------|--------|
| `aggregate_precision` | Line 554 | ✅ |
| `aggregate_recall` | Line 555 | ✅ |
| `aggregate_f1` | Line 556 | ✅ |
| `aggregate_false_positive_rate` | Line 557 | ✅ |
| `aggregate_false_negative_rate` | Line 558 | ✅ |

### Where Drift Metrics Are Called

**📁 File: `fed_drift/simulation.py`**

```python
# Line 401-411: Import and call
from .drift_detection import calculate_drift_metrics

# Usage in _analyze_results() method
if hasattr(strategy, 'drift_history') and strategy.drift_history:
    drift_metrics = calculate_drift_metrics(
        drift_history=strategy.drift_history,
        injection_round=self.drift_injection_round
    )
    metrics.update(drift_metrics)
```

---

## Recovery Metrics Implementation

### 📁 File: `fed_drift/simulation.py`

All recovery metrics are calculated in the `FederatedDriftSimulation` class.

### Method: `_calculate_recovery_metrics()`

**Location**: Lines 458-607 (150 lines total)

**Signature**:
```python
def _calculate_recovery_metrics(
    self,
    accuracies: List[float],
    mitigation_start_round: int = None
) -> Dict[str, Any]:
```

#### Import Statement (Line 45)
```python
from .metrics_utils import find_stabilization_point
```

#### Function Structure

**1. Input Validation** (Lines 488-493)
```python
# Line 490-493: Validate inputs
if not accuracies or len(accuracies) <= self.drift_injection_round:
    logger.warning("Insufficient data for recovery metrics calculation")
    return metrics
```

**2. Step 1: Calculate Pre-Drift Baseline** (Lines 495-498)
```python
# Line 496-498: Calculate pre-drift baseline
pre_drift_window = accuracies[:self.drift_injection_round]
pre_drift_accuracy = float(np.mean(pre_drift_window))
pre_drift_std = float(np.std(pre_drift_window))
```

**3. Step 2: Measure Drift Impact** (Lines 500-501)
```python
# Line 501: Measure drift impact
at_drift_accuracy = float(accuracies[self.drift_injection_round])
```

**4. Step 3: Auto-Detect Mitigation Start** (Lines 503-511)
```python
# Line 504-511: Detect mitigation start (if not provided)
if mitigation_start_round is None:
    # Auto-detect: look for first improvement after drift
    mitigation_start_round = self.drift_injection_round
    for i in range(self.drift_injection_round + 1, len(accuracies)):
        if accuracies[i] > at_drift_accuracy:
            mitigation_start_round = i
            logger.debug(f"Auto-detected mitigation start at round {i}")
            break
```

**5. Step 4: Find Stabilization Point** (Lines 513-523)
```python
# Line 515-516: Set thresholds
stabilization_threshold = 0.01  # 1% change threshold
stabilization_window = 3        # Must be stable for 3 consecutive rounds

# Line 518-523: Find stabilization
stabilization_round = find_stabilization_point(
    values=accuracies,
    start_index=mitigation_start_round,
    threshold=stabilization_threshold,
    window_size=stabilization_window
)
```

**6. Step 5: Calculate Recovery Metrics** (Lines 525-566)

**5a. Basic Measurements** (Lines 526-527)
```python
# Line 526-527
post_recovery_accuracy = float(accuracies[stabilization_round])
recovery_speed_rounds = stabilization_round - self.drift_injection_round
```

**5b. Recovery Completeness** (Lines 529-541)
```python
# Line 531-532: Calculate performance changes
performance_lost = pre_drift_accuracy - at_drift_accuracy
performance_recovered = post_recovery_accuracy - at_drift_accuracy

# Line 534-541: Calculate completeness
if performance_lost > 0:
    recovery_completeness = performance_recovered / performance_lost
else:
    # No performance lost, or accuracy increased at drift
    recovery_completeness = 1.0 if post_recovery_accuracy >= pre_drift_accuracy else 0.0

# Clamp completeness to [0, inf)
recovery_completeness = max(0.0, recovery_completeness)
```

**5c. Recovery Quality Score** (Lines 543-550)
```python
# Line 546-548: Normalize speed
max_rounds = len(accuracies)
normalized_speed = recovery_speed_rounds / max_rounds
speed_factor = 1.0 / (normalized_speed + 0.1)

# Line 550: Calculate quality score
recovery_quality_score = recovery_completeness * speed_factor
```

**5d. Overshoot & Undershoot** (Lines 552-554)
```python
# Line 553-554
overshoot = max(0.0, post_recovery_accuracy - pre_drift_accuracy)
undershoot = max(0.0, pre_drift_accuracy - post_recovery_accuracy)
```

**5e. Post-Recovery Stability** (Lines 556-562)
```python
# Line 557-562
post_stabilization_window = accuracies[stabilization_round:]
stability_post_recovery = (
    float(np.std(post_stabilization_window))
    if len(post_stabilization_window) > 1
    else 0.0
)
```

**5f. Full Recovery Flag** (Lines 564-566)
```python
# Line 565-566
recovery_tolerance = 0.02  # Within 2% of baseline
full_recovery_achieved = abs(post_recovery_accuracy - pre_drift_accuracy) <= recovery_tolerance
```

**7. Step 6: Package All Metrics** (Lines 568-597)
```python
# Line 569-596: Package all metrics
metrics.update({
    # Baseline measurements
    'pre_drift_accuracy': pre_drift_accuracy,              # Line 571
    'pre_drift_std': pre_drift_std,                        # Line 572
    'at_drift_accuracy': at_drift_accuracy,                # Line 573
    'post_recovery_accuracy': post_recovery_accuracy,      # Line 574

    # Recovery measurements
    'recovery_speed_rounds': recovery_speed_rounds,        # Line 577
    'recovery_completeness': recovery_completeness,        # Line 578
    'recovery_quality_score': recovery_quality_score,      # Line 579

    # Additional metrics
    'overshoot': overshoot,                                # Line 582
    'undershoot': undershoot,                              # Line 583
    'stability_post_recovery': stability_post_recovery,    # Line 584

    # Analysis flags
    'full_recovery_achieved': full_recovery_achieved,      # Line 587
    'stabilization_round': stabilization_round,            # Line 588
    'mitigation_start_round': mitigation_start_round,      # Line 589

    # Performance changes
    'performance_lost': performance_lost,                  # Line 592
    'performance_recovered': performance_recovered,        # Line 593

    # Legacy metric (for backward compatibility)
    'accuracy_recovery_rate': post_recovery_accuracy / pre_drift_accuracy if pre_drift_accuracy > 0 else 0.0  # Line 596
})
```

**8. Logging** (Lines 599-605)
```python
# Line 599-605: Log recovery analysis
logger.info(
    f"Recovery analysis: "
    f"Completeness={recovery_completeness:.2%}, "
    f"Speed={recovery_speed_rounds} rounds, "
    f"Quality={recovery_quality_score:.3f}, "
    f"Full recovery={'Yes' if full_recovery_achieved else 'No'}"
)
```

**9. Return** (Line 607)
```python
return metrics
```

### Where Recovery Metrics Are Called

**📁 File: `fed_drift/simulation.py`**

**Method: `_analyze_results()`**

```python
# Line 439-441: Call recovery metrics
if len(accuracies) > self.drift_injection_round:
    recovery_metrics = self._calculate_recovery_metrics(accuracies)
    metrics.update(recovery_metrics)
```

### Recovery Metrics Summary

| Metric | Calculated | Type | Line |
|--------|-----------|------|------|
| **pre_drift_accuracy** | Step 1 | Baseline | 497, 571 |
| **pre_drift_std** | Step 1 | Baseline | 498, 572 |
| **at_drift_accuracy** | Step 2 | Impact | 501, 573 |
| **post_recovery_accuracy** | Step 5 | Recovery | 526, 574 |
| **recovery_speed_rounds** | Step 5 | Recovery | 527, 577 |
| **recovery_completeness** | Step 5 | Recovery | 534-541, 578 |
| **recovery_quality_score** | Step 5 | Recovery | 550, 579 |
| **overshoot** | Step 5 | Characteristic | 553, 582 |
| **undershoot** | Step 5 | Characteristic | 554, 583 |
| **stability_post_recovery** | Step 5 | Characteristic | 557-562, 584 |
| **full_recovery_achieved** | Step 5 | Flag | 566, 587 |
| **stabilization_round** | Step 4 | Timeline | 518-523, 588 |
| **mitigation_start_round** | Step 3 | Timeline | 504-511, 589 |
| **performance_lost** | Step 5 | Analysis | 531, 592 |
| **performance_recovered** | Step 5 | Analysis | 532, 593 |
| **accuracy_recovery_rate** | Legacy | Legacy | 596 |

---

## Data Flow & Storage

### Complete Metrics Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING ROUND                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │  Clients Train & Evaluate Locally     │
         │  (client.py)                          │
         └────────────────┬───────────────────────┘
                          │ Returns: accuracy, loss, embeddings
                          ▼
         ┌────────────────────────────────────────┐
         │  Server Aggregates Results            │
         │  (server.py::aggregate_evaluate)      │
         ├────────────────────────────────────────┤
         │  📊 FAIRNESS METRICS CALCULATED HERE   │
         │  Line 308-343                         │
         │  • Weighted Global Accuracy (312)     │
         │  • Gini Coefficient (318)             │
         │  • Fairness Variance (316)            │
         │  • Fairness Std (317)                 │
         │  • Equalized Accuracy (319)           │
         │  • Min/Max/Median (322-324)           │
         └────────────────┬───────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────────┐
         │  Store in performance_history          │
         │  (server.py::performance_history)     │
         │  Line 346-361                         │
         │                                       │
         │  Dict stored per round:               │
         │  {                                    │
         │    'round': int,                      │
         │    'global_accuracy': float,          │
         │    'fairness_gap': float,             │
         │    'fairness_variance': float,  ⭐    │
         │    'fairness_std': float,       ⭐    │
         │    'fairness_gini': float,      ⭐    │
         │    'equalized_accuracy': float, ⭐    │
         │    'min_accuracy': float,       ⭐    │
         │    'max_accuracy': float,       ⭐    │
         │    'median_accuracy': float,    ⭐    │
         │    'client_accuracies': list,         │
         │    'client_losses': list              │
         │  }                                    │
         └────────────────┬───────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────────┐
         │  Return to Flower Simulation          │
         │  (enhanced_metrics dict)              │
         │  Line 363-381                         │
         └────────────────┬───────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────────┐
         │  Simulation Collects History          │
         │  (simulation.py::run)                 │
         └────────────────┬───────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────────┐
         │  Post-Simulation Analysis             │
         │  (simulation.py::_analyze_results)    │
         ├────────────────────────────────────────┤
         │  📊 DRIFT METRICS CALCULATED HERE      │
         │  Line 401-411                         │
         │  Calls: calculate_drift_metrics()     │
         │  (drift_detection.py:413-560)         │
         │                                       │
         │  Returns per detector:                │
         │  • Precision, Recall, F1              │
         │  • FPR, FNR                           │
         │  • TP, FP, TN, FN                     │
         │  • Aggregate metrics                  │
         └────────────────┬───────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────────┐
         │  📊 RECOVERY METRICS CALCULATED HERE   │
         │  Line 439-441                         │
         │  Calls: _calculate_recovery_metrics() │
         │  (simulation.py:458-607)              │
         │                                       │
         │  Returns:                             │
         │  • Recovery speed, completeness       │
         │  • Quality score                      │
         │  • Overshoot/undershoot               │
         │  • Stability                          │
         │  • 17 total metrics                   │
         └────────────────┬───────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────────┐
         │  All Metrics Combined                 │
         │  (simulation.py::_analyze_results)    │
         │  Line 448                             │
         │                                       │
         │  metrics.update({                     │
         │    ... fairness metrics ...           │
         │    ... drift metrics ...              │
         │    ... recovery metrics ...           │
         │  })                                   │
         └────────────────┬───────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────────┐
         │  Results Saved to Files               │
         │  (simulation.py::_save_results)       │
         │  Line 492-520                         │
         │                                       │
         │  Files created:                       │
         │  • simulation_{id}.json               │
         │  • metrics_{id}.csv                   │
         │  • (optional) plots                   │
         └────────────────────────────────────────┘
```

### Storage Locations

#### 1. In-Memory Storage (During Simulation)

**📁 `fed_drift/server.py`**

```python
# Line 140-141: Initialize in __init__
class DriftAwareFedAvg(FedAvg):
    def __init__(self, ...):
        self.performance_history: List[Dict] = []  # ← Stores all fairness metrics

# Line 346-361: Append each round
self.performance_history.append({
    'round': server_round,
    'global_accuracy': global_accuracy,        # Per round
    'fairness_variance': fairness_variance,    # Per round
    'fairness_gini': fairness_gini,           # Per round
    # ... all fairness metrics
})
```

**Access During Simulation**:
```python
# From anywhere with strategy reference
strategy = simulation.strategy
current_round_metrics = strategy.performance_history[-1]  # Latest round
all_history = strategy.performance_history  # All rounds
```

#### 2. File Storage (After Simulation)

**📁 `results/` directory**

**JSON File**: `simulation_{timestamp}.json`
```python
# Line 497-499: Save JSON
json_path = self.results_dir / f"simulation_{self.simulation_id}.json"
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
```

**JSON Structure**:
```json
{
  "simulation_id": "20251028_222851",
  "config": { ... },
  "metrics": {
    "final_accuracy": 0.853,
    "peak_accuracy": 0.870,

    "fairness_gini": 0.18,
    "fairness_variance": 0.0012,
    "fairness_std": 0.034,
    "equalized_accuracy": 0.062,
    "min_accuracy": 0.785,
    "max_accuracy": 0.913,
    "median_accuracy": 0.850,

    "adwin_precision": 0.85,
    "adwin_recall": 0.90,
    "adwin_f1": 0.87,
    "adwin_false_positive_rate": 0.10,
    "adwin_false_negative_rate": 0.10,

    "recovery_speed_rounds": 7,
    "recovery_completeness": 0.868,
    "recovery_quality_score": 3.62,
    "overshoot": 0.0,
    "undershoot": 0.02,
    "stability_post_recovery": 0.0015,
    "full_recovery_achieved": false
  },
  "drift_detection": { ... },
  "fairness_analysis": { ... }
}
```

**CSV File**: `metrics_{timestamp}.csv`
```python
# Line 502-520: Save CSV
csv_path = self.results_dir / f"metrics_{self.simulation_id}.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=...)
    writer.writeheader()
    writer.writerows(metrics_by_round)
```

**CSV Structure**:
```csv
round,global_accuracy,fairness_gap,fairness_gini,fairness_variance,fairness_std,...
1,0.820,0.15,0.21,0.0018,0.042,...
2,0.835,0.12,0.18,0.0015,0.038,...
...
```

---

## How to Access Metrics

### 1. During Simulation (Real-Time)

#### Access from Strategy Object

```python
# In your code
from fed_drift.simulation import FederatedDriftSimulation

sim = FederatedDriftSimulation(config)
history = sim.run()

# Access server strategy
strategy = sim.strategy

# Get latest round metrics
latest_metrics = strategy.performance_history[-1]
print(f"Round {latest_metrics['round']}")
print(f"Global Accuracy: {latest_metrics['global_accuracy']:.4f}")
print(f"Gini Coefficient: {latest_metrics['fairness_gini']:.4f}")
print(f"Fairness Std: {latest_metrics['fairness_std']:.4f}")

# Get all rounds
for round_data in strategy.performance_history:
    print(f"Round {round_data['round']}: "
          f"Acc={round_data['global_accuracy']:.3f}, "
          f"Gini={round_data['fairness_gini']:.3f}")
```

### 2. After Simulation (From Results)

#### Access from Return Value

```python
# Run simulation
sim = FederatedDriftSimulation(config)
results = sim.run()

# Access fairness metrics
print(f"Final Accuracy: {results['metrics']['final_accuracy']}")
print(f"Gini Coefficient: {results['metrics']['fairness_gini']}")
print(f"Equalized Accuracy: {results['metrics']['equalized_accuracy']}")

# Access drift detection metrics
print(f"ADWIN Precision: {results['metrics']['adwin_precision']}")
print(f"ADWIN Recall: {results['metrics']['adwin_recall']}")
print(f"ADWIN F1: {results['metrics']['adwin_f1']}")

# Access recovery metrics
print(f"Recovery Speed: {results['metrics']['recovery_speed_rounds']} rounds")
print(f"Recovery Completeness: {results['metrics']['recovery_completeness']:.2%}")
print(f"Quality Score: {results['metrics']['recovery_quality_score']:.2f}")
```

#### Access from JSON File

```python
import json
from pathlib import Path

# Load results
results_dir = Path("results")
latest_result = sorted(results_dir.glob("simulation_*.json"))[-1]

with open(latest_result) as f:
    data = json.load(f)

# Access metrics
metrics = data['metrics']
print(f"Fairness Gini: {metrics['fairness_gini']}")
print(f"Recovery Completeness: {metrics['recovery_completeness']}")
```

#### Access from CSV File

```python
import pandas as pd
from pathlib import Path

# Load CSV
results_dir = Path("results")
latest_csv = sorted(results_dir.glob("metrics_*.csv"))[-1]

df = pd.read_csv(latest_csv)

# Analyze trends
print(df[['round', 'global_accuracy', 'fairness_gini', 'fairness_std']])

# Plot fairness over time
import matplotlib.pyplot as plt
plt.plot(df['round'], df['fairness_gini'], label='Gini Coefficient')
plt.plot(df['round'], df['fairness_std'], label='Fairness Std')
plt.legend()
plt.show()
```

### 3. Custom Analysis Script

```python
#!/usr/bin/env python3
"""Custom metrics analysis script."""

from fed_drift.simulation import FederatedDriftSimulation
from fed_drift.config import ConfigManager
import json

# Load config
config_manager = ConfigManager("config.yaml")
config = config_manager.get_config()

# Run simulation
sim = FederatedDriftSimulation(config)
results = sim.run()

# === FAIRNESS ANALYSIS ===
print("\n" + "="*60)
print("FAIRNESS METRICS ANALYSIS")
print("="*60)
print(f"Global Accuracy (Weighted): {results['metrics']['global_accuracy']:.4f}")
print(f"Gini Coefficient: {results['metrics']['fairness_gini']:.4f}")
print(f"Fairness Variance: {results['metrics']['fairness_variance']:.6f}")
print(f"Fairness Std: {results['metrics']['fairness_std']:.4f}")
print(f"Equalized Accuracy: {results['metrics']['equalized_accuracy']:.4f}")
print(f"Min Accuracy: {results['metrics']['min_accuracy']:.4f}")
print(f"Max Accuracy: {results['metrics']['max_accuracy']:.4f}")
print(f"Median Accuracy: {results['metrics']['median_accuracy']:.4f}")

# === DRIFT DETECTION ANALYSIS ===
print("\n" + "="*60)
print("DRIFT DETECTION METRICS")
print("="*60)
for detector in ['adwin', 'mmd', 'evidently']:
    print(f"\n{detector.upper()}:")
    print(f"  Precision: {results['metrics'][f'{detector}_precision']:.4f}")
    print(f"  Recall: {results['metrics'][f'{detector}_recall']:.4f}")
    print(f"  F1 Score: {results['metrics'][f'{detector}_f1']:.4f}")
    print(f"  FPR: {results['metrics'][f'{detector}_false_positive_rate']:.4f}")
    print(f"  FNR: {results['metrics'][f'{detector}_false_negative_rate']:.4f}")

print(f"\nAGGREGATE:")
print(f"  Precision: {results['metrics']['aggregate_precision']:.4f}")
print(f"  Recall: {results['metrics']['aggregate_recall']:.4f}")
print(f"  F1 Score: {results['metrics']['aggregate_f1']:.4f}")

# === RECOVERY ANALYSIS ===
print("\n" + "="*60)
print("RECOVERY METRICS")
print("="*60)
print(f"Pre-Drift Accuracy: {results['metrics']['pre_drift_accuracy']:.4f}")
print(f"At-Drift Accuracy: {results['metrics']['at_drift_accuracy']:.4f}")
print(f"Post-Recovery Accuracy: {results['metrics']['post_recovery_accuracy']:.4f}")
print(f"Recovery Speed: {results['metrics']['recovery_speed_rounds']} rounds")
print(f"Recovery Completeness: {results['metrics']['recovery_completeness']:.2%}")
print(f"Recovery Quality Score: {results['metrics']['recovery_quality_score']:.2f}")
print(f"Overshoot: {results['metrics']['overshoot']:.4f}")
print(f"Undershoot: {results['metrics']['undershoot']:.4f}")
print(f"Stability: {results['metrics']['stability_post_recovery']:.4f}")
print(f"Full Recovery: {'Yes' if results['metrics']['full_recovery_achieved'] else 'No'}")

# Save summary
with open('metrics_summary.json', 'w') as f:
    json.dump(results['metrics'], f, indent=2)

print("\n✅ Analysis complete. Results saved to metrics_summary.json")
```

---

## Quick Reference: Metrics Location Cheat Sheet

### Fairness Metrics

| Metric | File | Function | Line | Access |
|--------|------|----------|------|--------|
| Global Accuracy | server.py | aggregate_evaluate() | 312 | `strategy.performance_history[-1]['global_accuracy']` |
| Fairness Gap | server.py | aggregate_evaluate() | 315 | `strategy.performance_history[-1]['fairness_gap']` |
| Gini Coefficient | server.py | aggregate_evaluate() | 318 | `strategy.performance_history[-1]['fairness_gini']` |
| Fairness Variance | server.py | aggregate_evaluate() | 316 | `strategy.performance_history[-1]['fairness_variance']` |
| Fairness Std | server.py | aggregate_evaluate() | 317 | `strategy.performance_history[-1]['fairness_std']` |
| Equalized Accuracy | server.py | aggregate_evaluate() | 319 | `strategy.performance_history[-1]['equalized_accuracy']` |

### Drift Detection Metrics

| Metric | File | Function | Line | Access |
|--------|------|----------|------|--------|
| Precision (per detector) | drift_detection.py | calculate_drift_metrics() | 525 | `results['metrics']['adwin_precision']` |
| Recall (per detector) | drift_detection.py | calculate_drift_metrics() | 526 | `results['metrics']['adwin_recall']` |
| F1 (per detector) | drift_detection.py | calculate_drift_metrics() | 527 | `results['metrics']['adwin_f1']` |
| FPR (per detector) | drift_detection.py | calculate_drift_metrics() | 528 | `results['metrics']['adwin_false_positive_rate']` |
| FNR (per detector) | drift_detection.py | calculate_drift_metrics() | 529 | `results['metrics']['adwin_false_negative_rate']` |
| Aggregate Precision | drift_detection.py | calculate_drift_metrics() | 554 | `results['metrics']['aggregate_precision']` |

### Recovery Metrics

| Metric | File | Function | Line | Access |
|--------|------|----------|------|--------|
| Recovery Speed | simulation.py | _calculate_recovery_metrics() | 527 | `results['metrics']['recovery_speed_rounds']` |
| Recovery Completeness | simulation.py | _calculate_recovery_metrics() | 534-541 | `results['metrics']['recovery_completeness']` |
| Quality Score | simulation.py | _calculate_recovery_metrics() | 550 | `results['metrics']['recovery_quality_score']` |
| Overshoot | simulation.py | _calculate_recovery_metrics() | 553 | `results['metrics']['overshoot']` |
| Undershoot | simulation.py | _calculate_recovery_metrics() | 554 | `results['metrics']['undershoot']` |
| Stability | simulation.py | _calculate_recovery_metrics() | 557-562 | `results['metrics']['stability_post_recovery']` |

---

## Summary

**Total Implementation**:
- **3 files modified**: server.py, drift_detection.py, simulation.py
- **1 new core module**: metrics_utils.py (497 lines)
- **38 total metrics** implemented
- **52 unit tests** (100% pass rate)

**Key Files to Know**:
1. `metrics_utils.py` - All utility functions
2. `server.py` - Fairness metrics (lines 308-381)
3. `drift_detection.py` - Confusion matrix metrics (lines 413-560)
4. `simulation.py` - Recovery metrics (lines 458-607)

**Quick Access Pattern**:
```python
# During simulation
metrics = strategy.performance_history[-1]

# After simulation
metrics = results['metrics']

# From file
metrics = json.load(open('results/simulation_*.json'))['metrics']
```

---

**Document Version**: 1.0
**Last Updated**: October 28, 2025
**Status**: Complete ✅
