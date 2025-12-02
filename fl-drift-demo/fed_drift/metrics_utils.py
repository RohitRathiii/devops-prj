"""
Utility functions for advanced evaluation metrics in federated learning.

This module provides utility functions for calculating fairness metrics, drift detection
metrics, and recovery metrics following 2024-2025 academic standards.

Functions:
    - calculate_gini_coefficient: Measures inequality across client accuracies
    - calculate_weighted_mean: Computes weighted average with proper error handling
    - calculate_fairness_variance: Calculates variance of client accuracies
    - calculate_fairness_std: Calculates standard deviation of client accuracies
    - calculate_equalized_accuracy: Measures maximum deviation from global accuracy
    - calculate_confusion_matrix_metrics: Computes precision, recall, F1, FPR, FNR
    - find_stabilization_point: Detects when metrics stabilize for recovery analysis
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_gini_coefficient(values: List[float]) -> float:
    """
    Calculate Gini coefficient to measure inequality across client accuracies.

    The Gini coefficient measures the degree of inequality in a distribution.
    A Gini coefficient of 0 represents perfect equality (all values are the same),
    while 1 represents maximum inequality.

    Implementation uses the Lorenz curve method:
    G = (n + 1 - 2 * sum(cumulative_sum) / total_sum) / n

    Args:
        values: List of accuracy values from different clients

    Returns:
        Gini coefficient (0-1), where 0 = perfect equality, 1 = maximum inequality

    Examples:
        >>> calculate_gini_coefficient([0.8, 0.8, 0.8])  # Perfect equality
        0.0
        >>> calculate_gini_coefficient([0.9, 0.5, 0.1])  # High inequality
        0.444...

    Edge Cases:
        - Empty list: Returns 0.0
        - Single value: Returns 0.0
        - All values equal: Returns 0.0
        - Negative values: Uses absolute values
    """
    if not values or len(values) == 0:
        logger.warning("calculate_gini_coefficient: Empty values list, returning 0.0")
        return 0.0

    if len(values) == 1:
        return 0.0

    # Convert to numpy array and take absolute values
    arr = np.abs(np.array(values))

    # Check if all values are equal
    if np.allclose(arr, arr[0]):
        return 0.0

    # Sort array for Gini calculation
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)

    # Calculate cumulative sum
    cumsum = np.cumsum(sorted_arr)

    # Handle edge case where sum is zero
    if cumsum[-1] == 0:
        logger.warning("calculate_gini_coefficient: Sum of values is zero, returning 0.0")
        return 0.0

    # Calculate Gini coefficient using Lorenz curve method
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    # Ensure result is in [0, 1] range
    gini = float(np.clip(gini, 0.0, 1.0))

    return gini


def calculate_weighted_mean(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted mean with comprehensive error handling.

    This function computes the weighted average of values, properly handling
    edge cases like zero weights, negative weights, and mismatched lengths.

    Formula: weighted_mean = sum(values * weights) / sum(weights)

    Args:
        values: List of values to average (e.g., client accuracies)
        weights: List of weights for each value (e.g., client dataset sizes)

    Returns:
        Weighted mean of the values

    Raises:
        ValueError: If values and weights have different lengths
        ValueError: If any weight is negative

    Examples:
        >>> calculate_weighted_mean([0.8, 0.9], [100, 200])
        0.8666...
        >>> calculate_weighted_mean([0.5, 0.7], [0, 100])  # Zero weight
        0.7

    Edge Cases:
        - All weights zero: Returns unweighted mean
        - Empty lists: Raises ValueError
        - Length mismatch: Raises ValueError
        - Negative weights: Raises ValueError
    """
    if len(values) != len(weights):
        raise ValueError(
            f"calculate_weighted_mean: Length mismatch - "
            f"values: {len(values)}, weights: {len(weights)}"
        )

    if len(values) == 0:
        raise ValueError("calculate_weighted_mean: Empty values list")

    # Convert to numpy arrays
    values_arr = np.array(values, dtype=np.float64)
    weights_arr = np.array(weights, dtype=np.float64)

    # Check for negative weights
    if np.any(weights_arr < 0):
        raise ValueError("calculate_weighted_mean: Negative weights detected")

    # Calculate total weight
    total_weight = np.sum(weights_arr)

    # Handle case where all weights are zero
    if total_weight == 0:
        logger.warning(
            "calculate_weighted_mean: All weights are zero, returning unweighted mean"
        )
        return float(np.mean(values_arr))

    # Calculate weighted mean
    weighted_sum = np.sum(values_arr * weights_arr)
    weighted_mean = weighted_sum / total_weight

    return float(weighted_mean)


def calculate_fairness_variance(values: List[float]) -> float:
    """
    Calculate variance of client accuracies to measure fairness dispersion.

    Variance measures how far values are spread from their mean. Lower variance
    indicates more fair distribution of performance across clients.

    Args:
        values: List of accuracy values from different clients

    Returns:
        Variance of the values

    Examples:
        >>> calculate_fairness_variance([0.8, 0.8, 0.8])
        0.0
        >>> calculate_fairness_variance([0.9, 0.7, 0.5])
        0.0266...

    Edge Cases:
        - Empty list: Returns 0.0
        - Single value: Returns 0.0
        - All values equal: Returns 0.0
    """
    if not values or len(values) == 0:
        logger.warning("calculate_fairness_variance: Empty values list, returning 0.0")
        return 0.0

    if len(values) == 1:
        return 0.0

    variance = float(np.var(values, ddof=1))  # Use sample variance (ddof=1)
    return variance


def calculate_fairness_std(values: List[float]) -> float:
    """
    Calculate standard deviation of client accuracies for fairness measurement.

    Standard deviation is the square root of variance and provides a measure
    in the same units as the original values. Lower std indicates better fairness.

    Args:
        values: List of accuracy values from different clients

    Returns:
        Standard deviation of the values

    Examples:
        >>> calculate_fairness_std([0.8, 0.8, 0.8])
        0.0
        >>> calculate_fairness_std([0.9, 0.7, 0.5])
        0.163...

    Edge Cases:
        - Empty list: Returns 0.0
        - Single value: Returns 0.0
        - All values equal: Returns 0.0
    """
    if not values or len(values) == 0:
        logger.warning("calculate_fairness_std: Empty values list, returning 0.0")
        return 0.0

    if len(values) == 1:
        return 0.0

    std = float(np.std(values, ddof=1))  # Use sample std (ddof=1)
    return std


def calculate_equalized_accuracy(
    client_accuracies: List[float],
    global_accuracy: float
) -> float:
    """
    Calculate equalized accuracy metric (maximum deviation from global accuracy).

    This metric measures the worst-case fairness by finding the maximum absolute
    deviation of any client from the global accuracy. Lower values indicate
    better fairness.

    Formula: max(|client_acc - global_acc|) for all clients

    Args:
        client_accuracies: List of accuracy values from different clients
        global_accuracy: Global (weighted) accuracy across all clients

    Returns:
        Maximum absolute deviation from global accuracy

    Examples:
        >>> calculate_equalized_accuracy([0.8, 0.82, 0.78], 0.8)
        0.02
        >>> calculate_equalized_accuracy([0.9, 0.7], 0.8)
        0.1

    Edge Cases:
        - Empty list: Returns 0.0
        - Single client: Returns absolute deviation
    """
    if not client_accuracies or len(client_accuracies) == 0:
        logger.warning(
            "calculate_equalized_accuracy: Empty client_accuracies, returning 0.0"
        )
        return 0.0

    # Calculate absolute deviations
    deviations = np.abs(np.array(client_accuracies) - global_accuracy)

    # Return maximum deviation
    max_deviation = float(np.max(deviations))
    return max_deviation


def calculate_confusion_matrix_metrics(
    true_positives: int,
    false_positives: int,
    true_negatives: int,
    false_negatives: int
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics from confusion matrix components.

    Computes precision, recall, F1 score, false positive rate (FPR),
    and false negative rate (FNR) from confusion matrix values.

    Metrics:
        - Precision = TP / (TP + FP) - Accuracy of positive predictions
        - Recall = TP / (TP + FN) - Coverage of actual positives
        - F1 = 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean
        - FPR = FP / (FP + TN) - False alarm rate
        - FNR = FN / (FN + TP) - Miss rate

    Args:
        true_positives: Number of correct positive predictions
        false_positives: Number of incorrect positive predictions
        true_negatives: Number of correct negative predictions
        false_negatives: Number of incorrect negative predictions

    Returns:
        Dictionary with keys: precision, recall, f1, false_positive_rate,
        false_negative_rate

    Examples:
        >>> calculate_confusion_matrix_metrics(80, 10, 85, 5)
        {'precision': 0.888..., 'recall': 0.941..., 'f1': 0.914...,
         'false_positive_rate': 0.105..., 'false_negative_rate': 0.058...}

    Edge Cases:
        - No positives predicted (TP + FP = 0): precision = 0.0
        - No actual positives (TP + FN = 0): recall = 0.0, FNR = 0.0
        - No actual negatives (FP + TN = 0): FPR = 0.0
        - Both precision and recall are 0: F1 = 0.0
    """
    # Calculate precision with zero division handling
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    # Calculate recall with zero division handling
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    # Calculate F1 score
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Calculate false positive rate
    false_positive_rate = (
        false_positives / (false_positives + true_negatives)
        if (false_positives + true_negatives) > 0
        else 0.0
    )

    # Calculate false negative rate
    false_negative_rate = (
        false_negatives / (false_negatives + true_positives)
        if (false_negatives + true_positives) > 0
        else 0.0
    )

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1_score),
        'false_positive_rate': float(false_positive_rate),
        'false_negative_rate': float(false_negative_rate)
    }


def find_stabilization_point(
    values: List[float],
    start_index: int = 0,
    threshold: float = 0.01,
    window_size: int = 3
) -> int:
    """
    Find the point where values stabilize using a sliding window approach.

    This function is used to determine when recovery is complete by finding
    the first point where values remain stable (changes below threshold)
    for a specified window size.

    Algorithm:
        1. Start from start_index
        2. For each position, examine a window of window_size values
        3. Calculate maximum absolute change within the window
        4. If max change < threshold, return the position
        5. If no stabilization found, return the last index

    Args:
        values: List of metric values (e.g., accuracy over rounds)
        start_index: Index to start searching from (default: 0)
        threshold: Maximum change allowed for stabilization (default: 0.01)
        window_size: Number of consecutive values to check (default: 3)

    Returns:
        Index where stabilization occurs, or last index if not found

    Examples:
        >>> find_stabilization_point([0.7, 0.75, 0.8, 0.81, 0.81, 0.82])
        3  # Stabilizes at index 3
        >>> find_stabilization_point([0.7, 0.8, 0.9, 1.0], threshold=0.05)
        5  # Never stabilizes, returns last index

    Edge Cases:
        - Empty list: Returns 0
        - List too short for window: Returns last index
        - start_index >= len(values): Returns last index
        - No stabilization found: Returns last index
    """
    if not values or len(values) == 0:
        logger.warning("find_stabilization_point: Empty values list, returning 0")
        return 0

    if start_index >= len(values):
        logger.warning(
            f"find_stabilization_point: start_index {start_index} >= "
            f"len(values) {len(values)}, returning last index"
        )
        return len(values) - 1

    if len(values) < start_index + window_size:
        logger.info(
            "find_stabilization_point: Not enough values for window, "
            "returning last index"
        )
        return len(values) - 1

    # Search for stabilization point
    for i in range(start_index, len(values) - window_size + 1):
        window = values[i:i + window_size]

        # Calculate maximum absolute change in the window
        max_change = np.max(np.abs(np.diff(window)))

        # Check if stabilized
        if max_change < threshold:
            logger.debug(
                f"find_stabilization_point: Stabilized at index {i} "
                f"with max_change={max_change:.4f}"
            )
            return i

    # No stabilization found
    logger.info(
        f"find_stabilization_point: No stabilization found with "
        f"threshold={threshold}, returning last index"
    )
    return len(values) - 1


# Validation functions for testing
def validate_metric_value(value: float, metric_name: str, min_val: float = 0.0, max_val: float = 1.0) -> bool:
    """
    Validate that a metric value is within expected range.

    Args:
        value: Metric value to validate
        metric_name: Name of the metric (for logging)
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        True if valid, False otherwise
    """
    if not (min_val <= value <= max_val):
        logger.warning(
            f"validate_metric_value: {metric_name}={value:.4f} "
            f"outside range [{min_val}, {max_val}]"
        )
        return False
    return True


def calculate_aggregate_fairness_score(
    fairness_variance: float,
    fairness_gini: float,
    equalized_accuracy: float
) -> float:
    """
    Calculate aggregate fairness score combining multiple fairness metrics.

    This provides a single fairness score that combines variance, Gini coefficient,
    and equalized accuracy. Lower values indicate better fairness.

    Args:
        fairness_variance: Variance of client accuracies
        fairness_gini: Gini coefficient (0-1)
        equalized_accuracy: Maximum deviation from global accuracy

    Returns:
        Aggregate fairness score (lower is better)
    """
    # Weighted combination (tunable based on importance)
    aggregate_score = (
        0.3 * fairness_variance +
        0.4 * fairness_gini +
        0.3 * equalized_accuracy
    )
    return float(aggregate_score)
