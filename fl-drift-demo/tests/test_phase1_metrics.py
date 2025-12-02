"""
Unit tests for Phase 1 metrics implementation.

Tests all new utility functions from metrics_utils.py including:
- Gini coefficient calculation
- Weighted mean calculation
- Fairness variance and standard deviation
- Equalized accuracy
- Confusion matrix metrics
- Stabilization point detection
"""

import pytest
import numpy as np
from fed_drift.metrics_utils import (
    calculate_gini_coefficient,
    calculate_weighted_mean,
    calculate_fairness_variance,
    calculate_fairness_std,
    calculate_equalized_accuracy,
    calculate_confusion_matrix_metrics,
    find_stabilization_point,
    validate_metric_value,
    calculate_aggregate_fairness_score
)


class TestGiniCoefficient:
    """Test cases for Gini coefficient calculation."""

    def test_perfect_equality(self):
        """Test that equal values yield Gini coefficient of 0."""
        values = [0.8, 0.8, 0.8, 0.8]
        gini = calculate_gini_coefficient(values)
        assert gini == pytest.approx(0.0, abs=1e-6)

    def test_perfect_inequality(self):
        """Test that maximum inequality yields high Gini coefficient."""
        values = [1.0, 0.0, 0.0, 0.0]
        gini = calculate_gini_coefficient(values)
        assert gini > 0.5  # Should be relatively high

    def test_moderate_inequality(self):
        """Test moderate inequality case."""
        values = [0.9, 0.8, 0.7, 0.6]
        gini = calculate_gini_coefficient(values)
        assert 0.0 < gini < 1.0

    def test_empty_list(self):
        """Test that empty list returns 0."""
        gini = calculate_gini_coefficient([])
        assert gini == 0.0

    def test_single_value(self):
        """Test that single value returns 0."""
        gini = calculate_gini_coefficient([0.8])
        assert gini == 0.0

    def test_negative_values(self):
        """Test that negative values are handled (absolute value)."""
        values = [-0.8, -0.8, -0.8]
        gini = calculate_gini_coefficient(values)
        assert gini == pytest.approx(0.0, abs=1e-6)

    def test_range_0_to_1(self):
        """Test that Gini coefficient is always in [0, 1]."""
        values = [0.95, 0.50, 0.10]
        gini = calculate_gini_coefficient(values)
        assert 0.0 <= gini <= 1.0


class TestWeightedMean:
    """Test cases for weighted mean calculation."""

    def test_basic_weighted_mean(self):
        """Test basic weighted mean calculation."""
        values = [0.8, 0.9]
        weights = [100, 200]
        result = calculate_weighted_mean(values, weights)
        expected = (0.8 * 100 + 0.9 * 200) / 300
        assert result == pytest.approx(expected, abs=1e-6)

    def test_equal_weights(self):
        """Test that equal weights give simple average."""
        values = [0.7, 0.8, 0.9]
        weights = [1.0, 1.0, 1.0]
        result = calculate_weighted_mean(values, weights)
        expected = np.mean(values)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_zero_weight(self):
        """Test handling of zero weight for one value."""
        values = [0.5, 0.9]
        weights = [0, 100]
        result = calculate_weighted_mean(values, weights)
        assert result == pytest.approx(0.9, abs=1e-6)

    def test_all_zero_weights(self):
        """Test that all zero weights returns unweighted mean."""
        values = [0.6, 0.7, 0.8]
        weights = [0, 0, 0]
        result = calculate_weighted_mean(values, weights)
        expected = np.mean(values)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_length_mismatch_error(self):
        """Test that mismatched lengths raise ValueError."""
        values = [0.8, 0.9]
        weights = [100]
        with pytest.raises(ValueError, match="Length mismatch"):
            calculate_weighted_mean(values, weights)

    def test_negative_weights_error(self):
        """Test that negative weights raise ValueError."""
        values = [0.8, 0.9]
        weights = [100, -50]
        with pytest.raises(ValueError, match="Negative weights"):
            calculate_weighted_mean(values, weights)

    def test_empty_lists_error(self):
        """Test that empty lists raise ValueError."""
        with pytest.raises(ValueError, match="Empty values"):
            calculate_weighted_mean([], [])


class TestFairnessVariance:
    """Test cases for fairness variance calculation."""

    def test_no_variance(self):
        """Test that equal values have zero variance."""
        values = [0.8, 0.8, 0.8]
        variance = calculate_fairness_variance(values)
        assert variance == pytest.approx(0.0, abs=1e-6)

    def test_positive_variance(self):
        """Test that different values have positive variance."""
        values = [0.7, 0.8, 0.9]
        variance = calculate_fairness_variance(values)
        assert variance > 0.0

    def test_empty_list(self):
        """Test that empty list returns 0."""
        variance = calculate_fairness_variance([])
        assert variance == 0.0

    def test_single_value(self):
        """Test that single value returns 0."""
        variance = calculate_fairness_variance([0.8])
        assert variance == 0.0

    def test_known_variance(self):
        """Test against known variance value."""
        values = [1.0, 2.0, 3.0]
        variance = calculate_fairness_variance(values)
        expected = np.var(values, ddof=1)
        assert variance == pytest.approx(expected, abs=1e-6)


class TestFairnessStd:
    """Test cases for fairness standard deviation calculation."""

    def test_no_std(self):
        """Test that equal values have zero std."""
        values = [0.8, 0.8, 0.8]
        std = calculate_fairness_std(values)
        assert std == pytest.approx(0.0, abs=1e-6)

    def test_positive_std(self):
        """Test that different values have positive std."""
        values = [0.7, 0.8, 0.9]
        std = calculate_fairness_std(values)
        assert std > 0.0

    def test_empty_list(self):
        """Test that empty list returns 0."""
        std = calculate_fairness_std([])
        assert std == 0.0

    def test_single_value(self):
        """Test that single value returns 0."""
        std = calculate_fairness_std([0.8])
        assert std == 0.0

    def test_known_std(self):
        """Test against known std value."""
        values = [1.0, 2.0, 3.0]
        std = calculate_fairness_std(values)
        expected = np.std(values, ddof=1)
        assert std == pytest.approx(expected, abs=1e-6)

    def test_std_variance_relationship(self):
        """Test that std is square root of variance."""
        values = [0.6, 0.7, 0.8, 0.9]
        std = calculate_fairness_std(values)
        variance = calculate_fairness_variance(values)
        assert std == pytest.approx(np.sqrt(variance), abs=1e-6)


class TestEqualizedAccuracy:
    """Test cases for equalized accuracy calculation."""

    def test_perfect_equality(self):
        """Test that all clients at global accuracy gives 0."""
        client_accuracies = [0.8, 0.8, 0.8]
        global_accuracy = 0.8
        result = calculate_equalized_accuracy(client_accuracies, global_accuracy)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_maximum_deviation(self):
        """Test that maximum deviation is correctly identified."""
        client_accuracies = [0.9, 0.8, 0.7]
        global_accuracy = 0.8
        result = calculate_equalized_accuracy(client_accuracies, global_accuracy)
        assert result == pytest.approx(0.1, abs=1e-6)  # max deviation is 0.9 - 0.8 = 0.1

    def test_symmetric_deviations(self):
        """Test with symmetric positive and negative deviations."""
        client_accuracies = [0.85, 0.75]
        global_accuracy = 0.8
        result = calculate_equalized_accuracy(client_accuracies, global_accuracy)
        assert result == pytest.approx(0.05, abs=1e-6)

    def test_empty_list(self):
        """Test that empty list returns 0."""
        result = calculate_equalized_accuracy([], 0.8)
        assert result == 0.0

    def test_single_client(self):
        """Test with single client."""
        client_accuracies = [0.75]
        global_accuracy = 0.8
        result = calculate_equalized_accuracy(client_accuracies, global_accuracy)
        assert result == pytest.approx(0.05, abs=1e-6)


class TestConfusionMatrixMetrics:
    """Test cases for confusion matrix metrics calculation."""

    def test_perfect_detection(self):
        """Test perfect drift detection (all correct)."""
        metrics = calculate_confusion_matrix_metrics(
            true_positives=10,
            false_positives=0,
            true_negatives=10,
            false_negatives=0
        )
        assert metrics['precision'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['recall'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['f1'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['false_positive_rate'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['false_negative_rate'] == pytest.approx(0.0, abs=1e-6)

    def test_no_detection(self):
        """Test case where nothing is detected."""
        metrics = calculate_confusion_matrix_metrics(
            true_positives=0,
            false_positives=0,
            true_negatives=10,
            false_negatives=10
        )
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['false_positive_rate'] == 0.0
        assert metrics['false_negative_rate'] == 1.0

    def test_balanced_case(self):
        """Test balanced confusion matrix."""
        metrics = calculate_confusion_matrix_metrics(
            true_positives=8,
            false_positives=2,
            true_negatives=8,
            false_negatives=2
        )

        expected_precision = 8 / 10  # 0.8
        expected_recall = 8 / 10  # 0.8
        expected_f1 = 2 * (0.8 * 0.8) / (0.8 + 0.8)  # 0.8
        expected_fpr = 2 / 10  # 0.2
        expected_fnr = 2 / 10  # 0.2

        assert metrics['precision'] == pytest.approx(expected_precision, abs=1e-6)
        assert metrics['recall'] == pytest.approx(expected_recall, abs=1e-6)
        assert metrics['f1'] == pytest.approx(expected_f1, abs=1e-6)
        assert metrics['false_positive_rate'] == pytest.approx(expected_fpr, abs=1e-6)
        assert metrics['false_negative_rate'] == pytest.approx(expected_fnr, abs=1e-6)

    def test_high_precision_low_recall(self):
        """Test high precision, low recall case."""
        metrics = calculate_confusion_matrix_metrics(
            true_positives=5,
            false_positives=0,
            true_negatives=10,
            false_negatives=10
        )
        assert metrics['precision'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['recall'] == pytest.approx(5/15, abs=1e-6)
        assert metrics['f1'] > 0.0

    def test_all_zeros(self):
        """Test edge case with all zeros."""
        metrics = calculate_confusion_matrix_metrics(
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0
        )
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['false_positive_rate'] == 0.0
        assert metrics['false_negative_rate'] == 0.0


class TestFindStabilizationPoint:
    """Test cases for stabilization point detection."""

    def test_immediate_stabilization(self):
        """Test case where values are immediately stable."""
        values = [0.80, 0.80, 0.80, 0.81, 0.80]
        result = find_stabilization_point(values, threshold=0.02, window_size=3)
        assert result == 0  # Stable from the start

    def test_delayed_stabilization(self):
        """Test case where stabilization occurs after some rounds."""
        values = [0.7, 0.75, 0.78, 0.80, 0.81, 0.81, 0.81]
        result = find_stabilization_point(values, threshold=0.01, window_size=3)
        assert result == 4  # Stabilizes at index 4

    def test_never_stabilizes(self):
        """Test case where values never stabilize."""
        values = [0.7, 0.75, 0.80, 0.85, 0.90]
        result = find_stabilization_point(values, threshold=0.01, window_size=3)
        assert result == len(values) - 1  # Returns last index

    def test_with_start_index(self):
        """Test with custom start index."""
        values = [0.5, 0.6, 0.7, 0.80, 0.81, 0.81, 0.81]
        result = find_stabilization_point(values, start_index=3, threshold=0.01, window_size=3)
        assert result == 4  # Stabilizes at index 4 (starting search from 3)

    def test_empty_list(self):
        """Test that empty list returns 0."""
        result = find_stabilization_point([], threshold=0.01, window_size=3)
        assert result == 0

    def test_list_too_short(self):
        """Test with list shorter than window size."""
        values = [0.8, 0.81]
        result = find_stabilization_point(values, threshold=0.01, window_size=3)
        assert result == len(values) - 1

    def test_start_index_out_of_bounds(self):
        """Test with start_index >= len(values)."""
        values = [0.8, 0.81, 0.82]
        result = find_stabilization_point(values, start_index=10, threshold=0.01, window_size=3)
        assert result == len(values) - 1


class TestValidateMetricValue:
    """Test cases for metric value validation."""

    def test_valid_value_in_range(self):
        """Test that valid value returns True."""
        assert validate_metric_value(0.5, "test_metric", 0.0, 1.0) is True

    def test_value_below_range(self):
        """Test that value below range returns False."""
        assert validate_metric_value(-0.1, "test_metric", 0.0, 1.0) is False

    def test_value_above_range(self):
        """Test that value above range returns False."""
        assert validate_metric_value(1.1, "test_metric", 0.0, 1.0) is False

    def test_boundary_values(self):
        """Test that boundary values are valid."""
        assert validate_metric_value(0.0, "test_metric", 0.0, 1.0) is True
        assert validate_metric_value(1.0, "test_metric", 0.0, 1.0) is True


class TestAggregateFairnessScore:
    """Test cases for aggregate fairness score calculation."""

    def test_perfect_fairness(self):
        """Test that perfect fairness gives low score."""
        score = calculate_aggregate_fairness_score(
            fairness_variance=0.0,
            fairness_gini=0.0,
            equalized_accuracy=0.0
        )
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_poor_fairness(self):
        """Test that poor fairness gives high score."""
        score = calculate_aggregate_fairness_score(
            fairness_variance=0.1,
            fairness_gini=0.5,
            equalized_accuracy=0.2
        )
        assert score > 0.0

    def test_weighted_combination(self):
        """Test that weights are applied correctly."""
        score = calculate_aggregate_fairness_score(
            fairness_variance=0.1,
            fairness_gini=0.2,
            equalized_accuracy=0.3
        )
        # Expected: 0.3 * 0.1 + 0.4 * 0.2 + 0.3 * 0.3 = 0.03 + 0.08 + 0.09 = 0.2
        assert score == pytest.approx(0.2, abs=1e-6)


class TestIntegration:
    """Integration tests for combined metric calculations."""

    def test_full_fairness_analysis(self):
        """Test complete fairness analysis workflow."""
        client_accuracies = [0.85, 0.80, 0.75, 0.70]
        sample_sizes = [100, 150, 120, 130]

        # Calculate all fairness metrics
        global_accuracy = calculate_weighted_mean(client_accuracies, sample_sizes)
        gini = calculate_gini_coefficient(client_accuracies)
        variance = calculate_fairness_variance(client_accuracies)
        std = calculate_fairness_std(client_accuracies)
        eq_accuracy = calculate_equalized_accuracy(client_accuracies, global_accuracy)

        # Verify all metrics are calculated
        assert 0.0 <= global_accuracy <= 1.0
        assert 0.0 <= gini <= 1.0
        assert variance >= 0.0
        assert std >= 0.0
        assert eq_accuracy >= 0.0

    def test_full_drift_detection_analysis(self):
        """Test complete drift detection analysis workflow."""
        # Simulate confusion matrix from drift detection
        tp, fp, tn, fn = 8, 2, 18, 2

        metrics = calculate_confusion_matrix_metrics(tp, fp, tn, fn)

        # Verify all metrics are present and valid
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1'] <= 1.0
        assert 0.0 <= metrics['false_positive_rate'] <= 1.0
        assert 0.0 <= metrics['false_negative_rate'] <= 1.0

    def test_recovery_analysis_workflow(self):
        """Test recovery analysis workflow."""
        # Simulate accuracy trajectory: baseline → drift → recovery
        accuracies = [
            0.85, 0.86, 0.85, 0.84,  # Pre-drift baseline (rounds 0-3)
            0.70,  # Drift injection (round 4)
            0.72, 0.75, 0.78, 0.82, 0.84, 0.85  # Recovery (rounds 5-10)
        ]

        # Find stabilization point
        stabilization_round = find_stabilization_point(
            accuracies,
            start_index=5,
            threshold=0.01,
            window_size=3
        )

        assert stabilization_round > 4  # Should stabilize after drift
        assert stabilization_round <= len(accuracies) - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
