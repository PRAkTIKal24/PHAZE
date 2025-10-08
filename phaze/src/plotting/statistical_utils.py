"""Statistical utilities for PHAZE plotting."""

import warnings
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


class StatisticalUtils:
    """Utilities for statistical analysis and error calculations."""

    @staticmethod
    def calculate_confidence_interval(
        data: List[float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """Calculate confidence interval for data.

        Args:
            data: List of numerical values
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        if not data:
            return 0.0, 0.0, 0.0

        data_array = np.array(data)
        mean = np.mean(data_array)
        n = len(data_array)

        if n == 1:
            return mean, mean, mean

        # Use t-distribution for small samples
        std_err = stats.sem(data_array)
        t_val = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        margin_error = t_val * std_err

        return mean, mean - margin_error, mean + margin_error

    @staticmethod
    def calculate_error_bars(
        data: List[List[float]],
        error_type: str = "std"
    ) -> Tuple[List[float], List[float]]:
        """Calculate error bars for multiple data series.

        Args:
            data: List of data series (each series is a list of values)
            error_type: Type of error bars ('std', 'sem', 'ci95')

        Returns:
            Tuple of (means, errors)
        """
        means = []
        errors = []

        for series in data:
            if not series:
                means.append(0.0)
                errors.append(0.0)
                continue

            series_array = np.array(series)
            mean = np.mean(series_array)
            means.append(mean)

            if error_type == "std":
                error = np.std(series_array, ddof=1)
            elif error_type == "sem":
                error = stats.sem(series_array)
            elif error_type == "ci95":
                _, lower, upper = StatisticalUtils.calculate_confidence_interval(
                    series, 0.95
                )
                error = max(mean - lower, upper - mean)
            else:
                raise ValueError(f"Unknown error_type: {error_type}")

            errors.append(error)

        return means, errors

    @staticmethod
    def detect_outliers(
        data: List[float],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> List[bool]:
        """Detect outliers in data.

        Args:
            data: List of numerical values
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            List of boolean values indicating outliers
        """
        if not data:
            return []

        data_array = np.array(data)

        if method == "iqr":
            q1, q3 = np.percentile(data_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (data_array < lower_bound) | (data_array > upper_bound)

        elif method == "zscore":
            z_scores = np.abs(stats.zscore(data_array))
            outliers = z_scores > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        return outliers.tolist()

    @staticmethod
    def filter_outliers(
        data: List[float],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> List[float]:
        """Remove outliers from data.

        Args:
            data: List of numerical values
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            List of values with outliers removed
        """
        if not data:
            return []

        outliers = StatisticalUtils.detect_outliers(data, method, threshold)
        return [
            val for val, is_outlier in zip(data, outliers, strict=False)
            if not is_outlier
        ]

    @staticmethod
    def test_normality(data: List[float]) -> Tuple[bool, float]:
        """Test if data follows normal distribution.

        Args:
            data: List of numerical values

        Returns:
            Tuple of (is_normal, p_value)
        """
        if len(data) < 3:
            return True, 1.0  # Assume normal for small samples

        # Use Shapiro-Wilk test for normality
        try:
            statistic, p_value = stats.shapiro(data)
            is_normal = p_value > 0.05  # 5% significance level
            return is_normal, p_value
        except Exception:
            # Fallback: assume normal if test fails
            return True, 1.0

    @staticmethod
    def compare_distributions(
        data1: List[float],
        data2: List[float],
        test: str = "auto"
    ) -> Tuple[bool, float, str]:
        """Compare two distributions for statistical significance.

        Args:
            data1: First dataset
            data2: Second dataset
            test: Statistical test to use ('auto', 'ttest', 'mannwhitney')

        Returns:
            Tuple of (is_significant, p_value, test_used)
        """
        if len(data1) < 3 or len(data2) < 3:
            return False, 1.0, "insufficient_data"

        # Choose test automatically if requested
        if test == "auto":
            normal1, _ = StatisticalUtils.test_normality(data1)
            normal2, _ = StatisticalUtils.test_normality(data2)

            if normal1 and normal2:
                test = "ttest"
            else:
                test = "mannwhitney"

        # Perform the chosen test
        try:
            if test == "ttest":
                statistic, p_value = stats.ttest_ind(data1, data2)
            elif test == "mannwhitney":
                statistic, p_value = stats.mannwhitneyu(
                    data1, data2, alternative='two-sided'
                )
            else:
                raise ValueError(f"Unknown test: {test}")

            is_significant = p_value < 0.05  # 5% significance level
            return is_significant, p_value, test

        except Exception as e:
            warnings.warn(f"Statistical test failed: {e}", stacklevel=2)
            return False, 1.0, "test_failed"

    @staticmethod
    def calculate_effect_size(
        data1: List[float],
        data2: List[float]
    ) -> Tuple[float, str]:
        """Calculate Cohen's d effect size between two groups.

        Args:
            data1: First dataset
            data2: Second dataset

        Returns:
            Tuple of (effect_size, interpretation)
        """
        if not data1 or not data2:
            return 0.0, "no_data"

        array1 = np.array(data1)
        array2 = np.array(data2)

        # Calculate pooled standard deviation
        n1, n2 = len(array1), len(array2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(array1, ddof=1) +
                             (n2 - 1) * np.var(array2, ddof=1)) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0, "no_variance"

        # Calculate Cohen's d
        cohens_d = (np.mean(array1) - np.mean(array2)) / pooled_std

        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return cohens_d, interpretation

    @staticmethod
    def aggregate_trials(
        trial_results: Dict[str, List[float]],
        aggregation: str = "mean"
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate results from multiple trials.

        Args:
            trial_results: Dictionary mapping metric names to lists of trial values
            aggregation: Aggregation method ('mean', 'median', 'robust')

        Returns:
            Dictionary with aggregated statistics for each metric
        """
        aggregated = {}

        for metric_name, values in trial_results.items():
            if not values:
                aggregated[metric_name] = {
                    'value': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
                continue

            values_array = np.array(values)

            if aggregation == "mean":
                central_value = np.mean(values_array)
            elif aggregation == "median":
                central_value = np.median(values_array)
            elif aggregation == "robust":
                # Use median after removing outliers
                filtered_values = StatisticalUtils.filter_outliers(values)
                central_value = (
                    np.median(filtered_values) if filtered_values
                    else np.median(values_array)
                )
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

            aggregated[metric_name] = {
                'value': float(central_value),
                'std': float(np.std(values_array, ddof=1)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'count': len(values)
            }

        return aggregated
