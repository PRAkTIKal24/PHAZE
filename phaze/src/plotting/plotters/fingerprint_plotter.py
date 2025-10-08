"""Plotter for fingerprinting algorithm performance analysis."""

from typing import Any, Dict, List

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from ..base_plotter import BasePlotter
from ..statistical_utils import StatisticalUtils


class FingerprintPlotter(BasePlotter):
    """Plotter for fingerprinting algorithm benchmark results.

    Generates comprehensive performance analysis plots for cryptographic primitives
    including time complexity, memory consumption, throughput, and comparative analysis.
    """

    def get_supported_plot_types(self) -> List[str]:
        """Get list of plot types supported by this plotter."""
        return [
            "time_complexity",
            "memory_usage",
            "throughput",
            "algorithm_comparison",
            "performance_distribution",
            "scaling_analysis"
        ]

    def get_required_data_fields(self) -> List[str]:
        """Get list of required data fields for this plotter."""
        return ["crypto_results"]

    def generate_plots(self, data: Dict[str, Any]) -> List[matplotlib.figure.Figure]:
        """Generate fingerprinting performance plots.

        Args:
            data: Benchmark data containing crypto_results

        Returns:
            List of matplotlib Figure objects
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data for FingerprintPlotter")

        crypto_results = data["crypto_results"]

        # Filter and organize data by algorithm
        fingerprint_data = self._organize_fingerprint_data(crypto_results)

        if not fingerprint_data:
            raise ValueError("No fingerprinting data found in crypto_results")

        figures = []

        # Generate each plot type
        figures.append(self._plot_time_complexity(fingerprint_data))
        figures.append(self._plot_memory_usage(fingerprint_data))
        figures.append(self._plot_throughput(fingerprint_data))
        figures.append(self._plot_algorithm_comparison(fingerprint_data))
        figures.append(self._plot_performance_distribution(fingerprint_data))
        figures.append(self._plot_scaling_analysis(fingerprint_data))

        return figures

    def _organize_fingerprint_data(
        self, crypto_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, List[Any]]]:
        """Organize crypto results by algorithm and operation.

        Args:
            crypto_results: List of CryptoBenchmarkResult dictionaries

        Returns:
            Nested dictionary: {algorithm: {metric: [values]}}
        """
        organized_data = {}

        for result in crypto_results:
            if not isinstance(result, dict):
                continue

            algo_name = result.get("primitive_name", "unknown")
            operation = result.get("operation", "default")

            # Create unique key for algorithm-operation combination
            key = f"{algo_name}_{operation}" if operation != "default" else algo_name

            if key not in organized_data:
                organized_data[key] = {
                    "input_sizes": [],
                    "execution_times": [],
                    "memory_usage": [],
                    "throughput": [],
                    "success_flags": []
                }

            # Only include successful results
            if result.get("success", False):
                organized_data[key]["input_sizes"].append(
                    result.get("input_size", 0)
                )
                organized_data[key]["execution_times"].append(
                    result.get("execution_time", 0)
                )
                organized_data[key]["memory_usage"].append(
                    result.get("memory_usage_mb", 0)
                )
                organized_data[key]["throughput"].append(
                    result.get("throughput_ops_per_sec", 0)
                )
                organized_data[key]["success_flags"].append(True)

        # Remove algorithms with no successful results
        organized_data = {k: v for k, v in organized_data.items() if v["input_sizes"]}

        return organized_data

    def _plot_time_complexity(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Plot execution time vs input size for different algorithms."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        colors = self.config.get_algorithm_colors()

        for i, (algo_name, algo_data) in enumerate(data.items()):
            input_sizes = algo_data["input_sizes"]
            exec_times = algo_data["execution_times"]

            if not input_sizes or not exec_times:
                continue

            # Group by input size and calculate statistics
            size_groups = {}
            for size, time in zip(input_sizes, exec_times, strict=False):
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(time)

            sizes = sorted(size_groups.keys())
            means, errors = StatisticalUtils.calculate_error_bars(
                [size_groups[size] for size in sizes]
            )

            # Get color for this algorithm
            algo_key = algo_name.split('_')[0].lower()  # Extract base algorithm name
            color = colors.get(
                algo_key,
                self.config.primary_colors[i % len(self.config.primary_colors)]
            )

            ax.errorbar(
                sizes, means, yerr=errors,
                marker='o', linewidth=2, markersize=6,
                label=algo_name.replace('_', ' ').title(),
                color=color, capsize=5
            )

        ax.set_xlabel('Input Size (bytes)')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Fingerprinting Algorithm Time Complexity')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_memory_usage(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Plot memory usage vs input size for different algorithms."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        colors = self.config.get_algorithm_colors()

        for i, (algo_name, algo_data) in enumerate(data.items()):
            input_sizes = algo_data["input_sizes"]
            memory_usage = algo_data["memory_usage"]

            if not input_sizes or not memory_usage:
                continue

            # Group by input size and calculate statistics
            size_groups = {}
            for size, mem in zip(input_sizes, memory_usage, strict=False):
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(mem)

            sizes = sorted(size_groups.keys())
            means, errors = StatisticalUtils.calculate_error_bars(
                [size_groups[size] for size in sizes]
            )

            # Get color for this algorithm
            algo_key = algo_name.split('_')[0].lower()
            color = colors.get(algo_key, self.config.primary_colors[i % len(self.config.primary_colors)])

            ax.errorbar(
                sizes, means, yerr=errors,
                marker='s', linewidth=2, markersize=6,
                label=algo_name.replace('_', ' ').title(),
                color=color, capsize=5
            )

        ax.set_xlabel('Input Size (bytes)')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Fingerprinting Algorithm Memory Consumption')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_throughput(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Plot throughput vs input size for different algorithms."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        colors = self.config.get_algorithm_colors()

        for i, (algo_name, algo_data) in enumerate(data.items()):
            input_sizes = algo_data["input_sizes"]
            throughput = algo_data["throughput"]

            if not input_sizes or not throughput:
                continue

            # Group by input size and calculate statistics
            size_groups = {}
            for size, tput in zip(input_sizes, throughput, strict=False):
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(tput)

            sizes = sorted(size_groups.keys())
            means, errors = StatisticalUtils.calculate_error_bars(
                [size_groups[size] for size in sizes]
            )

            # Get color for this algorithm
            algo_key = algo_name.split('_')[0].lower()
            color = colors.get(algo_key, self.config.primary_colors[i % len(self.config.primary_colors)])

            ax.errorbar(
                sizes, means, yerr=errors,
                marker='^', linewidth=2, markersize=6,
                label=algo_name.replace('_', ' ').title(),
                color=color, capsize=5
            )

        ax.set_xlabel('Input Size (bytes)')
        ax.set_ylabel('Throughput (operations/sec)')
        ax.set_title('Fingerprinting Algorithm Throughput')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_algorithm_comparison(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Create a comparison matrix of algorithm performance."""
        algorithms = list(data.keys())
        metrics = [
            'Mean Execution Time (s)',
            'Mean Memory (MB)',
            'Mean Throughput (ops/s)'
        ]

        # Calculate mean values for each algorithm
        comparison_data = []
        for algo_name in algorithms:
            algo_data = data[algo_name]

            mean_time = (
                np.mean(algo_data["execution_times"])
                if algo_data["execution_times"] else 0
            )
            mean_memory = (
                np.mean(algo_data["memory_usage"])
                if algo_data["memory_usage"] else 0
            )
            mean_throughput = (
                np.mean(algo_data["throughput"])
                if algo_data["throughput"] else 0
            )

            comparison_data.append([mean_time, mean_memory, mean_throughput])

        # Create grouped bar chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = self.config.get_color_cycle(len(algorithms))

        x_pos = np.arange(len(algorithms))

        for i, (metric, ax) in enumerate(zip(metrics, axes, strict=False)):
            values = [row[i] for row in comparison_data]

            bars = ax.bar(
                x_pos, values, color=colors, alpha=0.7,
                edgecolor='black', linewidth=0.5
            )

            # Add value labels on bars
            for bar, value in zip(bars, values, strict=False):
                height = bar.get_height()
                ax.annotate(f'{value:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

            ax.set_ylabel(metric)
            ax.set_title(f'Algorithm {metric}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([name.replace('_', ' ').title() for name in algorithms],
                              rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Fingerprinting Algorithm Performance Comparison')
        plt.tight_layout()
        return fig

    def _plot_performance_distribution(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Plot distribution of execution times for different algorithms."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        algorithms = list(data.keys())
        exec_time_data = [
            data[algo]["execution_times"] for algo in algorithms
            if data[algo]["execution_times"]
        ]
        labels = [
            algo.replace('_', ' ').title() for algo in algorithms
            if data[algo]["execution_times"]
        ]

        if not exec_time_data:
            # Create empty plot with message
            ax.text(0.5, 0.5, 'No execution time data available',
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('Execution Time Distribution')
            return fig

        # Create box plots
        box_plot = ax.boxplot(exec_time_data, labels=labels, patch_artist=True)

        # Color the boxes
        colors = self.config.get_color_cycle(len(exec_time_data))
        for patch, color in zip(box_plot['boxes'], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Execution Time Distribution by Algorithm')
        ax.set_yscale('log')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def _plot_scaling_analysis(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Analyze how algorithms scale with input size."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        colors = self.config.get_algorithm_colors()

        for i, (algo_name, algo_data) in enumerate(data.items()):
            input_sizes = algo_data["input_sizes"]
            exec_times = algo_data["execution_times"]
            throughput = algo_data["throughput"]

            if len(input_sizes) < 2:  # Need at least 2 points for scaling analysis
                continue

            # Sort by input size
            sorted_data = sorted(zip(input_sizes, exec_times, throughput, strict=False))
            sizes, times, tput = zip(*sorted_data, strict=False)

            # Get color for this algorithm
            algo_key = algo_name.split('_')[0].lower()
            color = colors.get(algo_key, self.config.primary_colors[i % len(self.config.primary_colors)])

            # Plot scaling of execution time
            ax1.plot(sizes, times, marker='o', linewidth=2, markersize=6,
                    label=algo_name.replace('_', ' ').title(), color=color)

            # Plot scaling of throughput
            ax2.plot(sizes, tput, marker='s', linewidth=2, markersize=6,
                    label=algo_name.replace('_', ' ').title(), color=color)

        # Configure first subplot (execution time scaling)
        ax1.set_xlabel('Input Size (bytes)')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Scaling')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Configure second subplot (throughput scaling)
        ax2.set_xlabel('Input Size (bytes)')
        ax2.set_ylabel('Throughput (operations/sec)')
        ax2.set_title('Throughput Scaling')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Algorithm Scaling Analysis')
        plt.tight_layout()
        return fig
