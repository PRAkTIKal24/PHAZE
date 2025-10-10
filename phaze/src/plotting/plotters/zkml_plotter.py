"""Plotter for zkML framework performance analysis."""

from typing import Any, Dict, List

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from ..base_plotter import BasePlotter
from ..statistical_utils import StatisticalUtils


class ZKMLPlotter(BasePlotter):
    """Plotter for zkML framework benchmark results.

    Generates comprehensive performance analysis plots for zero-knowledge ML frameworks
    including proof generation time, verification time, memory usage, and comparative
    analysis.
    """

    def get_supported_plot_types(self) -> List[str]:
        """Get list of plot types supported by this plotter."""
        return [
            "proof_generation_time",
            "verification_time", 
            "memory_usage",
            "cpu_gpu_memory_breakdown",
            "framework_comparison",
            "proof_size_analysis",
            "performance_matrix",
            "scaling_analysis",
        ]

    def get_required_data_fields(self) -> List[str]:
        """Get list of required data fields for this plotter."""
        return ["zkml_results"]

    def generate_plots(self, data: Dict[str, Any]) -> List[matplotlib.figure.Figure]:
        """Generate zkML framework performance plots.

        Args:
            data: Benchmark data containing zkml_results

        Returns:
            List of matplotlib Figure objects
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data for ZKMLPlotter")

        zkml_results = data["zkml_results"]

        # Organize data by framework and complexity
        framework_data = self._organize_zkml_data(zkml_results)

        if not framework_data:
            raise ValueError("No zkML framework data found in zkml_results")

        figures = []

        # Generate each plot type
        figures.append(self._plot_proof_generation_time(framework_data))
        figures.append(self._plot_verification_time(framework_data))
        figures.append(self._plot_memory_usage(framework_data))
        figures.append(self._plot_cpu_gpu_memory_breakdown(framework_data))
        figures.append(self._plot_framework_comparison(framework_data))
        figures.append(self._plot_proof_size_analysis(framework_data))
        figures.append(self._plot_performance_matrix(framework_data))
        figures.append(self._plot_scaling_analysis(framework_data))

        return figures

    def _organize_zkml_data(
        self, zkml_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, List[Any]]]:
        """Organize zkML results by framework.

        Args:
            zkml_results: List of BenchmarkResult dictionaries

        Returns:
            Nested dictionary: {framework: {metric: [values]}}
        """
        organized_data = {}

        for result in zkml_results:
            if not isinstance(result, dict):
                continue

            framework = result.get("framework", "unknown")

            if framework not in organized_data:
                organized_data[framework] = {
                    "complexities": [],
                    "input_sizes": [],
                    "model_parameters": [],
                    "setup_times": [],
                    "proof_times": [],
                    "verification_times": [],
                    "memory_usage": [],
                    "cpu_memory": [],
                    "gpu_memory": [],
                    "device_types": [],
                    "proof_sizes": [],
                    "success_flags": [],
                    "architectures": [],
                    "total_times": [],
                }

            # Only include successful results
            if result.get("success", False):
                organized_data[framework]["complexities"].append(
                    result.get("complexity", "unknown")
                )
                organized_data[framework]["input_sizes"].append(
                    result.get("input_size", 0)
                )
                organized_data[framework]["model_parameters"].append(
                    result.get("model_parameters", 0)
                )
                organized_data[framework]["setup_times"].append(
                    result.get("setup_time", 0)
                )
                organized_data[framework]["proof_times"].append(
                    result.get("proof_time", 0)
                )
                organized_data[framework]["verification_times"].append(
                    result.get("verification_time", 0)
                )
                organized_data[framework]["memory_usage"].append(
                    result.get("memory_usage_mb", 0)
                )
                organized_data[framework]["cpu_memory"].append(
                    result.get("cpu_memory_mb", 0)
                )
                organized_data[framework]["gpu_memory"].append(
                    result.get("gpu_memory_mb", 0)
                )
                organized_data[framework]["device_types"].append(
                    result.get("device_type", "cpu")
                )
                organized_data[framework]["proof_sizes"].append(
                    result.get("proof_size_bytes", 0)
                )
                organized_data[framework]["success_flags"].append(True)
                organized_data[framework]["architectures"].append(
                    result.get("architecture", "unknown")
                )
                organized_data[framework]["total_times"].append(
                    result.get("total_time", 0)
                )

        # Remove frameworks with no successful results
        organized_data = {k: v for k, v in organized_data.items() if v["proof_times"]}

        return organized_data

    def _plot_proof_generation_time(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Plot proof generation time vs model complexity for different frameworks."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        colors = self.config.get_framework_colors()

        for i, (framework, framework_data) in enumerate(data.items()):
            complexities = framework_data["complexities"]
            proof_times = framework_data["proof_times"]
            model_params = framework_data["model_parameters"]

            if not complexities or not proof_times:
                continue

            # Group by complexity and calculate statistics
            complexity_groups = {}
            param_groups = {}
            for complexity, time, params in zip(
                complexities, proof_times, model_params, strict=False
            ):
                if complexity not in complexity_groups:
                    complexity_groups[complexity] = []
                    param_groups[complexity] = []
                complexity_groups[complexity].append(time)
                param_groups[complexity].append(params)

            # Sort complexities in logical order
            complexity_order = ["minimal", "light", "medium", "heavy", "extreme"]
            sorted_complexities = [
                c for c in complexity_order if c in complexity_groups
            ]

            if not sorted_complexities:
                sorted_complexities = sorted(complexity_groups.keys())

            means, errors = StatisticalUtils.calculate_error_bars(
                [complexity_groups[c] for c in sorted_complexities]
            )

            # Use model parameters as x-axis for better scaling visualization
            x_values = [np.mean(param_groups[c]) for c in sorted_complexities]

            # Get color for this framework
            color = colors.get(
                framework,
                self.config.primary_colors[i % len(self.config.primary_colors)],
            )

            ax.errorbar(
                x_values,
                means,
                yerr=errors,
                marker="o",
                linewidth=2,
                markersize=8,
                label=framework.replace("_", " ").title(),
                color=color,
                capsize=5,
            )

        ax.set_xlabel("Model Parameters")
        ax.set_ylabel("Proof Generation Time (seconds)")
        ax.set_title("zkML Framework Proof Generation Performance")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_verification_time(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Plot verification time vs model complexity for different frameworks."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        colors = self.config.get_framework_colors()

        for i, (framework, framework_data) in enumerate(data.items()):
            complexities = framework_data["complexities"]
            verification_times = framework_data["verification_times"]
            model_params = framework_data["model_parameters"]

            if not complexities or not verification_times:
                continue

            # Group by complexity and calculate statistics
            complexity_groups = {}
            param_groups = {}
            for complexity, time, params in zip(
                complexities, verification_times, model_params, strict=False
            ):
                if complexity not in complexity_groups:
                    complexity_groups[complexity] = []
                    param_groups[complexity] = []
                complexity_groups[complexity].append(time)
                param_groups[complexity].append(params)

            # Sort complexities in logical order
            complexity_order = ["minimal", "light", "medium", "heavy", "extreme"]
            sorted_complexities = [
                c for c in complexity_order if c in complexity_groups
            ]

            if not sorted_complexities:
                sorted_complexities = sorted(complexity_groups.keys())

            means, errors = StatisticalUtils.calculate_error_bars(
                [complexity_groups[c] for c in sorted_complexities]
            )

            # Use model parameters as x-axis
            x_values = [np.mean(param_groups[c]) for c in sorted_complexities]

            # Get color for this framework
            color = colors.get(
                framework,
                self.config.primary_colors[i % len(self.config.primary_colors)],
            )

            ax.errorbar(
                x_values,
                means,
                yerr=errors,
                marker="s",
                linewidth=2,
                markersize=8,
                label=framework.replace("_", " ").title(),
                color=color,
                capsize=5,
            )

        ax.set_xlabel("Model Parameters")
        ax.set_ylabel("Verification Time (seconds)")
        ax.set_title("zkML Framework Verification Performance")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_memory_usage(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Plot memory usage vs model complexity for different frameworks."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        colors = self.config.get_framework_colors()

        for i, (framework, framework_data) in enumerate(data.items()):
            complexities = framework_data["complexities"]
            memory_usage = framework_data["memory_usage"]
            model_params = framework_data["model_parameters"]

            if not complexities or not memory_usage:
                continue

            # Group by complexity and calculate statistics
            complexity_groups = {}
            param_groups = {}
            for complexity, mem, params in zip(
                complexities, memory_usage, model_params, strict=False
            ):
                if complexity not in complexity_groups:
                    complexity_groups[complexity] = []
                    param_groups[complexity] = []
                complexity_groups[complexity].append(mem)
                param_groups[complexity].append(params)

            # Sort complexities in logical order
            complexity_order = ["minimal", "light", "medium", "heavy", "extreme"]
            sorted_complexities = [
                c for c in complexity_order if c in complexity_groups
            ]

            if not sorted_complexities:
                sorted_complexities = sorted(complexity_groups.keys())

            means, errors = StatisticalUtils.calculate_error_bars(
                [complexity_groups[c] for c in sorted_complexities]
            )

            # Use model parameters as x-axis
            x_values = [np.mean(param_groups[c]) for c in sorted_complexities]

            # Get color for this framework
            color = colors.get(
                framework,
                self.config.primary_colors[i % len(self.config.primary_colors)],
            )

            ax.errorbar(
                x_values,
                means,
                yerr=errors,
                marker="^",
                linewidth=2,
                markersize=8,
                label=framework.replace("_", " ").title(),
                color=color,
                capsize=5,
            )

        ax.set_xlabel("Model Parameters")
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title("zkML Framework Memory Consumption")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_cpu_gpu_memory_breakdown(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Plot CPU vs GPU memory usage with overlaid breakdown."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.figure_size[0], self.config.figure_size[1] * 1.5))
        colors = self.config.get_framework_colors()

        for i, (framework, framework_data) in enumerate(data.items()):
            complexities = framework_data["complexities"]
            cpu_memory = framework_data["cpu_memory"]
            gpu_memory = framework_data["gpu_memory"]
            device_types = framework_data["device_types"]

            if not complexities or not cpu_memory:
                continue

            # Group by complexity and calculate statistics
            complexity_groups_cpu = {}
            complexity_groups_gpu = {}
            device_type_groups = {}
            
            for complexity, cpu_mem, gpu_mem, device_type in zip(
                complexities, cpu_memory, gpu_memory, device_types, strict=False
            ):
                if complexity not in complexity_groups_cpu:
                    complexity_groups_cpu[complexity] = []
                    complexity_groups_gpu[complexity] = []
                    device_type_groups[complexity] = []
                complexity_groups_cpu[complexity].append(cpu_mem)
                complexity_groups_gpu[complexity].append(gpu_mem)
                device_type_groups[complexity].append(device_type)

            # Sort complexities in logical order
            complexity_order = ["minimal", "light", "medium", "heavy", "extreme"]
            sorted_complexities = [
                c for c in complexity_order if c in complexity_groups_cpu
            ]

            if not sorted_complexities:
                sorted_complexities = sorted(complexity_groups_cpu.keys())

            cpu_means, cpu_errors = StatisticalUtils.calculate_error_bars(
                [complexity_groups_cpu[c] for c in sorted_complexities]
            )
            gpu_means, gpu_errors = StatisticalUtils.calculate_error_bars(
                [complexity_groups_gpu[c] for c in sorted_complexities]
            )

            # Get color for this framework
            color = colors.get(
                framework,
                self.config.primary_colors[i % len(self.config.primary_colors)],
            )

            x_pos = np.arange(len(sorted_complexities))

            # Plot 1: Overlaid CPU/GPU memory
            ax1.errorbar(
                x_pos - 0.1 + i * 0.05,
                cpu_means,
                yerr=cpu_errors,
                marker="o",
                linewidth=2,
                markersize=6,
                label=f"{framework.replace('_', ' ').title()} (CPU)",
                color=color,
                alpha=0.7,
                capsize=3,
            )
            
            # Only plot GPU if there's actual GPU usage
            if any(gpu_means):
                ax1.errorbar(
                    x_pos + 0.1 + i * 0.05,
                    gpu_means,
                    yerr=gpu_errors,
                    marker="^",
                    linewidth=2,
                    markersize=6,
                    label=f"{framework.replace('_', ' ').title()} (GPU)",
                    color=color,
                    alpha=1.0,
                    linestyle="--",
                    capsize=3,
                )

            # Plot 2: Stacked bar chart
            ax2.bar(
                x_pos + i * 0.2,
                cpu_means,
                width=0.15,
                label=f"{framework.replace('_', ' ').title()} (CPU)" if i == 0 else "",
                color=color,
                alpha=0.7,
            )
            ax2.bar(
                x_pos + i * 0.2,
                gpu_means,
                width=0.15,
                bottom=cpu_means,
                label=f"{framework.replace('_', ' ').title()} (GPU)" if i == 0 else "",
                color=color,
                alpha=1.0,
                hatch="//",
            )

        # Configure plot 1 (overlaid)
        ax1.set_xlabel("Model Complexity")
        ax1.set_ylabel("Memory Usage (MB)")
        ax1.set_title("CPU vs GPU Memory Usage by Framework")
        ax1.set_xticks(range(len(sorted_complexities)))
        ax1.set_xticklabels([c.title() for c in sorted_complexities])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Configure plot 2 (stacked)
        ax2.set_xlabel("Model Complexity")
        ax2.set_ylabel("Total Memory Usage (MB)")
        ax2.set_title("Stacked CPU/GPU Memory Breakdown")
        ax2.set_xticks(range(len(sorted_complexities)))
        ax2.set_xticklabels([c.title() for c in sorted_complexities])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_framework_comparison(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Create a comparison of framework performance across metrics."""
        frameworks = list(data.keys())
        metrics = [
            "Mean Setup Time (s)",
            "Mean Proof Time (s)",
            "Mean Verification Time (s)",
            "Mean Memory (MB)",
        ]

        # Calculate mean values for each framework
        comparison_data = []
        for framework in frameworks:
            framework_data = data[framework]

            mean_setup = (
                np.mean(framework_data["setup_times"])
                if framework_data["setup_times"]
                else 0
            )
            mean_proof = (
                np.mean(framework_data["proof_times"])
                if framework_data["proof_times"]
                else 0
            )
            mean_verification = (
                np.mean(framework_data["verification_times"])
                if framework_data["verification_times"]
                else 0
            )
            mean_memory = (
                np.mean(framework_data["memory_usage"])
                if framework_data["memory_usage"]
                else 0
            )

            comparison_data.append(
                [mean_setup, mean_proof, mean_verification, mean_memory]
            )

        # Create grouped bar chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        colors = self.config.get_color_cycle(len(frameworks))

        x_pos = np.arange(len(frameworks))

        for i, (metric, ax) in enumerate(zip(metrics, axes, strict=False)):
            values = [row[i] for row in comparison_data]

            bars = ax.bar(
                x_pos, values, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5
            )

            # Add value labels on bars
            for bar, value in zip(bars, values, strict=False):
                height = bar.get_height()
                ax.annotate(
                    f"{value:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            ax.set_ylabel(metric)
            ax.set_title(f"Framework {metric}")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [name.replace("_", " ").title() for name in frameworks],
                rotation=45,
                ha="right",
            )
            ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle("zkML Framework Performance Comparison")
        plt.tight_layout()
        return fig

    def _plot_proof_size_analysis(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Plot proof size vs model complexity for different frameworks."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        colors = self.config.get_framework_colors()

        for i, (framework, framework_data) in enumerate(data.items()):
            model_params = framework_data["model_parameters"]
            proof_sizes = framework_data["proof_sizes"]

            if not model_params or not proof_sizes:
                continue

            # Filter out zero proof sizes (may indicate missing data)
            valid_data = [
                (p, s) for p, s in zip(model_params, proof_sizes, strict=False) if s > 0
            ]
            if not valid_data:
                continue

            params, sizes = zip(*valid_data, strict=False)

            # Get color for this framework
            color = colors.get(
                framework,
                self.config.primary_colors[i % len(self.config.primary_colors)],
            )

            ax.scatter(
                params,
                sizes,
                label=framework.replace("_", " ").title(),
                color=color,
                alpha=0.7,
                s=50,
            )

            # Add trend line if enough points
            if len(params) > 1:
                # Fit log-log linear trend
                log_params = np.log10(params)
                log_sizes = np.log10(sizes)
                coeffs = np.polyfit(log_params, log_sizes, 1)

                # Generate trend line
                x_trend = np.logspace(np.log10(min(params)), np.log10(max(params)), 50)
                y_trend = 10 ** (coeffs[0] * np.log10(x_trend) + coeffs[1])

                ax.plot(x_trend, y_trend, "--", color=color, alpha=0.8, linewidth=2)

        ax.set_xlabel("Model Parameters")
        ax.set_ylabel("Proof Size (bytes)")
        ax.set_title("zkML Framework Proof Size Analysis")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_performance_matrix(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Create a heatmap matrix of framework performance across complexities."""
        frameworks = list(data.keys())
        complexities = set()

        # Collect all complexities
        for framework_data in data.values():
            complexities.update(framework_data["complexities"])

        complexity_order = ["minimal", "light", "medium", "heavy", "extreme"]
        sorted_complexities = [c for c in complexity_order if c in complexities]
        if not sorted_complexities:
            sorted_complexities = sorted(complexities)

        # Create matrix of mean proof times
        matrix_data = np.zeros((len(frameworks), len(sorted_complexities)))

        for i, framework in enumerate(frameworks):
            framework_data = data[framework]

            for j, complexity in enumerate(sorted_complexities):
                # Find proof times for this framework-complexity combination
                proof_times = []
                for k, comp in enumerate(framework_data["complexities"]):
                    if comp == complexity:
                        proof_times.append(framework_data["proof_times"][k])

                if proof_times:
                    matrix_data[i, j] = np.mean(proof_times)
                else:
                    matrix_data[i, j] = np.nan

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(matrix_data, cmap="YlOrRd", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(sorted_complexities)))
        ax.set_yticks(np.arange(len(frameworks)))
        ax.set_xticklabels([c.title() for c in sorted_complexities])
        ax.set_yticklabels([f.replace("_", " ").title() for f in frameworks])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean Proof Generation Time (seconds)")

        # Add text annotations
        for i in range(len(frameworks)):
            for j in range(len(sorted_complexities)):
                if not np.isnan(matrix_data[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix_data[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=9,
                    )

        ax.set_title(
            "zkML Framework Performance Matrix\n(Proof Generation Time by Framework and Complexity)"
        )
        ax.set_xlabel("Model Complexity")
        ax.set_ylabel("zkML Framework")

        plt.tight_layout()
        return fig

    def _plot_scaling_analysis(
        self, data: Dict[str, Dict[str, List[Any]]]
    ) -> matplotlib.figure.Figure:
        """Analyze how frameworks scale with model complexity."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        colors = self.config.get_framework_colors()

        for i, (framework, framework_data) in enumerate(data.items()):
            model_params = framework_data["model_parameters"]
            proof_times = framework_data["proof_times"]
            verification_times = framework_data["verification_times"]

            if len(model_params) < 2:  # Need at least 2 points for scaling analysis
                continue

            # Sort by model parameters
            sorted_data = sorted(
                zip(model_params, proof_times, verification_times, strict=False)
            )
            params, p_times, v_times = zip(*sorted_data, strict=False)

            # Get color for this framework
            color = colors.get(
                framework,
                self.config.primary_colors[i % len(self.config.primary_colors)],
            )

            # Plot scaling of proof generation time
            ax1.plot(
                params,
                p_times,
                marker="o",
                linewidth=2,
                markersize=6,
                label=framework.replace("_", " ").title(),
                color=color,
            )

            # Plot scaling of verification time
            ax2.plot(
                params,
                v_times,
                marker="s",
                linewidth=2,
                markersize=6,
                label=framework.replace("_", " ").title(),
                color=color,
            )

        # Configure first subplot (proof time scaling)
        ax1.set_xlabel("Model Parameters")
        ax1.set_ylabel("Proof Generation Time (seconds)")
        ax1.set_title("Proof Generation Scaling")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Configure second subplot (verification time scaling)
        ax2.set_xlabel("Model Parameters")
        ax2.set_ylabel("Verification Time (seconds)")
        ax2.set_title("Verification Time Scaling")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle("zkML Framework Scaling Analysis")
        plt.tight_layout()
        return fig
