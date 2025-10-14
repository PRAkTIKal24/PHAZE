"""Main plotting orchestrator for PHAZE."""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_plotter import BasePlotter
from .plot_config import PlotConfig
from .plot_registry import PlotRegistry, get_global_registry

logger = logging.getLogger(__name__)


class PHAZEPlotSuite:
    """Main orchestrator for PHAZE plotting operations.

    Coordinates multiple plotters to generate comprehensive benchmark visualizations.
    """

    def __init__(
        self,
        config: Optional[PlotConfig] = None,
        registry: Optional[PlotRegistry] = None,
    ):
        """Initialize the plot suite.

        Args:
            config: Plot configuration. If None, uses default.
            registry: Plot registry. If None, uses global registry.
        """
        self.config = config or PlotConfig()
        self.registry = registry or get_global_registry()
        self.plotters: Dict[str, BasePlotter] = {}

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the plot suite."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def register_plotter(self, name: str, plotter: BasePlotter) -> None:
        """Register a custom plotter instance.

        Args:
            name: Name for the plotter
            plotter: Plotter instance
        """
        self.plotters[name] = plotter
        logger.info(f"Registered custom plotter: {name}")

    def get_available_plot_types(
        self, data: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Get available plot types, optionally filtered by data compatibility.

        Args:
            data: Optional data to check compatibility against

        Returns:
            List of available plot type names
        """
        if data is None:
            return self.registry.list_plot_types()

        # Get data fields for compatibility check
        data_fields = list(data.keys())
        return self.registry.get_compatible_plot_types(data_fields)

    def generate_plots(
        self,
        data: Dict[str, Any],
        plot_types: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        save_plots: bool = True,
    ) -> Dict[str, Any]:
        """Generate plots from benchmark data.

        Args:
            data: Benchmark data dictionary
            plot_types: List of plot types to generate. If None, generates all compatible.
            output_dir: Directory to save plots. If None and save_plots=True, uses 'plots'.
            save_plots: Whether to save plots to disk

        Returns:
            Dictionary containing generated plots and metadata
        """
        logger.info("Starting plot generation")

        # Determine which plot types to generate
        if plot_types is None:
            plot_types = self.get_available_plot_types(data)
            logger.info(f"Auto-detected {len(plot_types)} compatible plot types")
        else:
            # Validate requested plot types
            available = self.get_available_plot_types(data)
            invalid_types = [pt for pt in plot_types if pt not in available]
            if invalid_types:
                logger.warning(f"Invalid/incompatible plot types: {invalid_types}")
            plot_types = [pt for pt in plot_types if pt in available]

        if not plot_types:
            logger.warning("No compatible plot types found")
            return {
                "plots": {},
                "metadata": {"plot_types": [], "error": "No compatible plot types"},
            }

        # Validate data
        if not self._validate_data(data):
            logger.error("Data validation failed")
            return {
                "plots": {},
                "metadata": {"plot_types": [], "error": "Data validation failed"},
            }

        # Generate plots
        results = {
            "plots": {},
            "metadata": {
                "plot_types": plot_types,
                "config": asdict(self.config),
                "data_summary": self._summarize_data(data),
            },
        }

        for plot_type in plot_types:
            try:
                logger.info(f"Generating {plot_type} plots")

                # Get or create plotter
                if plot_type in self.plotters:
                    plotter = self.plotters[plot_type]
                else:
                    plotter = self.registry.create_plotter(plot_type, self.config)

                # Generate plots
                figures = plotter.generate_plots(data)

                # Save plots if requested
                saved_files = []
                if save_plots and figures:
                    if output_dir is None:
                        output_dir = Path("plots")

                    plot_output_dir = Path(output_dir) / plot_type
                    saved_files = plotter.save_plots(
                        figures, plot_output_dir, prefix=f"{plot_type}_"
                    )
                    logger.info(f"Saved {len(saved_files)} files for {plot_type}")

                results["plots"][plot_type] = {
                    "figures": figures,
                    "saved_files": [str(f) for f in saved_files],
                    "num_plots": len(figures),
                }

            except Exception as e:
                logger.error(f"Failed to generate {plot_type} plots: {e}")
                results["plots"][plot_type] = {
                    "figures": [],
                    "saved_files": [],
                    "num_plots": 0,
                    "error": str(e),
                }

        logger.info(
            f"Plot generation completed. Generated {len(results['plots'])} plot types"
        )
        return results

    def generate_comparative_plots(
        self,
        data: Dict[str, Any],
        components: List[str],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Generate comparative plots across multiple PHAZE components.

        Args:
            data: Benchmark data dictionary
            components: List of components to compare
            output_dir: Directory to save plots

        Returns:
            Dictionary containing generated comparative plots
        """
        logger.info(f"Generating comparative plots for components: {components}")

        # Try to find a comparative plotter
        comparative_types = [
            pt for pt in self.registry.list_plot_types() if "comparative" in pt
        ]

        if not comparative_types:
            logger.warning("No comparative plotter found")
            return {
                "plots": {},
                "metadata": {"error": "No comparative plotter available"},
            }

        # Use the first available comparative plotter
        plot_type = comparative_types[0]

        try:
            plotter = self.registry.create_plotter(plot_type, self.config)

            # Add component filter to data
            filtered_data = data.copy()
            filtered_data["_comparison_components"] = components

            figures = plotter.generate_plots(filtered_data)

            # Save plots
            saved_files = []
            if output_dir and figures:
                plot_output_dir = Path(output_dir) / "comparative"
                saved_files = plotter.save_plots(
                    figures, plot_output_dir, prefix="comparative_"
                )

            return {
                "plots": {
                    "comparative": {
                        "figures": figures,
                        "saved_files": [str(f) for f in saved_files],
                        "num_plots": len(figures),
                        "components": components,
                    }
                },
                "metadata": {"plot_type": plot_type, "components": components},
            }

        except Exception as e:
            logger.error(f"Failed to generate comparative plots: {e}")
            return {
                "plots": {},
                "metadata": {"error": str(e), "components": components},
            }

    def generate_summary_report(
        self,
        plot_results: Dict[str, Any],
        output_file: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate a summary report of plotting results.

        Args:
            plot_results: Results from generate_plots()
            output_file: Optional file to save report to

        Returns:
            Markdown-formatted report string
        """
        report_lines = [
            "# PHAZE Plotting Summary Report",
            "",
            f"**Generated on:** {self._get_timestamp()}",
            f"**Plot Style:** {self.config.style.value}",
            f"**Total Plot Types:** {len(plot_results.get('plots', {}))}",
            "",
        ]

        # Add metadata if available
        metadata = plot_results.get("metadata", {})
        if "data_summary" in metadata:
            data_summary = metadata["data_summary"]
            report_lines.extend(
                [
                    "## Data Summary",
                    "",
                    f"- **Total data points:** {data_summary.get('total_records', 'N/A')}",
                    f"- **Data fields:** {data_summary.get('num_fields', 'N/A')}",
                    f"- **Frameworks tested:** {', '.join(data_summary.get('frameworks', []))}",
                    "",
                ]
            )

        # Add plot type results
        plots = plot_results.get("plots", {})
        if plots:
            report_lines.extend(["## Generated Plots", ""])

            for plot_type, plot_data in plots.items():
                num_plots = plot_data.get("num_plots", 0)
                num_files = len(plot_data.get("saved_files", []))
                error = plot_data.get("error")

                if error:
                    status = f"❌ **Failed:** {error}"
                else:
                    status = (
                        f"✅ **Success:** {num_plots} plots, {num_files} files saved"
                    )

                report_lines.append(f"### {plot_type.title()}")
                report_lines.append(f"{status}")
                report_lines.append("")

                # List saved files
                saved_files = plot_data.get("saved_files", [])
                if saved_files:
                    report_lines.append("**Saved files:**")
                    for file_path in saved_files[:5]:  # Limit to first 5 files
                        report_lines.append(f"- `{file_path}`")
                    if len(saved_files) > 5:
                        report_lines.append(f"- ... and {len(saved_files) - 5} more")
                    report_lines.append("")

        # Add configuration details
        config_dict = metadata.get("config", {})
        if config_dict:
            report_lines.extend(
                [
                    "## Configuration",
                    "",
                    f"- **Style:** {config_dict.get('style', 'N/A')}",
                    f"- **DPI:** {config_dict.get('dpi', 'N/A')}",
                    f"- **Figure size:** {config_dict.get('figure_size', 'N/A')}",
                    f"- **Export formats:** {', '.join(config_dict.get('export_formats', []))}",
                    "",
                ]
            )

        report = "\n".join(report_lines)

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            logger.info(f"Report saved to {output_path}")

        return report

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate that data is suitable for plotting.

        Args:
            data: Data dictionary to validate

        Returns:
            True if data is valid
        """
        if not isinstance(data, dict):
            logger.error("Data must be a dictionary")
            return False

        if not data:
            logger.error("Data dictionary is empty")
            return False

        # Check for common required fields
        common_fields = ["results", "summary", "metadata"]
        if not any(field in data for field in common_fields):
            logger.warning("Data may not contain expected benchmark result fields")

        return True

    def _summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the input data.

        Args:
            data: Data dictionary to summarize

        Returns:
            Summary dictionary
        """
        summary = {"num_fields": len(data), "field_names": list(data.keys())}

        # Try to extract common information
        if "results" in data:
            results = data["results"]
            if isinstance(results, dict):
                summary["total_records"] = len(results)

                # Extract frameworks if available
                frameworks = set()
                for result in results.values():
                    if isinstance(result, dict) and "framework" in result:
                        frameworks.add(result["framework"])
                summary["frameworks"] = sorted(frameworks)

        return summary

    def _get_timestamp(self) -> str:
        """Get current timestamp as string.

        Returns:
            ISO format timestamp
        """
        from datetime import datetime

        return datetime.now().isoformat()
