import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional


def get_available_plot_types():
    """Get list of available plot types."""
    try:
        from .src.plotting.plot_registry import get_global_registry

        registry = get_global_registry()
        return registry.list_plot_types()
    except ImportError:
        return []


def get_plot_categories():
    """Get list of available plot categories."""
    try:
        from .src.plotting.plot_registry import get_global_registry

        registry = get_global_registry()
        return registry.list_categories()
    except ImportError:
        return []


async def run_training_command(
    config_file: Optional[str] = None,
    output_dir: str = "benchmark_results",
    architectures: Optional[List[str]] = None,
    complexities: Optional[List[str]] = None,
    epochs: Optional[int] = None,
    quick: bool = False,
    verbose: bool = False,
    model_path: Optional[str] = None,
) -> int:
    """Run training and benchmarking pipeline."""
    try:
        from .src.training_config import PHAZEConfig, load_config
        from .src.training_orchestrator import run_full_pipeline, run_pipeline_with_pretrained_model

        # Load configuration
        if config_file:
            config = PHAZEConfig.from_file(config_file)
        else:
            config = load_config()

        # Override configuration with command line arguments
        if output_dir != "benchmark_results":
            config.output.output_dir = output_dir

        if architectures:
            config.model.architectures = architectures

        if complexities:
            config.model.complexities = complexities

        if epochs is not None:
            config.training.epochs = epochs

        # Quick mode adjustments
        if quick:
            config.training.epochs = 1
            config.dataset.dataset_size = 500
            config.experiment.benchmark_iterations = 3
            config.experiment.seeds = [42, 123]  # Fewer seeds
            if not architectures:
                config.model.architectures = ["simple", "conv"]  # Subset
            if not complexities:
                config.model.complexities = ["minimal", "light"]  # Subset

        config.training.verbose = verbose

        # Validate configuration
        warnings = config.validate()
        if warnings:
            print("Configuration warnings:")
            for warning in warnings:
                print(f"  - {warning}")
            print()

        if verbose:
            print(f"Configuration loaded: {config.project_name} v{config.version}")
            print(f"Output directory: {config.output.output_dir}")
            print(f"Architectures: {config.model.architectures}")
            print(f"Complexities: {config.model.complexities}")
            print(f"Training epochs: {config.training.epochs}")
            print(f"Dataset size: {config.dataset.dataset_size}")
            print()

        # Run pipeline
        if model_path:
            print(f"Using pre-trained model: {model_path}")
            # Load and process pre-trained model
            from .src.model_architectures import load_pretrained_model
            model_info = load_pretrained_model(model_path)
            
            # Run zkML and crypto benchmarking with plotting
            print("Running zkML and crypto benchmarking with pre-trained model...")
            results = await run_pipeline_with_pretrained_model(config, model_info)
            print("Pipeline completed successfully!")
        else:
            print("Running complete pipeline...")
            results = await run_full_pipeline(config)
            print("Pipeline completed successfully!")

        return 0

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


async def run_plotting_command(
    plot_types: Optional[List[str]] = None,
    components: Optional[List[str]] = None,
    output_dir: str = "plots",
    data_file: Optional[str] = None,
    style: str = "neurips",
    formats: Optional[List[str]] = None,
    trials: int = 10,
    verbose: bool = False,
    frameworks: Optional[List[str]] = None,
    algorithms: Optional[List[str]] = None,
    complexities_plot: Optional[List[str]] = None,
) -> int:
    """Run plotting command with specified parameters."""
    try:
        from .src.comprehensive_benchmark import ComprehensiveBenchmarkSuite
        from .src.plotting import PHAZEPlotSuite, PlotConfig, PlotStyle

        # Create plot configuration
        style_enum = PlotStyle.NEURIPS
        if style.lower() == "publication":
            style_enum = PlotStyle.PUBLICATION
        elif style.lower() == "presentation":
            style_enum = PlotStyle.PRESENTATION
        elif style.lower() == "web":
            style_enum = PlotStyle.WEB

        config = PlotConfig(
            style=style_enum,
            export_formats=formats or ["png", "pdf"],
            num_trials=trials,
        )

        # Initialize plot suite
        plot_suite = PHAZEPlotSuite(config)

        if verbose:
            print(f"Plot suite initialized with style: {style}")
            print(f"Output directory: {output_dir}")
            print(f"Export formats: {config.export_formats}")

        # Get data for plotting
        if data_file:
            # Load data from file
            if verbose:
                print(f"Loading data from: {data_file}")

            data_path = Path(data_file)
            if not data_path.exists():
                print(f"Error: Data file not found: {data_file}", file=sys.stderr)
                return 1

            with open(data_path, "r") as f:
                data = json.load(f)
        else:
            # Run benchmarks to generate data
            if verbose:
                print("Running benchmarks to generate plotting data...")

            benchmark_suite = ComprehensiveBenchmarkSuite(output_dir)

            # Configure benchmark parameters based on requested plot types
            zkml_config = {
                "architectures": ["simple", "multi_exit"],
                "complexities": complexities_plot or ["light", "medium", "heavy"],
                "input_sizes": [10, 50, 100],
                "num_trials": max(3, trials // 3),  # Fewer trials for benchmarking
                "frameworks": frameworks or ["ezkl", "risc_zero"],
            }

            crypto_config = {
                "rabin_input_sizes": [64, 256, 1024],
                "shamir_secret_sizes": [32, 64, 128],
                "num_trials": trials,
                "algorithms": algorithms or ["rabin", "shamir"],
            }

            # Run comprehensive benchmarks
            data = await benchmark_suite.run_full_benchmark_suite(
                zkml_config, crypto_config
            )

        # Generate plots
        if verbose:
            print(f"Generating plots for types: {plot_types or 'all available'}")

        if components:
            # Generate comparative plots
            results = plot_suite.generate_comparative_plots(
                data, components, output_dir
            )
        else:
            # Generate standard plots
            results = plot_suite.generate_plots(
                data, plot_types, output_dir, save_plots=True
            )

        # Generate summary report
        report = plot_suite.generate_summary_report(
            results, Path(output_dir) / "plotting_report.md"
        )

        if verbose:
            print("\nPlotting Summary:")
            print("=" * 50)
            print(report)

        print(f"\nPlots generated successfully in: {output_dir}")
        print(f"Report saved to: {Path(output_dir) / 'plotting_report.md'}")

        return 0

    except Exception as e:
        print(f"Error during plotting: {e}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def main():
    """
    Main function for the phaze CLI.
    """
    # Get available plot types
    available_plot_types = get_available_plot_types()
    plot_categories = get_plot_categories()

    # Format plot types for help text
    if available_plot_types:
        plot_type_names = [f"  {name}" for name in available_plot_types]
        plot_type_list = chr(10).join(plot_type_names)
    else:
        plot_type_list = "  No plot types available"

    parser = argparse.ArgumentParser(
        description=(
            "PHAZE: Probabilistic Hashing And ZKML-based Early-exit models "
            "for low latency inference at LHC."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available plot types:
{plot_type_list}

Example usage:
  # Run full pipeline (default)
  uv run phaze                                            # Full pipeline in quick mode (default)
  uv run phaze --train                                    # Full pipeline with default config
  uv run phaze --config my_config.py                     # Full pipeline with custom config
  uv run phaze --architectures simple,conv --epochs 1    # Full pipeline with specific models
  uv run phaze --model-path model.pth                    # Use pre-trained model for zkML/crypto benchmarking

  # Generate plots
  uv run phaze --plot fingerprint --output plots/fingerprint/
  uv run phaze --plot zkml-proof --frameworks ezkl,risc_zero --trials 10
  uv run phaze --plot all --style publication --output plots/comprehensive/

  # Comparative analysis
  uv run phaze --plot comparative --components fingerprint,zkml-proof

  # List available options
  uv run phaze --list-plot-types

  # Legacy benchmarks (deprecated - use phaze-legacy instead)
  uv run phaze-legacy basic_benchmark --mode standard --quick
  uv run phaze-legacy risc_zero --mode standalone

Default behavior (if no options specified):
  uv run phaze  # Full pipeline in quick mode (fast training + benchmarking + plots)
        """,
    )

    # Training-related arguments
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run full training and benchmarking pipeline (not quick mode)",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (.py, .yml, or .yaml)",
    )

    parser.add_argument(
        "--architectures",
        type=str,
        help="Comma-separated list of model architectures (e.g., 'simple,conv,transformer')",
    )

    parser.add_argument(
        "--complexities-train",
        type=str,
        help="Comma-separated list of model complexities for training (e.g., 'minimal,light,medium')",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: reduced epochs, dataset size, and model variants for testing",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to pre-trained model file (.pth, .pt) - skips training and runs zkML/crypto benchmarking",
    )

    # Plotting-related arguments
    parser.add_argument(
        "--plot",
        type=str,
        choices=available_plot_types + ["all", "comparative"],
        help="Generate plots of specified type (or 'all' for all types, 'comparative' for cross-component analysis)",
    )

    parser.add_argument(
        "--list-plot-types",
        action="store_true",
        help="List all available plot types and exit",
    )

    parser.add_argument(
        "--components",
        type=str,
        help="Comma-separated list of components for comparative plotting (e.g., 'fingerprint,zkml-proof')",
    )

    parser.add_argument(
        "--frameworks",
        type=str,
        help="Comma-separated list of zkML frameworks to include (e.g., 'ezkl,risc_zero')",
    )

    parser.add_argument(
        "--algorithms",
        type=str,
        help="Comma-separated list of fingerprinting algorithms to include (e.g., 'rabin,shamir')",
    )

    parser.add_argument(
        "--complexities-plot",
        type=str,
        help="Comma-separated list of model complexities for plotting (e.g., 'light,medium,heavy')",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of statistical trials per configuration (default: 10)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots)",
    )

    parser.add_argument(
        "--data-file",
        type=str,
        help="Load benchmark data from JSON file instead of running new benchmarks",
    )

    parser.add_argument(
        "--style",
        type=str,
        choices=["publication", "presentation", "web", "neurips"],
        default="neurips",
        help="Plot style (default: neurips)",
    )

    parser.add_argument(
        "--format",
        type=str,
        help="Comma-separated list of export formats (e.g., 'png,pdf,svg')",
    )

    # General arguments
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output."
    )

    # Parse arguments
    args = parser.parse_args()

    if args.list_plot_types:
        print("Available plot types:")
        if available_plot_types:
            for plot_type in available_plot_types:
                print(f"  {plot_type}")
            print(f"\nPlot categories: {', '.join(plot_categories)}")
        else:
            print("  No plot types available")
        return 0

    if args.verbose:
        print("Verbose mode enabled.")

    # Handle training commands (including when config is provided)
    if args.train or args.model_path or args.config:
        if args.verbose:
            if args.model_path:
                print("Running training command: with pre-trained model")
            elif args.config:
                print("Running training command: with custom config")
            else:
                print("Running training command: full pipeline")

        # Parse training-specific arguments
        architectures = args.architectures.split(",") if args.architectures else None
        complexities = args.complexities_train.split(",") if args.complexities_train else None

        # Run training asynchronously
        return asyncio.run(
            run_training_command(
                config_file=args.config,
                output_dir=args.output,
                architectures=architectures,
                complexities=complexities,
                epochs=args.epochs,
                quick=args.quick,
                verbose=args.verbose,
                model_path=args.model_path,
            )
        )

    # Handle plotting commands
    if args.plot:
        if args.verbose:
            print(f"Running plotting command: {args.plot}")

        # Parse plot-specific arguments
        plot_types = None if args.plot in ["all", "comparative"] else [args.plot]
        components = args.components.split(",") if args.components else None
        formats = args.format.split(",") if args.format else None
        frameworks = args.frameworks.split(",") if args.frameworks else None
        algorithms = args.algorithms.split(",") if args.algorithms else None
        complexities_plot = args.complexities_plot.split(",") if args.complexities_plot else None

        # Run plotting asynchronously
        return asyncio.run(
            run_plotting_command(
                plot_types=plot_types,
                components=components,
                output_dir=args.output,
                data_file=args.data_file,
                style=args.style,
                formats=formats,
                trials=args.trials,
                verbose=args.verbose,
                frameworks=frameworks,
                algorithms=algorithms,
                complexities_plot=complexities_plot,
            )
        )

    # Default behavior: run quick training pipeline
    if args.verbose:
        print(
            "No command specified, running default: "
            "full pipeline in quick mode (fast training + benchmarking + plots)"
        )

    return asyncio.run(
        run_training_command(
            config_file=None,
            output_dir="benchmark_results",
            architectures=None,
            complexities=None,
            epochs=None,
            quick=True,  # Default to quick mode
            verbose=args.verbose,
            model_path=None,
        )
    )


if __name__ == "__main__":
    main()
