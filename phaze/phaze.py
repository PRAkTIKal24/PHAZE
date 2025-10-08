import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


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


async def run_plotting_command(
    plot_types: Optional[List[str]] = None,
    components: Optional[List[str]] = None,
    output_dir: str = "plots",
    data_file: Optional[str] = None,
    style: str = "neurips",
    formats: Optional[List[str]] = None,
    trials: int = 10,
    verbose: bool = False
) -> int:
    """Run plotting command with specified parameters."""
    try:
        from .src.plotting import PHAZEPlotSuite, PlotConfig, PlotStyle
        from .src.comprehensive_benchmark import ComprehensiveBenchmarkSuite
        
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
            num_trials=trials
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
            
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            # Run benchmarks to generate data
            if verbose:
                print("Running benchmarks to generate plotting data...")
            
            benchmark_suite = ComprehensiveBenchmarkSuite(output_dir)
            
            # Configure benchmark parameters based on requested plot types
            zkml_config = {
                "architectures": ["simple", "multi_exit"],
                "complexities": ["light", "medium", "heavy"],
                "input_sizes": [10, 50, 100],
                "num_trials": max(3, trials // 3)  # Fewer trials for benchmarking
            }
            
            crypto_config = {
                "rabin_input_sizes": [64, 256, 1024],
                "shamir_secret_sizes": [32, 64, 128],
                "num_trials": trials
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


def get_available_benchmarks():
    """Get list of available benchmarks from the examples directory."""
    examples_dir = Path(__file__).parent.parent / "examples"
    if not examples_dir.exists():
        return {}

    benchmarks = {}
    for file_path in examples_dir.glob("run_*.py"):
        # Extract benchmark name from filename:
        # run_basic_benchmark.py -> basic_benchmark
        benchmark_name = file_path.name[4:-3]  # Remove "run_" prefix and ".py" suffix
        benchmarks[benchmark_name] = str(file_path)

    return benchmarks


def get_benchmark_help(benchmark_name, benchmark_path):
    """Get help text for a specific benchmark by running it with --help."""
    try:
        cmd = [sys.executable, benchmark_path, "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return f"Help not available for {benchmark_name}"


def run_benchmark(benchmark_name, benchmark_path, benchmark_args):
    """Run the specified benchmark with given arguments."""
    # Construct the command to run the benchmark
    cmd = [sys.executable, benchmark_path] + benchmark_args

    try:
        # Run the benchmark script
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark '{benchmark_name}': {e}", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print(f"Error: Benchmark file not found: {benchmark_path}", file=sys.stderr)
        return 1


def main():
    """
    Main function for the phaze CLI.
    """
    # Get available benchmarks and plot types
    available_benchmarks = get_available_benchmarks()
    available_plot_types = get_available_plot_types()
    plot_categories = get_plot_categories()

    # Format benchmark list for help text
    if available_benchmarks:
        benchmark_names = [f"  {name}" for name in available_benchmarks.keys()]
        benchmark_list = chr(10).join(benchmark_names)
    else:
        benchmark_list = "  No benchmarks found"
    
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
Available benchmarks:
{benchmark_list}

Available plot types:
{plot_type_list}

Example usage:
  # Run benchmarks
  uv run phaze -b basic_benchmark --mode standard --quick
  uv run phaze --benchmark risc_zero --mode standalone
  
  # Generate plots
  uv run phaze --plot fingerprint --output plots/fingerprint/
  uv run phaze --plot zkml-proof --frameworks ezkl,risc_zero --trials 10
  uv run phaze --plot all --style publication --output plots/comprehensive/
  
  # Comparative analysis
  uv run phaze --plot comparative --components fingerprint,zkml-proof
  
  # List available options
  uv run phaze --list-benchmarks
  uv run phaze --list-plot-types
  uv run phaze --help-benchmark basic_benchmark

Default behavior (if no options specified):
  uv run phaze  # Runs basic_benchmark with --mode all --output-dir plots/
        """,
    )

    # Benchmark-related arguments
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        choices=list(available_benchmarks.keys()),
        help="Run a specific benchmark from the examples directory",
    )

    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List all available benchmarks and exit",
    )

    parser.add_argument(
        "--help-benchmark",
        type=str,
        choices=list(available_benchmarks.keys()),
        help="Show help for a specific benchmark",
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
        "--complexity-range",
        type=str,
        help="Comma-separated list of model complexities to test (e.g., 'light,medium,heavy')",
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

    # Parse known args to separate PHAZE args from benchmark args
    args, benchmark_args = parser.parse_known_args()

    if args.list_benchmarks:
        print("Available benchmarks:")
        if available_benchmarks:
            for name, path in available_benchmarks.items():
                print(f"  {name}: {path}")
        else:
            print("  No benchmarks found in examples directory")
        return 0
    
    if args.list_plot_types:
        print("Available plot types:")
        if available_plot_types:
            for plot_type in available_plot_types:
                print(f"  {plot_type}")
            print(f"\nPlot categories: {', '.join(plot_categories)}")
        else:
            print("  No plot types available")
        return 0

    if args.help_benchmark:
        if args.help_benchmark not in available_benchmarks:
            print(
                f"Error: Benchmark '{args.help_benchmark}' not found.",
                file=sys.stderr,
            )
            return 1

        benchmark_path = available_benchmarks[args.help_benchmark]
        help_text = get_benchmark_help(args.help_benchmark, benchmark_path)
        print(f"Help for benchmark '{args.help_benchmark}':")
        print("=" * 50)
        print(help_text)
        return 0

    if args.verbose:
        print("Verbose mode enabled.")

    # Handle plotting commands
    if args.plot:
        if args.verbose:
            print(f"Running plotting command: {args.plot}")
        
        # Parse plot-specific arguments
        plot_types = None if args.plot in ["all", "comparative"] else [args.plot]
        components = args.components.split(",") if args.components else None
        formats = args.format.split(",") if args.format else None
        
        # Run plotting asynchronously
        return asyncio.run(run_plotting_command(
            plot_types=plot_types,
            components=components,
            output_dir=args.output,
            data_file=args.data_file,
            style=args.style,
            formats=formats,
            trials=args.trials,
            verbose=args.verbose
        ))

    # Handle benchmark execution
    if args.benchmark:
        if args.benchmark not in available_benchmarks:
            print(f"Error: Benchmark '{args.benchmark}' not found.", file=sys.stderr)
            print(
                f"Available benchmarks: {', '.join(available_benchmarks.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

        benchmark_path = available_benchmarks[args.benchmark]
        if args.verbose:
            print(f"Running benchmark: {args.benchmark}")
            if benchmark_args:
                print(f"Benchmark arguments: {' '.join(benchmark_args)}")
            print(f"Executing: {benchmark_path} {' '.join(benchmark_args)}")

        return run_benchmark(args.benchmark, benchmark_path, benchmark_args)

    # Default behavior: run basic_benchmark with default arguments
    if "basic_benchmark" in available_benchmarks:
        default_args = ["--mode", "all", "--output-dir", "plots/"]
        if args.verbose:
            print(
                "No benchmark specified, running default: "
                "basic_benchmark --mode all --output-dir plots/"
            )
            print(
                f"Executing: {available_benchmarks['basic_benchmark']} "
                f"{' '.join(default_args)}"
            )

        return run_benchmark(
            "basic_benchmark",
            available_benchmarks["basic_benchmark"],
            default_args,
        )
    else:
        print("Error: Default benchmark 'basic_benchmark' not found.", file=sys.stderr)
        parser.print_help()
        return 1


if __name__ == "__main__":
    main()
