import argparse
import subprocess
import sys
from pathlib import Path


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
    # Get available benchmarks
    available_benchmarks = get_available_benchmarks()

    # Format benchmark list for help text
    if available_benchmarks:
        benchmark_names = [f"  {name}" for name in available_benchmarks.keys()]
        benchmark_list = chr(10).join(benchmark_names)
    else:
        benchmark_list = "  No benchmarks found"

    parser = argparse.ArgumentParser(
        description=(
            "PHAZE: Probabilistic Hashing And ZKML-based Early-exit models "
            "for low latency inference at LHC."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available benchmarks:
{benchmark_list}

Example usage:
  uv run phaze -b basic_benchmark --mode standard --quick
  uv run phaze --benchmark risc_zero --mode standalone
  uv run phaze --list-benchmarks
  uv run phaze --help-benchmark basic_benchmark

Default behavior (if no benchmark specified):
  uv run phaze  # Runs basic_benchmark with --mode all --output-dir plots/
        """,
    )

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
        if args.benchmark:
            print(f"Running benchmark: {args.benchmark}")
            if benchmark_args:
                print(f"Benchmark arguments: {' '.join(benchmark_args)}")

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
