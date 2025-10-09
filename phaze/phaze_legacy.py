#!/usr/bin/env python3
"""
PHAZE Legacy Benchmark CLI

This module provides access to the legacy benchmark system for backward compatibility.
For new projects, use the main `phaze` command which provides the modern training pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def get_available_benchmarks():
    """Get list of available legacy benchmarks."""
    legacy_dir = Path(__file__).parent.parent / "legacy" / "examples"
    if not legacy_dir.exists():
        return {}

    benchmarks = {}
    for file_path in legacy_dir.glob("run_*.py"):
        # Convert run_basic_benchmark.py -> basic_benchmark
        benchmark_name = file_path.stem.replace("run_", "")
        benchmarks[benchmark_name] = str(file_path)

    return benchmarks


def get_benchmark_help(benchmark_name: str, benchmark_path: str) -> str:
    """Get help text for a specific benchmark."""
    try:
        result = subprocess.run(
            [sys.executable, benchmark_path, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error getting help for {benchmark_name}: {e}"


def run_benchmark(benchmark_name: str, benchmark_path: str, args: List[str]) -> int:
    """Run a specific benchmark with given arguments."""
    try:
        cmd = [sys.executable, benchmark_path] + args
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as e:
        print(f"Error running benchmark {benchmark_name}: {e}", file=sys.stderr)
        return 1


def main():
    """Main entry point for legacy PHAZE benchmarks."""

    available_benchmarks = get_available_benchmarks()
    benchmark_choices = list(available_benchmarks.keys())

    parser = argparse.ArgumentParser(
        description="PHAZE Legacy Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
⚠️  LEGACY SYSTEM WARNING ⚠️

This is the legacy benchmark system. For new projects, use:
  uv run phaze              # Modern training pipeline
  uv run phaze --help       # See modern options

Available legacy benchmarks:
{chr(10).join(f"  {name}" for name in benchmark_choices)}

Example usage:
  # Run legacy basic benchmark
  uv run phaze-legacy basic_benchmark --mode standard --quick

  # Run legacy RISC Zero benchmark
  uv run phaze-legacy risc_zero --mode standalone

  # Get help for specific legacy benchmark
  uv run phaze-legacy --help-benchmark basic_benchmark

Migrate to modern system:
  uv run phaze                    # Replaces most legacy functionality
  uv run phaze --train            # Full training pipeline
  uv run phaze --plot all         # Publication-ready plots
        """,
    )

    parser.add_argument(
        "benchmark",
        nargs="?",
        choices=benchmark_choices,
        help="Legacy benchmark to run",
    )

    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List all available legacy benchmarks and exit",
    )

    parser.add_argument(
        "--help-benchmark",
        choices=benchmark_choices,
        help="Show help for a specific legacy benchmark",
    )

    # Parse known args to allow passing through benchmark-specific arguments
    args, benchmark_args = parser.parse_known_args()

    # Handle list benchmarks
    if args.list_benchmarks:
        print("Available legacy benchmarks:")
        if available_benchmarks:
            for name, path in available_benchmarks.items():
                print(f"  {name}: {path}")
        else:
            print("  No legacy benchmarks found")
        return 0

    # Handle benchmark help
    if args.help_benchmark:
        if args.help_benchmark not in available_benchmarks:
            print(
                f"Error: Legacy benchmark '{args.help_benchmark}' not found.",
                file=sys.stderr,
            )
            return 1

        benchmark_path = available_benchmarks[args.help_benchmark]
        help_text = get_benchmark_help(args.help_benchmark, benchmark_path)
        print(f"Help for legacy benchmark '{args.help_benchmark}':")
        print("=" * 50)
        print(help_text)
        return 0

    # Handle benchmark execution
    if args.benchmark:
        if args.benchmark not in available_benchmarks:
            print(
                f"Error: Legacy benchmark '{args.benchmark}' not found.",
                file=sys.stderr,
            )
            return 1

        benchmark_path = available_benchmarks[args.benchmark]
        return run_benchmark(args.benchmark, benchmark_path, benchmark_args)

    # No benchmark specified - show help and migration guidance
    print("⚠️  LEGACY SYSTEM WARNING ⚠️")
    print()
    print("You're using the legacy benchmark system. Consider migrating to:")
    print("  uv run phaze              # Modern training pipeline")
    print("  uv run phaze --help       # See all modern options")
    print()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
