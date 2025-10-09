#!/usr/bin/env python3
"""
Example script to run PHAZE benchmarks with RISC Zero backend.

This script demonstrates how to use the RISC Zero backend
with the PHAZE benchmarking suite
to evaluate the performance of RISC Zero zkVM for machine learning inference proofs.
"""

import argparse
import asyncio
import json
import os
import sys

import torch

# Make sure we can import phaze modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from legacy.src.comprehensive_benchmark import (
    ComprehensiveBenchmarkSuite,
)
from phaze.src.model_architectures import PHAZEModelFactory
from phaze.src.rust_zkml_backend import RustRiscZeroBackend
from phaze.src.zkml_framework_interface import ZKMLFramework
from phaze.src.zkml_integration import PHAZEZKMLIntegration


async def run_risc_zero_standalone_example():
    """Run a simple standalone example with RISC Zero backend."""
    print("=== Running RISC Zero Standalone Example ===")

    # Create a simple model
    model = PHAZEModelFactory.create_early_exit_model("simple", complexity="light")

    # Create input data
    input_data = torch.randn(1, 10)

    # Run model to get expected output
    with torch.no_grad():
        expected_output = model(input_data)

    # Create RISC Zero backend
    backend = RustRiscZeroBackend()

    # Setup
    setup_result = backend.setup(
        {
            "model_type": "simple_early_exit",
            "input_size": "10",
            "output_size": "5",
        }
    )
    print(f"Setup result: {setup_result}")

    # Generate proof
    print("Generating proof...")
    proof = backend.prove(input_data, model.state_dict())
    print(
        f"Proof generated: {proof['framework']} proof with "
        f"{len(proof['proof_data'])} bytes"
    )

    # Verify proof
    print("Verifying proof...")
    verification_result = backend.verify(proof, expected_output)
    print(f"Verification result: {verification_result}")

    return verification_result


async def run_risc_zero_integration_example():
    """Example demonstrating RISC Zero integration with PHAZE for secure ML."""
    print("\n=== Running RISC Zero Integration Example ===")

    # Create integration manager
    integration = PHAZEZKMLIntegration()

    # Create and register a model
    model = integration.create_and_register_model(
        "risc_zero_test_model",
        architecture="simple",
        complexity="light",
        model_type="early_exit",
    )
    print(f"Created model: {model}")

    # Create sample input
    input_data = torch.randn(1, 10)

    # Import the backend factory function
    from phaze.src.zkml_backends import create_backend

    # Create RISC Zero backend through the backend factory
    backend = create_backend(ZKMLFramework.RISC_ZERO, model, "risc_zero_test")
    print(f"Created backend: {backend}")

    # Setup the backend
    await backend.setup(input_data)
    print("Backend setup complete")

    # Generate a proof
    print("Generating proof...")
    proof, output = await backend.generate_proof(input_data)
    print(f"Proof generated with output shape: {len(output)}")

    # Verify the proof
    print("Verifying proof...")
    is_valid = await backend.verify_proof(proof, input_data)
    print(f"Verification result: {is_valid}")

    return is_valid


async def run_risc_zero_benchmark():
    """Run benchmarks for RISC Zero and compare with other frameworks."""
    print("\n=== Running RISC Zero Benchmarks ===")

    # Create temporary directory for results
    import tempfile

    results_dir = tempfile.mkdtemp()
    print(f"Results will be saved to: {results_dir}")

    # Create the benchmark suite
    suite = ComprehensiveBenchmarkSuite(results_dir)

    # Register RISC Zero backend
    suite.zkml_benchmark.rust_manager.backends["risc_zero"] = RustRiscZeroBackend()

    # Directly use the benchmark_framework method to benchmark only RISC Zero
    print("Running RISC Zero benchmarks...")

    try:
        # Directly benchmark RISC Zero
        risc_zero_results = await suite.zkml_benchmark.benchmark_framework(
            framework_name="risc_zero",
            architecture="simple",
            complexity="light",
            input_size=10,
            output_size=5,
            num_trials=1,
        )

        print(f"RISC Zero benchmark results: {len(risc_zero_results)} entries")
        for result in risc_zero_results:
            print(
                f"  - {result.test_name}: Success={result.success}, "
                f"Time={result.total_time:.4f}s, Memory={result.memory_usage_mb:.2f}MB"
            )
            if not result.success:
                print(f"    Error: {result.error_message}")

        # Create a small results structure to generate a report
        import time

        results = {
            "zkml_results": [r.__dict__ for r in risc_zero_results],
            "crypto_results": [],
            "timestamp": time.time(),
            "summary": {
                "zkml_summary": {
                    "frameworks_tested": ["risc_zero"],
                    "architectures_tested": ["simple"],
                    "complexities_tested": ["light"],
                    "total_tests": len(risc_zero_results),
                    "successful_tests": sum(1 for r in risc_zero_results if r.success),
                },
                "crypto_summary": {
                    "primitives_tested": [],
                    "total_tests": 0,
                    "successful_tests": 0,
                },
            },
        }

        # Generate a report
        report = suite.generate_report(results)
        report_path = os.path.join(results_dir, "benchmark_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Benchmark report saved to: {report_path}")

        # Also save raw results
        results_path = os.path.join(results_dir, "benchmark_results.json")
        with open(results_path, "w") as f:
            # Convert non-serializable items to strings
            serializable_results = json.loads(
                json.dumps(results, default=lambda o: str(o))
            )
            json.dump(serializable_results, f, indent=2)
        print(f"Raw results saved to: {results_path}")

        return True
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main function to run the examples."""
    parser = argparse.ArgumentParser(
        description="Run PHAZE benchmarks with RISC Zero backend"
    )
    parser.add_argument(
        "--mode",
        choices=["standalone", "integration", "benchmark", "all"],
        default="all",
        help="Which example mode to run",
    )
    args = parser.parse_args()

    if args.mode in ["standalone", "all"]:
        await run_risc_zero_standalone_example()

    if args.mode in ["integration", "all"]:
        await run_risc_zero_integration_example()

    if args.mode in ["benchmark", "all"]:
        await run_risc_zero_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
