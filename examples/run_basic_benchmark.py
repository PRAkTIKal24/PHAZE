#!/usr/bin/env python3
"""
Example script demonstrating the PHAZE comprehensive benchmarking system.

This script shows how to use the benchmarking framework to evaluate different
zkML frameworks, model architectures, and cryptographic primitives.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import torch

# Add the parent directory to the path so we can import phaze
sys.path.insert(0, str(Path(__file__).parent.parent))

from phaze.src.comprehensive_benchmark import (
    ComprehensiveBenchmarkSuite,
    run_phaze_benchmarks,
)
from phaze.src.model_architectures import PHAZEModelFactory
from phaze.src.rust_zkml_backend import RustRiscZeroBackend


async def run_standard_benchmarks(output_dir, quick_mode=True):
    """Run the standard PHAZE benchmarks."""
    print("🚀 Starting PHAZE Comprehensive Benchmark Example")
    print("=" * 60)

    print(f"📁 Output directory: {output_dir}")
    if quick_mode:
        print("⚡ Running in quick mode for demonstration...")
    else:
        print("🔬 Running full benchmark suite...")
    print()

    try:
        # Add RISC Zero to the benchmarks
        from phaze.src.comprehensive_benchmark import ComprehensiveBenchmarkSuite
        from phaze.src.rust_zkml_backend import RustRiscZeroBackend

        # Create benchmark suite
        suite = ComprehensiveBenchmarkSuite(output_dir)

        # Register RISC Zero backend
        suite.zkml_benchmark.rust_manager.backends["risc_zero"] = RustRiscZeroBackend()

        # Add RISC Zero test explicitly
        result = await run_phaze_benchmarks(output_dir, quick_mode=quick_mode)

        # Add a standalone RISC Zero test to ensure it's included
        print("🔬 Adding RISC Zero to standard benchmarks...")
        risc_result = await suite.zkml_benchmark.benchmark_framework(
            framework_name="risc_zero",
            architecture="simple",
            complexity="light",
            input_size=10,
            output_size=5,
            num_trials=1,
        )

        # Include RISC Zero results in the results
        result["results"]["zkml_results"].extend([r.__dict__ for r in risc_result])

        # Update the summary to include RISC Zero
        if "frameworks_tested" in result["results"]["summary"]["zkml_summary"]:
            if (
                "risc_zero"
                not in result["results"]["summary"]["zkml_summary"]["frameworks_tested"]
            ):
                result["results"]["summary"]["zkml_summary"][
                    "frameworks_tested"
                ].append("risc_zero")

        print("✅ Benchmark completed successfully!")
        print()

        # Display summary
        results = result["results"]
        summary = results.get("summary", {})

        print("📊 BENCHMARK SUMMARY")
        print("-" * 30)

        overall = summary.get("overall_summary", {})
        print(f"Total tests: {overall.get('total_tests', 0)}")
        print(f"Successful tests: {overall.get('successful_tests', 0)}")
        print(f"Success rate: {overall.get('overall_success_rate', 0):.1%}")
        print()

        # zkML summary
        zkml_summary = summary.get("zkml_summary", {})
        if zkml_summary:
            print("🔐 zkML Framework Results:")
            print(
                f"  - Frameworks tested: {
                    ', '.join(zkml_summary.get('frameworks_tested', []))
                }"
            )
            print(f"  - Success rate: {zkml_summary.get('success_rate', 0):.1%}")
            print(f"  - Avg proof time: {zkml_summary.get('avg_proof_time', 0):.3f}s")
            print(
                f"  - Avg memory usage: {zkml_summary.get('avg_memory_usage', 0):.1f}MB"
            )
            print()

        # Crypto summary
        crypto_summary = summary.get("crypto_summary", {})
        if crypto_summary:
            print("🔒 Cryptographic Primitive Results:")
            print(
                f"  - Primitives tested: {
                    ', '.join(crypto_summary.get('primitives_tested', []))
                }"
            )
            print(f"  - Success rate: {crypto_summary.get('success_rate', 0):.1%}")
            avg_throughput = crypto_summary.get('avg_throughput', 0)
            print(
                f"  - Avg throughput: {avg_throughput:.1f} ops/sec"
            )
            print()

        # Show some individual results
        print("🔍 Sample Individual Results:")
        print("-" * 30)

        zkml_results = results.get("zkml_results", [])
        if zkml_results:
            successful_zkml = [r for r in zkml_results if r.get("success", False)]
            if successful_zkml:
                sample = successful_zkml[0]
                print(
                    f"zkML Test: {sample.get('framework', 'Unknown')} - {
                        sample.get('architecture', 'Unknown')
                    }"
                )
                print(f"  Setup: {sample.get('setup_time', 0):.3f}s")
                print(f"  Proof: {sample.get('proof_time', 0):.3f}s")
                print(f"  Verify: {sample.get('verification_time', 0):.3f}s")
                print(f"  Memory: {sample.get('memory_usage_mb', 0):.1f}MB")
                print()

        crypto_results = results.get("crypto_results", [])
        if crypto_results:
            successful_crypto = [r for r in crypto_results if r.get("success", False)]
            if successful_crypto:
                sample = successful_crypto[0]
                print(
                    f"Crypto Test: {sample.get('primitive_name', 'Unknown')} - {
                        sample.get('operation', 'Unknown')
                    }"
                )
                print(f"  Execution: {sample.get('execution_time', 0):.6f}s")
                throughput = sample.get('throughput_ops_per_sec', 0)
                print(
                    f"  Throughput: {throughput:.1f} ops/sec"
                )
                print(f"  Input size: {sample.get('input_size', 0)} bytes")
                print()

        print("📁 Files generated:")
        output_path = Path(output_dir)
        for file_path in output_path.glob("*"):
            if file_path.is_file():
                print(f"  - {file_path.name}")

        print()
        print("🎉 Example completed! Check the output directory for detailed results.")
        return 0

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


async def run_risc_zero_standalone(output_dir):
    """Run a standalone RISC Zero example."""
    print("🚀 Running RISC Zero Standalone Example")
    print("=" * 60)

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
    print(f"🔧 Setup result: {setup_result}")

    # Generate proof
    print("🔒 Generating proof...")
    proof = backend.prove(input_data, model.state_dict())
    print(
        f"✅ Proof generated: {proof['framework']} proof with {
            len(proof['proof_data'])
        } bytes"
    )

    # Verify proof
    print("🔍 Verifying proof...")
    verification_result = backend.verify(proof, expected_output)
    print(f"🔐 Verification result: {verification_result}")

    # Save results to output directory
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/risc_zero_standalone_result.json", "w") as f:
        # Convert proof to serializable format
        serializable_proof = {
            "framework": proof["framework"],
            "proof_data_length": len(proof["proof_data"]),
            "verification_result": verification_result,
        }
        json.dump(serializable_proof, f, indent=2)

    print(f"📝 Results saved to {output_dir}/risc_zero_standalone_result.json")
    return 0


async def run_risc_zero_benchmark(output_dir):
    """Run benchmarks for RISC Zero framework."""
    print("🚀 Running RISC Zero Benchmarks")
    print("=" * 60)

    print(f"📁 Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Create the benchmark suite
    suite = ComprehensiveBenchmarkSuite(output_dir)

    # Register RISC Zero backend
    suite.zkml_benchmark.rust_manager.backends["risc_zero"] = RustRiscZeroBackend()

    # Directly use the benchmark_framework method to benchmark only RISC Zero
    print("📊 Running RISC Zero benchmarks...")

    try:
        # Benchmark RISC Zero with different configurations
        architectures = ["simple", "multi_exit"]
        complexities = ["light", "medium"]
        input_sizes = [10, 50]

        all_results = []

        for architecture in architectures:
            for complexity in complexities:
                for input_size in input_sizes:
                    print(
                        f"🔄 Benchmarking: {architecture}/{complexity}, input_size={
                            input_size
                        }"
                    )

                    results = await suite.zkml_benchmark.benchmark_framework(
                        framework_name="risc_zero",
                        architecture=architecture,
                        complexity=complexity,
                        input_size=input_size,
                        output_size=5,
                        num_trials=1,
                    )

                    all_results.extend(results)

        print(f"✅ RISC Zero benchmark completed with {len(all_results)} tests")

        # Count successful tests
        successful = [r for r in all_results if r.success]
        success_percentage = len(successful) / len(all_results)
        print(
            f"✅ Successful tests: {len(successful)}/{len(all_results)} "
            f"({success_percentage:.1%})"
        )

        # Calculate average metrics
        if successful:
            avg_setup_time = sum(r.setup_time for r in successful) / len(successful)
            avg_proof_time = sum(r.proof_time for r in successful) / len(successful)
            avg_verify_time = sum(r.verification_time for r in successful) / len(
                successful
            )
            avg_memory = sum(r.memory_usage_mb for r in successful) / len(successful)

            print(f"⏱️ Average setup time: {avg_setup_time:.4f}s")
            print(f"⏱️ Average proof time: {avg_proof_time:.4f}s")
            print(f"⏱️ Average verification time: {avg_verify_time:.4f}s")
            print(f"💾 Average memory usage: {avg_memory:.2f}MB")

        # Create a results structure
        results_dict = {
            "zkml_results": [r.__dict__ for r in all_results],
            "timestamp": asyncio.get_event_loop().time(),
            "summary": {
                "zkml_summary": {
                    "frameworks_tested": ["risc_zero"],
                    "architectures_tested": architectures,
                    "complexities_tested": complexities,
                    "total_tests": len(all_results),
                    "successful_tests": len(successful),
                    "success_rate": len(successful) / len(all_results)
                    if all_results
                    else 0,
                    "avg_setup_time": avg_setup_time if successful else 0,
                    "avg_proof_time": avg_proof_time if successful else 0,
                    "avg_verification_time": avg_verify_time if successful else 0,
                    "avg_memory_usage": avg_memory if successful else 0,
                }
            },
        }

        # Generate a report
        report = suite.generate_report(results_dict)
        report_path = os.path.join(output_dir, "risc_zero_benchmark_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"📝 Benchmark report saved to: {report_path}")

        # Also save raw results
        results_path = os.path.join(output_dir, "risc_zero_benchmark_results.json")
        with open(results_path, "w") as f:
            # Convert non-serializable items to strings
            serializable_results = json.loads(
                json.dumps(results_dict, default=lambda o: str(o))
            )
            json.dump(serializable_results, f, indent=2)
        print(f"📝 Raw results saved to: {results_path}")

        return 0
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


async def run_extended_benchmark(output_dir):
    """Run extended benchmarks with all frameworks including RISC Zero."""
    print("🚀 Starting PHAZE Comprehensive Benchmark Example")
    print("=" * 60)

    print(f"📁 Output directory: {output_dir}")
    print("🔬 Running extended benchmark with all frameworks including RISC Zero...")

    try:
        # Create benchmark suite
        from phaze.src.comprehensive_benchmark import ComprehensiveBenchmarkSuite
        from phaze.src.rust_zkml_backend import RustRiscZeroBackend

        # Create the benchmark suite
        suite = ComprehensiveBenchmarkSuite(output_dir)

        # Register RISC Zero backend
        suite.zkml_benchmark.rust_manager.backends["risc_zero"] = RustRiscZeroBackend()

        # Run the benchmark directly with minimal configuration
        # Use the run_phaze_benchmarks function to ensure compatibility
        result = await run_phaze_benchmarks(output_dir, quick_mode=True)

        # Explicitly run RISC Zero benchmarks
        print("🔬 Running RISC Zero benchmarks...")
        risc_zero_results = []

        # Test RISC Zero with different input sizes
        for input_size in [10, 50]:
            results = await suite.zkml_benchmark.benchmark_framework(
                framework_name="risc_zero",
                architecture="simple",
                complexity="light",
                input_size=input_size,
                output_size=5,
                num_trials=1,
            )
            risc_zero_results.extend(results)

        # Add the RISC Zero results to the overall results
        result["results"]["zkml_results"].extend(
            [r.__dict__ for r in risc_zero_results]
        )

        # Update the summary to include RISC Zero
        if "frameworks_tested" in result["results"]["summary"]["zkml_summary"]:
            if (
                "risc_zero"
                not in result["results"]["summary"]["zkml_summary"]["frameworks_tested"]
            ):
                result["results"]["summary"]["zkml_summary"][
                    "frameworks_tested"
                ].append("risc_zero")

        # Print summary
        print("✅ Benchmark completed successfully!")
        print()

        # Display summary
        results = result["results"]
        summary = results.get("summary", {})

        print("📊 BENCHMARK SUMMARY")
        print("-" * 30)

        overall = summary.get("overall_summary", {})
        print(f"Total tests: {overall.get('total_tests', 0)}")
        print(f"Successful tests: {overall.get('successful_tests', 0)}")
        print(f"Success rate: {overall.get('overall_success_rate', 0):.1%}")
        print()

        # zkML summary
        zkml_summary = summary.get("zkml_summary", {})
        if zkml_summary:
            print("🔐 zkML Framework Results:")
            print(
                f"  - Frameworks tested: {
                    ', '.join(zkml_summary.get('frameworks_tested', []))
                }"
            )
            print(f"  - Success rate: {zkml_summary.get('success_rate', 0):.1%}")
            print(f"  - Avg proof time: {zkml_summary.get('avg_proof_time', 0):.3f}s")
            print(
                f"  - Avg memory usage: {zkml_summary.get('avg_memory_usage', 0):.1f}MB"
            )
            print()

        return 0

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


async def main():
    """Parse arguments and run the selected benchmarks."""
    parser = argparse.ArgumentParser(description="PHAZE Benchmarking Examples")
    parser.add_argument(
        "--mode",
        choices=["standard", "risc-zero", "extended", "all"],
        default="standard",
        help="Which benchmark mode to run",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/phaze_benchmark_example",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run in quick mode with limited tests"
    )
    args = parser.parse_args()

    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)

    exit_code = 0

    if args.mode in ["standard", "all"]:
        output_dir = os.path.join(args.output_dir, "standard")
        exit_code |= await run_standard_benchmarks(output_dir, args.quick)

    if args.mode in ["risc-zero", "all"]:
        # Run both standalone and benchmark for RISC Zero
        standalone_dir = os.path.join(args.output_dir, "risc_zero_standalone")
        exit_code |= await run_risc_zero_standalone(standalone_dir)

        benchmark_dir = os.path.join(args.output_dir, "risc_zero_benchmark")
        exit_code |= await run_risc_zero_benchmark(benchmark_dir)

    if args.mode in ["extended", "all"]:
        extended_dir = os.path.join(args.output_dir, "extended")
        exit_code |= await run_extended_benchmark(extended_dir)

    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
