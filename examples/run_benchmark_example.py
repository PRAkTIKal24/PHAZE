#!/usr/bin/env python3
"""
Example script demonstrating the PHAZE comprehensive benchmarking system.

This script shows how to use the benchmarking framework to evaluate different
zkML frameworks, model architectures, and cryptographic primitives.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import phaze
sys.path.insert(0, str(Path(__file__).parent.parent))

from phaze.src.comprehensive_benchmark import run_phaze_benchmarks


async def main():
    """Run the PHAZE benchmark example."""
    print("🚀 Starting PHAZE Comprehensive Benchmark Example")
    print("=" * 60)

    # Create output directory
    output_dir = "/tmp/phaze_benchmark_example"
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")
    print("⚡ Running in quick mode for demonstration...")
    print()

    try:
        # Run benchmarks in quick mode
        result = await run_phaze_benchmarks(output_dir, quick_mode=True)

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
                f"  - Frameworks tested: {', '.join(zkml_summary.get('frameworks_tested', []))}"
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
                f"  - Primitives tested: {', '.join(crypto_summary.get('primitives_tested', []))}"
            )
            print(f"  - Success rate: {crypto_summary.get('success_rate', 0):.1%}")
            print(
                f"  - Avg throughput: {crypto_summary.get('avg_throughput', 0):.1f} ops/sec"
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
                    f"zkML Test: {sample.get('framework', 'Unknown')} - {sample.get('architecture', 'Unknown')}"
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
                    f"Crypto Test: {sample.get('primitive_name', 'Unknown')} - {sample.get('operation', 'Unknown')}"
                )
                print(f"  Execution: {sample.get('execution_time', 0):.6f}s")
                print(
                    f"  Throughput: {sample.get('throughput_ops_per_sec', 0):.1f} ops/sec"
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

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
