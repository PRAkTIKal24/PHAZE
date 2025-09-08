import asyncio
import time

import numpy as np
import torch

from .crypto_primitives import RabinFingerprint
from .early_exit_models import SimpleEarlyExitModel
from .zkml_integration import SimpleFullModel, ZKMLProverVerifier


async def run_early_exit_benchmark(
    model: SimpleEarlyExitModel, input_data: torch.Tensor, num_iterations: int = 100
):
    """Benchmarks the inference time of an M_early model."""
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = model(input_data)
    end_time = time.perf_counter()
    avg_time_ms = (end_time - start_time) / num_iterations * 1000
    return avg_time_ms


async def run_hashing_benchmark(
    fingerprinter: RabinFingerprint, data_vector: list, num_iterations: int = 100
):
    """Benchmarks the time taken for polynomial hashing."""
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = fingerprinter.compute_hash(data_vector)
    end_time = time.perf_counter()
    avg_time_ms = (end_time - start_time) / num_iterations * 1000
    return avg_time_ms


async def run_zkml_benchmark(
    zkml_pv: ZKMLProverVerifier, input_data: torch.Tensor, num_iterations: int = 10
):
    """Benchmarks the proving and verification time for a ZKML system."""
    proving_times = []
    verification_times = []

    for _ in range(num_iterations):
        # Proving
        start_time = time.perf_counter()
        proof, _ = await zkml_pv.generate_proof(input_data)
        end_time = time.perf_counter()
        proving_times.append((end_time - start_time) * 1000)

        # Verification
        start_time = time.perf_counter()
        _ = await zkml_pv.verify_proof(proof, input_data)
        end_time = time.perf_counter()
        verification_times.append((end_time - start_time) * 1000)

    avg_proving_time_ms = np.mean(proving_times)
    avg_verification_time_ms = np.mean(verification_times)

    return avg_proving_time_ms, avg_verification_time_ms


async def run_full_pipeline_benchmark(num_iterations: int = 100):
    """Runs a conceptual full pipeline benchmark for PHAZE components."""
    results = {}

    # M_early benchmark
    early_model = SimpleEarlyExitModel()
    early_input = torch.randn(1, 10)
    results["m_early_inference_ms"] = await run_early_exit_benchmark(
        early_model, early_input, num_iterations
    )

    # Hashing benchmark
    fingerprinter = RabinFingerprint(field_size=2**31 - 1, degree=10)
    hash_data = [np.random.randint(0, 100) for _ in range(10)]
    results["hashing_ms"] = await run_hashing_benchmark(
        fingerprinter, hash_data, num_iterations
    )

    # ZKML benchmark
    full_model = SimpleFullModel()
    zkml_input = torch.randn(1, 10)
    zkml_pv = ZKMLProverVerifier(full_model, "ezkl")

    # Setup ezkl once
    await zkml_pv.setup(zkml_input)

    proving_time, verification_time = await run_zkml_benchmark(
        zkml_pv, zkml_input, num_iterations=10
    )  # ZKML is more expensive, fewer iterations
    results["zkml_proving_ms"] = proving_time
    results["zkml_verification_ms"] = verification_time

    # Cleanup ezkl artifacts after all benchmarks
    zkml_pv.cleanup()

    return results


if __name__ == "__main__":
    print("Running PHAZE full pipeline benchmark...")
    benchmark_results = asyncio.run(run_full_pipeline_benchmark(num_iterations=50))
    print("\n--- Benchmark Results ---")
    for key, value in benchmark_results.items():
        print(f"{key}: {value:.4f} ms")
