"""
Enhanced crypto benchmarking for PHAZE framework.

This module provides comprehensive benchmarking of cryptographic primitives
including fingerprinting algorithms, with detailed performance metrics.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn

from .crypto_primitives import RabinFingerprint, ShamirSecretSharing
from .training_config import PHAZEConfig

logger = logging.getLogger(__name__)


class CryptoMemoryProfiler:
    """Memory profiler for crypto operations."""

    def __init__(self):
        self.start_memory = 0
        self.peak_memory = 0
        self.end_memory = 0
        self.monitoring = False

    def start_monitoring(self):
        """Start memory monitoring."""
        self.start_memory = self._get_memory_usage()
        self.peak_memory = self.start_memory
        self.monitoring = True

    def update_peak(self):
        """Update peak memory usage."""
        if self.monitoring:
            current_memory = self._get_memory_usage()
            self.peak_memory = max(self.peak_memory, current_memory)

    def stop_monitoring(self):
        """Stop memory monitoring and return stats."""
        if not self.monitoring:
            return None

        self.end_memory = self._get_memory_usage()
        self.monitoring = False

        return {
            "start_mb": self.start_memory,
            "peak_mb": self.peak_memory,
            "end_mb": self.end_memory,
            "peak_increase_mb": self.peak_memory - self.start_memory,
            "net_increase_mb": self.end_memory - self.start_memory,
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class EnhancedCryptoBenchmark:
    """Enhanced benchmarking for cryptographic primitives and fingerprinting."""

    def __init__(self, config: PHAZEConfig):
        """Initialize benchmarker with configuration.

        Args:
            config: PHAZE configuration object
        """
        self.config = config
        self.crypto_results = []
        self.failed_benchmarks = []
        self.degree = config.experiment.polynomial_degree
        self.field_size = config.experiment.field_size

        # Initialize crypto systems
        self.crypto_systems = self._initialize_crypto_systems()

    def _initialize_crypto_systems(self) -> Dict[str, Any]:
        """Initialize cryptographic systems based on configuration."""
        systems = {}

        for algorithm in self.config.experiment.hashing_algorithms:
            if algorithm == "rabin":
                systems["rabin"] = {
                    "instance": RabinFingerprint(
                        field_size=self.field_size, degree=self.degree
                    ),
                    "operations": ["hash"],
                }
            elif algorithm == "shamir":
                systems["shamir"] = {
                    "instance": ShamirSecretSharing(threshold=3, num_shares=5),
                    "operations": ["share", "reconstruct"],
                }
            # Future: more algo down here

        logger.info(
            f"Initialized {len(systems)} crypto systems: {list(systems.keys())}"
        )
        return systems

    def extract_model_outputs(
        self, model: nn.Module, model_info: Dict[str, Any], num_samples: int = 100
    ) -> List[np.ndarray]:
        """Extract outputs from model for fingerprinting.

        Args:
            model: PyTorch model
            model_info: Model information
            num_samples: Number of sample inputs to generate

        Returns:
            List of model output arrays
        """
        model.eval()
        outputs = []

        with torch.no_grad():
            for _ in range(num_samples):
                # Generate sample input based on architecture
                if model_info["architecture"] == "conv":
                    input_channels = model_info.get("input_channels", 1)
                    spatial_size = model_info.get("spatial_size", 28)
                    sample_input = torch.randn(
                        1, input_channels, spatial_size, spatial_size
                    )
                else:
                    input_size = model_info.get("input_size", 784)
                    sample_input = torch.randn(1, input_size)

                # Get model output
                output = model(sample_input)
                outputs.append(output.cpu().numpy().flatten())

        logger.info(f"Extracted {len(outputs)} outputs from {model_info['model_id']}")
        return outputs

    def prepare_fingerprint_data(
        self, outputs: List[np.ndarray], target_sizes: Optional[List[int]] = None
    ) -> List[Tuple[int, bytes]]:
        """Prepare model outputs for fingerprinting with different input sizes.

        Args:
            outputs: List of model output arrays
            target_sizes: List of target byte sizes for fingerprinting

        Returns:
            List of (size, data) tuples for fingerprinting
        """
        if target_sizes is None:
            # Default sizes for testing different input complexities
            target_sizes = [32, 64, 128, 256, 512, 1024, 2048]

        fingerprint_data = []

        # Concatenate all outputs
        all_outputs = np.concatenate(outputs, axis=0)

        # Normalize and convert to bytes
        normalized_outputs = (all_outputs * 255).astype(np.uint8)

        for target_size in target_sizes:
            if len(normalized_outputs) >= target_size:
                # Take first target_size bytes
                data = normalized_outputs[:target_size].tobytes()
            else:
                # Repeat data to reach target size
                repeat_factor = (target_size // len(normalized_outputs)) + 1
                repeated_data = np.tile(normalized_outputs, repeat_factor)
                data = repeated_data[:target_size].tobytes()

            fingerprint_data.append((target_size, data))

        return fingerprint_data

    def benchmark_crypto_algorithm(
        self,
        algorithm_name: str,
        operation: str,
        fingerprint_data: List[Tuple[int, bytes]],
    ) -> List[Dict[str, Any]]:
        """Benchmark a specific crypto algorithm and operation.

        Args:
            algorithm_name: Name of the algorithm ("rabin", "shamir", etc.)
            operation: Operation to benchmark ("hash", "share", "reconstruct")
            fingerprint_data: List of (size, data) tuples

        Returns:
            List of benchmark results
        """
        if algorithm_name not in self.crypto_systems:
            logger.error(f"Algorithm {algorithm_name} not available")
            return []

        crypto_system = self.crypto_systems[algorithm_name]["instance"]
        results = []

        logger.info(f"Benchmarking {algorithm_name} {operation} operation")

        for input_size, data in fingerprint_data:
            result = {
                "primitive_name": algorithm_name,
                "operation": operation,
                "input_size": input_size,
                "success": False,
                "execution_time": 0.0,
                "memory_usage_mb": 0.0,
                "throughput_ops_per_sec": 0.0,
                "output_size": 0,
                "error_message": None,
            }

            try:
                # Benchmark the operation
                execution_times = []
                memory_stats = []

                for _iteration in range(self.config.experiment.benchmark_iterations):
                    memory_profiler = CryptoMemoryProfiler()
                    memory_profiler.start_monitoring()

                    start_time = time.perf_counter()

                    # Execute the operation
                    if algorithm_name == "rabin" and operation == "hash":
                        output = crypto_system.compute_hash(list(data))
                        result["output_size"] = len(str(output))

                    elif algorithm_name == "shamir" and operation == "share":
                        shares = crypto_system.generate_shares(data)
                        result["output_size"] = sum(len(share[1]) for share in shares)

                    elif algorithm_name == "shamir" and operation == "reconstruct":
                        # First generate shares, then reconstruct
                        shares = crypto_system.generate_shares(data)
                        reconstructed = crypto_system.reconstruct_secret(
                            shares[: crypto_system.threshold]
                        )
                        result["output_size"] = len(reconstructed)

                    else:
                        raise ValueError(
                            f"Unknown operation {operation} for {algorithm_name}"
                        )

                    memory_profiler.update_peak()
                    end_time = time.perf_counter()

                    execution_time = end_time - start_time
                    execution_times.append(execution_time)

                    memory_stat = memory_profiler.stop_monitoring()
                    memory_stats.append(memory_stat)

                # Calculate statistics
                result["execution_time"] = np.mean(execution_times)
                result["execution_time_std"] = np.std(execution_times)
                result["execution_time_min"] = np.min(execution_times)
                result["execution_time_max"] = np.max(execution_times)

                # Memory statistics
                if memory_stats:
                    peak_memories = [m["peak_mb"] for m in memory_stats]
                    result["memory_usage_mb"] = np.mean(peak_memories)
                    result["memory_usage_std"] = np.std(peak_memories)

                # Throughput calculation
                if result["execution_time"] > 0:
                    result["throughput_ops_per_sec"] = 1.0 / result["execution_time"]

                result["success"] = True

                logger.info(
                    f"  Size {input_size}: {result['execution_time']:.4f}s, "
                    f"{result['memory_usage_mb']:.1f}MB, "
                    f"{result['throughput_ops_per_sec']:.1f} ops/sec"
                )

            except Exception as e:
                logger.error(f"  Size {input_size} failed: {e}")
                result["error_message"] = str(e)
                self.failed_benchmarks.append(result)

            results.append(result)

            if result["success"]:
                self.crypto_results.append(result)

        return results

    def benchmark_model_fingerprinting(
        self, model_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Benchmark fingerprinting for a specific model.

        Args:
            model_info: Model information dictionary (must include 'model' key)

        Returns:
            List of benchmark results
        """
        if "model" not in model_info:
            logger.error(
                f"Model object missing for {model_info.get('model_id', 'unknown')}"
            )
            return []

        model = model_info["model"]
        model_id = model_info["model_id"]

        logger.info(f"Starting fingerprint benchmark for {model_id}")

        # Extract model outputs
        outputs = self.extract_model_outputs(model, model_info)

        # Prepare fingerprint data
        fingerprint_data = self.prepare_fingerprint_data(outputs)

        all_results = []

        # Benchmark each algorithm and operation
        for algorithm_name, algorithm_info in self.crypto_systems.items():
            for operation in algorithm_info["operations"]:
                results = self.benchmark_crypto_algorithm(
                    algorithm_name, operation, fingerprint_data
                )

                # Add model information to results
                for result in results:
                    result["model_id"] = model_id
                    result["model_architecture"] = model_info["architecture"]
                    result["model_complexity"] = model_info["complexity"]
                    result["model_parameters"] = model_info["parameters"]

                all_results.extend(results)

        logger.info(
            f"Completed fingerprint benchmark for {model_id}: "
            f"{len(all_results)} total operations"
        )

        return all_results

    def benchmark_multiple_models(
        self, models_info: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Benchmark fingerprinting for multiple models.

        Args:
            models_info: List of model information dictionaries

        Returns:
            List of all benchmark results
        """
        logger.info(f"Starting batch fingerprint benchmark: {len(models_info)} models")

        all_results = []

        for model_info in models_info:
            try:
                results = self.benchmark_model_fingerprinting(model_info)
                all_results.extend(results)

            except Exception as e:
                logger.error(
                    f"Failed to benchmark {model_info.get('model_id', 'unknown')}: {e}"
                )

                # Create error result
                error_result = {
                    "model_id": model_info.get("model_id", "unknown"),
                    "primitive_name": "unknown",
                    "operation": "unknown",
                    "success": False,
                    "error_message": str(e),
                }
                all_results.append(error_result)
                self.failed_benchmarks.append(error_result)

        logger.info("Batch fingerprint benchmark completed:")
        logger.info(f"  Successful operations: {len(self.crypto_results)}")
        logger.info(f"  Failed operations: {len(self.failed_benchmarks)}")

        return all_results

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary statistics of crypto benchmark results."""
        if not self.crypto_results:
            return {"error": "No successful crypto benchmarks"}

        summary = {
            "total_operations": len(self.crypto_results),
            "total_failed": len(self.failed_benchmarks),
            "algorithms": {},
            "operations": {},
            "input_size_analysis": {},
            "overall_stats": {},
        }

        # Group by algorithm
        algorithm_groups = {}
        for result in self.crypto_results:
            algorithm = result["primitive_name"]
            if algorithm not in algorithm_groups:
                algorithm_groups[algorithm] = []
            algorithm_groups[algorithm].append(result)

        for algorithm, results in algorithm_groups.items():
            execution_times = [r["execution_time"] for r in results]
            throughputs = [r["throughput_ops_per_sec"] for r in results]
            memory_usage = [r["memory_usage_mb"] for r in results]

            summary["algorithms"][algorithm] = {
                "count": len(results),
                "avg_execution_time": np.mean(execution_times),
                "avg_throughput": np.mean(throughputs),
                "avg_memory_usage_mb": np.mean(memory_usage),
                "execution_time_range": [
                    np.min(execution_times),
                    np.max(execution_times),
                ],
            }

        # Group by operation
        operation_groups = {}
        for result in self.crypto_results:
            operation = result["operation"]
            if operation not in operation_groups:
                operation_groups[operation] = []
            operation_groups[operation].append(result)

        for operation, results in operation_groups.items():
            execution_times = [r["execution_time"] for r in results]

            summary["operations"][operation] = {
                "count": len(results),
                "avg_execution_time": np.mean(execution_times),
                "algorithms": list(set(r["primitive_name"] for r in results)),
            }

        # Input size analysis
        size_groups = {}
        for result in self.crypto_results:
            size = result["input_size"]
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(result)

        for size, results in size_groups.items():
            execution_times = [r["execution_time"] for r in results]
            throughputs = [r["throughput_ops_per_sec"] for r in results]

            summary["input_size_analysis"][size] = {
                "count": len(results),
                "avg_execution_time": np.mean(execution_times),
                "avg_throughput": np.mean(throughputs),
            }

        # Overall statistics
        all_execution_times = [r["execution_time"] for r in self.crypto_results]
        all_throughputs = [r["throughput_ops_per_sec"] for r in self.crypto_results]
        all_memory_usage = [r["memory_usage_mb"] for r in self.crypto_results]

        summary["overall_stats"] = {
            "avg_execution_time": np.mean(all_execution_times),
            "avg_throughput": np.mean(all_throughputs),
            "avg_memory_usage_mb": np.mean(all_memory_usage),
            "total_input_sizes": len(set(r["input_size"] for r in self.crypto_results)),
            "execution_time_range": [
                np.min(all_execution_times),
                np.max(all_execution_times),
            ],
        }

        return summary

    def save_benchmark_results(self, output_dir) -> str:
        """Save crypto benchmark results to file.

        Args:
            output_dir: Output directory

        Returns:
            Path to saved results file
        """
        import json
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / "crypto_benchmark_results.json"

        results_data = {
            "config": self.config.to_dict(),
            "successful_benchmarks": self.crypto_results,
            "failed_benchmarks": self.failed_benchmarks,
            "summary": self.get_benchmark_summary(),
            "timestamp": time.time(),
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(f"Crypto benchmark results saved to {results_file}")
        return str(results_file)


# Convenience functions
def benchmark_model_fingerprints(
    config: PHAZEConfig, models_info: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Benchmark fingerprinting for a list of models.

    Args:
        config: PHAZE configuration
        models_info: List of model information dictionaries

    Returns:
        List of benchmark results
    """
    benchmarker = EnhancedCryptoBenchmark(config)
    return benchmarker.benchmark_multiple_models(models_info)


if __name__ == "__main__":
    # Example usage
    from .model_architectures import ModelComplexity, PHAZEModelFactory
    from .training_config import create_default_config

    config = create_default_config()
    config.experiment.benchmark_iterations = 3  # Quick test

    # Create test model
    model = PHAZEModelFactory.create_full_model(
        "simple", ModelComplexity.MINIMAL, 784, 10
    )

    model_info = {
        "model_id": "test_simple_minimal",
        "model": model,
        "architecture": "simple",
        "complexity": "minimal",
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_size": 784,
        "output_size": 10,
    }

    benchmarker = EnhancedCryptoBenchmark(config)
    results = benchmarker.benchmark_model_fingerprinting(model_info)

    print(f"Crypto benchmark completed: {len(results)} operations")

    successful = [r for r in results if r["success"]]
    print(f"Successful operations: {len(successful)}")

    if successful:
        avg_time = np.mean([r["execution_time"] for r in successful])
        avg_throughput = np.mean([r["throughput_ops_per_sec"] for r in successful])
        print(f"Average execution time: {avg_time:.4f}s")
        print(f"Average throughput: {avg_throughput:.1f} ops/sec")
