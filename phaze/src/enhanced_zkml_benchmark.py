"""
Enhanced ZK proof benchmarking for PHAZE framework.

This module provides comprehensive benchmarking of zero-knowledge machine learning
frameworks including ezkl and risc-zero, with detailed performance metrics.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import torch
import torch.nn as nn

from .training_config import PHAZEConfig
from .zkml_integration import ZKMLProverVerifier

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Utility class for monitoring memory usage during operations."""

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
        """Get current memory usage in MB (CPU + GPU if available)."""
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Add GPU memory if available
        gpu_memory = 0.0
        try:
            if hasattr(self, 'device_type'):
                if self.device_type == "cuda" and torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
                elif self.device_type == "mps" and torch.backends.mps.is_available():
                    # MPS doesn't have direct memory query, use approximation
                    gpu_memory = torch.mps.current_allocated_memory() / 1024 / 1024
        except (AttributeError, RuntimeError):
            # Fallback if GPU memory query fails
            pass
            
        return cpu_memory + gpu_memory

    def get_detailed_memory_stats(self) -> dict:
        """Get detailed breakdown of CPU and GPU memory usage."""
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024 / 1024
        
        stats = {
            "cpu_memory_mb": cpu_memory,
            "gpu_memory_mb": 0.0,
            "total_memory_mb": cpu_memory,
            "device_type": getattr(self, "device_type", "cpu")
        }
        
        try:
            if hasattr(self, 'device_type'):
                if self.device_type == "cuda" and torch.cuda.is_available():
                    gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                    gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                    stats.update({
                        "gpu_memory_mb": gpu_allocated,
                        "gpu_reserved_mb": gpu_reserved,
                        "total_memory_mb": cpu_memory + gpu_allocated,
                        "gpu_device_name": torch.cuda.get_device_name()
                    })
                elif self.device_type == "mps" and torch.backends.mps.is_available():
                    gpu_memory = torch.mps.current_allocated_memory() / 1024 / 1024
                    stats.update({
                        "gpu_memory_mb": gpu_memory,
                        "total_memory_mb": cpu_memory + gpu_memory,
                        "gpu_device_name": "Apple Metal GPU"
                    })
        except (AttributeError, RuntimeError):
            pass
            
        return stats


class EnhancedZKMLBenchmark:
    """Enhanced benchmarking for zkML frameworks."""

    def __init__(self, config: PHAZEConfig):
        """Initialize benchmarker with configuration.

        Args:
            config: PHAZE configuration object
        """
        self.config = config
        # Enhanced device selection supporting MPS (Mac M1), CUDA, and CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_type = "mps"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"
        self.benchmark_results = []
        self.failed_benchmarks = []

    def create_sample_input(self, model_info: Dict[str, Any]) -> torch.Tensor:
        """Create sample input tensor for the model.

        Args:
            model_info: Model information dictionary

        Returns:
            Sample input tensor
        """
        architecture = model_info["architecture"]

        if architecture == "conv":
            input_channels = model_info.get("input_channels", 1)
            spatial_size = model_info.get("spatial_size", 28)
            return torch.randn(1, input_channels, spatial_size, spatial_size)
        else:
            input_size = model_info.get("input_size", 784)
            return torch.randn(1, input_size)

    async def benchmark_single_model(
        self, model: nn.Module, model_info: Dict[str, Any], framework: str = "ezkl"
    ) -> Dict[str, Any]:
        """Benchmark a single model with specified framework.

        Args:
            model: PyTorch model to benchmark
            model_info: Model information dictionary
            framework: zkML framework to use ("ezkl" or "risc_zero")

        Returns:
            Benchmark results dictionary
        """
        model_id = model_info["model_id"]
        logger.info(f"Benchmarking {model_id} with {framework}")

        # Initialize result structure
        result = {
            "model_id": model_id,
            "framework": framework,
            "architecture": model_info["architecture"],
            "complexity": model_info["complexity"],
            "model_parameters": model_info["parameters"],
            "success": False,
            "error_message": None,
            "setup_time": 0.0,
            "proof_time": 0.0,
            "verification_time": 0.0,
            "total_time": 0.0,
            "memory_usage_mb": 0.0,
            "cpu_memory_mb": 0.0,
            "gpu_memory_mb": 0.0,
            "device_type": getattr(self, "device_type", "cpu"),
            "proof_size_bytes": 0,
            "setup_memory": {},
            "proof_memory": {},
            "verification_memory": {},
        }

        try:
            # Create sample input
            sample_input = self.create_sample_input(model_info)
            sample_input = sample_input.to(self.device)

            # Initialize zkML system
            if framework == "ezkl":
                zkml_system = ZKMLProverVerifier(model, framework)
            elif framework == "risc_zero":
                zkml_system = await self._create_risc_zero_system(model, model_info)
            else:
                raise ValueError(f"Unknown framework: {framework}")

            # Benchmark setup phase\n            setup_start_time = time.time()\n            memory_profiler = MemoryProfiler()\n            # Pass device info to memory profiler\n            if hasattr(self, 'device_type'):\n                memory_profiler.device_type = self.device_type\n            memory_profiler.start_monitoring()

            try:
                if framework == "ezkl":
                    await zkml_system.async_setup(sample_input)
                else:
                    await self._setup_risc_zero(zkml_system, sample_input)

                memory_profiler.update_peak()
                setup_time = time.time() - setup_start_time
                setup_memory = memory_profiler.stop_monitoring()

                result["setup_time"] = setup_time
                result["setup_memory"] = setup_memory

                logger.info(f"  Setup completed in {setup_time:.2f}s")

            except Exception as e:
                logger.error(f"  Setup failed: {e}")
                result["error_message"] = f"Setup failed: {str(e)}"
                return result

            # Benchmark proof generation
            proof_times = []
            proof_memory_stats = []

            for i in range(self.config.experiment.benchmark_iterations):
                memory_profiler = MemoryProfiler()
                memory_profiler.start_monitoring()

                proof_start_time = time.time()

                try:
                    if framework == "ezkl":
                        proof, output = await zkml_system.generate_proof(sample_input)
                    else:
                        proof, output = await self._generate_risc_zero_proof(
                            zkml_system, sample_input
                        )

                    memory_profiler.update_peak()
                    proof_time = time.time() - proof_start_time
                    proof_memory = memory_profiler.stop_monitoring()

                    proof_times.append(proof_time)
                    proof_memory_stats.append(proof_memory)

                    # Store proof for verification and size measurement
                    if i == 0:  # Store first proof for verification
                        stored_proof = proof
                        if isinstance(proof, dict) and "proof" in proof:
                            # Estimate proof size (rough approximation)
                            proof_str = json.dumps(proof)
                            result["proof_size_bytes"] = len(proof_str.encode("utf-8"))

                except Exception as e:
                    logger.error(f"  Proof generation {i + 1} failed: {e}")
                    if i == 0:  # If first proof fails, stop benchmarking
                        result["error_message"] = f"Proof generation failed: {str(e)}"
                        return result
                    # Otherwise continue with successful proofs
                    continue

            if not proof_times:
                result["error_message"] = "No successful proof generations"
                return result

            # Calculate proof generation statistics
            result["proof_time"] = np.mean(proof_times)
            result["proof_time_std"] = np.std(proof_times)
            result["proof_time_min"] = np.min(proof_times)
            result["proof_time_max"] = np.max(proof_times)

            # Average memory usage during proof generation
            if proof_memory_stats:
                result["proof_memory"] = {
                    "peak_mb": np.mean([m["peak_mb"] for m in proof_memory_stats]),
                    "peak_increase_mb": np.mean(
                        [m["peak_increase_mb"] for m in proof_memory_stats]
                    ),
                }

            logger.info(
                f"  Proof generation: {result['proof_time']:.2f}s avg "
                f"({len(proof_times)} successful)"
            )

            # Benchmark verification
            verification_times = []
            verification_memory_stats = []

            for i in range(self.config.experiment.benchmark_iterations):
                memory_profiler = MemoryProfiler()
                memory_profiler.start_monitoring()

                verification_start_time = time.time()

                try:
                    if framework == "ezkl":
                        verified = await zkml_system.verify_proof(
                            stored_proof, sample_input
                        )
                    else:
                        verified = await self._verify_risc_zero_proof(
                            zkml_system, stored_proof, sample_input
                        )

                    memory_profiler.update_peak()
                    verification_time = time.time() - verification_start_time
                    verification_memory = memory_profiler.stop_monitoring()

                    verification_times.append(verification_time)
                    verification_memory_stats.append(verification_memory)

                    if not verified and i == 0:
                        logger.warning(f"  Proof verification failed for {model_id}")

                except Exception as e:
                    logger.error(f"  Verification {i + 1} failed: {e}")
                    continue

            if verification_times:
                # Calculate verification statistics
                result["verification_time"] = np.mean(verification_times)
                result["verification_time_std"] = np.std(verification_times)
                result["verification_time_min"] = np.min(verification_times)
                result["verification_time_max"] = np.max(verification_times)

                # Average memory usage during verification
                if verification_memory_stats:
                    result["verification_memory"] = {
                        "peak_mb": np.mean(
                            [m["peak_mb"] for m in verification_memory_stats]
                        ),
                        "peak_increase_mb": np.mean(
                            [m["peak_increase_mb"] for m in verification_memory_stats]
                        ),
                    }

                logger.info(
                    f"  Verification: {result['verification_time']:.2f}s avg "
                    f"({len(verification_times)} successful)"
                )

            # Calculate total time and overall memory usage
            result["total_time"] = (
                result["setup_time"]
                + result["proof_time"]
                + result["verification_time"]
            )

            # Overall memory usage (peak across all phases)
            peak_memories = []
            if result["setup_memory"]:
                peak_memories.append(result["setup_memory"]["peak_mb"])
            if result["proof_memory"]:
                peak_memories.append(result["proof_memory"]["peak_mb"])
            if result["verification_memory"]:
                peak_memories.append(result["verification_memory"]["peak_mb"])

            if peak_memories:
                result["memory_usage_mb"] = max(peak_memories)
                
                # Get detailed memory breakdown from the peak phase
                peak_profiler = MemoryProfiler()
                if hasattr(self, 'device_type'):
                    peak_profiler.device_type = self.device_type
                detailed_stats = peak_profiler.get_detailed_memory_stats()
                result["cpu_memory_mb"] = detailed_stats["cpu_memory_mb"]
                result["gpu_memory_mb"] = detailed_stats["gpu_memory_mb"]

            result["success"] = True
            logger.info("  ✅ Benchmark completed successfully")
            logger.info(f"     Total time: {result['total_time']:.2f}s")
            logger.info(f"     Peak memory: {result['memory_usage_mb']:.1f}MB")

            # Cleanup
            if hasattr(zkml_system, "cleanup"):
                zkml_system.cleanup()

        except Exception as e:
            logger.error(f"  ❌ Benchmark failed: {e}")
            result["error_message"] = str(e)

        return result

    async def _create_risc_zero_system(
        self, model: nn.Module, model_info: Dict[str, Any]
    ):
        """Create risc-zero zkML system using Rust backend."""
        from .rust_zkml_backend import RustRiscZeroBackend
        
        # Create the RISC Zero backend
        risc_zero_system = RiscZeroBackendWrapper(model, model_info)
        logger.info("Using RISC Zero backend implementation")
        return risc_zero_system

    async def _setup_risc_zero(self, system, sample_input: torch.Tensor):
        """Setup risc-zero system."""
        await system.setup(sample_input)

    async def _generate_risc_zero_proof(self, system, sample_input: torch.Tensor):
        """Generate proof with risc-zero."""
        return await system.generate_proof(sample_input)

    async def _verify_risc_zero_proof(self, system, proof, sample_input: torch.Tensor):
        """Verify proof with risc-zero."""
        return await system.verify_proof(proof, sample_input)

    async def benchmark_multiple_models(
        self, models_info: List[Dict[str, Any]], frameworks: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Benchmark multiple models across specified frameworks.

        Args:
            models_info: List of model information dictionaries (should include 'model' key)
            frameworks: List of frameworks to test (defaults to config)

        Returns:
            List of benchmark results
        """
        if frameworks is None:
            frameworks = self.config.experiment.zkml_frameworks

        logger.info(
            f"Starting batch benchmark: {len(models_info)} models, "
            f"{len(frameworks)} frameworks"
        )

        all_results = []

        for model_info in models_info:
            if "model" not in model_info:
                logger.error(
                    f"Model object missing for {model_info.get('model_id', 'unknown')}"
                )
                continue

            model = model_info["model"]

            for framework in frameworks:
                try:
                    result = await self.benchmark_single_model(
                        model, model_info, framework
                    )
                    all_results.append(result)

                    if result["success"]:
                        self.benchmark_results.append(result)
                    else:
                        self.failed_benchmarks.append(result)

                except Exception as e:
                    logger.error(
                        f"Failed to benchmark {model_info.get('model_id')} "
                        f"with {framework}: {e}"
                    )

                    failed_result = {
                        "model_id": model_info.get("model_id", "unknown"),
                        "framework": framework,
                        "success": False,
                        "error_message": str(e),
                    }
                    all_results.append(failed_result)
                    self.failed_benchmarks.append(failed_result)

        logger.info("Batch benchmark completed:")
        logger.info(f"  Successful: {len(self.benchmark_results)}")
        logger.info(f"  Failed: {len(self.failed_benchmarks)}")

        return all_results

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary statistics of benchmark results."""
        if not self.benchmark_results:
            return {"error": "No successful benchmarks"}

        summary = {
            "total_benchmarks": len(self.benchmark_results),
            "total_failed": len(self.failed_benchmarks),
            "frameworks": {},
            "architectures": {},
            "overall_stats": {},
        }

        # Group by framework
        framework_groups = {}
        for result in self.benchmark_results:
            framework = result["framework"]
            if framework not in framework_groups:
                framework_groups[framework] = []
            framework_groups[framework].append(result)

        for framework, results in framework_groups.items():
            proof_times = [r["proof_time"] for r in results]
            verification_times = [r["verification_time"] for r in results]
            memory_usage = [r["memory_usage_mb"] for r in results]

            summary["frameworks"][framework] = {
                "count": len(results),
                "avg_proof_time": np.mean(proof_times),
                "avg_verification_time": np.mean(verification_times),
                "avg_memory_usage_mb": np.mean(memory_usage),
                "success_rate": len(results)
                / (
                    len(results)
                    + len(
                        [
                            f
                            for f in self.failed_benchmarks
                            if f["framework"] == framework
                        ]
                    )
                ),
            }

        # Group by architecture
        arch_groups = {}
        for result in self.benchmark_results:
            arch = result["architecture"]
            if arch not in arch_groups:
                arch_groups[arch] = []
            arch_groups[arch].append(result)

        for arch, results in arch_groups.items():
            proof_times = [r["proof_time"] for r in results]
            parameters = [r["model_parameters"] for r in results]

            summary["architectures"][arch] = {
                "count": len(results),
                "avg_proof_time": np.mean(proof_times),
                "avg_parameters": np.mean(parameters),
                "parameter_range": [np.min(parameters), np.max(parameters)],
            }

        # Overall statistics
        all_proof_times = [r["proof_time"] for r in self.benchmark_results]
        all_verification_times = [
            r["verification_time"] for r in self.benchmark_results
        ]
        all_memory_usage = [r["memory_usage_mb"] for r in self.benchmark_results]

        summary["overall_stats"] = {
            "avg_proof_time": np.mean(all_proof_times),
            "avg_verification_time": np.mean(all_verification_times),
            "avg_memory_usage_mb": np.mean(all_memory_usage),
            "proof_time_range": [np.min(all_proof_times), np.max(all_proof_times)],
            "verification_time_range": [
                np.min(all_verification_times),
                np.max(all_verification_times),
            ],
        }

        return summary

    def save_benchmark_results(self, output_dir: Path) -> str:
        """Save benchmark results to file.

        Args:
            output_dir: Output directory

        Returns:
            Path to saved results file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / "zkml_benchmark_results.json"

        results_data = {
            "config": self.config.to_dict(),
            "successful_benchmarks": self.benchmark_results,
            "failed_benchmarks": self.failed_benchmarks,
            "summary": self.get_benchmark_summary(),
            "timestamp": time.time(),
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(f"Benchmark results saved to {results_file}")
        return str(results_file)


class RiscZeroBackendWrapper:
    """Wrapper for the Rust RISC Zero backend to work with the benchmark system."""

    def __init__(self, model: nn.Module, model_info: Dict[str, Any]):
        self.model = model
        self.model_info = model_info
        from .rust_zkml_backend import RustRiscZeroBackend
        self.backend = RustRiscZeroBackend()
        self.is_setup = False

    async def setup(self, sample_input: torch.Tensor):
        """Setup the RISC Zero backend with model parameters."""
        # Extract model parameters
        params = {
            "model_type": self.model_info.get("architecture", "simple"),
            "input_size": str(sample_input.numel()),
            "output_size": str(self.model_info.get("output_size", 10)),
            "complexity": self.model_info.get("complexity", "minimal"),
        }
        
        # Setup the backend
        setup_result = self.backend.setup(params)
        self.is_setup = True

    async def generate_proof(self, sample_input: torch.Tensor):
        """Generate proof using the RISC Zero backend."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")
        
        # Get model state dict as weights
        model_weights = self.model.state_dict()
        
        # Generate proof using the Rust backend
        proof_dict = self.backend.prove(sample_input, model_weights)
        
        # Run actual model to get expected output for comparison
        with torch.no_grad():
            output = self.model(sample_input)
        
        return proof_dict, output.numpy().tolist()

    async def verify_proof(self, proof, sample_input: torch.Tensor):
        """Verify proof using the RISC Zero backend."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")
        
        # Get expected outputs
        with torch.no_grad():
            expected_outputs = self.model(sample_input)
        
        # Verify using the Rust backend
        return self.backend.verify(proof, expected_outputs)


# Convenience functions
async def benchmark_model_list(
    config: PHAZEConfig, models_info: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Benchmark a list of models with all configured frameworks.

    Args:
        config: PHAZE configuration
        models_info: List of model information dictionaries

    Returns:
        List of benchmark results
    """
    benchmarker = EnhancedZKMLBenchmark(config)
    return await benchmarker.benchmark_multiple_models(models_info)


if __name__ == "__main__":
    # Example usage
    import asyncio

    from .model_architectures import ModelComplexity, PHAZEModelFactory
    from .training_config import create_default_config

    async def test_benchmark():
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

        benchmarker = EnhancedZKMLBenchmark(config)
        result = await benchmarker.benchmark_single_model(model, model_info, "ezkl")

        print("Benchmark result:")
        print(f"  Success: {result['success']}")
        if result["success"]:
            print(f"  Setup time: {result['setup_time']:.3f}s")
            print(f"  Proof time: {result['proof_time']:.3f}s")
            print(f"  Verification time: {result['verification_time']:.3f}s")
            print(f"  Memory usage: {result['memory_usage_mb']:.1f}MB")
        else:
            print(f"  Error: {result['error_message']}")

    asyncio.run(test_benchmark())
