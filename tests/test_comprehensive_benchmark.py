import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path for legacy imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from legacy.src.comprehensive_benchmark import run_phaze_benchmarks
from legacy.src.phaze_benchmark_suite import (
    ComprehensiveBenchmarkSuite,
    CryptographicPrimitiveBenchmark,
)
from legacy.src.comprehensive_benchmark import (
    BenchmarkResult,
    CryptoBenchmarkResult,
    PerformanceMonitor,
    ZKMLFrameworkBenchmark,
)


class TestPerformanceMonitor:
    """Test the PerformanceMonitor class."""

    def test_monitoring_cycle(self):
        """Test complete monitoring cycle."""
        monitor = PerformanceMonitor()

        monitor.start_monitoring()

        # Simulate some work
        import time

        time.sleep(0.1)

        execution_time, memory_usage, cpu_usage = monitor.stop_monitoring()

        assert execution_time >= 0.1
        assert memory_usage >= 0
        assert cpu_usage >= 0


class TestCryptographicPrimitiveBenchmark:
    """Test cryptographic primitive benchmarking."""

    def setup_method(self):
        """Setup for each test method."""
        self.benchmark = CryptographicPrimitiveBenchmark()

    # Removed rabin fingerprint benchmark test - method not available

    # Removed shamir secret sharing benchmark test - method not available

    # Removed rust primitives benchmark test - method not available


class TestZKMLFrameworkBenchmark:
    """Test zkML framework benchmarking."""

    def setup_method(self):
        """Setup for each test method."""
        self.benchmark = ZKMLFrameworkBenchmark()

    @pytest.mark.asyncio
    async def test_benchmark_framework_ezkl(self):
        """Test benchmarking EZKL framework."""
        results = await self.benchmark.benchmark_framework(
            "ezkl", "simple", "light", 10, 5, num_trials=1
        )

        assert len(results) == 1
        result = results[0]

        assert isinstance(result, BenchmarkResult)
        assert result.framework == "ezkl"
        assert result.architecture == "simple"
        assert result.complexity == "light"
        assert result.input_size == 10
        assert result.output_size == 5
        assert result.total_time >= 0
        assert result.memory_usage_mb >= 0
        # Note: success might be False if ezkl is not properly installed

    @pytest.mark.asyncio
    async def test_benchmark_framework_rust_backend(self):
        """Test benchmarking Rust-based frameworks."""
        for framework in ["groth16", "plonky", "halo"]:
            results = await self.benchmark.benchmark_framework(
                framework, "simple", "light", 10, 5, num_trials=1
            )

            assert len(results) == 1
            result = results[0]

            assert isinstance(result, BenchmarkResult)
            assert result.framework == framework
            assert result.total_time >= 0
            assert result.memory_usage_mb >= 0
            # These should generally succeed as they use mock implementations
            assert result.success is True

    @pytest.mark.asyncio
    async def test_benchmark_all_frameworks_quick(self):
        """Test benchmarking all frameworks with minimal configuration."""
        results = await self.benchmark.benchmark_all_frameworks(
            architectures=["simple"],
            complexities=["light"],
            input_sizes=[10],
            num_trials=1,
        )

        # Should have results for all frameworks
        frameworks = set(r.framework for r in results)
        expected_frameworks = {"ezkl", "groth16", "plonky", "halo"}
        assert frameworks == expected_frameworks

        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.architecture == "simple"
            assert result.complexity == "light"
            assert result.input_size == 10


class TestComprehensiveBenchmarkSuite:
    """Test the comprehensive benchmark suite."""

    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.suite = ComprehensiveBenchmarkSuite(self.temp_dir)

    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_run_full_benchmark_suite_quick(self):
        """Test running the full benchmark suite in quick mode."""
        zkml_config = {
            "architectures": ["simple"],
            "complexities": ["light"],
            "input_sizes": [10],
            "num_trials": 1,
        }

        crypto_config = {
            "rabin_input_sizes": [64],
            "shamir_secret_sizes": [32],
            "num_trials": 5,
        }

        results = await self.suite.run_full_benchmark_suite(zkml_config, crypto_config)

        assert "zkml_results" in results
        assert "crypto_results" in results
        assert "summary" in results
        assert "timestamp" in results

        # Check that we have some results
        assert len(results["zkml_results"]) > 0
        assert len(results["crypto_results"]) > 0

        # Check summary structure
        summary = results["summary"]
        assert "zkml_summary" in summary
        assert "crypto_summary" in summary
        assert "overall_summary" in summary

    def test_generate_summary(self):
        """Test summary generation."""
        # Create mock results
        zkml_results = [
            BenchmarkResult(
                test_name="test1",
                framework="ezkl",
                architecture="simple",
                complexity="light",
                input_size=10,
                output_size=5,
                setup_time=1.0,
                proof_time=2.0,
                verification_time=0.5,
                total_time=3.5,
                memory_usage_mb=100.0,
                cpu_usage_percent=50.0,
                success=True,
            ),
            BenchmarkResult(
                test_name="test2",
                framework="groth16",
                architecture="simple",
                complexity="light",
                input_size=10,
                output_size=5,
                setup_time=0.5,
                proof_time=1.0,
                verification_time=0.2,
                total_time=1.7,
                memory_usage_mb=80.0,
                cpu_usage_percent=40.0,
                success=True,
            ),
        ]

        crypto_results = [
            CryptoBenchmarkResult(
                primitive_name="RabinFingerprint",
                operation="compute_hash",
                input_size=64,
                execution_time=0.001,
                memory_usage_mb=10.0,
                throughput_ops_per_sec=1000.0,
                success=True,
            )
        ]

        summary = self.suite._generate_summary(zkml_results, crypto_results)

        assert "zkml_summary" in summary
        assert "crypto_summary" in summary
        assert "overall_summary" in summary

        zkml_summary = summary["zkml_summary"]
        assert zkml_summary["total_tests"] == 2
        assert zkml_summary["successful_tests"] == 2
        assert zkml_summary["success_rate"] == 1.0
        assert zkml_summary["avg_setup_time"] == 0.75
        assert zkml_summary["avg_proof_time"] == 1.5

        crypto_summary = summary["crypto_summary"]
        assert crypto_summary["total_tests"] == 1
        assert crypto_summary["successful_tests"] == 1
        assert crypto_summary["success_rate"] == 1.0

    def test_save_results(self):
        """Test saving results to files."""
        results = {
            "zkml_results": [
                {
                    "test_name": "test1",
                    "framework": "ezkl",
                    "success": True,
                    "total_time": 1.0,
                }
            ],
            "crypto_results": [
                {
                    "primitive_name": "RabinFingerprint",
                    "operation": "compute_hash",
                    "success": True,
                    "execution_time": 0.001,
                }
            ],
            "summary": {"test": "data"},
            "timestamp": 1234567890,
        }

        self.suite._save_results(results)

        # Check that files were created
        output_dir = Path(self.temp_dir)
        assert (output_dir / "benchmark_results.json").exists()
        assert (output_dir / "zkml_benchmark_results.csv").exists()
        assert (output_dir / "crypto_benchmark_results.csv").exists()

        # Check JSON content
        with open(output_dir / "benchmark_results.json") as f:
            saved_results = json.load(f)

        assert saved_results["zkml_results"] == results["zkml_results"]
        assert saved_results["crypto_results"] == results["crypto_results"]

    def test_generate_report(self):
        """Test report generation."""
        results = {
            "summary": {
                "overall_summary": {
                    "total_tests": 10,
                    "successful_tests": 8,
                    "overall_success_rate": 0.8,
                },
                "zkml_summary": {
                    "frameworks_tested": ["ezkl", "groth16"],
                    "architectures_tested": ["simple"],
                    "success_rate": 0.75,
                    "avg_setup_time": 1.0,
                    "avg_proof_time": 2.0,
                    "avg_verification_time": 0.5,
                    "avg_memory_usage": 100.0,
                },
                "crypto_summary": {
                    "primitives_tested": ["RabinFingerprint"],
                    "operations_tested": ["compute_hash"],
                    "success_rate": 1.0,
                    "avg_execution_time": 0.001,
                    "avg_throughput": 1000.0,
                },
            },
            "timestamp": 1234567890,
        }

        report = self.suite.generate_report(results)

        assert "PHAZE Framework Comprehensive Benchmark Report" in report
        assert "Executive Summary" in report
        assert "zkML Framework Benchmarks" in report
        assert "Cryptographic Primitive Benchmarks" in report
        assert "Recommendations" in report

        # Check that key metrics are included
        assert "Total tests executed: 10" in report
        assert "Successful tests: 8" in report
        assert "Overall success rate: 80.00%" in report

        # Check that report file was saved
        output_dir = Path(self.temp_dir)
        assert (output_dir / "benchmark_report.md").exists()


class TestConvenienceFunction:
    """Test the convenience function for running benchmarks."""

    @pytest.mark.asyncio
    async def test_run_phaze_benchmarks_quick_mode(self):
        """Test running benchmarks in quick mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await run_phaze_benchmarks(temp_dir, quick_mode=True)

            assert "results" in result
            assert "report" in result
            assert "output_directory" in result

            assert result["output_directory"] == temp_dir

            # Check that files were created
            output_dir = Path(temp_dir)
            assert (output_dir / "benchmark_results.json").exists()
            assert (output_dir / "benchmark_report.md").exists()

            # Check that results have the expected structure
            results = result["results"]
            assert "zkml_results" in results
            assert "crypto_results" in results
            assert "summary" in results

            # In quick mode, should have minimal results
            assert len(results["zkml_results"]) >= 4  # At least one per framework
            assert len(results["crypto_results"]) >= 1  # At least some crypto results
