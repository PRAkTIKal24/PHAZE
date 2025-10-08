# PHAZE: Probabilistic Hashing And Zero-knowledge proofs for Early-exit models

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/PRAkTIKal24/PHAZE/actions/workflows/pytest.yml/badge.svg)](https://github.com/PRAkTIKal24/PHAZE/actions/workflows/pytest.yml)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

PHAZE is a comprehensive framework for benchmarking and evaluating zero-knowledge machine learning (zkML) systems with support for early-exit models and privacy-preserving inference at the Large Hadron Collider (LHC). The framework provides tools for comparing different zkML frameworks, cryptographic primitives, and model architectures with a focus on low-latency ML inference for high energy physics applications.

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Development Installation](#development-installation)
- [💡 Usage Examples](#-usage-examples)
  - [PHAZE CLI (Recommended)](#phaze-cli-recommended)
  - [Direct Python Examples](#direct-python-examples)
  - [Model Creation and Testing](#model-creation-and-testing)
  - [Zero-Knowledge ML Integration](#zero-knowledge-ml-integration)
  - [Cryptographic Primitives](#cryptographic-primitives)
- [🏗️ Architecture](#️-architecture)
  - [Project Structure](#project-structure)
  - [Model Architecture System](#model-architecture-system)
  - [zkML Backend Integration](#zkml-backend-integration)
- [📊 Benchmarking](#-benchmarking)
  - [Performance Metrics](#performance-metrics)
  - [Typical Performance (Reference Hardware)](#typical-performance-reference-hardware)
  - [Running Custom Benchmarks](#running-custom-benchmarks)
- [🧪 Testing](#-testing)
  - [Running Tests](#running-tests)
  - [Test Structure](#test-structure)
- [🔧 Configuration](#-configuration)
  - [Environment Variables](#environment-variables)
  - [Dependency Groups](#dependency-groups)
- [🚨 Known Limitations](#-known-limitations)
  - [Current Constraints](#current-constraints)
  - [Performance Considerations](#performance-considerations)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [📞 Support](#-support)

## ✨ Features

### 🔐 Zero-Knowledge ML Support
- **EZKL Integration**: Full PyTorch model compilation and proof generation
- **RISC Zero**: High-performance zkVM integration for ML inference
- **Groth16, Plonky, Halo**: Mock implementations for comparative benchmarking
- **Unified Interface**: Consistent API across different zkML backends

### 🧠 Model Architectures  
- **Early-Exit Models**: Adaptive inference with confidence-based termination
- **Multi-Exit Networks**: Multiple decision points for optimal latency/accuracy trade-offs
- **Convolutional Models**: CNN architectures for image-like data
- **Transformer Models**: Attention-based architectures for sequence data
- **Complexity Scaling**: Minimal to extreme complexity levels for performance testing

### 🔒 Cryptographic Primitives
- **Rabin Fingerprinting**: Polynomial-based hashing for data integrity verification
- **Shamir Secret Sharing**: Threshold secret sharing with configurable parameters
- **Field Operations**: Finite field arithmetic with multiple field sizes
- **Hash Functions**: Rust-optimized SHA256 and Keccak256 implementations

### 📊 Comprehensive Benchmarking
- **Performance Metrics**: Setup time, proof generation, verification time, memory usage
- **Scalability Testing**: Performance across different input sizes and model complexities
- **Framework Comparison**: Side-by-side evaluation of zkML approaches
- **Automated Reporting**: Markdown reports with performance recommendations

## 🚀 Quick Start

### Installation

**Prerequisites**: Python 3.10+, [uv](https://docs.astral.sh/uv/) package manager

```bash
# Clone the repository
git clone https://github.com/PRAkTIKal24/PHAZE.git
cd PHAZE

# Option 1: Automated installation (recommended)
./install_dev.sh

# Option 2: Manual installation
uv pip install -e .
uv run python dev_setup.py

# Verify installation and CLI
uv run phaze --help
uv run phaze --list-benchmarks
```

> **Note**: PHAZE uses a mixed Python/Rust architecture. The additional setup step (`dev_setup.py`) is required to ensure the CLI works correctly with editable installs.

### Development Installation

```bash
# Install with all development dependencies
uv sync --group dev --group test

# Setup editable install (required for CLI)
uv run python dev_setup.py

# Verify installation
uv run python -c "import phaze; print(f'PHAZE version: {phaze.__version__}')"

# Run tests
uv run pytest
```

## 💡 Usage Examples

### PHAZE CLI (Recommended)

**List Available Benchmarks**
```bash
uv run phaze --list-benchmarks
```

**Run Default Benchmarks**
```bash
# Runs basic_benchmark with --mode all --output-dir plots/
uv run phaze
```

**Run Specific Benchmarks**
```bash
# Run basic benchmark in quick mode
uv run phaze -b basic_benchmark --mode standard --quick

# Run RISC Zero benchmarks
uv run phaze -b risc_zero --mode standalone

# Get help for a specific benchmark
uv run phaze --help-benchmark basic_benchmark
```

### Direct Python Examples

**Quick Start Example**
```bash
# Run standard benchmarks in quick mode
uv run python examples/run_basic_benchmark.py --mode standard --quick

# Run RISC Zero specific benchmarks  
uv run python examples/run_risc_zero.py

# Run comprehensive benchmarks (all frameworks)
uv run python examples/run_basic_benchmark.py --mode all
```

**Using the Python API**
```python
import asyncio
from phaze import run_phaze_benchmarks

async def main():
    # Run comprehensive benchmarks
    results = await run_phaze_benchmarks(
        output_dir="./benchmark_results",
        quick_mode=True  # For demonstration
    )
    
    # Print summary
    summary = results["results"]["summary"]
    print(f"Success rate: {summary['overall_summary']['overall_success_rate']:.1%}")
    print(f"Frameworks tested: {summary['zkml_summary']['frameworks_tested']}")

asyncio.run(main())
```

### Model Creation and Testing

```python
from phaze import PHAZEModelFactory, ModelComplexity
import torch

# Create different model architectures
simple_model = PHAZEModelFactory.create_early_exit_model(
    architecture="simple",
    complexity=ModelComplexity.LIGHT,
    input_size=128,
    output_size=10
)

conv_model = PHAZEModelFactory.create_early_exit_model(
    architecture="conv",
    complexity=ModelComplexity.MEDIUM,
    input_size=32*32*3,  # CIFAR-10 sized input
    output_size=10
)

multi_exit_model = PHAZEModelFactory.create_early_exit_model(
    architecture="multi_exit",
    complexity=ModelComplexity.HEAVY,
    input_size=256,
    output_size=100
)

# Test inference
input_data = torch.randn(1, 128)
output = simple_model(input_data)
confidence = simple_model.get_confidence(input_data)

print(f"Output shape: {output.shape}")
print(f"Confidence: {confidence.item():.3f}")
```

### Zero-Knowledge ML Integration

```python
import asyncio
from phaze import create_backend, ZKMLFramework
from phaze.src.model_architectures import PHAZEModelFactory
import torch

async def zkml_example():
    # Create a model
    model = PHAZEModelFactory.create_early_exit_model("simple", "light")
    
    # Create zkML backend (EZKL, RISC Zero, etc.)
    backend = create_backend(ZKMLFramework.EZKL, model, "test_circuit")
    
    # Prepare input
    input_data = torch.randn(1, 10)
    
    # Setup (compile model, generate circuits)
    await backend.setup(input_data)
    
    # Generate proof
    proof, output = await backend.generate_proof(input_data)
    
    # Verify proof
    is_valid = await backend.verify_proof(proof, input_data)
    
    print(f"Proof generated: {len(proof)} bytes")
    print(f"Verification: {'✅ Valid' if is_valid else '❌ Invalid'}")

asyncio.run(zkml_example())
```

### Cryptographic Primitives

```python
from phaze import RabinFingerprint, ShamirSecretSharing

# Rabin Fingerprinting for data integrity
rabin = RabinFingerprint(field_size=2**31 - 1, degree=5)
data = [1, 2, 3, 4, 5]
hash_value = rabin.compute_hash(data)
print(f"Rabin hash: {hash_value}")

# Shamir Secret Sharing
sss = ShamirSecretSharing(threshold=3, num_shares=5, field_size=2**31 - 1)
secret_data = b"my_secret_key_data"

# Generate shares
shares = sss.generate_shares(secret_data)
print(f"Generated {len(shares)} shares")

# Reconstruct from subset
reconstructed = sss.reconstruct_secret(shares[:3])  # Use any 3 shares
print(f"Reconstruction: {'✅ Success' if reconstructed == secret_data else '❌ Failed'}")
```

## 🏗️ Architecture

### Project Structure

```
phaze/
├── __init__.py                     # Main package exports
├── phaze.py                        # CLI interface  
└── src/
    ├── model_architectures.py      # Model factory and definitions
    ├── zkml_integration.py         # zkML framework integration
    ├── zkml_backends.py            # Concrete zkML implementations
    ├── zkml_framework_interface.py # Abstract zkML interfaces
    ├── rust_zkml_backend.py        # Rust-Python bindings
    ├── crypto_primitives.py        # Cryptographic implementations
    ├── comprehensive_benchmark.py  # Benchmarking system
    ├── phaze_benchmark_suite.py    # Benchmark configurations
    └── benchmarking.py             # Legacy benchmark functions
```

### Model Architecture System

PHAZE supports multiple neural network architectures optimized for zkML:

- **Simple Early-Exit**: Basic feedforward with confidence-based early termination
- **Convolutional Early-Exit**: CNN architecture for image data with multiple exit points  
- **Transformer Early-Exit**: Attention-based models with adaptive computation
- **Multi-Exit**: Networks with multiple decision points for latency optimization

Each architecture supports complexity scaling from **minimal** (fastest proofs) to **extreme** (highest accuracy).

### zkML Backend Integration

The framework provides a unified interface for different zero-knowledge proof systems:

| Backend | Status | Use Case | Performance |
|---------|--------|----------|-------------|
| **EZKL** | ✅ Full | Production ML | High setup cost, fast verification |
| **RISC Zero** | ✅ Full | General computation | Balanced performance |
| **Groth16** | 🧪 Mock | Research/comparison | Fastest verification |
| **Plonky** | 🧪 Mock | Research/comparison | Moderate performance |
| **Halo** | 🧪 Mock | Research/comparison | No trusted setup |

## 📊 Benchmarking

### Performance Metrics

PHAZE tracks comprehensive performance metrics:

- **Setup Time**: Model compilation and circuit generation
- **Proof Generation**: Time to create zero-knowledge proofs  
- **Verification Time**: Time to verify proofs
- **Memory Usage**: Peak memory consumption during operations
- **Success Rate**: Percentage of successful operations
- **Throughput**: Operations per second for crypto primitives

### Typical Performance (Reference Hardware)

| Framework | Setup | Proof Gen | Verification | Memory |
|-----------|-------|-----------|--------------|--------|
| EZKL (Simple) | ~2-5s | ~1-10s | ~0.1-1s | ~100-500MB |
| RISC Zero | ~1-3s | ~2-8s | ~0.1-0.8s | ~120-400MB |
| Groth16* | ~0.1-1s | ~0.1-2s | ~0.01-0.1s | ~50-200MB |

*Mock implementation - actual performance may vary

### Running Custom Benchmarks

```python
from phaze.src.comprehensive_benchmark import ComprehensiveBenchmarkSuite

# Create benchmark suite
suite = ComprehensiveBenchmarkSuite("./my_benchmarks")

# Configure test parameters
config = {
    "architectures": ["simple", "conv", "multi_exit"],
    "complexities": ["light", "medium", "heavy"], 
    "input_sizes": [10, 50, 100, 500],
    "frameworks": ["ezkl", "risc_zero", "groth16"],
    "num_trials": 3
}

# Run benchmarks
results = await suite.run_full_benchmark_suite(config)

# Generate report
report = suite.generate_report(results)
print(report)
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=phaze

# Run specific test modules
uv run pytest tests/test_model_architectures.py -v
uv run pytest tests/test_zkml_integration.py -v 
uv run pytest tests/test_benchmarking.py -v

# Run tests in parallel
uv run pytest -n auto
```

### Test Structure

The test suite provides comprehensive coverage:

- **Unit Tests**: Individual component functionality (105 tests)
- **Integration Tests**: Cross-component interactions
- **Benchmark Tests**: Performance validation and regression testing  
- **Mock Tests**: Simulated zkML backend behavior

Current test coverage: **75%** with 105 passing tests.

## 🔧 Configuration

### Environment Variables

```bash
# Optional environment configuration
export PHAZE_LOG_LEVEL=INFO          # Logging level
export PHAZE_CACHE_DIR=/tmp/phaze    # Cache directory for models/proofs
export PHAZE_RUST_LOG=debug          # Rust component logging
```

### Dependency Groups

The project uses uv dependency groups for different use cases:

```bash
# Development dependencies
uv sync --group dev

# Testing dependencies  
uv sync --group test

# Minimal CI dependencies
uv sync --group ci

# Production dependencies only
uv sync --no-dev
```

## 🚨 Known Limitations

### Current Constraints

1. **EZKL Integration**: Requires proper EZKL installation and Structured Reference String (SRS) setup
2. **Large Models**: Memory requirements scale significantly with model complexity
3. **Platform Support**: Rust bindings require compilation on target platform
4. **Mock Backends**: Some zkML frameworks use simulated implementations for testing

### Performance Considerations

- **Model Complexity**: Choose appropriate complexity based on latency requirements
- **Input Size**: Larger inputs significantly increase proof generation time
- **Memory Usage**: Monitor system resources for complex models
- **Framework Selection**: Different backends have varying performance characteristics

### Troubleshooting

**Common Issues:**

```bash
# EZKL installation problems
pip install ezkl --no-binary ezkl

# Rust compilation issues  
rustup update
cd rust_bindings && cargo clean && cargo build --release

# Memory issues with large models
# Use quick_mode for initial testing
uv run python examples/run_benchmark_example.py --quick

# Import errors
# Ensure proper installation
uv sync && uv run python -c "import phaze; print('OK')"
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/PHAZE.git
cd PHAZE

# Set up development environment
uv sync --group dev --group test

# Install pre-commit hooks (optional)
uv tool install pre-commit
pre-commit install

# Run tests to verify setup
uv run pytest
```

### Code Quality

The project uses modern Python tooling:

- **Ruff**: Linting and formatting (`uv run ruff check .`)
- **pytest**: Testing framework with asyncio support
- **Type hints**: Comprehensive type annotations
- **Docstrings**: Full API documentation

### Pull Request Process

1. Create a feature branch from `main`
2. Make changes with tests
3. Ensure all tests pass: `uv run pytest`
4. Check code quality: `uv run ruff check .`
5. Submit pull request with description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EZKL Team**: Pioneering work in zero-knowledge machine learning
- **RISC Zero**: High-performance zkVM development
- **Rust Community**: Cryptographic primitives and performance optimizations  
- **PyTorch Team**: Deep learning framework foundation
- **uv/Astral**: Modern Python packaging and dependency management

## 📞 Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/PRAkTIKal24/PHAZE/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/PRAkTIKal24/PHAZE/discussions)
- **Documentation**: Comprehensive API docs in source code
- **Examples**: Check the `examples/` directory for usage patterns

---

**PHAZE** - Advancing privacy-preserving machine learning for high-energy physics through comprehensive zero-knowledge benchmarking.

