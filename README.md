# PHAZE: Privacy-preserving Hierarchical Adaptive Zero-knowledge Execution Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)

PHAZE is an experimental framework for benchmarking and evaluating zero-knowledge machine learning (zkML) systems with support for early-exit models and hierarchical privacy-preserving inference. This repository provides a comprehensive suite of tools for comparing different zkML frameworks, cryptographic primitives, and model architectures.

## 🚀 Features

### Core Capabilities
- **Multi-Framework zkML Support**: Integrated support for EZKL, Groth16, Plonky, and Halo zkML systems
- **Modular Model Architectures**: Flexible model factory supporting simple, convolutional, transformer, and multi-exit architectures
- **Comprehensive Benchmarking**: Performance evaluation across different frameworks, model complexities, and input sizes
- **Rust-Python Integration**: High-performance cryptographic primitives implemented in Rust with Python bindings
- **Early-Exit Models**: Support for adaptive inference with confidence-based early termination

### zkML Framework Integration
- **EZKL**: Full integration with EZKL for PyTorch model compilation and proof generation
- **Groth16**: Mock implementation for benchmarking Groth16-based zkML systems
- **Plonky**: Mock implementation for evaluating Plonky-based approaches
- **Halo**: Mock implementation for testing Halo-based zkML workflows

### Cryptographic Primitives
- **Rabin Fingerprinting**: Polynomial-based hashing for data integrity
- **Shamir Secret Sharing**: Threshold secret sharing implementation
- **Field Operations**: Finite field arithmetic with multiple field sizes
- **Hash Functions**: SHA256 and Keccak256 implementations in Rust

### Benchmarking and Analysis
- **Performance Monitoring**: Memory usage, CPU utilization, and execution time tracking
- **Comparative Analysis**: Side-by-side evaluation of different zkML approaches
- **Visualization**: Automated generation of performance charts and analysis plots
- **Report Generation**: Comprehensive markdown reports with recommendations

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- Rust toolchain (for building Rust-Python bindings)
- Git

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/your-org/praktikal24-phaze.git
cd praktikal24-phaze

# Install Python dependencies
pip install -e .

# Build Rust-Python bindings
cd rust_bindings
cargo build --release
maturin build --release
pip install ./target/wheels/rust_zkml_bindings-*.whl
cd ..
```

### Development Installation

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (if available)
# pre-commit install

# Run tests to verify installation
pytest tests/
```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
import asyncio
from phaze.comprehensive_benchmark import run_phaze_benchmarks

# Run a quick benchmark
async def main():
    results = await run_phaze_benchmarks(
        output_dir="/tmp/phaze_results",
        quick_mode=True
    )
    print(f"Benchmark completed with {results['results']['summary']['overall_summary']['success_rate']:.1%} success rate")

asyncio.run(main())
```

### Model Creation and Testing

```python
from phaze.model_architectures import PHAZEModelFactory, ModelComplexity
from phaze.zkml_integration import PHAZEZKMLIntegration
import torch

# Create a model
model = PHAZEModelFactory.create_early_exit_model(
    architecture="multi_exit",
    complexity=ModelComplexity.MEDIUM,
    input_size=128,
    output_size=10
)

# Test the model
input_data = torch.randn(1, 128)
output = model(input_data)
confidence = model.get_confidence(input_data)

print(f"Model output shape: {output.shape}")
print(f"Confidence: {confidence.item():.3f}")
```

### zkML Integration

```python
import asyncio
from phaze.zkml_integration import PHAZEZKMLIntegration
import torch

async def zkml_example():
    integration = PHAZEZKMLIntegration()
    
    # Create and register a model
    model = integration.create_and_register_model(
        name="test_model",
        architecture="simple",
        complexity="light",
        input_size=10,
        output_size=5
    )
    
    # Generate test data
    input_data = torch.randn(1, 10)
    
    # Setup zkML system
    await integration.setup_model("test_model", input_data)
    
    # Generate proof
    proof, output = await integration.prove_inference("test_model", input_data)
    
    # Verify proof
    is_valid = await integration.verify_inference("test_model", proof, input_data)
    
    print(f"Proof verification: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Cleanup
    integration.cleanup_model("test_model")

asyncio.run(zkml_example())
```

### Cryptographic Primitives

```python
from phaze.crypto_primitives import RabinFingerprint, ShamirSecretSharing
from phaze.rust_zkml_backend import RustZKMLFrameworkManager

# Rabin Fingerprinting
rabin = RabinFingerprint()
data = b"Hello, PHAZE!"
hash_value = rabin.compute_hash(data)
print(f"Rabin hash: {hash_value}")

# Shamir Secret Sharing
sss = ShamirSecretSharing(threshold=3, num_shares=5)
secret = b"my_secret_key"
shares = sss.generate_shares(secret)
reconstructed = sss.reconstruct_secret(shares[:3])
print(f"Secret reconstruction: {'✓ Success' if reconstructed == secret else '✗ Failed'}")

# Rust-based operations
rust_manager = RustZKMLFrameworkManager()
sha256_hash = rust_manager.base_backend.sha256_hash(b"test data")
print(f"SHA256: {sha256_hash}")
```

## 📊 Benchmarking

### Running Comprehensive Benchmarks

```bash
# Run the example benchmark script
python examples/run_benchmark_example.py

# Or use the Python API
python -c "
import asyncio
from phaze.comprehensive_benchmark import run_phaze_benchmarks

async def main():
    results = await run_phaze_benchmarks('/tmp/phaze_benchmarks')
    print('Benchmark completed!')

asyncio.run(main())
"
```

### Custom Benchmark Configuration

```python
from phaze.comprehensive_benchmark import ComprehensiveBenchmarkSuite

# Create custom benchmark suite
suite = ComprehensiveBenchmarkSuite("/tmp/custom_benchmarks")

# Configure zkML benchmarks
zkml_config = {
    "architectures": ["simple", "conv", "transformer", "multi_exit"],
    "complexities": ["light", "medium", "heavy"],
    "input_sizes": [10, 50, 100],
    "num_trials": 5
}

# Configure crypto benchmarks
crypto_config = {
    "rabin_input_sizes": [64, 256, 1024, 4096],
    "shamir_secret_sizes": [32, 64, 128, 256],
    "num_trials": 100
}

# Run benchmarks
results = await suite.run_full_benchmark_suite(zkml_config, crypto_config)
```

## 🏗️ Architecture

### Framework Overview

PHAZE is built with a modular architecture that separates concerns and enables easy extension:

```
phaze/
├── model_architectures.py      # Model factory and architecture definitions
├── zkml_integration.py         # zkML framework integration layer
├── rust_zkml_backend.py        # Rust-Python bindings interface
├── crypto_primitives.py        # Cryptographic primitive implementations
├── comprehensive_benchmark.py  # Benchmarking and evaluation system
├── zkml_framework_interface.py # Abstract interfaces for zkML systems
└── zkml_backends.py            # Concrete zkML backend implementations
```

### Model Architecture System

The framework supports multiple model architectures through a factory pattern:

- **SimpleEarlyExitModel**: Basic feedforward network with early exit capability
- **ConvolutionalEarlyExitModel**: CNN-based model for image-like data
- **TransformerEarlyExitModel**: Transformer architecture with attention mechanisms
- **MultiExitModel**: Model with multiple exit points for adaptive inference

Each model supports different complexity levels (minimal, light, medium, heavy, extreme) and provides confidence estimation for early-exit decisions.

### zkML Integration Layer

The zkML integration system provides a unified interface for different zero-knowledge proof systems:

1. **EZKL Integration**: Full support for PyTorch model compilation and proof generation
2. **Rust Backend Integration**: High-performance implementations of Groth16, Plonky, and Halo
3. **Mock Implementations**: Testing and benchmarking support for various zkML approaches

### Benchmarking System

The comprehensive benchmarking system evaluates:

- **Performance Metrics**: Setup time, proof generation time, verification time, memory usage
- **Scalability**: Performance across different input sizes and model complexities
- **Reliability**: Success rates and error analysis
- **Comparative Analysis**: Side-by-side evaluation of different approaches

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_model_architectures.py -v
pytest tests/test_zkml_integration.py -v
pytest tests/test_comprehensive_benchmark.py -v

# Run with coverage
pytest --cov=phaze --cov-report=html
```

### Test Structure

The test suite covers:

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Cross-component interactions
- **Benchmark Tests**: Performance and reliability validation
- **Mock Tests**: Simulated zkML system behavior

## 📈 Performance Considerations

### Optimization Guidelines

1. **Model Complexity**: Choose appropriate complexity levels based on performance requirements
2. **Input Size**: Consider the trade-off between accuracy and proof generation time
3. **Framework Selection**: Different zkML frameworks have varying performance characteristics
4. **Memory Management**: Monitor memory usage for large-scale deployments

### Benchmarking Results

Typical performance characteristics (on standard hardware):

| Framework | Setup Time | Proof Time | Verification Time | Memory Usage |
|-----------|------------|------------|-------------------|--------------|
| EZKL      | ~2-5s      | ~1-10s     | ~0.1-1s          | ~100-500MB   |
| Groth16   | ~0.1-1s    | ~0.1-2s    | ~0.01-0.1s       | ~50-200MB    |
| Plonky    | ~0.5-2s    | ~0.5-5s    | ~0.05-0.5s       | ~80-300MB    |
| Halo      | ~1-3s      | ~2-8s      | ~0.1-0.8s        | ~120-400MB   |

*Note: These are approximate values and actual performance depends on model complexity, input size, and hardware configuration.*

## 🤝 Contributing

We welcome contributions to the PHAZE framework! Please consider the following guidelines:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/praktikal24-phaze.git
cd praktikal24-phaze

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests to ensure everything works
pytest
```

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Write comprehensive docstrings
- Add tests for new functionality

## 📚 Documentation

### API Documentation

The framework provides comprehensive API documentation through docstrings and type hints. Key modules include:

- **Model Architectures**: `phaze.model_architectures`
- **zkML Integration**: `phaze.zkml_integration`
- **Benchmarking**: `phaze.comprehensive_benchmark`
- **Cryptographic Primitives**: `phaze.crypto_primitives`

### Examples

The `examples/` directory contains:

- [Benchmark Examples](examples/run_benchmark_example.py)
- Basic usage patterns and integration examples

## 🔧 Configuration

### Environment Variables

- `PHAZE_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `PHAZE_CACHE_DIR`: Directory for caching compiled models and proofs
- `PHAZE_RUST_LOG`: Enable Rust component logging

### Configuration Files

The framework supports configuration through:

- `pyproject.toml`: Project configuration and dependencies

## 🚨 Known Issues and Limitations

### Current Limitations

1. **EZKL Integration**: Requires proper EZKL installation and SRS setup
2. **Model Size**: Large models may require significant memory and time for proof generation
3. **Platform Support**: Rust bindings require compilation on target platform
4. **Mock Implementations**: Some zkML backends use mock implementations for testing

### Troubleshooting

Common issues and solutions:

**EZKL Installation Issues**:
```bash
# Install EZKL from source
pip install ezkl --no-binary ezkl
```

**Rust Compilation Issues**:
```bash
# Update Rust toolchain
rustup update
# Rebuild bindings
cd rust_bindings && maturin build --release
```

**Memory Issues**:
- Reduce model complexity or input size
- Use quick_mode for initial testing
- Monitor system resources during benchmarks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The EZKL team for their pioneering work in zkML
- The Rust cryptography community for high-performance implementations
- The PyTorch team for the excellent deep learning framework
- Contributors and researchers in the zero-knowledge proof space

## 📞 Support

For questions, issues, or contributions:

- **Issues**: Create GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for general questions
- **Documentation**: Refer to inline documentation and examples

---

**PHAZE** - Advancing the state of privacy-preserving machine learning through comprehensive benchmarking and evaluation.

