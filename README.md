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
- [📊 Performance Analysis & Plotting](#-performance-analysis--plotting)
  - [Publication-Ready Plots](#publication-ready-plots)
  - [Plot Types and Styles](#plot-types-and-styles)
  - [Statistical Analysis](#statistical-analysis)
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

### 📈 Publication-Ready Plotting
- **Statistical Analysis**: Error bars, confidence intervals, significance testing
- **Multiple Plot Types**: Time complexity, memory usage, throughput, comparative analysis
- **Publication Styles**: NeurIPS, IEEE/ACM, presentation, and web-optimized formats
- **Export Formats**: High-resolution PNG, PDF, SVG for papers and presentations
- **Colorblind-Friendly**: Accessible color schemes and consistent styling

## 🚀 Quick Start

### Installation

**Prerequisites**: Python 3.10+, [uv](https://docs.astral.sh/uv/) package manager

First check for uv and install it if it doesn't exist. Alternative: see [uv docs](https://docs.astral.sh/uv/getting-started/installation/) to install uv.

```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/PRAkTIKal24/PHAZE.git
cd PHAZE

# Option 1: Automated installation (recommended)
./install_dev.sh

# Verify installation and CLI
uv run phaze --help
uv run phaze --list-benchmarks
```

Alternatively, you can manually install the `maturin` based rust backend and then added the required `.pth` file that the `dev_setup.py` file will take care for you.

```bash
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
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh
# Alternative: pip install uv

# Install with all development dependencies
uv sync --group dev --group test

# Setup editable install (required for CLI)
uv run python dev_setup.py

# Verify installation and CLI
uv run phaze --help
uv run phaze --list-benchmarks

# Run tests
uv run pytest
```

## 💡 Usage Examples

### PHAZE CLI (Recommended)

**List Available Benchmarks**
```bash
uv run phaze --list-benchmarks
uv run phaze --list-plot-types  # List available plot types
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

**Generate Publication-Ready Plots**
```bash
# List available plot types
uv run phaze --list-plot-types

# Generate fingerprinting algorithm performance plots
uv run phaze --plot fingerprint --output plots/fingerprint/ --trials 10

# Generate zkML framework performance plots
uv run phaze --plot zkml-proof --style neurips --format png,pdf --output plots/zkml/

# Generate all available plots
uv run phaze --plot all --style publication --output plots/comprehensive/

# Comparative analysis across components
uv run phaze --plot comparative --components fingerprint,zkml-proof --output plots/comparison/

# Use existing benchmark data instead of running new benchmarks
uv run phaze --plot fingerprint --data-file benchmark_results.json --output plots/

# Different plot styles for different use cases
uv run phaze --plot zkml-proof --style presentation --output plots/presentation/  # For slides
uv run phaze --plot fingerprint --style neurips --output plots/paper/           # For papers
uv run phaze --plot all --style web --format png --output plots/web/            # For websites
```

### Direct Python Examples
> **Note**: This is not the recommended way to use PHAZE. Use the PHAZE CLI above instead.

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

## 📊 Performance Analysis & Plotting

PHAZE provides comprehensive plotting capabilities for analyzing benchmark results with publication-ready visualizations. The plotting system supports statistical analysis, multiple export formats, and various styling options optimized for research papers, presentations, and web display.

### Publication-Ready Plots

**Available Plot Types:**
- **Fingerprinting Analysis**: Time complexity, memory usage, throughput, and algorithm comparisons
- **zkML Framework Analysis**: Proof generation/verification performance, memory consumption, proof size analysis
- **Statistical Analysis**: Error bars, confidence intervals, distribution analysis, scaling analysis
- **Comparative Analysis**: Cross-component performance comparisons and framework rankings

**Quick Start with Plotting:**
```bash
# Generate all available plots with publication quality
uv run phaze --plot all --style neurips --output plots/paper/

# Generate specific analysis plots
uv run phaze --plot fingerprint --trials 10 --output plots/crypto/
uv run phaze --plot zkml-proof --style presentation --output plots/zkml/

# Use existing benchmark data
uv run phaze --plot fingerprint --data-file benchmark_results.json --style publication
```

### Plot Types and Styles

**Fingerprinting Performance Plots:**
```bash
# Time complexity analysis across algorithms
uv run phaze --plot fingerprint --algorithms rabin,shamir --complexity-range light,medium,heavy

# Memory and throughput analysis
uv run phaze --plot fingerprint --trials 20 --style neurips --format png,pdf
```

**zkML Framework Performance Plots:**
```bash
# Proof generation and verification analysis
uv run phaze --plot zkml-proof --frameworks ezkl,risc_zero --style publication

# Framework comparison matrix and scaling analysis
uv run phaze --plot zkml-verify --complexity-range light,medium --trials 10
```

**Cross-Component Comparative Analysis:**
```bash
# Compare performance across different PHAZE components
uv run phaze --plot comparative --components fingerprint,zkml-proof,zkml-verify

# Generate comprehensive analysis with all metrics
uv run phaze --plot all --trials 15 --style neurips --output plots/comprehensive/
```

### Statistical Analysis

**Built-in Statistical Features:**
- **Error Bars**: Standard deviation, standard error, 95% confidence intervals
- **Distribution Analysis**: Box plots, violin plots, outlier detection
- **Significance Testing**: Statistical comparison between frameworks and algorithms
- **Effect Size Calculation**: Cohen's d for practical significance assessment

**Plot Styling Options:**
- **`neurips`**: NeurIPS conference style with serif fonts and publication formatting
- **`publication`**: IEEE/ACM paper format with high-resolution output
- **`presentation`**: Large fonts and clear visuals optimized for slides
- **`web`**: Web-friendly styling with smaller file sizes

**Export Formats:**
```bash
# Multiple format export
uv run phaze --plot all --format png,pdf,svg --output plots/

# High-resolution for publications
uv run phaze --plot fingerprint --style publication --format pdf
```

**Advanced Usage Examples:**
```python
# Using the Python API for custom plotting
from phaze import PHAZEPlotSuite, PlotConfig, PlotStyle
import asyncio

async def custom_plotting():
    # Create custom plot configuration
    config = PlotConfig(
        style=PlotStyle.NEURIPS,
        dpi=300,
        export_formats=['png', 'pdf'],
        confidence_level=0.95
    )
    
    # Initialize plot suite
    suite = PHAZEPlotSuite(config)
    
    # Load benchmark data
    with open('benchmark_results.json', 'r') as f:
        data = json.load(f)
    
    # Generate plots
    results = suite.generate_plots(data, ['fingerprint', 'zkml-proof'])
    
    # Generate report
    report = suite.generate_summary_report(results, 'plotting_report.md')
    print("Plots generated successfully!")

asyncio.run(custom_plotting())
```

**Plot Customization:**
```bash
# Custom trials and output settings
uv run phaze --plot zkml-proof --trials 25 --output custom_plots/ --verbose

# Specific framework and complexity combinations
uv run phaze --plot zkml-proof --frameworks ezkl,risc_zero --complexity-range medium,heavy

# Generate plots with existing data to save time
uv run phaze --plot all --data-file previous_benchmark.json --style neurips
```

## 🏗️ Architecture

### Project Structure

```
PHAZE/
├── README.md                       # Project documentation
├── pyproject.toml                  # Python project configuration
├── dev_setup.py                    # Editable install setup script
├── install_dev.sh                  # Automated installation script
├── uv.lock                         # Dependency lock file
│
├── phaze/                          # Main Python package
│   ├── __init__.py                 # Package exports and version
│   ├── phaze.py                    # CLI interface and entry point
│   └── src/                        # Core implementation modules
│       ├── model_architectures.py      # Model factory and definitions
│       ├── zkml_integration.py         # zkML framework integration
│       ├── zkml_backends.py            # Concrete zkML implementations
│       ├── zkml_framework_interface.py # Abstract zkML interfaces
│       ├── rust_zkml_backend.py        # Rust-Python bindings
│       ├── crypto_primitives.py        # Cryptographic implementations
│       ├── early_exit_models.py        # Early-exit model implementations
│       ├── comprehensive_benchmark.py  # Benchmarking system
│       ├── phaze_benchmark_suite.py    # Benchmark configurations
│       ├── benchmarking.py             # Legacy benchmark functions
│       └── mock_rust_zkml_bindings.py  # Mock Rust bindings for testing
│
├── rust_bindings/                  # Rust extension module
│   ├── Cargo.toml                  # Rust project configuration
│   ├── src/lib.rs                  # Rust implementation
│   ├── risc0_guest/                # RISC Zero guest program
│   └── target/                     # Rust build artifacts
│
├── examples/                       # Usage examples and benchmarks
│   ├── run_basic_benchmark.py      # Standard benchmarking example
│   └── run_risc_zero.py           # RISC Zero specific example
│
├── tests/                          # Test suite
│   ├── test_*.py                   # Comprehensive test coverage
│   └── ...                        # (20+ test files)
│
├── docs/                           # Documentation
│   ├── conf.py                     # Sphinx configuration
│   ├── API_REFERENCE.md           # API documentation
│   ├── RISCZERO_INTEGRATION.md    # RISC Zero integration guide
│   └── _build/                     # Generated documentation
│
└── .github/                       # CI/CD configuration
    └── workflows/                  # GitHub Actions workflows
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

