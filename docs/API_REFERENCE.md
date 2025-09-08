# PHAZE Framework API Reference

This document provides comprehensive API reference documentation for the PHAZE framework components.

## Table of Contents

1. [Model Architectures](#model-architectures)
2. [zkML Integration](#zkml-integration)
3. [Cryptographic Primitives](#cryptographic-primitives)
4. [Benchmarking System](#benchmarking-system)
5. [Rust-Python Bindings](#rust-python-bindings)
6. [Framework Interfaces](#framework-interfaces)

## Model Architectures

### PHAZEModelFactory

Factory class for creating different model architectures with configurable complexity levels.

#### Methods

##### `create_early_exit_model(architecture, complexity, model_type, **kwargs)`

Creates an early-exit model with specified architecture and complexity.

**Parameters:**
- `architecture` (str): Model architecture type ("simple", "conv", "transformer", "multi_exit")
- `complexity` (ModelComplexity): Complexity level (MINIMAL, LIGHT, MEDIUM, HEAVY, EXTREME)
- `model_type` (str): Model type ("early" or "full")
- `**kwargs`: Additional model-specific parameters

**Returns:**
- Model instance implementing the early-exit interface

**Example:**
```python
from phaze.model_architectures import PHAZEModelFactory, ModelComplexity

model = PHAZEModelFactory.create_early_exit_model(
    architecture="multi_exit",
    complexity=ModelComplexity.MEDIUM,
    input_size=128,
    output_size=10
)
```

##### `create_full_model(architecture, complexity, **kwargs)`

Creates a full model without early-exit capabilities.

**Parameters:**
- `architecture` (str): Model architecture type
- `complexity` (ModelComplexity): Complexity level
- `**kwargs`: Additional model-specific parameters

**Returns:**
- Standard PyTorch model instance

##### `get_available_architectures()`

Returns list of available model architectures.

**Returns:**
- `List[str]`: Available architecture names

##### `get_complexity_levels()`

Returns list of available complexity levels.

**Returns:**
- `List[ModelComplexity]`: Available complexity levels

### Model Classes

#### SimpleEarlyExitModel

Basic feedforward neural network with early-exit capability.

**Constructor Parameters:**
- `input_size` (int): Input dimension
- `output_size` (int): Output dimension
- `complexity` (ModelComplexity): Model complexity level
- `confidence_threshold` (float): Threshold for early exit decision

**Methods:**

##### `forward(x, exit_layer=None)`

Forward pass through the model.

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `exit_layer` (int, optional): Specific exit layer to use

**Returns:**
- `torch.Tensor`: Model output

##### `get_confidence(x)`

Estimate confidence for the given input.

**Parameters:**
- `x` (torch.Tensor): Input tensor

**Returns:**
- `torch.Tensor`: Confidence score

##### `should_exit_early(x)`

Determine if early exit should be taken.

**Parameters:**
- `x` (torch.Tensor): Input tensor

**Returns:**
- `bool`: Whether to exit early

#### ConvolutionalEarlyExitModel

CNN-based model for image-like data with multiple exit points.

**Constructor Parameters:**
- `input_channels` (int): Number of input channels
- `output_size` (int): Output dimension
- `complexity` (ModelComplexity): Model complexity level
- `image_size` (int): Input image size

**Methods:**
Similar to SimpleEarlyExitModel with additional CNN-specific functionality.

#### TransformerEarlyExitModel

Transformer architecture with attention mechanisms and early exits.

**Constructor Parameters:**
- `input_size` (int): Input dimension
- `output_size` (int): Output dimension
- `complexity` (ModelComplexity): Model complexity level
- `num_heads` (int): Number of attention heads
- `sequence_length` (int): Maximum sequence length

#### MultiExitModel

Advanced model with multiple intermediate classifiers and adaptive inference.

**Constructor Parameters:**
- `input_size` (int): Input dimension
- `output_size` (int): Output dimension
- `num_exits` (int): Number of exit points
- `complexity` (ModelComplexity): Model complexity level

**Methods:**

##### `adaptive_forward(x, confidence_threshold=0.8)`

Adaptive forward pass with confidence-based early termination.

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `confidence_threshold` (float): Confidence threshold for early exit

**Returns:**
- `Tuple[torch.Tensor, int]`: (output, exit_layer_used)

##### `get_all_exits(x)`

Get outputs from all exit points.

**Parameters:**
- `x` (torch.Tensor): Input tensor

**Returns:**
- `List[torch.Tensor]`: Outputs from all exits

## zkML Integration

### PHAZEZKMLIntegration

Main integration class for zkML frameworks.

#### Methods

##### `create_and_register_model(name, architecture, complexity, model_type, **kwargs)`

Create and register a model for zkML operations.

**Parameters:**
- `name` (str): Model identifier
- `architecture` (str): Model architecture
- `complexity` (str): Complexity level
- `model_type` (str): Model type
- `**kwargs`: Additional parameters

**Returns:**
- Model instance

##### `async setup_model(model_name, input_data)`

Setup zkML system for the specified model.

**Parameters:**
- `model_name` (str): Registered model name
- `input_data` (torch.Tensor): Sample input data

**Returns:**
- Setup result information

##### `async prove_inference(model_name, input_data)`

Generate zero-knowledge proof for model inference.

**Parameters:**
- `model_name` (str): Registered model name
- `input_data` (torch.Tensor): Input data

**Returns:**
- `Tuple[Any, torch.Tensor]`: (proof, output)

##### `async verify_inference(model_name, proof, input_data)`

Verify zero-knowledge proof.

**Parameters:**
- `model_name` (str): Registered model name
- `proof` (Any): Proof to verify
- `input_data` (torch.Tensor): Original input data

**Returns:**
- `bool`: Verification result

##### `cleanup_model(model_name)`

Clean up resources for the specified model.

**Parameters:**
- `model_name` (str): Model to clean up

### ZKMLProverVerifier

Low-level interface for zkML operations.

#### Methods

##### `async setup(model, input_data, framework="ezkl")`

Setup zkML system.

##### `async prove(model, input_data, framework="ezkl")`

Generate proof.

##### `async verify(proof, public_inputs, framework="ezkl")`

Verify proof.

## Cryptographic Primitives

### RabinFingerprint

Polynomial-based hashing for data integrity.

#### Constructor

```python
RabinFingerprint(field_size=2**31 - 1, degree=100)
```

**Parameters:**
- `field_size` (int): Size of the finite field
- `degree` (int): Maximum polynomial degree

#### Methods

##### `compute_hash(data_vector, challenge_point=None)`

Compute Rabin fingerprint hash.

**Parameters:**
- `data_vector` (Union[list, bytes]): Input data
- `challenge_point` (int, optional): Evaluation point

**Returns:**
- `int`: Hash value

### ShamirSecretSharing

Threshold secret sharing implementation.

#### Constructor

```python
ShamirSecretSharing(threshold, num_shares, prime=2**31 - 1)
```

**Parameters:**
- `threshold` (int): Minimum shares needed for reconstruction
- `num_shares` (int): Total number of shares
- `prime` (int): Prime for finite field operations

#### Methods

##### `generate_shares(secret)`

Generate shares for the secret.

**Parameters:**
- `secret` (bytes): Secret to be shared

**Returns:**
- `List[Tuple[int, bytes]]`: List of (x, y) share tuples

##### `reconstruct_secret(shares)`

Reconstruct secret from shares.

**Parameters:**
- `shares` (List[Tuple[int, bytes]]): Share tuples

**Returns:**
- `bytes`: Reconstructed secret

## Benchmarking System

### ComprehensiveBenchmarkSuite

Main benchmarking orchestration class.

#### Constructor

```python
ComprehensiveBenchmarkSuite(output_dir="/tmp/phaze_benchmarks")
```

**Parameters:**
- `output_dir` (str): Directory for benchmark results

#### Methods

##### `async run_full_benchmark_suite(zkml_config=None, crypto_config=None)`

Run comprehensive benchmark suite.

**Parameters:**
- `zkml_config` (Dict, optional): zkML benchmark configuration
- `crypto_config` (Dict, optional): Crypto benchmark configuration

**Returns:**
- `Dict[str, Any]`: Benchmark results

##### `generate_report(results)`

Generate comprehensive benchmark report.

**Parameters:**
- `results` (Dict): Benchmark results

**Returns:**
- `str`: Markdown report content

### ZKMLFrameworkBenchmark

Benchmark zkML frameworks.

#### Methods

##### `async benchmark_framework(framework_name, architecture, complexity, input_size, output_size, num_trials=3)`

Benchmark specific framework configuration.

**Parameters:**
- `framework_name` (str): Framework to benchmark
- `architecture` (str): Model architecture
- `complexity` (str): Model complexity
- `input_size` (int): Input dimension
- `output_size` (int): Output dimension
- `num_trials` (int): Number of benchmark trials

**Returns:**
- `List[BenchmarkResult]`: Benchmark results

##### `async benchmark_all_frameworks(architectures=None, complexities=None, input_sizes=None, num_trials=3)`

Benchmark all available frameworks.

**Returns:**
- `List[BenchmarkResult]`: Comprehensive benchmark results

### CryptographicPrimitiveBenchmark

Benchmark cryptographic primitives.

#### Methods

##### `benchmark_rabin_fingerprint(input_sizes=None, num_trials=100)`

Benchmark Rabin fingerprint operations.

##### `benchmark_shamir_secret_sharing(secret_sizes=None, num_trials=50)`

Benchmark Shamir secret sharing operations.

##### `benchmark_rust_primitives()`

Benchmark Rust-based cryptographic primitives.

### Data Classes

#### BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    test_name: str
    framework: str
    architecture: str
    complexity: str
    input_size: int
    output_size: int
    setup_time: float
    proof_time: float
    verification_time: float
    total_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    accuracy: Optional[float] = None
    model_parameters: Optional[int] = None
    proof_size_bytes: Optional[int] = None
```

#### CryptoBenchmarkResult

```python
@dataclass
class CryptoBenchmarkResult:
    primitive_name: str
    operation: str
    input_size: int
    execution_time: float
    memory_usage_mb: float
    throughput_ops_per_sec: float
    success: bool
    error_message: Optional[str] = None
```

## Rust-Python Bindings

### RustZKMLFrameworkManager

Manager for Rust-based zkML frameworks.

#### Methods

##### `get_backend(framework_name)`

Get backend for specified framework.

**Parameters:**
- `framework_name` (str): Framework name ("groth16", "plonky", "halo")

**Returns:**
- Backend instance

### Backend Classes

#### Groth16Backend

Mock Groth16 implementation.

##### `setup(constraint_count)`

Setup Groth16 system.

##### `prove(witness)`

Generate Groth16 proof.

##### `verify(proof)`

Verify Groth16 proof.

#### PlonkyBackend

Mock Plonky implementation.

##### `setup(circuit_size)`

Setup Plonky system.

##### `prove(witness)`

Generate Plonky proof.

##### `verify(proof)`

Verify Plonky proof.

#### HaloBackend

Mock Halo implementation.

##### `setup(k_value)`

Setup Halo system.

##### `prove(witness)`

Generate Halo proof.

##### `verify(proof)`

Verify Halo proof.

### Base Backend

#### RustZKMLBaseBackend

Base backend with cryptographic primitives.

##### `sha256_hash(data)`

Compute SHA256 hash.

**Parameters:**
- `data` (bytes): Input data

**Returns:**
- `str`: Hex-encoded hash

##### `keccak256_hash(data)`

Compute Keccak256 hash.

**Parameters:**
- `data` (bytes): Input data

**Returns:**
- `str`: Hex-encoded hash

##### `benchmark_field_operations(field_size, num_operations)`

Benchmark finite field operations.

**Parameters:**
- `field_size` (str): Field size as string
- `num_operations` (int): Number of operations

**Returns:**
- `Dict[str, float]`: Operation timing results

## Framework Interfaces

### ZKMLFrameworkInterface

Abstract base class for zkML framework implementations.

#### Abstract Methods

##### `async setup_model(model, input_data)`

Setup model for zkML operations.

##### `async generate_proof(model, input_data)`

Generate zero-knowledge proof.

##### `async verify_proof(proof, public_inputs)`

Verify zero-knowledge proof.

##### `get_framework_info()`

Get framework information.

##### `cleanup()`

Clean up framework resources.

## Utility Functions

### Convenience Functions

#### `run_phaze_benchmarks(output_dir, quick_mode=False)`

Convenience function for running PHAZE benchmarks.

**Parameters:**
- `output_dir` (str): Output directory
- `quick_mode` (bool): Whether to run in quick mode

**Returns:**
- `Dict[str, Any]`: Benchmark results and metadata

#### `create_simple_early_exit_model(input_size, output_size, complexity="light")`

Create a simple early-exit model.

#### `create_simple_full_model(input_size, output_size, complexity="light")`

Create a simple full model.

## Error Handling

### Custom Exceptions

The framework defines several custom exceptions for better error handling:

- `PHAZEModelError`: Model-related errors
- `ZKMLIntegrationError`: zkML integration errors
- `BenchmarkError`: Benchmarking-related errors
- `CryptographicError`: Cryptographic operation errors

### Error Handling Patterns

All async methods use proper exception handling:

```python
try:
    result = await integration.prove_inference("model", input_data)
except ZKMLIntegrationError as e:
    logger.error(f"zkML operation failed: {e}")
    # Handle error appropriately
```

## Configuration

### Environment Variables

- `PHAZE_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `PHAZE_CACHE_DIR`: Directory for caching compiled models and proofs
- `PHAZE_RUST_LOG`: Enable Rust component logging

### Configuration Classes

#### ModelComplexity

Enumeration of model complexity levels:

```python
class ModelComplexity(Enum):
    MINIMAL = "minimal"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    EXTREME = "extreme"
```

## Performance Considerations

### Memory Management

- Use context managers for resource cleanup
- Monitor memory usage during benchmarks
- Implement lazy loading for large models

### Async Operations

- All zkML operations are asynchronous
- Use proper async/await patterns
- Handle timeouts appropriately

### Benchmarking Best Practices

- Run multiple trials for statistical significance
- Monitor system resources during benchmarks
- Use appropriate sample sizes for reliable results

---

This API reference provides comprehensive documentation for all public interfaces in the PHAZE framework. For additional examples and usage patterns, refer to the examples directory and the main README documentation.

