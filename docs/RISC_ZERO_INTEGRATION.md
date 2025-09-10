# RISC Zero Integration Guide

This document describes the RISC Zero zkVM integration in the PHAZE framework, enabling privacy-preserving machine learning inference using zero-knowledge proofs.

## Overview

RISC Zero is a zero-knowledge virtual machine (zkVM) that provides a general-purpose platform for generating proofs of arbitrary computation. The PHAZE framework now includes a complete RISC Zero backend implementation that allows you to generate zero-knowledge proofs of neural network inference.

## Key Features

- **RISC-V zkVM**: Utilizes RISC Zero's RISC-V based virtual machine
- **STARK Proofs**: Generates STARK-based zero-knowledge proofs
- **Guest Programs**: Supports Rust-based guest programs for ML execution
- **Receipt System**: Implements RISC Zero's receipt-based verification
- **Journal Support**: Extracts public outputs through RISC Zero's journal system

## Quick Start

### Basic Usage

```python
import asyncio
import torch
import torch.nn as nn
from phaze.src.zkml_backends import RiscZeroBackend

async def basic_example():
    # Create a simple model
    model = nn.Linear(10, 1)
    
    # Initialize RISC Zero backend
    backend = RiscZeroBackend(model, "my_model")
    
    # Setup the zkVM
    input_data = torch.randn(1, 10)
    await backend.setup(input_data)
    
    # Generate proof
    proof, output = await backend.generate_proof(input_data)
    
    # Verify proof
    verified = await backend.verify_proof(proof, input_data)
    
    print(f"Verification: {verified}")
    print(f"Output: {output}")
    
    # Cleanup
    backend.cleanup()

# Run the example
asyncio.run(basic_example())
```

### Integration with PHAZE Benchmarking

```python
from phaze.src.zkml_backends import create_backend
from phaze.src.zkml_framework_interface import ZKMLFramework

# Create backend through factory
backend = create_backend(ZKMLFramework.RISC_ZERO, model, "benchmark_test")

# Use with benchmark runner
from phaze.src.phaze_benchmark_suite import ZKMLBenchmarkRunner
runner = ZKMLBenchmarkRunner()
runner.register_backend(backend)
```

## Technical Details

### Proof Structure

RISC Zero proofs in PHAZE follow this structure:

```json
{
  "receipt": {
    "seal": "risc_zero_seal_xxxxx",
    "guest_id": "risc_zero_guest_xxxx", 
    "journal_digest": "journal_digest_xxxxx"
  },
  "proof_system": "STARK",
  "curve": "RISC-V",
  "zkvm_version": "0.18.0"
}
```

### Journal Format

The journal contains public outputs from the guest program:

```json
{
  "model_output": [0.12345],
  "input_commitment": "input_hash_xxxxx",
  "execution_metadata": {
    "cycles": 1000000,
    "guest_id": "risc_zero_guest_xxxx"
  }
}
```

### File Artifacts

The backend creates several temporary files during operation:

- `{name}_model_artifacts`: Serialized model weights and architecture
- `{name}_guest.bin`: Compiled guest program binary
- `{name}_proof.json`: Generated proof data
- `{name}_journal.json`: Public outputs and metadata
- `{name}_receipt.json`: Complete receipt for verification
- `{name}_input.json`: Input data for the zkVM

## Architecture

### Guest Program Simulation

The current implementation simulates the RISC Zero guest program compilation and execution process. In a production deployment, this would involve:

1. **Rust Guest Program**: A Rust program that executes the ML model
2. **Compilation**: Building the guest program to RISC-V bytecode
3. **zkVM Execution**: Running the model inside the zkVM
4. **Proof Generation**: Creating STARK proofs of correct execution

### Verification Process

The verification process checks:

1. **Proof Structure**: Validates the receipt format
2. **Guest ID**: Ensures the correct guest program was executed
3. **Seal Verification**: Validates the cryptographic seal
4. **Journal Consistency**: Checks output consistency

## Configuration

### Framework Information

```python
backend = RiscZeroBackend(model, "test")
info = backend.get_framework_info()
```

Returns:
```python
{
    "framework": "RISC Zero",
    "version": "0.18.0", 
    "backend": "RISC-V zkVM",
    "proof_system": "STARK",
    "guest_support": "Rust",
    "curve": "RISC-V ISA",
    "status": "active_implementation"
}
```

## Error Handling

The backend includes comprehensive error handling:

```python
try:
    await backend.generate_proof(input_data)
except RuntimeError as e:
    if "Backend not setup" in str(e):
        print("Call setup() first")
```

## Performance Considerations

### Optimization Tips

1. **Model Size**: Smaller models have faster proof generation
2. **Input Batching**: Batch inputs when possible for efficiency
3. **Cleanup**: Always call `cleanup()` to remove temporary files
4. **Async Operations**: Use proper async/await patterns

### Benchmarking

The RISC Zero backend integrates with PHAZE's benchmarking system:

```python
from phaze.src.phaze_benchmark_suite import ZKMLBenchmarkRunner

runner = ZKMLBenchmarkRunner()
runner.register_backend(risc_zero_backend)
# Benchmark will measure setup, proving, and verification times
```

## Comparison with Other Backends

| Feature | RISC Zero | EZKL | Groth16 |
|---------|-----------|------|---------|
| Proof System | STARK | KZG | Groth16 |
| General Purpose | ✅ | ❌ | ❌ |
| Setup Required | Minimal | SRS | Trusted Setup |
| Proof Size | Large | Medium | Small |
| Verification | Fast | Fast | Very Fast |
| Flexibility | High | Medium | Low |

## Future Enhancements

Potential improvements for the RISC Zero integration:

1. **Real Guest Programs**: Implement actual Rust guest programs
2. **Optimized Circuits**: Add ML-specific optimizations
3. **Batch Proving**: Support for batch proof generation
4. **Hardware Acceleration**: GPU acceleration for proof generation
5. **Custom Instructions**: RISC-V extensions for ML operations

## Troubleshooting

### Common Issues

**Import Errors**:
```python
# Use the correct import path
from phaze.src.zkml_backends import RiscZeroBackend
```

**Setup Failures**:
```python
# Ensure input data is properly formatted
input_data = torch.randn(1, 10)  # Correct shape
await backend.setup(input_data)
```

**Verification Failures**:
```python
# Ensure the same backend instance is used
proof, _ = await backend.generate_proof(input_data)
verified = await backend.verify_proof(proof, input_data)  # Same backend
```

## Examples

See `examples/risc_zero_demo.py` for a complete working example demonstrating all features of the RISC Zero integration.

## Contributing

When contributing to the RISC Zero backend:

1. Maintain compatibility with the `ZKMLBackendInterface`
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Follow the existing async patterns
5. Ensure proper resource cleanup

For more information about RISC Zero, visit: https://www.risczero.com/