# PHAZE Model Training & Inference Integration Plan

## Overview

This plan outlines the integration of minimal training and inference capabilities for the PHAZE framework to generate real benchmark data for plotting. The focus is on **fast results for compute time and memory plots** rather than model accuracy or performance, enabling meaningful visualizations with the existing plotting infrastructure.

## Objectives

1. **Minimal Training Pipeline**: Train models briefly (5 epochs) on small MNIST subsets
2. **Multi-Architecture Support**: Train all available model architectures with different complexities
3. **ONNX Export**: Export trained models for ezkl compatibility
4. **ZK Proof Generation**: Create proofs for M_full models across both ezkl and risc-zero
5. **Early Exit Model Creation**: Generate multiple M_early variants from M_full models
6. **Fingerprinting Data**: Generate probabilistic hashes using all available algorithms
7. **Statistical Reproducibility**: Run experiments with multiple seeds for error bars
8. **Real Plotting Data**: Generate data compatible with existing `FingerprintPlotter` and `ZKMLPlotter`

## Core Components to Implement

### 1. Training Configuration System

**File**: `phaze/src/training_config.py`

```python
@dataclass
class TrainingConfig:
    # Training parameters
    epochs: int = 5                    # Minimal training
    batch_size: int = 64              # Small batches for speed
    learning_rate: float = 0.001      # Standard learning rate
    dataset_size: int = 1000          # Small MNIST subset
    
    # Model configuration
    architectures: List[str] = field(default_factory=lambda: ["simple", "conv", "transformer", "multi_exit"])
    complexities: List[str] = field(default_factory=lambda: ["minimal", "light", "medium", "heavy"])
    
    # Reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])
    
    # Output configuration
    save_models: bool = True
    export_onnx: bool = True
    output_dir: str = "trained_models"
```

### 2. MNIST Training Module

**File**: `phaze/src/mnist_trainer.py`

Key features:
- Fast MNIST subset loading (1000 samples per class)
- Model factory integration for all architectures
- Minimal training loop (5 epochs, early stopping for time)
- Automatic ONNX export after training
- Model metadata saving (parameters, complexity, etc.)

### 3. Multi-Exit Model Generator

**File**: `phaze/src/multi_exit_generator.py`

Functions:
- Take largest available M_full model (e.g., heavy complexity transformer)
- Create multiple M_early variants by truncating at different layers
- Generate models with 25%, 50%, 75% of original parameters
- Maintain compatibility with fingerprinting requirements

### 4. Enhanced ZK Proof Benchmarking

**File**: `phaze/src/enhanced_zkml_benchmark.py`

Capabilities:
- Support both ezkl and risc-zero backends
- Batch proof generation for multiple models
- Memory usage tracking during proof generation
- Proof size measurement
- Setup time vs proof time separation

### 5. Comprehensive Fingerprinting Benchmark

**File**: `phaze/src/enhanced_crypto_benchmark.py`

Features:
- All available hashing algorithms (Rabin, Shamir, plus any additional)
- Variable input sizes (model weights as input)
- Memory usage tracking
- Throughput measurements
- Multiple M_early model fingerprinting

### 6. Training Orchestrator

**File**: `phaze/src/training_orchestrator.py`

Master coordinator that:
- Manages training across all architecture/complexity combinations
- Handles multiple seed runs for statistical analysis
- Coordinates ONNX export and ZK setup
- Generates M_early variants
- Runs comprehensive benchmarking
- Outputs data in format compatible with existing plotters

## Detailed Implementation Plan

### Phase 1: Core Training Infrastructure (Week 1)

1. **Training Configuration System**
   - Implement `TrainingConfig` dataclass
   - Add validation and defaults
   - Integration with existing model factory

2. **MNIST Training Module**
   - Lightweight MNIST data loader
   - Generic training loop for all model types
   - Progress tracking and early stopping
   - Model serialization

3. **ONNX Export Integration**
   - Extend existing ONNX export in `zkml_integration.py`
   - Add batch export capabilities
   - Validation of exported models

### Phase 2: Multi-Exit and Benchmarking (Week 1-2)

4. **Multi-Exit Generator**
   - Layer analysis for different architectures
   - Automatic truncation logic
   - M_early variant generation with parameter counting

5. **Enhanced ZK Benchmarking**
   - Extend existing `ZKMLProverVerifier` class
   - Add risc-zero backend support
   - Memory profiling integration
   - Batch processing capabilities

6. **Enhanced Crypto Benchmarking**
   - Extend existing crypto primitives
   - Add more hashing algorithms if needed
   - Model weight fingerprinting
   - Comprehensive metrics collection

### Phase 3: Orchestration and Integration (Week 2)

7. **Training Orchestrator**
   - Master workflow coordinator
   - Parallel processing where possible
   - Progress reporting and logging
   - Error handling and recovery

8. **Plotting Data Integration**
   - Ensure data format compatibility with existing plotters
   - Add any missing data fields for comprehensive plots
   - Validation of generated plot data

## Data Flow Architecture

```
1. TrainingOrchestrator
   ├── Load MNIST subset (1000 samples)
   ├── For each (architecture, complexity, seed):
   │   ├── Train model (5 epochs)
   │   ├── Export to ONNX
   │   └── Save model metadata
   │
   ├── Select largest M_full model
   ├── Generate M_early variants (25%, 50%, 75%)
   │
   ├── ZK Proof Benchmarking:
   │   ├── Setup ZK systems (ezkl + risc-zero)
   │   ├── Generate proofs for all M_full models
   │   └── Collect timing/memory data
   │
   ├── Fingerprinting Benchmarking:
   │   ├── Extract weights from all M_early models
   │   ├── Run all hashing algorithms
   │   └── Collect timing/memory/throughput data
   │
   └── Generate plotting data:
       ├── Format for FingerprintPlotter
       ├── Format for ZKMLPlotter
       └── Save to benchmark results
```

## Expected Outputs

### Model Artifacts
- 20 trained models (4 architectures × 5 complexities)
- 20 ONNX exported models
- 3-6 M_early variants per architecture
- Model metadata and parameter counts

### Benchmark Data Structure

```python
{
    "training_results": {
        "model_id": {
            "architecture": str,
            "complexity": str,
            "parameters": int,
            "training_time": float,
            "final_loss": float,
            "onnx_export_time": float
        }
    },
    "zkml_results": [
        {
            "framework": "ezkl",
            "model_id": str,
            "complexity": str,
            "model_parameters": int,
            "setup_time": float,
            "proof_time": float,
            "verification_time": float,
            "memory_usage_mb": float,
            "proof_size_bytes": int,
            "success": bool
        }
    ],
    "crypto_results": [
        {
            "primitive_name": "rabin_fingerprint",
            "operation": "hash",
            "input_size": int,
            "execution_time": float,
            "memory_usage_mb": float,
            "throughput_ops_per_sec": float,
            "success": bool
        }
    ]
}
```

### Generated Plots
- **Fingerprinting Plots**: Time complexity, memory usage, throughput, algorithm comparison
- **ZK Proof Plots**: Proof generation time, verification time, memory usage, framework comparison
- **Model Scaling Plots**: Parameter count vs performance metrics
- **Statistical Plots**: Error bars from multiple seed runs

## Configuration Files

### `training_config.yaml`
```yaml
training:
  epochs: 5
  batch_size: 64
  learning_rate: 0.001
  dataset_size: 1000

models:
  architectures: ["simple", "conv", "transformer", "multi_exit"]
  complexities: ["minimal", "light", "medium", "heavy"]

experiments:
  seeds: [42, 123, 456, 789, 999]
  zkml_frameworks: ["ezkl", "risc_zero"]
  hashing_algorithms: ["rabin", "shamir"]

output:
  save_models: true
  export_onnx: true
  generate_plots: true
  output_dir: "benchmark_results"
```

## Success Criteria

1. **Training Pipeline**: Successfully train all 20 model variants in under 30 minutes total
2. **ZK Integration**: Generate proofs for at least 80% of trained models
3. **Fingerprinting**: Complete hashing benchmarks for all M_early variants
4. **Plot Generation**: Produce all 13 plot types (6 fingerprinting + 7 zkML)
5. **Reproducibility**: Consistent results across multiple seed runs
6. **Performance**: Complete full pipeline in under 2 hours on standard hardware

## Implementation Timeline

- **Days 1-2**: Core training infrastructure and MNIST integration
- **Days 3-4**: Multi-exit generation and enhanced benchmarking
- **Days 5-6**: Training orchestrator and data integration
- **Day 7**: Testing, validation, and documentation

## Next Steps

1. **Review and Approval**: Review this plan and suggest modifications
2. **Implementation Priority**: Decide on implementation order based on dependencies
3. **Resource Allocation**: Determine if parallel development is possible
4. **Testing Strategy**: Define validation criteria for each component

This plan focuses on generating meaningful benchmark data quickly while maintaining compatibility with the existing plotting infrastructure. The emphasis on minimal training and small datasets ensures fast iteration while still producing realistic performance characteristics for the plotting system.