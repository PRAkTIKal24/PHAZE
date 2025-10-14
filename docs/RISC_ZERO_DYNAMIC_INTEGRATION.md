# PHAZE Dynamic RISC Zero Integration

This document describes the new dynamic RISC Zero integration system that automatically generates guest programs for any model architecture in the PHAZE model factory.

## Overview

The new system replaces the hardcoded RISC Zero implementation with a flexible, architecture-aware approach that:

- ✅ **Builds once, runs anywhere**: Pre-build guest programs for all model architectures during setup
- ✅ **Automatic model support**: Any model from the PHAZE factory gets RISC Zero support automatically
- ✅ **Extensible architecture**: Easy to add new model types without manual RISC Zero coding
- ✅ **Build-time optimization**: Generate optimized guest programs for each architecture

## Multi-Dataset Support

The dynamic RISC Zero integration supports multiple datasets automatically:

### Supported Datasets
- **MNIST**: 28×28 grayscale handwritten digits (784 inputs, 10 classes)
- **CIFAR-10**: 32×32 RGB objects (3072 inputs, 10 classes)
- **CIFAR-100**: 32×32 RGB objects (3072 inputs, 100 classes)
- **Fashion-MNIST**: 28×28 grayscale clothing (784 inputs, 10 classes)
- **ImageNet**: 224×224 RGB objects (150,528 inputs, 1000 classes)
- **Tiny ImageNet**: 64×64 RGB objects (12,288 inputs, 200 classes)

### Dataset-Specific Building

```bash
# Build for specific dataset
python setup_phaze_complete.py --dataset cifar10

# Or rebuild for different dataset
uv run phaze-risc-build build --dataset cifar100

# List supported datasets
uv run phaze-risc-build datasets

# Scan factory for dataset
uv run phaze-risc-build scan --dataset imagenet
```

### Custom Datasets

```python
from phaze.src.dataset_config import create_custom_dataset_config

# Define your dataset
custom_config = create_custom_dataset_config(
    name="my_dataset",
    input_size=1024,     # Your input size
    output_size=20,      # Your number of classes
    input_channels=3,    # RGB
    spatial_size=32,     # 32x32 images
    description="My custom dataset"
)

# Build guest programs
from phaze.src.risc_zero_codegen import RiscZeroArchitectureRegistry
registry = RiscZeroArchitectureRegistry()
registry.register_all_factory_models(custom_config.to_dict())
```

### Components

1. **RiscZeroArchitectureRegistry**: Manages model architecture registration and metadata
2. **RiscZeroTemplateEngine**: Generates guest programs from Rust templates
3. **RiscZeroBuildManager**: Handles compilation of guest programs using cargo-risczero
4. **RiscZeroBackendWrapper**: Runtime selection of appropriate guest programs

### File Structure

```
rust_bindings/
├── guest_templates/          # Rust templates for different architectures
│   ├── simple_model.rs.template
│   ├── conv_model.rs.template
│   ├── transformer_model.rs.template
│   └── multi_exit_model.rs.template
├── guest_programs/           # Generated guest programs
│   ├── architecture_registry.json
│   ├── guest_simple_minimal/
│   ├── guest_simple_light/
│   └── ...
└── risc0_guest/             # Legacy guest program (still present)
```

## Build Process

### Initial Setup
```bash
# Full setup with dynamic RISC Zero support
python setup_phaze_complete.py

# This will:
# 1. Install RISC Zero toolchain
# 2. Scan PHAZE model factory for all architectures
# 3. Generate guest programs for each architecture-complexity combination
# 4. Build all guest programs using cargo-risczero
```

### Adding New Models

When you add a new model architecture to the PHAZE model factory:

```bash
# Rebuild guest programs for new architectures
uv run phaze-risc-build build

# Check what's registered
uv run phaze-risc-build list

# Scan for new architectures
uv run phaze-risc-build scan
```

## Usage

### Runtime Selection

The system automatically selects the appropriate guest program based on model metadata:

```python
from phaze.src.enhanced_zkml_benchmark import RiscZeroBackendWrapper

# Create wrapper - automatically detects architecture
wrapper = RiscZeroBackendWrapper(model, model_info)

# Setup - finds and loads appropriate guest program
await wrapper.setup(sample_input)

# Generate proof - uses architecture-specific guest program
proof, output = await wrapper.generate_proof(sample_input)
```

### Model Architecture Support

Currently supported architectures:
- **simple**: Feed-forward networks
- **conv**: Convolutional networks  
- **transformer**: Transformer-based models
- **multi_exit**: Multi-exit models

Each architecture supports all complexity levels (minimal, light, medium, heavy, extreme).

## Benefits

### For Users
- **Zero manual RISC Zero coding**: Just add PyTorch models to the factory
- **Automatic optimization**: Each model gets a specialized guest program
- **Build once workflow**: Set up once, use with any supported model
- **Clear error messages**: Know exactly which models are supported

### For Developers
- **Template-based generation**: Easy to add new architecture types
- **Extensible design**: Clean separation of concerns
- **Version management**: Track which models are built and supported
- **Debugging tools**: Utilities to inspect and manage guest programs

## Templates

### Adding New Architecture Templates

1. Create a new template in `rust_bindings/guest_templates/`:
```rust
// my_new_arch.rs.template
#![no_std]
extern crate alloc;

// Architecture-specific implementation
```

2. Update the template mapping in `RiscZeroTemplateEngine._get_template_name()`

3. Add weight conversion logic in `RiscZeroBackendWrapper._convert_*_weights()`

4. Rebuild guest programs:
```bash
uv run phaze-risc-build build
```

## Error Handling

### Common Issues

**"Architecture not registered"**
- Solution: Run `uv run phaze-risc-build build`

**"Guest program not found"**  
- Solution: Check RISC Zero toolchain installation, then rebuild

**"cargo-risczero not found"**
- Solution: Install RISC Zero toolchain: `python setup_phaze_complete.py`

## Migration from Legacy System

The legacy hardcoded RISC Zero implementation is completely replaced. No backward compatibility is maintained as requested.

### Before (Legacy)
- Hardcoded 784->128->10 architecture
- Manual weight padding for smaller models
- Single guest program for all models

### After (Dynamic)
- Architecture-aware guest program selection
- Optimized guest programs for each model type
- Automatic registration and building

## Testing

Test the integration:
```bash
# Run integration test
uv run python test_risc_zero_integration.py

# Build and test specific architecture
uv run phaze-risc-build build
uv run phaze-risc-build list

# Run full PHAZE pipeline
uv run phaze --config phaze_config.py
```

## Performance Considerations

- **Build time**: Initial setup takes longer (builds multiple guest programs)
- **Runtime**: Faster execution (optimized guest programs per architecture)
- **Storage**: More disk space (multiple guest programs vs. single hardcoded one)
- **Memory**: Better memory usage (no weight padding needed)

## Future Enhancements

- **Dynamic complexity scaling**: Generate guest programs with variable layer sizes
- **Hot-reloading**: Rebuild guest programs without full reinstall
- **Cross-compilation**: Build guest programs for different target platforms
- **Optimization profiles**: Different optimization levels for guest programs