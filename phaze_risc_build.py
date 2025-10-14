#!/usr/bin/env python3
"""
PHAZE RISC Zero build utility.

This utility manages dynamic guest program generation for RISC Zero support.
Use this when you add new model architectures to the PHAZE model factory.
"""

import argparse
import sys
from pathlib import Path

def build_guest_programs(dataset_name="mnist"):
    """Build guest programs for all registered architectures.
    
    Args:
        dataset_name: Name of the dataset to build for
    """
    try:
        from phaze.src.risc_zero_codegen import RiscZeroArchitectureRegistry, RiscZeroBuildManager
        from phaze.src.dataset_config import get_dataset_config, list_supported_datasets
        
        print(f"🏗️  Building RISC Zero guest programs for dataset: {dataset_name}")
        
        # Get dataset configuration
        try:
            dataset_config = get_dataset_config(dataset_name).to_dict()
            print(f"📊 Dataset info: {dataset_config['input_size']} inputs → {dataset_config['output_size']} classes")
        except ValueError as e:
            print(f"❌ {e}")
            print(f"💡 Supported datasets: {', '.join(list_supported_datasets())}")
            return False
        
        # Initialize registry and scan for models
        registry = RiscZeroArchitectureRegistry()
        newly_registered = registry.register_all_factory_models(dataset_config)
        
        if newly_registered > 0:
            print(f"📝 Registered {newly_registered} new model architectures")
        else:
            print("📝 All model architectures already registered")
        
        # Build all guest programs
        build_manager = RiscZeroBuildManager(registry)
        results = build_manager.build_all_guest_programs()
        
        # Report results
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        print(f"\n📊 Build Results:")
        for arch_key, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {arch_key}")
        
        print(f"\n🎯 Successfully built {success_count}/{total_count} guest programs for {dataset_name}")
        
        if success_count == 0:
            print("❌ No guest programs were built successfully")
            print("💡 Check that the RISC Zero toolchain is installed: rzup install")
            return False
        elif success_count < total_count:
            print("⚠️  Some guest programs failed to build")
            print("💡 PHAZE will work with the successfully built architectures")
        else:
            print("🎉 All guest programs built successfully!")
        
        return success_count > 0
        
    except ImportError as e:
        print(f"❌ Failed to import PHAZE modules: {e}")
        print("💡 Make sure PHAZE is installed: uv pip install -e .")
        return False
    except Exception as e:
        print(f"❌ Error building guest programs: {e}")
        return False

def list_architectures():
    """List all registered model architectures."""
    try:
        from phaze.src.risc_zero_codegen import RiscZeroArchitectureRegistry
        
        registry = RiscZeroArchitectureRegistry()
        architectures = registry.list_architectures()
        
        if not architectures:
            print("📝 No architectures registered yet")
            print("💡 Run 'build-guest-programs' to register and build architectures")
            return
        
        print(f"📝 Registered architectures ({len(architectures)}):")
        for arch_key in sorted(architectures):
            arch_info = registry.get_architecture_info(arch_key)
            print(f"  • {arch_key}")
            if arch_info:
                print(f"    Input: {arch_info['input_size']}, Output: {arch_info['output_size']}")
                print(f"    Parameters: {arch_info['parameters']:,}")
                
                # Check if guest program exists
                from phaze.src.risc_zero_codegen import RiscZeroBuildManager
                build_manager = RiscZeroBuildManager(registry)
                guest_path = build_manager.get_guest_program_path(arch_key)
                if guest_path:
                    print(f"    Guest program: ✅ {guest_path.name}")
                else:
                    print(f"    Guest program: ❌ Not built")
            print()
            
    except ImportError as e:
        print(f"❌ Failed to import PHAZE modules: {e}")
    except Exception as e:
        print(f"❌ Error listing architectures: {e}")

def scan_factory(dataset_name="mnist"):
    """Scan the model factory for available architectures.
    
    Args:
        dataset_name: Name of the dataset to scan for
    """
    try:
        from phaze.src.risc_zero_codegen import RiscZeroArchitectureRegistry
        from phaze.src.model_architectures import PHAZEModelFactory
        from phaze.src.dataset_config import get_dataset_config, list_supported_datasets
        
        print(f"🔍 Scanning PHAZE model factory for dataset: {dataset_name}")
        
        # Get dataset configuration
        try:
            dataset_config = get_dataset_config(dataset_name).to_dict()
            print(f"📊 Dataset: {dataset_config['description']}")
            print(f"📊 Input: {dataset_config['input_size']} ({dataset_config['input_channels']}×{dataset_config['spatial_size']}×{dataset_config['spatial_size']})")
            print(f"📊 Output: {dataset_config['output_size']} classes")
        except ValueError as e:
            print(f"❌ {e}")
            print(f"💡 Supported datasets: {', '.join(list_supported_datasets())}")
            return
        
        architectures = PHAZEModelFactory.get_available_architectures()
        complexities = PHAZEModelFactory.get_complexity_levels()
        
        print(f"📋 Available architectures: {', '.join(architectures)}")
        print(f"📋 Available complexities: {', '.join(c.value for c in complexities)}")
        
        total_combinations = len(architectures) * len(complexities)
        print(f"📊 Total architecture-complexity combinations: {total_combinations}")
        
        # Check registration status
        registry = RiscZeroArchitectureRegistry()
        registered_count = len(registry.list_architectures())
        
        print(f"📝 Currently registered: {registered_count}")
        
        if registered_count < total_combinations:
            print(f"💡 Run 'phaze-risc-build build --dataset {dataset_name}' to register missing architectures")
        
    except ImportError as e:
        print(f"❌ Failed to import PHAZE modules: {e}")
    except Exception as e:
        print(f"❌ Error scanning factory: {e}")

def list_datasets():
    """List all supported datasets."""
    try:
        from phaze.src.dataset_config import list_supported_datasets, get_dataset_config
        
        print("📊 Supported datasets:")
        for name in list_supported_datasets():
            config = get_dataset_config(name)
            print(f"  • {name}")
            print(f"    {config.description}")
            print(f"    Input: {config.input_size} ({config.input_channels}×{config.spatial_size}×{config.spatial_size})")
            print(f"    Output: {config.output_size} classes")
            print()
            
    except ImportError as e:
        print(f"❌ Failed to import PHAZE modules: {e}")
    except Exception as e:
        print(f"❌ Error listing datasets: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="PHAZE RISC Zero build utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  phaze-risc-build build                    # Build guest programs for MNIST (default)
  phaze-risc-build build --dataset cifar10  # Build for CIFAR-10
  phaze-risc-build list                     # List registered architectures
  phaze-risc-build scan --dataset imagenet  # Scan factory for ImageNet
  phaze-risc-build datasets                 # List supported datasets
  
Use this utility when you add new model architectures or switch datasets
to ensure RISC Zero compatibility.
        """
    )
    
    parser.add_argument(
        "command",
        choices=["build", "list", "scan", "datasets"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        default="mnist",
        help="Dataset to build/scan for (default: mnist)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    print("🦀 PHAZE RISC Zero Build Utility")
    print("=" * 40)
    
    success = True
    
    if args.command == "build":
        success = build_guest_programs(args.dataset)
    elif args.command == "list":
        list_architectures()
    elif args.command == "scan":
        scan_factory(args.dataset)
    elif args.command == "datasets":
        list_datasets()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()