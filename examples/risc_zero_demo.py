#!/usr/bin/env python3
"""
RISC Zero Integration Example for PHAZE

This example demonstrates the new RISC Zero backend integration,
showing how to use RISC Zero's zkVM for privacy-preserving ML inference.
"""

import asyncio
import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path so we can import phaze
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phaze.src.zkml_backends import RiscZeroBackend


async def main():
    print("🚀 RISC Zero Integration Demo")
    print("=" * 50)
    
    # Create a simple neural network
    print("1. Creating a simple linear model...")
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    # Initialize RISC Zero backend
    print("2. Initializing RISC Zero backend...")
    backend = RiscZeroBackend(model, "demo")
    
    # Show framework information
    info = backend.get_framework_info()
    print(f"   Framework: {info['framework']}")
    print(f"   Version: {info['version']}")
    print(f"   Proof System: {info['proof_system']}")
    print(f"   Backend: {info['backend']}")
    print(f"   Status: {info['status']}")
    
    # Create sample input
    print("\n3. Preparing input data...")
    input_data = torch.randn(1, 10)
    print(f"   Input shape: {input_data.shape}")
    
    # Setup the zkVM
    print("\n4. Setting up RISC Zero zkVM...")
    await backend.setup(input_data)
    print(f"   ✓ Guest ID: {backend.guest_id}")
    print(f"   ✓ zkVM Binary: {backend.zkvm_binary_path}")
    
    # Generate proof
    print("\n5. Generating zero-knowledge proof...")
    proof, output = await backend.generate_proof(input_data)
    print(f"   ✓ Proof generated")
    print(f"   ✓ Output: {output}")
    print(f"   ✓ Proof system: {proof['proof_system']}")
    print(f"   ✓ Guest ID: {proof['receipt']['guest_id']}")
    
    # Verify proof
    print("\n6. Verifying proof...")
    verified = await backend.verify_proof(proof, input_data)
    print(f"   ✓ Verification result: {verified}")
    
    # Compare with regular inference
    print("\n7. Comparing with regular inference...")
    with torch.no_grad():
        regular_output = model(input_data)
    
    zkvm_tensor = torch.tensor(output).reshape(regular_output.shape)
    match = torch.allclose(zkvm_tensor, regular_output, atol=1e-6)
    print(f"   ✓ Outputs match: {match}")
    print(f"   ✓ Regular: {regular_output.item():.6f}")
    print(f"   ✓ zkVM: {zkvm_tensor.item():.6f}")
    
    # Cleanup
    print("\n8. Cleaning up...")
    backend.cleanup()
    print("   ✓ Cleanup completed")
    
    print("\n🎉 RISC Zero integration demo completed successfully!")
    print("\nKey Benefits:")
    print("  • Privacy-preserving ML inference")
    print("  • Zero-knowledge proofs of correct computation")
    print("  • STARK-based proof system")
    print("  • Compatible with existing PHAZE architecture")


if __name__ == "__main__":
    asyncio.run(main())