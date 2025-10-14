#!/bin/bash

# Build script for RISC Zero guest program
# Prerequisites: RISC Zero toolchain must be installed via rzup

set -e

echo "🔨 Building RISC Zero guest program..."

# Verify prerequisites
if ! command -v cargo-risczero &> /dev/null; then
    echo "❌ cargo-risczero not found. Installing via rzup..."
    if ! command -v rzup &> /dev/null; then
        echo "❌ rzup not found. Please run ../setup_risc_zero.sh first"
        exit 1
    fi
    rzup install cargo-risczero
fi

if ! command -v rzup &> /dev/null; then
    echo "❌ rzup not found. Please run ../setup_risc_zero.sh first"
    exit 1
fi

echo "✅ RISC Zero toolchain detected"

# Ensure we have the Rust toolchain
echo "🦀 Installing RISC Zero Rust toolchain..."
rzup install rust || echo "⚠️  Rust toolchain installation may have issues, continuing..."

# Navigate to guest directory
cd "$(dirname "$0")/risc0_guest"

# Build the guest program
echo "🔨 Building guest program..."
cargo risczero build

echo "✅ Guest program built successfully!"
echo "📁 ELF location: target/riscv32im-risc0-zkvm-elf/release/risc0_guest"

# Verify the ELF was created
if [ -f "target/riscv32im-risc0-zkvm-elf/release/risc0_guest" ]; then
    echo "✅ ELF file verified"
    ls -la target/riscv32im-risc0-zkvm-elf/release/risc0_guest
else
    echo "❌ ELF file not found!"
    exit 1
fi

echo ""
echo "🎉 RISC Zero guest program build complete!"
echo "💡 You can now run PHAZE with full RISC Zero support"