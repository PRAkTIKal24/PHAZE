#!/bin/bash

# Build script for RISC Zero guest program
# This requires cargo-risczero to be installed: cargo install cargo-risczero

set -e

echo "Building RISC Zero guest program..."

# Check if cargo-risczero is installed
if ! command -v cargo-risczero &> /dev/null; then
    echo "cargo-risczero not found. Installing..."
    cargo install cargo-risczero
fi

# Install the RISC Zero toolchain
echo "Installing RISC Zero toolchain..."
cargo risczero install

# Build the guest program
echo "Building guest program..."
cd risc0_guest
cargo risczero build

echo "Guest program built successfully!"
echo "ELF location: risc0_guest/target/riscv32im-risc0-zkvm-elf/release/risc0_guest"