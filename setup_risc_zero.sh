#!/bin/bash

# RISC Zero Setup Script for PHAZE
# This script provides a fresh installation of RISC Zero toolchain

set -e

echo "🔧 Setting up RISC Zero for PHAZE..."

# Check if rzup is installed
if ! command -v rzup &> /dev/null; then
    echo "📦 Installing RISC Zero using rzup..."
    curl -L https://risczero.com/install | bash
    source "$HOME/.bashrc"
    echo "✅ rzup installed successfully"
else
    echo "✅ rzup is already installed"
fi

# Install RISC Zero components
echo "📦 Installing RISC Zero components..."
rzup install cargo-risczero
rzup install r0vm

# Verify installation
echo "🔍 Verifying installation..."
rzup show

# Check if cargo-risczero is available
if command -v cargo-risczero &> /dev/null; then
    echo "✅ cargo-risczero is available"
    cargo-risczero --version
else
    echo "❌ cargo-risczero not found in PATH"
    exit 1
fi

echo "🎉 RISC Zero setup complete!"
echo ""
echo "Next steps:"
echo "1. Build the guest program: ./build_guest.sh"
echo "2. Run tests: python -m pytest tests/test_risc_zero_*.py -v"