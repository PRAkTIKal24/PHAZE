#!/bin/bash
# Development installation script for PHAZE
# This script installs the package in editable mode and sets up the proper paths

set -e

echo "🔧 Installing PHAZE in development mode..."

# Install the package in editable mode
echo "📦 Installing package..."
uv pip install -e .

# Run the development setup script
echo "⚙️  Setting up editable install paths..."
uv run python dev_setup.py

echo "✅ Development installation complete!"
echo ""
echo "🚀 You can now use:"
echo "   uv run phaze --help"
echo "   uv run phaze --list-benchmarks"