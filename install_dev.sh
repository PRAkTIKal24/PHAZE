#!/bin/bash
# Development installation script for PHAZE
# This script installs the package in editable mode and sets up the proper paths

set -e

echo "🔧 Installing PHAZE in development mode..."

# Check if uv is installed, install if not
if ! command -v uv &> /dev/null; then
    echo "📥 uv not found. Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo "❌ Failed to install uv. Please install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
    echo "✅ uv installed successfully!"
else
    echo "✅ uv is already installed"
fi

# Create venv for installation
uv venv
source .venv/bin/activate

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