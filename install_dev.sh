#!/bin/bash#!/bin/bash

# Development installation script for PHAZE# Development installation script for PHAZE

# This script provides multiple installation options

set -e

set -e

echo "🔧 PHAZE Development Installation"

echo "================================="echo "🔧 PHAZE Development Installation"

echo "================================="

# Check if comprehensive setup exists and prefer it

if [ -f "setup_phaze_complete.py" ]; then# Check if the comprehensive setup script exists

    echo "🚀 Using comprehensive setup (recommended)..."if [ -f "setup_phaze_complete.py" ]; then

    echo ""    echo "🚀 Using comprehensive PHAZE setup (includes RISC Zero)..."

    echo "Choose installation:"    echo ""

    echo "  1) Full setup with RISC Zero (recommended)"    echo "Choose installation mode:"

    echo "  2) Quick setup without RISC Zero"    echo "  1) Full setup with RISC Zero (recommended for development)"

    echo "  3) Manual uv-only setup"    echo "  2) Quick setup without RISC Zero (faster, reduced functionality)"

    echo ""    echo "  3) Legacy installation (uv only)"

    read -p "Enter choice (1-3) [1]: " choice    echo ""

    choice=${choice:-1}    read -p "Enter choice (1-3) [1]: " choice

        choice=${choice:-1}

    case $choice in    

        1)    case $choice in

            echo "🦀 Full setup with RISC Zero..."        1)

            python setup_phaze_complete.py --development            echo "🦀 Installing with full RISC Zero support..."

            exit 0            python setup_phaze_complete.py --development

            ;;            ;;

        2)        2)

            echo "⚡ Quick setup..."            echo "⚡ Installing without RISC Zero (quick setup)..."

            python setup_phaze_complete.py --development --skip-risc-zero            python setup_phaze_complete.py --development --skip-risc-zero

            exit 0            ;;

            ;;        3)

        3)            echo "� Using legacy installation..."

            echo "📦 Manual setup..."            # Fall through to legacy installation

            ;;            ;;

        *)        *)

            echo "❌ Invalid choice. Using full setup..."            echo "❌ Invalid choice. Using full setup..."

            python setup_phaze_complete.py --development            python setup_phaze_complete.py --development

            exit 0            ;;

            ;;    esac

    esac    

fi    # Exit if we used the comprehensive setup

    if [ "$choice" != "3" ]; then

echo "📦 Manual uv installation..."        exit 0

    fi

# Install uv if neededfi

if ! command -v uv &> /dev/null; then

    echo "📥 Installing uv..."echo "📦 Running legacy installation..."

    curl -LsSf https://astral.sh/uv/install.sh | sh

    export PATH="$HOME/.cargo/bin:$PATH"# Check if uv is installed, install if not

fiif ! command -v uv &> /dev/null; then

    echo "📥 uv not found. Installing uv package manager..."

# Install package    curl -LsSf https://astral.sh/uv/install.sh | sh

echo "🐍 Setting up environment..."    

uv venv && source .venv/bin/activate    # Add uv to PATH for current session

uv pip install -e .    export PATH="$HOME/.cargo/bin:$PATH"

uv run python dev_setup.py    

    # Verify installation

echo "✅ Installation complete!"    if ! command -v uv &> /dev/null; then

echo "🚀 Try: uv run phaze --help"        echo "❌ Failed to install uv. Please install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
    echo "✅ uv installed successfully!"
else
    echo "✅ uv is already installed"
fi

# Create venv for installation
echo "🐍 Setting up virtual environment..."
uv venv
source .venv/bin/activate

# Install the package in editable mode
echo "📦 Installing package in editable mode..."
uv pip install -e .

# Run the development setup script
echo "⚙️ Setting up editable install paths..."
uv run python dev_setup.py

echo "✅ Legacy development installation complete!"
echo ""
echo "⚠️  Note: This installation does not include RISC Zero support."
echo "   For full functionality, run: python setup_phaze_complete.py"
echo ""
echo "🚀 You can now use:"
echo "   uv run phaze --help"
echo "   uv run phaze-legacy --list-benchmarks"