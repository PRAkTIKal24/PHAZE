#!/usr/bin/env python3
"""
Complete PHAZE setup script with RISC Zero integration.

This script handles:
1. RISC Zero toolchain installation
2. Rust bindings compilation with maturin
3. Guest program building
4. Python package installation in editable mode

Usage:
    python setup_phaze_complete.py [--skip-risc-zero] [--development]
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None, check=True, capture_output=False):
    """Run a command and optionally show output in real-time."""
    if isinstance(cmd, str):
        cmd = cmd.split()

    print(f"Running: {' '.join(cmd)}")

    if capture_output:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"❌ Command failed: {result.stderr}")
            raise RuntimeError(f"Command failed with return code {result.returncode}")
        return result
    else:
        result = subprocess.run(cmd, cwd=cwd)
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed with return code {result.returncode}")
        return result


def check_command_exists(cmd):
    """Check if a command exists in PATH."""
    return shutil.which(cmd) is not None


def setup_uv():
    """Ensure uv is installed and available."""
    print("🔧 Checking uv installation...")

    if check_command_exists("uv"):
        print("✅ uv is already installed")
        return True

    print("📥 Installing uv...")
    try:
        if platform.system() == "Windows":
            # Windows installation
            install_cmd = 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
            run_command(install_cmd, check=True)
        else:
            # Unix-like systems
            install_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
            run_command(install_cmd, check=True)

        # Add to PATH for current session
        home = os.path.expanduser("~")
        cargo_bin = os.path.join(home, ".cargo", "bin")
        if cargo_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"

        if check_command_exists("uv"):
            print("✅ uv installed successfully")
            return True
        else:
            print("❌ uv installation failed")
            return False

    except Exception as e:
        print(f"❌ Error installing uv: {e}")
        return False


def setup_risc_zero():
    """Install RISC Zero toolchain using rzup."""
    print("🦀 Setting up RISC Zero toolchain...")

    # Check if rzup is already installed
    if check_command_exists("rzup"):
        print("✅ rzup is already installed")
    else:
        print("📥 Installing rzup...")
        try:
            install_cmd = "curl -L https://risczero.com/install | bash"
            run_command(install_cmd, check=True)

            # Add to PATH for current session
            home = os.path.expanduser("~")
            rzup_bin = os.path.join(home, ".rzup", "bin")
            if rzup_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{rzup_bin}:{os.environ.get('PATH', '')}"

        except Exception as e:
            print(f"❌ Error installing rzup: {e}")
            return False

    # Install RISC Zero toolchain components
    try:
        print("🔧 Installing RISC Zero toolchain components...")
        run_command(["rzup", "install"], check=True)

        print("🦀 Installing cargo-risczero...")
        run_command(["rzup", "install", "cargo-risczero"], check=True)

        print("🦀 Installing Rust toolchain for RISC Zero...")
        run_command(["rzup", "install", "rust"], check=True)

        print("✅ RISC Zero toolchain setup complete")
        return True

    except Exception as e:
        print(f"❌ Error setting up RISC Zero toolchain: {e}")
        return False


def build_guest_program():
    """Build the RISC Zero guest program."""
    print("🔨 Building RISC Zero guest program...")

    project_root = Path(__file__).parent
    guest_dir = project_root / "rust_bindings" / "risc0_guest"

    if not guest_dir.exists():
        print(f"❌ Guest directory not found: {guest_dir}")
        return False

    try:
        # Build the guest program
        run_command(["cargo", "risczero", "build"], cwd=guest_dir, check=True)

        # Verify the ELF was created (check both possible locations)
        elf_paths = [
            guest_dir / "target" / "riscv32im-risc0-zkvm-elf" / "release" / "risc0_guest",
            guest_dir / "target" / "riscv32im-risc0-zkvm-elf" / "docker" / "risc0_guest.bin",
            guest_dir / "target" / "riscv32im-risc0-zkvm-elf" / "docker" / "risc0_guest",
        ]
        
        for elf_path in elf_paths:
            if elf_path.exists():
                print(f"✅ Guest program built successfully: {elf_path}")
                return True
        
        print("❌ Guest ELF file not found after build")
        print(f"   Checked paths:")
        for path in elf_paths:
            print(f"   - {path} (exists: {path.exists()})")
        return False

    except Exception as e:
        print(f"❌ Error building guest program: {e}")
        return False


def build_rust_bindings():
    """Build Rust bindings using maturin."""
    print("🔨 Building Rust bindings...")

    project_root = Path(__file__).parent
    rust_dir = project_root / "rust_bindings"

    if not rust_dir.exists():
        print(f"❌ Rust bindings directory not found: {rust_dir}")
        return False

    try:
        # Check and add required Rust targets (especially for Apple Silicon)
        print("🎯 Setting up Rust targets...")
        try:
            # Get current target
            result = run_command(
                ["rustc", "--version", "--verbose"], capture_output=True, check=True
            )

            # Check if we're on Apple Silicon but using wrong target
            if platform.machine() == "arm64" and platform.system() == "Darwin":
                print("🍎 Detected Apple Silicon Mac")

                # Add the correct target
                run_command(
                    ["rustup", "target", "add", "aarch64-apple-darwin"], check=False
                )

                # Check if default is wrong (x86_64 on arm64)
                default_result = run_command(
                    ["rustup", "default"], capture_output=True, check=False
                )
                if (
                    "x86_64" in default_result.stdout
                    and "apple-darwin" in default_result.stdout
                ):
                    print("🔄 Switching from x86_64 to native Apple Silicon target...")
                    run_command(
                        ["rustup", "default", "stable-aarch64-apple-darwin"],
                        check=False,
                    )

            elif "x86_64-apple-darwin" in result.stdout:
                # On Intel Mac, ensure we have the target
                print("💻 Detected Intel Mac, adding x86_64-apple-darwin target...")
                run_command(
                    ["rustup", "target", "add", "x86_64-apple-darwin"], check=False
                )

        except Exception as e:
            # If target detection fails, try adding common targets
            print(f"⚠️ Target detection failed ({e}), adding common targets...")
            run_command(
                ["rustup", "target", "add", "aarch64-apple-darwin"], check=False
            )
            run_command(["rustup", "target", "add", "x86_64-apple-darwin"], check=False)

        # # Clean previous builds
        # print("🧹 Cleaning previous builds...")
        # run_command(["cargo", "clean"], cwd=rust_dir, check=True)

        # Check if maturin is available via uv
        try:
            run_command(
                ["uv", "run", "maturin", "--version"], check=True, capture_output=True
            )
            maturin_cmd = ["uv", "run", "maturin"]
            print("✅ Using uv run maturin")
        except Exception:
            # Fallback to direct maturin
            if check_command_exists("maturin"):
                maturin_cmd = ["maturin"]
                print("✅ Using direct maturin")
            else:
                print("📥 Installing maturin...")
                run_command(["uv", "pip", "install", "maturin"], check=True)
                maturin_cmd = ["uv", "run", "maturin"]

        # Build and install in development mode
        print("🔨 Building with maturin...")
        build_cmd = maturin_cmd + ["develop", "--release"]
        run_command(build_cmd, cwd=rust_dir, check=True)

        print("✅ Rust bindings built successfully")
        return True

    except Exception as e:
        print(f"❌ Error building Rust bindings: {e}")
        return False


def install_python_package(development=True):
    """Install the Python package."""
    print("📦 Installing Python package...")

    project_root = Path(__file__).parent

    try:
        # Create/update virtual environment
        print("🐍 Setting up virtual environment...")
        run_command(["uv", "venv"], cwd=project_root, check=True)

        if development:
            # Install in editable mode
            print("🔧 Installing in development mode...")
            run_command(
                ["uv", "pip", "install", "-e", "."], cwd=project_root, check=True
            )

            # Run dev setup
            print("⚙️ Running development setup...")
            run_command(
                ["uv", "run", "python", "dev_setup.py"], cwd=project_root, check=True
            )
        else:
            # Regular installation
            print("📦 Installing package...")
            run_command(["uv", "pip", "install", "."], cwd=project_root, check=True)

        print("✅ Python package installed successfully")
        return True

    except Exception as e:
        print(f"❌ Error installing Python package: {e}")
        return False


def verify_installation():
    """Verify that everything is working correctly."""
    print("🧪 Verifying installation...")

    try:
        # Test import
        result = run_command(
            [
                "uv",
                "run",
                "python",
                "-c",
                "import phaze; import rust_zkml_bindings; "
                "info = rust_zkml_bindings.get_binding_info(); "
                "print(f'✅ PHAZE installation verified: {info}')",
            ],
            capture_output=True,
            check=True,
        )

        print(result.stdout.strip())

        # Test CLI
        result = run_command(
            ["uv", "run", "phaze", "--help"], capture_output=True, check=True
        )
        print("✅ PHAZE CLI is working")

        return True

    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Complete PHAZE setup with RISC Zero")
    parser.add_argument(
        "--skip-risc-zero",
        action="store_true",
        help="Skip RISC Zero toolchain installation",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        default=True,
        help="Install in development mode (default)",
    )
    parser.add_argument(
        "--production", action="store_true", help="Install in production mode"
    )

    args = parser.parse_args()

    if args.production:
        args.development = False

    print("🚀 PHAZE Complete Setup")
    print("=" * 50)

    success = True

    # Step 1: Setup uv
    if not setup_uv():
        print("❌ Failed to setup uv")
        success = False

    # Step 2: Setup RISC Zero (optional)
    if not args.skip_risc_zero and success:
        if not setup_risc_zero():
            print("❌ Failed to setup RISC Zero toolchain")
            print("💡 You can continue without RISC Zero by using --skip-risc-zero")
            success = False

    # Step 3: Install Python package
    if success:
        if not install_python_package(development=args.development):
            print("❌ Failed to install Python package")
            success = False

    # Step 4: Build Rust bindings
    if success:
        if not build_rust_bindings():
            print("❌ Failed to build Rust bindings")
            success = False

    # Step 5: Build guest program (if RISC Zero is available)
    if not args.skip_risc_zero and success:
        if not build_guest_program():
            print("❌ Failed to build guest program")
            print("💡 PHAZE will work with reduced RISC Zero functionality")

    # Step 6: Verify installation
    if success:
        if not verify_installation():
            print("❌ Installation verification failed")
            success = False

    print("\n" + "=" * 50)
    if success:
        print("🎉 PHAZE setup completed successfully!")
        print("\n🚀 You can now use:")
        print("   uv run phaze --help")
        print("   uv run phaze-legacy --list-benchmarks")
        print(
            "   uv run python -c \"import rust_zkml_bindings; print('RISC Zero ready!')\""
        )
    else:
        print("❌ Setup failed. Please check the errors above.")
        print("\n🔧 You can try:")
        print("   python setup_phaze_complete.py --skip-risc-zero  # Skip RISC Zero")
        print("   ./install_dev.sh  # Use the existing simple script")
        sys.exit(1)


if __name__ == "__main__":
    main()
