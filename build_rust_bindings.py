#!/usr/bin/env python3
"""
Build and install script for PHAZE Rust bindings.
This script builds the Rust library and installs it in the Python environment.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run a command and show output in real-time."""
    print(f"Running: {' '.join(cmd)}")
    
    # Use subprocess.run without capture_output to show real-time output
    result = subprocess.run(cmd, cwd=cwd)
    
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")
    
    return result

def check_rust_installation():
    """Check if Rust and Cargo are installed."""
    try:
        result = run_command(["cargo", "--version"], check=False)
        if result.returncode == 0:
            print(f"✅ Rust/Cargo found")
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def install_rust():
    """Install Rust using rustup."""
    print("🦀 Installing Rust...")
    
    # Download and run rustup installer
    if sys.platform == "win32":
        print("Please install Rust manually from https://rustup.rs/")
        return False
    else:
        try:
            # Download rustup installer
            curl_cmd = [
                "curl", "--proto", "=https", "--tlsv1.2", "-sSf", 
                "https://sh.rustup.rs", "-o", "/tmp/rustup-init.sh"
            ]
            run_command(curl_cmd)
            
            # Make it executable and run
            run_command(["chmod", "+x", "/tmp/rustup-init.sh"])
            run_command(["/tmp/rustup-init.sh", "-y", "--default-toolchain", "stable"])
            
            # Source the environment
            home = os.path.expanduser("~")
            cargo_env = os.path.join(home, ".cargo", "env")
            if os.path.exists(cargo_env):
                # Add cargo to PATH for this session
                cargo_bin = os.path.join(home, ".cargo", "bin")
                if cargo_bin not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"
            
            return check_rust_installation()
            
        except Exception as e:
            print(f"❌ Rust installation failed: {e}")
            return False

def build_rust_bindings():
    """Build the Rust bindings."""
    print("🔨 Building Rust bindings...")
    
    # Get the project root directory
    project_root = Path(__file__).parent
    rust_dir = project_root / "rust_bindings"
    
    if not rust_dir.exists():
        raise RuntimeError(f"Rust bindings directory not found: {rust_dir}")
    
    # Clean previous builds
    print("🧹 Cleaning previous builds...")
    run_command(["cargo", "clean"], cwd=rust_dir)
    
    # Build the library
    print("🔨 Building Rust library...")
    run_command(["cargo", "build", "--release"], cwd=rust_dir)
    
    # Find the built library
    target_dir = rust_dir / "target" / "release"
    
    # Look for the shared library
    lib_patterns = [
        "librust_zkml_bindings.so",     # Linux
        "librust_zkml_bindings.dylib",  # macOS
        "rust_zkml_bindings.dll",       # Windows
    ]
    
    built_lib = None
    for pattern in lib_patterns:
        lib_path = target_dir / pattern
        if lib_path.exists():
            built_lib = lib_path
            break
    
    if not built_lib:
        raise RuntimeError(f"Built library not found in {target_dir}")
    
    print(f"✅ Built library found: {built_lib}")
    
    # Install the library
    install_library(built_lib)

def install_library(lib_path):
    """Install the built library into the Python environment."""
    print("📦 Installing library into Python environment...")
    
    # Get the site-packages directory
    import site
    site_packages = site.getsitepackages()[0]
    
    # Create the target filename (Python expects .so even on macOS)
    if sys.platform == "win32":
        target_name = "rust_zkml_bindings.pyd"
    else:
        target_name = "rust_zkml_bindings.so"
    
    target_path = Path(site_packages) / target_name
    
    # Copy the library
    print(f"📁 Copying {lib_path} -> {target_path}")
    shutil.copy2(lib_path, target_path)
    
    # Verify the installation
    try:
        import rust_zkml_bindings
        info = rust_zkml_bindings.get_binding_info()
        print(f"✅ Installation successful! Binding info: {info}")
        return True
    except ImportError as e:
        print(f"❌ Installation failed: {e}")
        return False

def main():
    """Main function."""
    print("🚀 PHAZE Rust Bindings Build Script")
    print("=" * 50)
    
    try:
        # Check if we already have working bindings
        try:
            import rust_zkml_bindings
            info = rust_zkml_bindings.get_binding_info()
            if info.get("implementation") == "real_rust_bindings":
                print(f"✅ Real Rust bindings already installed: {info}")
                return
        except ImportError:
            pass
        
        # Check if Rust is installed
        if not check_rust_installation():
            print("🦀 Rust not found. Installing...")
            if not install_rust():
                print("\n❌ Failed to install Rust. Please install manually:")
                print("   Visit: https://rustup.rs/")
                print("   Run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
                print("   Then run this script again.")
                sys.exit(1)
        
        # Build and install
        build_rust_bindings()
        print("\n🎉 Build and installation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        print("\n🔧 Alternative approaches:")
        print("1. Install Rust manually: https://rustup.rs/")
        print("2. Use the pre-built Python wheels (if available)")
        print("3. For now, PHAZE will use mock implementations")
        sys.exit(1)

if __name__ == "__main__":
    main()