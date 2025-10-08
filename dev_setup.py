#!/usr/bin/env python3
"""
Development setup script for PHAZE.
Run this after installing the package in editable mode to ensure proper path setup.

Usage:
    python dev_setup.py

Or automatically after installation:
    uv pip install -e . && python dev_setup.py
"""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path


def get_site_packages():
    """Get the site-packages directory for the current Python environment."""
    try:
        # Try to get it from sysconfig
        return sysconfig.get_path("purelib")
    except Exception:
        # Fallback for virtual environments
        return os.path.join(
            sys.prefix,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
        )


def setup_editable_install():
    """Setup editable install for PHAZE mixed Python/Rust project."""
    print("🔧 Setting up PHAZE editable install...")

    # Project root
    project_root = Path(__file__).parent.absolute()
    print(f"   Project root: {project_root}")

    # Get site-packages directory
    site_packages = get_site_packages()
    print(f"   Site-packages: {site_packages}")

    if not os.path.exists(site_packages):
        print(f"❌ Error: Site-packages directory not found: {site_packages}")
        return False

    # Create .pth file
    pth_file = Path(site_packages) / "__phaze_editable__.pth"

    try:
        # Check if already exists with correct path
        if pth_file.exists():
            existing_path = pth_file.read_text().strip()
            if existing_path == str(project_root):
                print("✅ Editable install already configured correctly")
                return True
            else:
                print(f"🔄 Updating existing .pth file (was: {existing_path})")

        # Write the .pth file
        pth_file.write_text(str(project_root))
        print(f"✅ Created editable install .pth file: {pth_file}")

        # Verify the setup works
        print("🧪 Testing import...")
        import_cmd = "import phaze; from phaze.phaze import main; print('✅ Success')"
        test_result = subprocess.run(
            [sys.executable, "-c", import_cmd],
            capture_output=True,
            text=True,
        )

        if test_result.returncode == 0:
            print(test_result.stdout.strip())
            print("🎉 PHAZE editable install setup complete!")
            return True
        else:
            print(f"❌ Import test failed: {test_result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error setting up editable install: {e}")
        return False


def main():
    """Main function."""
    print("PHAZE Development Setup")
    print("=" * 50)

    if not setup_editable_install():
        print("\n❌ Setup failed. You may need to manually create the .pth file:")
        pth_path = f"{get_site_packages()}/__phaze_editable__.pth"
        print(f"   echo '{Path(__file__).parent.absolute()}' > {pth_path}")
        sys.exit(1)

    print("\n🚀 Ready to use PHAZE CLI:")
    print("   uv run phaze --help")
    print("   uv run phaze --list-benchmarks")


if __name__ == "__main__":
    main()
