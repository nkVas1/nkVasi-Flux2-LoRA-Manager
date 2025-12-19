#!/usr/bin/env python
"""
Setup script for training environment (Embedded Python Compatible).
Installs packages to separate directory instead of creating venv.

Usage:
    python setup_training_env.py
    python setup_training_env.py --force  # Force reinstall
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from venv_manager import StandalonePackageManager


def main():
    parser = argparse.ArgumentParser(description="Setup Flux2 training packages")
    parser.add_argument("--force", action="store_true", help="Force reinstall packages")
    args = parser.parse_args()
    
    print("=" * 70)
    print("FLUX2 LORA MANAGER - TRAINING PACKAGES SETUP")
    print("=" * 70)
    print()
    print("NOTE: This method works with embedded Python (no venv module needed)")
    print()
    
    manager = StandalonePackageManager()
    
    print(f"Packages will be installed to:")
    print(f"  {manager.libs_dir}")
    print()
    
    if manager.libs_exist() and not args.force:
        print("Training packages already exist.")
        print("Use --force to reinstall.")
        print()
        
        # Verify existing installation
        print("Verifying existing packages...")
        all_ok, messages = manager.verify_installation()
        
        for msg in messages:
            print(f"  {msg}")
        
        if all_ok:
            print("\n✓ Packages are ready for training")
            return 0
        else:
            print("\n✗ Some packages have issues, use --force to reinstall")
            return 1
    
    # Install packages
    print("Installing training packages...")
    print("This will take 5-10 minutes (downloading ~2GB)...")
    print()
    
    success, msg = manager.setup_training_packages(force_reinstall=args.force)
    
    if not success:
        print(f"\n✗ Setup failed: {msg}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Try running as administrator")
        print("  3. Check disk space (need 5GB+)")
        return 1
    
    print(f"\n✓ {msg}")
    print()
    
    # Show verification
    print("Verification results:")
    all_ok, messages = manager.verify_installation()
    
    for msg in messages:
        print(f"  {msg}")
    
    print()
    print("=" * 70)
    
    if all_ok:
        print("✓ SETUP COMPLETE - Ready for training!")
        print()
        print("Next steps:")
        print("  1. Configure training in ComfyUI")
        print("  2. Run training workflow")
        print("  3. Monitor in Flux2 Training Monitor")
    else:
        print("✗ SETUP INCOMPLETE")
        print()
        print("Some packages failed to install.")
        print("Training may still work with partial installation.")
    
    print("=" * 70)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

