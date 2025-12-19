#!/usr/bin/env python
"""
Setup script for training environment.
Run this once to create isolated venv with correct dependencies.

Usage:
    python setup_training_env.py
    python setup_training_env.py --force  # Force recreate
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from venv_manager import VirtualEnvManager


def main():
    parser = argparse.ArgumentParser(description="Setup Flux2 training environment")
    parser.add_argument("--force", action="store_true", help="Force recreate environment")
    args = parser.parse_args()
    
    print("=" * 70)
    print("FLUX2 LORA MANAGER - TRAINING ENVIRONMENT SETUP")
    print("=" * 70)
    print()
    
    manager = VirtualEnvManager()
    
    print(f"Virtual environment will be created at:")
    print(f"  {manager.venv_dir}")
    print()
    
    if manager.venv_exists() and not args.force:
        print("Virtual environment already exists.")
        print("Use --force to recreate.")
        print()
        
        # Verify existing installation
        print("Verifying existing installation...")
        all_ok, messages = manager.verify_installation()
        
        for msg in messages:
            print(f"  {msg}")
        
        if all_ok:
            print("\n✓ Environment is ready for training")
            return 0
        else:
            print("\n✗ Environment has issues, use --force to recreate")
            return 1
    
    # Create environment
    print("Creating virtual environment...")
    success, msg = manager.setup_training_env(force_recreate=args.force)
    
    if not success:
        print(f"\n✗ Setup failed: {msg}")
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
        print("  1. Configure your training in ComfyUI")
        print("  2. Run training workflow")
        print("  3. Monitor progress in Flux2 Training Monitor")
    else:
        print("✗ SETUP INCOMPLETE - Some packages failed to install")
        print()
        print("Try:")
        print("  1. Check internet connection")
        print("  2. Run: python setup_training_env.py --force")
        print("  3. Check error messages above")
    
    print("=" * 70)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
