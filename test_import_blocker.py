"""
Test script to verify import blocker works correctly
Run this BEFORE running training to verify setup

Usage:
    cd ComfyUI-Flux2-LoRA-Manager
    python test_import_blocker.py
"""

import sys
import os

# Add plugin to path
plugin_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(plugin_path, "src"))

print("=" * 60)
print("TESTING IMPORT BLOCKER SYSTEM")
print("=" * 60)

# Test 1: Install blockers
print("\n[TEST 1] Installing import blockers...")
try:
    from import_blocker import install_import_blockers, verify_blockers_active
    install_import_blockers()
    print("✓ Import blocker module loaded")
except ImportError as e:
    print(f"✗ Failed to import blocker: {e}")
    sys.exit(1)

# Test 2: Verify they work
print("\n[TEST 2] Verifying blockers are active...")
if verify_blockers_active():
    print("✓ Blockers verified")
else:
    print("✗ Blockers NOT working!")
    sys.exit(1)

# Test 3: Try to import blocked modules
print("\n[TEST 3] Attempting to import blocked modules...")
try:
    import triton
    if hasattr(triton, '__file__') and triton.__file__ == '<blocked>':
        print("✓ triton import blocked successfully")
    else:
        print("✗ triton import NOT blocked (real module loaded)")
except ImportError:
    print("✓ triton not installed (blocker not needed)")

try:
    import bitsandbytes
    if hasattr(bitsandbytes, '__file__') and bitsandbytes.__file__ == '<blocked>':
        print("✓ bitsandbytes import blocked successfully")
    else:
        print("✗ bitsandbytes import NOT blocked (real module loaded)")
except ImportError:
    print("✓ bitsandbytes not installed (blocker not needed)")

# Test 4: Environment check
print("\n[TEST 4] Running environment check...")
try:
    from environment_checker import print_environment_report
    all_ok = print_environment_report()
except ImportError as e:
    print(f"✗ Environment checker not available: {e}")
    all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("✓ ALL TESTS PASSED - System ready for training")
else:
    print("⚠ SOME TESTS FAILED - Check output above")
print("=" * 60)
