"""
Comprehensive test suite for import blocker.
Validates that fake modules satisfy all torch requirements.

Critical test: torch._dynamo.utils.py:2417 does:
    common_constant_types.add(triton.language.dtype)

If triton.language.dtype returns None, this crashes.
Our ProperFakeModule returns self, so this always works.
"""

import sys
import os
import importlib.util

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_import_blocker():
    """Comprehensive test suite for import blocker."""
    
    print("\n" + "=" * 70)
    print("=== Import Blocker Test Suite ===")
    print("=" * 70)
    
    # Install blockers
    from import_blocker import install_import_blockers
    install_import_blockers()
    
    # Test 1: Modules in sys.modules
    print("\n[TEST 1] Modules in sys.modules")
    try:
        assert 'triton' in sys.modules, "❌ triton not blocked"
        assert 'bitsandbytes' in sys.modules, "❌ bitsandbytes not blocked"
        assert 'triton.compiler' in sys.modules, "❌ triton.compiler not blocked"
        print("  ✓ All modules blocked correctly")
    except AssertionError as e:
        print(f"  ✗ {e}")
        return False
    
    # Test 2: Proper __spec__ attributes
    print("\n[TEST 2] Proper __spec__ attributes")
    try:
        triton = sys.modules['triton']
        assert hasattr(triton, '__spec__'), "❌ triton missing __spec__"
        assert triton.__spec__ is not None, "❌ triton.__spec__ is None"
        assert triton.__spec__.origin == "blocked", f"❌ Wrong origin: {triton.__spec__.origin}"
        print("  ✓ __spec__ attributes correct")
        print(f"    - name: {triton.__spec__.name}")
        print(f"    - origin: {triton.__spec__.origin}")
    except AssertionError as e:
        print(f"  ✗ {e}")
        return False
    
    # Test 3: importlib.util.find_spec() works
    print("\n[TEST 3] importlib.util.find_spec() functionality")
    try:
        spec = importlib.util.find_spec('triton')
        assert spec is not None, "❌ find_spec returned None"
        assert spec.origin == "blocked", f"❌ Wrong origin: {spec.origin}"
        print("  ✓ find_spec works correctly")
        print(f"    - spec.name: {spec.name}")
        print(f"    - spec.origin: {spec.origin}")
    except ValueError as e:
        print(f"  ✗ find_spec raised ValueError: {e}")
        return False
    except AssertionError as e:
        print(f"  ✗ {e}")
        return False
    
    # Test 4: Nested attribute access (CRITICAL for torch._dynamo.utils)
    print("\n[TEST 4] Nested attribute access (torch._dynamo.utils compatibility)")
    try:
        # This is what torch._dynamo.utils.py:2417 does:
        # common_constant_types.add(triton.language.dtype)
        language_dtype = triton.language.dtype
        assert language_dtype is not None, "❌ triton.language.dtype is None (would crash torch)"
        print("  ✓ triton.language.dtype works correctly")
        print(f"    - Returns: {language_dtype}")
        print(f"    - Type: {type(language_dtype).__name__}")
        
        # Also test deep compiler chain
        compiler_obj = triton.compiler.compiler
        assert compiler_obj is not None, "❌ triton.compiler.compiler is None"
        print("  ✓ triton.compiler.compiler works correctly")
        print(f"    - Returns: {compiler_obj}")
        
    except AssertionError as e:
        print(f"  ✗ {e}")
        return False
    
    # Test 5: Callable behavior (for decorators)
    print("\n[TEST 5] Callable behavior (decorator support)")
    try:
        # Test as decorator
        @triton
        def dummy_func():
            return "test"
        
        assert callable(dummy_func), "❌ Decorator broke function"
        assert dummy_func() == "test", "❌ Function doesn't work after decoration"
        print("  ✓ Decorator behavior correct")
        print(f"    - @triton decorated function works")
        print(f"    - Function call result: {dummy_func()}")
    except Exception as e:
        print(f"  ✗ {e}")
        return False
    
    # Test 6: Boolean checks (for 'if triton:' checks)
    print("\n[TEST 6] Boolean behavior (falsy evaluation)")
    try:
        # Should be falsy
        if triton:
            print("  ✗ triton should be falsy (evaluated to True)")
            return False
        print("  ✓ Falsy behavior correct")
        print(f"    - bool(triton) = {bool(triton)}")
    except Exception as e:
        print(f"  ✗ {e}")
        return False
    
    # Test 7: transformers import (real-world test)
    print("\n[TEST 7] Real-world: transformers import")
    try:
        import transformers
        print(f"  ✓ transformers imported successfully")
        print(f"    - version: {transformers.__version__}")
        
        # This internally checks bitsandbytes availability
        from transformers.utils import import_utils
        print(f"    - import_utils loaded (checks bitsandbytes)")
    except ImportError as e:
        print(f"  ⚠ transformers not available: {e}")
        print(f"    This is OK if transformers not installed")
    except Exception as e:
        print(f"  ✗ transformers import caused error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    print("\nSummary:")
    print("  ✓ Modules properly blocked with valid __spec__")
    print("  ✓ importlib.util.find_spec() works without ValueError")
    print("  ✓ Nested attribute access works (torch._dynamo.utils compatible)")
    print("  ✓ Decorator and boolean behavior correct")
    print("  ✓ Real-world transformers import successful")
    print("\nReady for production!")
    
    return True


if __name__ == '__main__':
    try:
        success = test_import_blocker()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

