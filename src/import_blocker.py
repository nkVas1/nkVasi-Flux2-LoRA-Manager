"""
Production-grade import blocker with proper ModuleSpec.
Ensures blocked modules are invisible to importlib.util.find_spec().

Key insight: transformers checks module availability using importlib.util.find_spec().
If a module has __spec__ = None, find_spec() raises ValueError.
Our ProperFakeModule includes proper ModuleSpec to avoid this.
"""

import sys
import os
from types import ModuleType
import importlib.util
from importlib.machinery import ModuleSpec

_BLOCKERS_INSTALLED = False


class ProperFakeModule(ModuleType):
    """
    Production-grade fake module with nested attribute support.
    Returns self for ALL attribute access to satisfy deep chains like:
    triton.language.dtype, triton.compiler.compiler.AttrsDescriptor
    
    CRITICAL: torch._dynamo.utils.py:2417 does:
        common_constant_types.add(triton.language.dtype)
    If triton.language.dtype returns None, this crashes.
    Our solution: __getattr__ returns self, so any chain always returns a module object.
    """
    
    def __init__(self, name):
        super().__init__(name)
        
        # Create proper __spec__ for importlib
        self.__spec__ = ModuleSpec(
            name=name,
            loader=None,
            origin="blocked",
            is_package=False
        )
        
        self.__file__ = None
        self.__path__ = None
        self.__package__ = name.rpartition('.')[0] if '.' in name else ''
    
    def __getattr__(self, item):
        """
        CRITICAL: Return self (not None) for nested attribute access.
        This allows triton.language.dtype to work:
        - triton.language -> returns self
        - (self).dtype -> returns self
        Result: triton.language.dtype = self (an object, not None)
        """
        return self
    
    def __call__(self, *args, **kwargs):
        """Make callable (for decorators like @triton.jit)."""
        # If called with a function, return it unchanged (decorator pattern)
        if args and callable(args[0]):
            return args[0]
        # Otherwise return a dummy wrapper
        def wrapper(func):
            return func
        return wrapper
    
    def __bool__(self):
        """Make falsy for checks like 'if triton:'"""
        return False
    
    def __repr__(self):
        """Clear representation showing it's a fake module."""
        return f"<ProperFakeModule '{self.__name__}' (blocked)>"


def install_import_blockers():
    """
    Install production-grade import blockers.
    These modules have proper __spec__ and pass importlib checks.
    """
    global _BLOCKERS_INSTALLED
    
    if _BLOCKERS_INSTALLED:
        return
    
    print("[IMPORT-BLOCKER] Installing production import blockers...")
    
    # Block problematic modules with proper fake modules
    blocked_modules = [
        'triton',
        'triton.language',
        'triton.compiler',
        'triton.compiler.compiler',
        'triton.runtime',
        'bitsandbytes',
        'bitsandbytes.nn',
        'bitsandbytes.optim',
    ]
    
    for mod_name in blocked_modules:
        if mod_name not in sys.modules:
            fake_mod = ProperFakeModule(mod_name)
            sys.modules[mod_name] = fake_mod
            print(f"[IMPORT-BLOCKER]   ✓ Blocked {mod_name}")
    
    # Environment variables for extra safety
    os.environ["TRITON_ENABLED"] = "0"
    os.environ["DISABLE_TRITON"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCH_INDUCTOR_DISABLE"] = "1"
    
    _BLOCKERS_INSTALLED = True
    print("[IMPORT-BLOCKER] ✓ All blockers installed with proper __spec__")


def verify_blockers_active() -> bool:
    """
    Verify that blockers work correctly.
    Tests both importlib.find_spec and nested attribute access.
    
    Critical tests:
    1. importlib can find fake modules
    2. Nested attribute access works (triton.language.dtype)
    3. Deep chains work (triton.compiler.compiler)
    """
    try:
        import importlib.util
        import sys
        
        # Test 1: importlib can find fake modules
        spec = importlib.util.find_spec('triton')
        if spec is None or spec.origin != "blocked":
            print("[IMPORT-BLOCKER] ⚠ Triton spec verification failed")
            return False
        
        # Test 2: Module is in sys.modules
        triton = sys.modules.get('triton')
        if triton is None:
            print("[IMPORT-BLOCKER] ⚠ Triton not in sys.modules")
            return False
        
        # Test 3: Nested attribute access works (CRITICAL for torch._dynamo.utils)
        test_attr = triton.language.dtype
        if test_attr is None:
            print("[IMPORT-BLOCKER] ⚠ triton.language.dtype returned None")
            return False
        
        # Test 4: Deep compiler chain (for torch._inductor)
        compiler_test = triton.compiler.compiler
        if compiler_test is None:
            print("[IMPORT-BLOCKER] ⚠ triton.compiler.compiler returned None")
            return False
        
        # All tests passed
        print("[IMPORT-BLOCKER] ✓ Triton properly blocked (importlib-verified)")
        print("[IMPORT-BLOCKER] ✓ Nested attributes working (language.dtype, compiler.compiler)")
        return True
        
    except ValueError as e:
        print(f"[IMPORT-BLOCKER] ⚠ ValueError during verification: {e}")
        return False
    except Exception as e:
        print(f"[IMPORT-BLOCKER] ⚠ Verification error: {e}")
        import traceback
        traceback.print_exc()
        return False


def patch_diffusers_quantizers():
    """No longer needed with proper fake modules."""
    pass
