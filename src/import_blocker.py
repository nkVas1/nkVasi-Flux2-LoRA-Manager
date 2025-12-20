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
    Fake module that passes importlib checks.
    Has proper __spec__ to avoid ValueError in find_spec().
    
    This is crucial because transformers/utils/import_utils.py does:
        spec = importlib.util.find_spec(pkg_name)
        if spec is None: ...
    
    If our fake module has __spec__ = None, this raises ValueError.
    """
    
    def __init__(self, name):
        super().__init__(name)
        
        # Create fake spec to satisfy importlib
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
        """Return None for any attribute (safe for checks)."""
        return None
    
    def __call__(self, *args, **kwargs):
        """Make callable (for decorators like @triton.jit)."""
        return lambda x: x


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
    Verify that importlib.util.find_spec() returns fake modules.
    This is the critical test - if find_spec() returns None or raises,
    transformers/diffusers will fail when checking for bitsandbytes.
    """
    try:
        import importlib.util
        
        # Test triton blocking
        spec = importlib.util.find_spec('triton')
        if spec is not None and spec.origin == "blocked":
            print("[IMPORT-BLOCKER] ✓ Triton properly blocked (importlib-verified)")
            return True
        
        print("[IMPORT-BLOCKER] ⚠ Verification inconclusive (spec not properly set)")
        return False
        
    except ValueError as e:
        print(f"[IMPORT-BLOCKER] ⚠ ValueError during verification: {e}")
        return False
    except Exception as e:
        print(f"[IMPORT-BLOCKER] ⚠ Verification error: {e}")
        return False


def patch_diffusers_quantizers():
    """No longer needed with proper fake modules."""
    pass
