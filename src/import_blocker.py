"""
Production-grade import blocker with Package Support.
Fixes 'No module named triton.backends' on Windows.

Key insight: PyTorch tries to import nested submodules like triton.backends.
We must block ALL possible submodule paths, not just top-level modules.
"""

import sys
import os
from types import ModuleType
import importlib.util
from importlib.machinery import ModuleSpec

_BLOCKERS_INSTALLED = False


class ProperFakeModule(ModuleType):
    """
    Fake module that can act as a package (folder) or a module (file).
    
    - If is_package=True: has __path__ = [] (can have submodules)
    - If is_package=False: __path__ = None (terminal module)
    
    Critical: Returns self for ALL attribute access (nested chains like
    triton.language.dtype work correctly).
    """
    
    def __init__(self, name, is_package=False):
        super().__init__(name)
        
        # Create proper __spec__ for importlib
        self.__spec__ = ModuleSpec(
            name=name,
            loader=None,
            origin="blocked",
            is_package=is_package
        )
        
        self.__file__ = None
        
        # If it's a package, it needs a __path__ (list of search paths)
        # This tells Python "this module can have submodules"
        if is_package:
            self.__path__ = []
            self.__package__ = name
        else:
            self.__path__ = None
            self.__package__ = name.rpartition('.')[0] if '.' in name else ''
    
    def __getattr__(self, item):
        """
        Return self for any attribute access.
        Allows chains like triton.language.dtype to work.
        Ensures torch._dynamo.utils:2417 doesn't crash.
        """
        return self
    
    def __call__(self, *args, **kwargs):
        """Make callable (for decorators like @triton.jit)."""
        if args and callable(args[0]):
            return args[0]
        return lambda x: x
    
    def __bool__(self):
        """Falsy for 'if triton:' checks."""
        return False
    
    def __repr__(self):
        """Clear representation."""
        pkg_str = " (package)" if self.__path__ is not None else ""
        return f"<ProperFakeModule '{self.__name__}'{pkg_str} (blocked)>"


def install_import_blockers():
    """
    Install blockers including nested subpackages like triton.backends.
    
    Critical modules to block:
    - triton (top package)
    - triton.language (submodule)
    - triton.compiler (subpackage)
    - triton.compiler.compiler (nested submodule)
    - triton.backends (subpackage, was causing 'No module named' error)
    - triton.backends.compiler (nested under backends)
    - triton.runtime (submodule)
    - bitsandbytes (top package)
    - bitsandbytes.nn (submodule)
    - bitsandbytes.optim (submodule)
    - xformers (broken DLL load issue on Windows)
    - xformers.ops (forces diffusers to use standard sdpa attention)
    """
    global _BLOCKERS_INSTALLED
    
    if _BLOCKERS_INSTALLED:
        return
    
    print("[IMPORT-BLOCKER] Installing production import blockers (Package-Aware)...")
    
    # List of all modules to block.
    # Order doesn't matter much, but we must list ALL possible imports.
    targets = [
        'triton',
        'triton.language',
        'triton.compiler',
        'triton.compiler.compiler',
        'triton.runtime',
        'triton.backends',          # <--- CRITICAL FIX: Was missing
        'triton.backends.compiler', # <--- CRITICAL FIX: Was missing
        'bitsandbytes',
        'bitsandbytes.nn',
        'bitsandbytes.optim',
        'xformers',                 # <--- NEW: Block broken xformers DLL issue
        'xformers.ops',             # <--- NEW: Forces diffusers to use sdpa instead
    ]
    
    for name in targets:
        # Check if any other target starts with "name." 
        # If yes, then "name" is a package (has submodules)
        is_pkg = any(t.startswith(name + ".") for t in targets)
        
        if name not in sys.modules:
            fake = ProperFakeModule(name, is_package=is_pkg)
            sys.modules[name] = fake
            pkg_type = "package" if is_pkg else "module"
            print(f"[IMPORT-BLOCKER]   ✓ Blocked {name} ({pkg_type})")
    
    # Environment variables to discourage PyTorch from looking for triton
    os.environ["TRITON_ENABLED"] = "0"
    os.environ["DISABLE_TRITON"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCH_INDUCTOR_DISABLE"] = "1"
    
    _BLOCKERS_INSTALLED = True
    print("[IMPORT-BLOCKER] ✓ All blockers installed with package support")


def verify_blockers_active() -> bool:
    """
    Verify that blockers work correctly.
    Tests:
    1. Basic modules in sys.modules
    2. Subpackages like triton.backends exist
    3. Nested attribute access works
    """
    try:
        import sys
        
        # Test 1: Top-level modules
        if sys.modules.get('triton') is None:
            print("[IMPORT-BLOCKER] ⚠ triton not in sys.modules")
            return False
        
        # Test 2: Subpackage (critical fix)
        if sys.modules.get('triton.backends') is None:
            print("[IMPORT-BLOCKER] ⚠ triton.backends missing (critical for torch._inductor)")
            return False
        
        # Test 3: Nested submodule
        if sys.modules.get('triton.backends.compiler') is None:
            print("[IMPORT-BLOCKER] ⚠ triton.backends.compiler missing")
            return False
        
        # Test 4: Nested attribute access (torch._dynamo.utils compatibility)
        triton_mod = sys.modules['triton']
        if triton_mod.language.dtype is None:
            print("[IMPORT-BLOCKER] ⚠ triton.language.dtype is None")
            return False
        
        # All tests passed
        print("[IMPORT-BLOCKER] ✓ All blockers verified")
        print("[IMPORT-BLOCKER] ✓ Package hierarchy intact (triton.backends, etc)")
        print("[IMPORT-BLOCKER] ✓ Nested attributes working (language.dtype)")
        return True
        
    except Exception as e:
        print(f"[IMPORT-BLOCKER] ⚠ Verification error: {e}")
        import traceback
        traceback.print_exc()
        return False


def patch_diffusers_quantizers():
    """Placeholder for diffusers patching (no longer needed)."""
    pass
