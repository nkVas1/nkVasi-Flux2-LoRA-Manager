"""
Advanced Import Blocker for Embedded Python Environments

Blocks problematic packages (triton, bitsandbytes) that require
Python.h and full development environment.

This module uses Python's meta_path hooks to intercept imports
BEFORE they try to compile C extensions.
"""

import sys
import types
import warnings
from typing import Optional, Any


class ProblematicModuleBlocker:
    """
    Meta path finder/loader that blocks imports of modules
    that cause compilation errors in embedded Python.
    """
    
    # Modules that require full Python dev environment
    BLOCKED_MODULES = {
        'triton': 'Triton requires Python.h (use standard PyTorch operations)',
        'bitsandbytes': 'bitsandbytes requires compilation (quantization disabled)',
        'bitsandbytes.nn': 'bitsandbytes.nn blocked (embedded Python limitation)',
        'bitsandbytes.triton': 'bitsandbytes.triton blocked (compilation not supported)',
    }
    
    # Modules to return dummy implementations for
    DUMMY_MODULES = {
        'triton',
        'bitsandbytes',
    }
    
    def find_spec(self, fullname, path=None, target=None):
        """
        Called by Python import system to check if this finder handles the module.
        """
        # Check if this is a blocked module or submodule
        for blocked in self.BLOCKED_MODULES.keys():
            if fullname == blocked or fullname.startswith(blocked + '.'):
                return self._create_dummy_spec(fullname)
        return None
    
    def find_module(self, fullname, path=None):
        """
        Legacy find_module for Python < 3.4 compatibility.
        """
        for blocked in self.BLOCKED_MODULES.keys():
            if fullname == blocked or fullname.startswith(blocked + '.'):
                return self
        return None
    
    def load_module(self, fullname):
        """
        Load a dummy module instead of the real one.
        """
        if fullname in sys.modules:
            return sys.modules[fullname]
        
        # Create dummy module
        module = types.ModuleType(fullname)
        module.__package__ = fullname.rpartition('.')[0]
        module.__loader__ = self
        module.__file__ = '<blocked>'
        module.__path__ = []
        
        # Add warning attribute
        reason = self.BLOCKED_MODULES.get(fullname, 'Module blocked in embedded Python')
        module.__doc__ = f"BLOCKED: {reason}"
        
        # Add dummy attributes that might be accessed
        module.__all__ = []
        
        # Register in sys.modules to prevent re-import
        sys.modules[fullname] = module
        
        # Warn user (only once per module)
        if not hasattr(module, '_warned'):
            warnings.warn(
                f"Module '{fullname}' was blocked: {reason}",
                ImportWarning,
                stacklevel=2
            )
            module._warned = True
        
        return module
    
    def _create_dummy_spec(self, fullname):
        """Create a ModuleSpec for the dummy module."""
        try:
            from importlib.machinery import ModuleSpec
            return ModuleSpec(fullname, self, is_package=True)
        except ImportError:
            # Python < 3.4 doesn't have ModuleSpec
            return None


class DiffusersQuantizerPatcher:
    """
    Patches diffusers to skip quantizer imports that cause
    bitsandbytes/triton compilation errors.
    """
    
    @staticmethod
    def patch():
        """
        Monkey-patch diffusers to prevent quantizer imports.
        Must be called BEFORE diffusers is imported.
        """
        import sys
        
        # Check if diffusers is already imported
        if 'diffusers' in sys.modules:
            print("[IMPORT-PATCH] Warning: diffusers already imported, patching may be incomplete")
        
        # Create a custom import hook for diffusers.quantizers
        class DiffusersQuantizerBlocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith('diffusers.quantizers'):
                    return self._create_safe_spec(fullname)
                return None
            
            def find_module(self, fullname, path=None):
                if fullname.startswith('diffusers.quantizers'):
                    return self
                return None
            
            def load_module(self, fullname):
                if fullname in sys.modules:
                    return sys.modules[fullname]
                
                # Create minimal module that doesn't import bitsandbytes
                module = types.ModuleType(fullname)
                module.__package__ = fullname.rpartition('.')[0]
                module.__loader__ = self
                module.__file__ = '<safe-patched>'
                module.__path__ = []
                module.__all__ = []
                
                # Add dummy classes if this is the main quantizers module
                if fullname == 'diffusers.quantizers':
                    # Create dummy quantizer classes that do nothing
                    class DummyQuantizer:
                        def __init__(self, *args, **kwargs):
                            pass
                    
                    module.DiffusersQuantizer = DummyQuantizer
                    module.DiffusersAutoQuantizer = DummyQuantizer
                
                sys.modules[fullname] = module
                return module
            
            def _create_safe_spec(self, fullname):
                try:
                    from importlib.machinery import ModuleSpec
                    return ModuleSpec(fullname, self, is_package=True)
                except ImportError:
                    return None
        
        # Install the quantizer blocker
        sys.meta_path.insert(0, DiffusersQuantizerBlocker())
        print("[IMPORT-PATCH] Diffusers quantizer imports patched")


def install_import_blockers():
    """
    Install all import blockers at the very beginning of script execution.
    This MUST be called before any ML library imports.
    """
    # Install problematic module blocker
    blocker = ProblematicModuleBlocker()
    
    # Remove old instances if they exist (for re-runs)
    sys.meta_path = [hook for hook in sys.meta_path 
                     if not isinstance(hook, ProblematicModuleBlocker)]
    
    # Insert at the very beginning (highest priority)
    sys.meta_path.insert(0, blocker)
    
    print("[IMPORT-BLOCKER] Installed: triton, bitsandbytes imports will be blocked")
    
    # Patch diffusers
    DiffusersQuantizerPatcher.patch()
    
    # Set environment variables as additional safety layer
    import os
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    os.environ["DISABLE_TRITON"] = "1"
    os.environ["DISABLE_BITSANDBYTES_WARN"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    print("[IMPORT-BLOCKER] Environment variables set for embedded Python safety")


def verify_blockers_active():
    """
    Test that import blockers are working correctly.
    Returns True if blockers are active, False otherwise.
    """
    try:
        # Try to import a blocked module
        import triton
        # If we got here, blocker didn't work (triton is real module)
        if hasattr(triton, '__file__') and triton.__file__ != '<blocked>':
            print("[IMPORT-BLOCKER] WARNING: Blockers not active! Real triton imported")
            return False
    except ImportError:
        # This is also OK - module simply doesn't exist
        pass
    
    print("[IMPORT-BLOCKER] Verification passed")
    return True


# Auto-install if this module is imported directly
if __name__ != "__main__":
    # This executes when module is imported (not when run as script)
    # Useful for "import import_blocker" pattern
    install_import_blockers()
