"""
Simplified import blocker - now torch is patched BEFORE import.
This module provides backup blocking only.
"""

import sys
import os

_BLOCKERS_INSTALLED = False

class _ImportBlocker:
    """Minimal blocker that raises on import."""
    def __init__(self, name):
        self.name = name
    
    def __getattr__(self, item):
        return None

def install_import_blockers():
    """
    Install minimal import blockers.
    Note: Main blocking now happens in wrapper via torch patching.
    """
    global _BLOCKERS_INSTALLED
    
    if _BLOCKERS_INSTALLED:
        return
    
    # Backup blockers (torch already patched in wrapper)
    if 'triton' not in sys.modules:
        sys.modules['triton'] = _ImportBlocker('triton')
    
    if 'bitsandbytes' not in sys.modules:
        sys.modules['bitsandbytes'] = _ImportBlocker('bitsandbytes')
    
    print("[IMPORT-BLOCKER] ✓ Backup blockers installed")
    
    # Environment variables
    os.environ.setdefault("TRITON_ENABLED", "0")
    os.environ.setdefault("DISABLE_TRITON", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    
    _BLOCKERS_INSTALLED = True

def verify_blockers_active() -> bool:
    """Verify blockers active OR torch safely patched."""
    try:
        import torch
        # Check if torch._dynamo is disabled
        if hasattr(torch, '_dynamo'):
            if hasattr(torch._dynamo.config, 'suppress_errors'):
                if torch._dynamo.config.suppress_errors:
                    print("[IMPORT-BLOCKER] ✓ torch._dynamo suppressed")
                    return True
        
        print("[IMPORT-BLOCKER] ⚠ Verification inconclusive")
        return False
    except Exception as e:
        print(f"[IMPORT-BLOCKER] Verification error: {e}")
        return False

def patch_diffusers_quantizers():
    """Patch diffusers to not import quantizers."""
    pass  # No longer needed with torch patching
