"""
Compatibility wrappers for kohya-ss/sd-scripts.

This module provides safe patches and compatibility fixes for:
- LoRA network attributes (train_t5xxl, train_clip_l, etc.)
- Safe diagnostics for LoRA module composition
- Version detection and logging

Allows custom trainers to work with various sd-scripts versions without
directly modifying sd-scripts source code.
"""

import importlib
import logging

logger = logging.getLogger(__name__)


def patch_lora_flags():
    """
    Patch LoRANetwork class to add missing attributes that sd-scripts expects.
    
    Problem: older/different versions of sd-scripts may expect LoRANetwork
    to have attributes like train_t5xxl, train_clip_l, etc., but the actual
    LoRA module doesn't always define them. This causes AttributeError when
    sd-scripts tries to access these flags.
    
    Solution: Dynamically add these attributes to the class with safe defaults.
    """
    try:
        lora_mod = importlib.import_module("networks.lora")
    except ImportError as e:
        print(f"[COMPAT] Cannot import networks.lora: {e}")
        return False

    net_cls = getattr(lora_mod, "LoRANetwork", None)
    if net_cls is None:
        print("[COMPAT] LoRANetwork class not found in networks.lora")
        return False

    patched = []
    
    # Add missing attributes that flux_train_network.py might expect
    if not hasattr(net_cls, "train_t5xxl"):
        net_cls.train_t5xxl = False
        patched.append("train_t5xxl")

    if not hasattr(net_cls, "train_clip_l"):
        net_cls.train_clip_l = False
        patched.append("train_clip_l")

    if not hasattr(net_cls, "train_unet"):
        net_cls.train_unet = True
        patched.append("train_unet")

    if patched:
        print(f"[COMPAT] ✓ Patched LoRANetwork with: {', '.join(patched)}")
        return True
    else:
        print("[COMPAT] ✓ LoRANetwork already has all expected attributes")
        return True


def log_lora_module_stats(network):
    """
    Log LoRA module composition statistics for debugging.
    
    Args:
        network: LoRANetwork instance
    
    Helps diagnose why optimizer gets empty parameter lists or why
    certain modules aren't being trained.
    """
    if not hasattr(network, "__dict__"):
        print("[COMPAT] Cannot inspect network structure")
        return

    enc_count = getattr(network, "text_encoder_lora_count", 0)
    unet_count = getattr(network, "unet_lora_count", 0)
    
    total = enc_count + unet_count
    
    print(f"[COMPAT] LoRA module composition:")
    print(f"         - Text Encoders: {enc_count} modules")
    print(f"         - U-Net: {unet_count} modules")
    print(f"         - Total: {total} modules")
    
    if total == 0:
        print("[COMPAT] ⚠ WARNING: No trainable LoRA modules created!")
        print("[COMPAT]     This will cause: ValueError: optimizer got an empty parameter list")
        return False
    
    return True


def check_sd_scripts_structure(sd_scripts_dir: str) -> bool:
    """
    Verify that sd-scripts directory contains expected structure.
    
    Args:
        sd_scripts_dir: Path to sd-scripts root
    
    Returns:
        True if structure looks valid
    """
    import os
    
    if not os.path.isdir(sd_scripts_dir):
        print(f"[COMPAT] ERROR: sd-scripts dir not found: {sd_scripts_dir}")
        return False
    
    required_files = [
        "flux_train_network.py",
    ]
    
    required_dirs = [
        "library",
    ]
    
    missing = []
    for fname in required_files:
        if not os.path.exists(os.path.join(sd_scripts_dir, fname)):
            missing.append(f"file: {fname}")
    
    for dname in required_dirs:
        if not os.path.isdir(os.path.join(sd_scripts_dir, dname)):
            missing.append(f"dir: {dname}")
    
    if missing:
        print(f"[COMPAT] ERROR: sd-scripts missing components: {', '.join(missing)}")
        return False
    
    print(f"[COMPAT] ✓ sd-scripts structure validated")
    return True
