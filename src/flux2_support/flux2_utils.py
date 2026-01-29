"""
Flux.2 Utilities - Model Loading and Configuration
Handles loading Flux.2 models with proper architecture support.

Key Features:
- Loads Flux.2 checkpoints with 6144 hidden size
- Bypasses CLIP/T5 loading (Flux.2 uses Mistral, which requires cached embeddings)
- Supports FP8 quantization for VRAM optimization
- Provides dummy encoders to satisfy pipeline structure
"""

import torch
import logging
import os
from typing import Tuple, Optional, Union, Any

# Try to import from sd-scripts library
try:
    from library import flux_utils as base_utils
    from library.utils import setup_logging
    LIBRARY_AVAILABLE = True
except ImportError:
    LIBRARY_AVAILABLE = False
    logging.basicConfig(level=logging.INFO)

from . import flux2_models

if LIBRARY_AVAILABLE:
    setup_logging()
else:
    pass

logger = logging.getLogger(__name__)


def load_flow_model(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
    model_type: str = "flux2_dev",
) -> Tuple[bool, Any]:
    """
    Load Flux.2 flow (transformer) model from checkpoint.
    
    Args:
        ckpt_path: Path to model checkpoint (safetensors)
        dtype: Target data type (e.g., torch.bfloat16)
        device: Device to load on
        disable_mmap: Whether to disable memory-mapped loading
        model_type: Model variant ("flux2_dev" or "dev")
        
    Returns:
        Tuple of (is_schnell, model)
        is_schnell: False for Dev model
        model: Loaded Flux.2 model instance
    """
    logger.info(f"[FLUX2] Building Flux.2 model ({model_type})")
    
    # Get Flux.2 configuration
    config = flux2_models.get_flux2_config(model_type)
    params = config.params if hasattr(config, 'params') else config.get('params')
    
    if params is None:
        raise ValueError(f"Could not extract params from config for {model_type}")
    
    # Create model on meta device to avoid memory overhead during initialization
    logger.info("[FLUX2] Creating model on meta device")
    with torch.device("meta"):
        model = flux2_models.Flux2(params)
        
        # Set dtype if specified
        if dtype is not None:
            model = model.to(dtype)
    
    # Load state dict from checkpoint
    logger.info(f"[FLUX2] Loading state dict from {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    
    try:
        if LIBRARY_AVAILABLE:
            # Use robust loading from sd-scripts
            sd = base_utils.load_safetensors(
                ckpt_path, 
                device=device, 
                disable_mmap=disable_mmap, 
                dtype=dtype
            )
        else:
            # Fallback: use safetensors directly
            from safetensors.torch import load_file
            sd = load_file(ckpt_path, device=str(device))
            if dtype is not None:
                sd = {k: v.to(dtype) for k, v in sd.items()}
    except Exception as e:
        logger.error(f"[FLUX2] Failed to load checkpoint: {e}")
        raise
    
    # Standardize state dict keys
    # Remove 'model.diffusion_model.' prefix if present (from diffusers format)
    logger.info("[FLUX2] Standardizing state dict keys")
    for key in list(sd.keys()):
        new_key = key.replace("model.diffusion_model.", "")
        if new_key != key:
            sd[new_key] = sd.pop(key)
    
    # Load state dict into model
    logger.info("[FLUX2] Loading state dict into model")
    try:
        info = model.load_state_dict(sd, strict=False, assign=True)
        logger.info(f"[FLUX2] Loaded successfully: {info}")
    except Exception as e:
        logger.error(f"[FLUX2] Failed to load state dict: {e}")
        raise
    
    # Move to target device
    logger.info(f"[FLUX2] Moving model to {device}")
    model = model.to(device)
    
    # is_schnell = False (this is Dev model)
    return False, model


def create_dummy_encoder(dims: int = 4096) -> torch.nn.Module:
    """
    Create a dummy encoder module that satisfies the pipeline interface.
    
    Flux.2 uses Mistral as text encoder, which requires cached embeddings.
    This dummy encoder is used to satisfy the pipeline structure when
    cache_text_encoder_outputs is enabled.
    
    Args:
        dims: Embedding dimension (default 4096 for Mistral)
        
    Returns:
        Dummy module that returns zeros with correct shape
    """
    class DummyEncoder(torch.nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.hidden_size = hidden_size
            self.config = type('obj', (object,), {'hidden_size': hidden_size})()
            
        def forward(self, input_ids, attention_mask=None, **kwargs):
            """Return dummy embeddings with correct shape."""
            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]
            # Return zeros with shape (batch, seq_len, hidden_size)
            return torch.zeros(batch_size, seq_length, self.hidden_size)
    
    return DummyEncoder(dims)


def load_text_encoders(
    args: Any,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
) -> Tuple[Any, Any]:
    """
    Load text encoders for Flux.2 (or provide dummies for cached mode).
    
    Flux.2 uses Mistral Small 3.1 as text encoder. Since sd-scripts doesn't
    support Mistral yet, this function:
    1. Logs a warning about Mistral dependency
    2. Returns dummy encoders for use with cache_text_encoder_outputs mode
    
    Args:
        args: Training arguments
        dtype: Target data type
        device: Device to load on
        
    Returns:
        Tuple of (clip_l_encoder, t5_encoder) - both dummies for now
    """
    logger.warning(
        "[FLUX2] Flux.2 uses Mistral Small 3.1 as text encoder. "
        "sd-scripts does not support Mistral yet. "
        "Returning dummy encoders for use with cache_text_encoder_outputs=True. "
        "IMPORTANT: Ensure your dataset has pre-cached embeddings!"
    )
    
    # Create dummy encoders
    # Both return 4096-dim embeddings (Mistral hidden size)
    clip_l = create_dummy_encoder(4096)
    t5 = create_dummy_encoder(4096)
    
    return clip_l, t5


def validate_flux2_compatibility(model_path: str) -> bool:
    """
    Validate that a model checkpoint is compatible with Flux.2 architecture.
    
    Checks for expected keys and shapes in state dict.
    
    Args:
        model_path: Path to checkpoint
        
    Returns:
        True if model appears to be Flux.2 compatible
    """
    logger.info(f"[FLUX2] Validating model compatibility: {model_path}")
    
    try:
        if LIBRARY_AVAILABLE:
            from library import flux_utils as base_utils
            sd = base_utils.load_safetensors(model_path, device="cpu")
        else:
            from safetensors.torch import load_file
            sd = load_file(model_path)
        
        # Standardize keys
        for key in list(sd.keys()):
            new_key = key.replace("model.diffusion_model.", "")
            if new_key != key:
                sd[new_key] = sd.pop(key)
        
        # Check for expected Flux.2 architecture markers
        expected_keys = [
            "img_in.weight",     # Input projection
            "time_in.in_layers", # Time embedding
            "context_in",        # Context projection
            "double_blocks",     # Double stream blocks
            "single_blocks",     # Single stream blocks
        ]
        
        found_keys = set(sd.keys())
        markers_found = sum(1 for key in expected_keys if any(k.startswith(key) for k in found_keys))
        
        if markers_found >= 4:  # At least 4 of 5 expected key groups
            logger.info("[FLUX2] Model appears to be Flux.2 compatible ✓")
            
            # Try to detect hidden size from model weights
            if "double_blocks.0.img_attn.to_q.weight" in found_keys:
                weight = sd["double_blocks.0.img_attn.to_q.weight"]
                # Shape is (hidden_size, hidden_size)
                hidden_size = weight.shape[-1]
                logger.info(f"[FLUX2] Detected hidden size: {hidden_size}")
                
                if hidden_size == 6144:
                    logger.info("[FLUX2] Confirmed Flux.2 Dev architecture (hidden_size=6144) ✓")
                    return True
                elif hidden_size == 3072:
                    logger.warning("[FLUX2] Model appears to be Flux.1 (hidden_size=3072), not Flux.2")
                    return False
            
            return True
        else:
            logger.warning(f"[FLUX2] Model structure doesn't match Flux.2 (found {markers_found}/5 markers)")
            return False
            
    except Exception as e:
        logger.error(f"[FLUX2] Validation failed: {e}")
        return False


# Reference: Expected model sizes
FLUX2_MODEL_INFO = {
    "FLUX.2-dev": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-dev",
        "size_gb": 48.4,
        "params": "24.1B",
        "dtype_vram_bf16": "~48GB",
        "dtype_vram_fp16": "~48GB",
        "dtype_vram_fp8": "~24GB",
        "lora_rank_recommended": 32,
        "training_batch_size_8gb": "Not recommended",
        "training_batch_size_24gb": 1,
    }
}
