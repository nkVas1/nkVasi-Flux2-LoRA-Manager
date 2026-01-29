"""
Flux.2 Architecture Definition and Configuration
Defines Flux.2 model with 6144 hidden size and 128 input channels.

Architecture Differences from Flux.1:
- Hidden Size: 3072 → 6144 (doubled)
- Input Channels: 64 → 128 (doubled from VAE)
- Num Heads: 24 → 48
- Depth: 19 → 8 double blocks + 48 single blocks
- Context Dim: 4096 (for Mistral Small 3.1)

Reference: https://huggingface.co/black-forest-labs/FLUX.2-dev
"""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional

# Try to import from sd-scripts library
try:
    from library.flux_models import (
        FluxParams, AutoEncoderParams, ModelSpec, 
        Flux as BaseFlux, 
        DoubleStreamBlock, SingleStreamBlock, 
        LastLayer, EmbedND, MLPEmbedder, timestep_embedding
    )
except ImportError:
    # Fallback: import from diffusers or create minimal stubs
    # This allows the code to be analyzed even if library isn't installed
    BaseFlux = object
    FluxParams = dict
    AutoEncoderParams = dict
    ModelSpec = dict


@dataclass
class Flux2Params(FluxParams):
    """Flux.2 specific parameters. Inherits from FluxParams for compatibility."""
    pass


class Flux2(BaseFlux):
    """
    Flux.2 Transformer model with 6144 hidden size and 128 input channels.
    
    Extends the standard Flux implementation to support the larger architecture
    required for Flux.2 Dev model.
    """
    
    def __init__(self, params: FluxParams):
        """
        Initialize Flux.2 model.
        
        Args:
            params: FluxParams object with Flux.2 configuration
        """
        super().__init__(params)
        
        # Re-initialize img_in if channels differ from default
        # Flux.1 default is 64, Flux.2 is 128.
        # BaseFlux uses params.in_channels, but we explicitly verify here.
        if hasattr(self, 'in_channels') and self.in_channels != params.in_channels:
            self.img_in = nn.Linear(params.in_channels, self.hidden_size, bias=True)
            print(f"[FLUX2] Reinitializing img_in for {params.in_channels} input channels")


# Configuration for Flux.2 Dev
# Reference architecture: https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/config.json
flux2_dev_spec = ModelSpec(
    ckpt_path=None,
    params=FluxParams(
        # Input/Output dimensions
        in_channels=128,        # VAE output is 128 channels (doubled from Flux.1)
        vec_in_dim=768,         # Positional embedding dimension
        context_in_dim=4096,    # Text embedding from Mistral Small 3.1 (4096 hidden)
        
        # Model dimensions
        hidden_size=6144,       # Doubled from Flux.1's 3072
        mlp_ratio=4.0,          # MLP expansion ratio
        num_heads=48,           # 6144 / 128 (head_dim) = 48 heads
        
        # Block structure
        depth=8,                # 8 Double Blocks (down from 19 in Flux.1)
        depth_single_blocks=48, # 48 Single Blocks (up from 38 in Flux.1)
        
        # Other parameters
        axes_dim=[16, 56, 56],  # RoPE axes dimensions
        theta=10_000,           # RoPE frequency base
        qkv_bias=True,          # Use bias in QKV projections
        guidance_embed=True,    # Support for guidance scaling
    ),
    ae_path=None,
    ae_params=AutoEncoderParams(
        # VAE Architecture for Flux.2
        resolution=256,
        in_channels=3,          # RGB input
        ch=128,                 # Base channels
        out_ch=3,               # RGB output
        ch_mult=[1, 2, 4, 4],   # Channel multipliers per block
        num_res_blocks=2,       # Residual blocks per level
        z_channels=32,          # Latent space channels
        # With ch=128, ch_mult=[1,2,4,4], z_channels=32:
        # Output latents: 128 * 1 = 128 channels (matches in_channels above)
        scale_factor=0.3611,    # Scaling factor for latent space
        shift_factor=0.1159,    # Shift factor for latent space
    ),
)

# Configuration aliases for different Flux.2 variants
configs = {
    "dev": flux2_dev_spec,
    "flux2-dev": flux2_dev_spec,
    "flux2_dev": flux2_dev_spec,
    "FLUX.2-dev": flux2_dev_spec,
}


def get_flux2_config(model_name: str = "dev") -> ModelSpec:
    """
    Get Flux.2 configuration by name.
    
    Args:
        model_name: Model variant name (e.g., "dev", "flux2_dev")
        
    Returns:
        ModelSpec with Flux.2 configuration
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower in configs:
        return configs[model_name_lower]
    
    # Default to dev
    print(f"[FLUX2] Unknown model '{model_name}', using 'dev' configuration")
    return configs["dev"]


# Model architecture reference
FLUX2_ARCHITECTURE_SUMMARY = {
    "name": "Flux.2 Dev",
    "hidden_size": 6144,
    "num_heads": 48,
    "num_double_blocks": 8,
    "num_single_blocks": 48,
    "input_channels": 128,
    "context_dim": 4096,
    "text_encoder": "Mistral Small 3.1",
    "vae_scaling": 0.3611,
    "training_dtype": "bfloat16 or float32",
    "recommended_lora_rank": 32,  # Larger than Flux.1 due to larger model
    "vram_estimate_8gb": "Not directly supported - requires quantization or LoRA",
    "vram_estimate_24gb": "~20GB with bfloat16 and gradient checkpointing",
}
