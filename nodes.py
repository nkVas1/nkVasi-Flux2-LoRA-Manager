"""
ComfyUI Entry Point - Node Registration with Senior-Level Architecture
Imports and registers all node classes from the src module.

Architecture Notes:
- Separation of Concerns: Model, Dataset, Process are independent modules
- Type Safety: Custom RETURN_TYPES (TRAIN_FLUX_MODELS, TRAIN_DATASET) prevent workflow errors
- Scalability: Each node is self-contained, easy to extend (e.g., FluxTrainValidation)
- Senior-Level Design: Composable nodes that form a complete training pipeline
"""

from .src.config_gen import Flux2_8GB_Configurator
from .src.process import Flux2_Runner, Flux2_Stopper
import json
import os
from pathlib import Path


# ============================================================================
# Custom Type Definitions for Type-Safe Workflows
# ============================================================================

class FluxTrainModelSelect:
    """
    Senior-level node: Model selection and configuration.
    Separates model setup from training logic (Single Responsibility Principle).
    
    Output: TRAIN_FLUX_MODELS type - can be consumed by training executor.
    """
    
    def __init__(self):
        self.models_dir = Path("./models")
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define input ports with validation."""
        return {
            "required": {
                "transformer_name": (["flux1-dev", "flux1-schnell"], {
                    "default": "flux1-dev",
                    "tooltip": "Base FLUX model variant"
                }),
                "vae_name": ("STRING", {
                    "default": "ae.safetensors",
                    "tooltip": "VAE checkpoint filename"
                }),
                "clip_l_name": ("STRING", {
                    "default": "clip_l.safetensors",
                    "tooltip": "CLIP-L text encoder"
                }),
                "t5_name": ("STRING", {
                    "default": "t5xxl.safetensors",
                    "tooltip": "T5-XXL text encoder"
                }),
            },
            "optional": {
                "fp8_base": (["disable", "enabled"], {
                    "default": "disable",
                    "tooltip": "Load base model in FP8 quantization"
                }),
                "compile": (["disable", "enabled"], {
                    "default": "disable",
                    "tooltip": "Use torch.compile for optimization"
                }),
            }
        }
    
    RETURN_TYPES = ("TRAIN_FLUX_MODELS",)
    RETURN_NAMES = ("flux_models",)
    FUNCTION = "select_models"
    CATEGORY = "training/flux"
    
    def select_models(self, transformer_name, vae_name, clip_l_name, t5_name, 
                      fp8_base="disable", compile="disable"):
        """
        Build model configuration dictionary.
        
        Returns:
            tuple: (models_config_dict,) where config contains all model paths and settings
        """
        config = {
            "transformer": transformer_name,
            "vae": vae_name,
            "clip_l": clip_l_name,
            "t5": t5_name,
            "fp8_quantization": fp8_base == "enabled",
            "torch_compile": compile == "enabled",
        }
        
        print(f"[FluxTrainModelSelect] ✓ Model config: {transformer_name} + {vae_name}")
        return (config,)


class FluxTrainDatasetConfig:
    """
    Senior-level node: Dataset configuration and validation.
    Separates data setup from training logic (Single Responsibility Principle).
    
    Output: TRAIN_DATASET type - can be consumed by training executor.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define dataset input ports."""
        return {
            "required": {
                "image_dir": ("STRING", {
                    "default": "./datasets/my_dataset",
                    "tooltip": "Path to training images directory"
                }),
                "resolution": ("INT", {
                    "default": 768,
                    "min": 512,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Training image resolution (H=W)"
                }),
                "repeats": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Epochs - how many times to repeat dataset"
                }),
            },
            "optional": {
                "caption_extension": ("STRING", {
                    "default": ".txt",
                    "tooltip": "Caption file extension (.txt, .cap, etc)"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "tooltip": "Batch size (lower = less VRAM)"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "Random seed for reproducibility"
                }),
            }
        }
    
    RETURN_TYPES = ("TRAIN_DATASET",)
    RETURN_NAMES = ("dataset_config",)
    FUNCTION = "configure_dataset"
    CATEGORY = "training/flux"
    
    def configure_dataset(self, image_dir, resolution, repeats, caption_extension=".txt",
                         batch_size=1, seed=42):
        """
        Build and validate dataset configuration.
        
        Args:
            image_dir: Path to images
            resolution: Training resolution
            repeats: Number of epochs
            caption_extension: File extension for captions
            batch_size: Batch size for training
            seed: Random seed
        
        Returns:
            tuple: (dataset_config,) where config is validated and ready for training
        """
        # Validation
        image_path = Path(image_dir)
        if not image_path.exists():
            print(f"[FluxTrainDatasetConfig] ⚠ Warning: {image_dir} doesn't exist yet")
        
        config = {
            "image_directory": str(image_path.absolute()),
            "resolution": resolution,
            "repeats": repeats,
            "caption_ext": caption_extension,
            "batch_size": batch_size,
            "seed": seed,
            "total_images": len(list(image_path.glob("*.jpg"))) + len(list(image_path.glob("*.png"))) 
                           if image_path.exists() else 0,
        }
        
        print(f"[FluxTrainDatasetConfig] ✓ Dataset: {config['total_images']} images @ {resolution}x{resolution}")
        return (config,)


class FluxTrainExecutor:
    """
    (Placeholder for future) Senior-level node: Training orchestration.
    Consumes TRAIN_FLUX_MODELS and TRAIN_DATASET, executes training.
    
    Future additions:
    - FluxTrainValidation: Preview generation during training
    - FluxLoRAMerge: Merge trained LoRA with base model
    - FluxCheckpointManager: Save/load checkpoints
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_models": ("TRAIN_FLUX_MODELS",),
                "dataset_config": ("TRAIN_DATASET",),
                "learning_rate": ("FLOAT", {
                    "default": 0.0001,
                    "min": 0.00001,
                    "max": 0.001,
                    "step": 0.00001,
                }),
                "steps": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 100000,
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute_training"
    CATEGORY = "training/flux"
    
    def execute_training(self, flux_models, dataset_config, learning_rate, steps):
        """Start training with given configuration."""
        return ("Training executor - use Flux2_Run_External instead",)


# ============================================================================
# Node Registration Dictionary
# ============================================================================

NODE_CLASS_MAPPINGS = {
    # Original nodes
    "Flux2_8GB_Config": Flux2_8GB_Configurator,
    "Flux2_Run_External": Flux2_Runner,
    "Flux2_Stop": Flux2_Stopper,
    
    # New modular nodes (ETAP 2 - Senior Architecture)
    "FluxTrainModelSelect": FluxTrainModelSelect,
    "FluxTrainDatasetConfig": FluxTrainDatasetConfig,
    "FluxTrainExecutor": FluxTrainExecutor,
}

# Display names with emojis for UI
NODE_DISPLAY_NAME_MAPPINGS = {
    # Original nodes
    "Flux2_8GB_Config": "🛠️ FLUX.2 Config (Low VRAM)",
    "Flux2_Run_External": "🚀 Start Training (External)",
    "Flux2_Stop": "🛑 Emergency Stop",
    
    # New modular nodes (organized in workflow)
    "FluxTrainModelSelect": "🤖 [1] Select Models",
    "FluxTrainDatasetConfig": "📁 [2] Configure Dataset",
    "FluxTrainExecutor": "⚙️ [3] Execute Training",
}
