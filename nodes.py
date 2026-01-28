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
# OpenArt-Compatible Training Nodes (New Architecture - ЕТАП Б)
# ============================================================================

class FluxTrainModelSelect:
    """
    Selects model components for Flux LoRA training.
    
    Purpose: Separation of concerns - model selection is independent
    from dataset configuration and training parameters.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer_name": ("STRING", {
                    "default": "flux1-dev.safetensors",
                    "tooltip": "Flux transformer model filename"
                }),
                "vae_name": ("STRING", {
                    "default": "ae.safetensors",
                    "tooltip": "VAE model filename"
                }),
                "clip_l_name": ("STRING", {
                    "default": "clip_l.safetensors",
                    "tooltip": "CLIP-L text encoder"
                }),
                "t5_name": ("STRING", {
                    "default": "t5xxl.safetensors",
                    "tooltip": "T5-XXL text encoder (T5 XL for schnell)"
                }),
                "fp8_base": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Enable FP8",
                    "label_off": "Disable FP8",
                    "tooltip": "Load base models in FP8 to save VRAM"
                }),
            }
        }
    
    RETURN_TYPES = ("TRAIN_FLUX_MODELS",)
    RETURN_NAMES = ("flux_models",)
    FUNCTION = "select_models"
    CATEGORY = "FluxTrainer/Config"
    
    def select_models(self, transformer_name, vae_name, clip_l_name, t5_name, fp8_base):
        """Build model configuration dictionary."""
        config = {
            "transformer": transformer_name,
            "vae": vae_name,
            "clip_l": clip_l_name,
            "t5": t5_name,
            "fp8_base": fp8_base,
        }
        print(f"[FluxTrainModelSelect] ✓ {transformer_name} + VAE ({fp8_base=})")
        return (config,)


class FluxTrainDatasetConfig:
    """
    Configures a single dataset for Flux LoRA training.
    
    Purpose: Separation of concerns - dataset config independent from models.
    Can be reused across multiple training runs with different model selections.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_dir": ("STRING", {
                    "default": "path/to/training/images",
                    "tooltip": "Path to folder with training images (JPG/PNG)"
                }),
                "resolution": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Training resolution (square: HxW)"
                }),
                "repeats": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Dataset repeats (epochs)"
                }),
                "caption_extension": ("STRING", {
                    "default": ".txt",
                    "tooltip": "Extension for caption files (.txt, .cap)"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Batch size (lower = less VRAM)"
                }),
            }
        }
    
    RETURN_TYPES = ("TRAIN_DATASET",)
    RETURN_NAMES = ("dataset_config",)
    FUNCTION = "configure"
    CATEGORY = "FluxTrainer/Config"
    
    def configure(self, image_dir, resolution, repeats, caption_extension, batch_size):
        """Build dataset configuration dictionary."""
        config = {
            "image_dir": image_dir,
            "resolution": resolution,
            "repeats": repeats,
            "caption_extension": caption_extension,
            "batch_size": batch_size,
        }
        print(f"[FluxTrainDatasetConfig] ✓ Dataset: {image_dir} @ {resolution}x{resolution}")
        return (config,)


class FluxTrainValidationSettings:
    """
    Validation/preview settings for Flux LoRA training.
    
    Purpose: Configure validation prompts and generation parameters.
    Separated from training config for cleaner architecture.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "validation_steps": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Frequency of validation runs"
                }),
                "validation_prompts": ("STRING", {
                    "default": "a portrait of a person",
                    "multiline": True,
                    "tooltip": "Prompts for validation (one per line)"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2147483647,
                }),
            }
        }
    
    RETURN_TYPES = ("VALSETTINGS",)
    RETURN_NAMES = ("validation_config",)
    FUNCTION = "configure"
    CATEGORY = "FluxTrainer/Config"
    
    def configure(self, validation_steps, validation_prompts, width, height, guidance_scale, seed):
        """Build validation configuration."""
        config = {
            "steps": validation_steps,
            "prompts": validation_prompts.split('\n') if validation_prompts else [],
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        print(f"[FluxTrainValidationSettings] ✓ Every {validation_steps} steps")
        return (config,)


class InitFluxLoRATraining:
    """
    Main orchestration node: Initializes Flux LoRA training pipeline.
    
    Purpose: Combines models, dataset, and parameters into a training context.
    This is the entry point for the new Senior-level training architecture.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_models": ("TRAIN_FLUX_MODELS",),
                "dataset": ("TRAIN_DATASET",),
                "max_train_steps": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 100000,
                    "tooltip": "Total training steps"
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.0001,
                    "min": 0.00001,
                    "max": 0.01,
                    "step": 0.00001,
                    "tooltip": "Base learning rate for LoRA training"
                }),
                "output_dir": ("STRING", {
                    "default": "output/flux_lora",
                    "tooltip": "Output directory for LoRA weights"
                }),
                "lora_name": ("STRING", {
                    "default": "my_flux_lora",
                    "tooltip": "Name for the output LoRA (without extension)"
                }),
                "optimizer": (["adafactor", "adamw", "sgd"], {
                    "default": "adafactor",
                    "tooltip": "Optimizer type (adafactor for low VRAM)"
                }),
            },
            "optional": {
                "validation_config": ("VALSETTINGS", {
                    "tooltip": "Optional validation settings"
                }),
            }
        }
    
    RETURN_TYPES = ("NETWORKTRAINER",)
    RETURN_NAMES = ("trainer_context",)
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer/Core"
    
    def init_training(self, flux_models, dataset, max_train_steps, learning_rate, 
                     output_dir, lora_name, optimizer, validation_config=None):
        """
        Initialize training context with all parameters.
        
        Returns a NETWORKTRAINER type object that can be consumed by
        training execution nodes.
        """
        trainer = {
            "type": "FluxLoRA",
            "models": flux_models,
            "dataset": dataset if isinstance(dataset, list) else [dataset],
            "validation": validation_config or {"steps": 0, "prompts": []},
            "config": {
                "max_train_steps": max_train_steps,
                "learning_rate": learning_rate,
                "output_dir": output_dir,
                "output_name": lora_name,
                "optimizer": optimizer,
            },
            "status": "initialized",
            "version": "1.0",
        }
        
        print(f"[InitFluxLoRATraining] ✓ Training context initialized")
        print(f"  - Model: {flux_models.get('transformer', 'unknown')}")
        print(f"  - Steps: {max_train_steps} @ lr={learning_rate}")
        print(f"  - Output: {output_dir}/{lora_name}")
        
        return (trainer,)


# ============================================================================
# Node Registration Dictionary
# ============================================================================

NODE_CLASS_MAPPINGS = {
    # Original nodes
    "Flux2_8GB_Config": Flux2_8GB_Configurator,
    "Flux2_Run_External": Flux2_Runner,
    "Flux2_Stop": Flux2_Stopper,
    
    # New modular nodes (ЕТАП Б - Senior Architecture)
    "FluxTrainModelSelect": FluxTrainModelSelect,
    "FluxTrainDatasetConfig": FluxTrainDatasetConfig,
    "FluxTrainValidationSettings": FluxTrainValidationSettings,
    "InitFluxLoRATraining": InitFluxLoRATraining,
}

# Display names with emojis for UI
NODE_DISPLAY_NAME_MAPPINGS = {
    # Original nodes
    "Flux2_8GB_Config": "🛠️ FLUX.2 Config (Low VRAM)",
    "Flux2_Run_External": "🚀 Start Training (External)",
    "Flux2_Stop": "🛑 Emergency Stop",
    
    # New modular nodes (OpenArt-compatible architecture)
    "FluxTrainModelSelect": "🤖 Flux Model Selector",
    "FluxTrainDatasetConfig": "📁 Flux Dataset Config",
    "FluxTrainValidationSettings": "🔍 Flux Validation Settings",
    "InitFluxLoRATraining": "⚙️ Init Flux LoRA Training",
}
