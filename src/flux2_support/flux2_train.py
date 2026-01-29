"""
Flux.2 Training Script
Main entry point for Flux.2 LoRA training with proper architecture support.

This script extends the standard flux_train_network.py to support Flux.2 Dev
with 6144 hidden size and 128 input channels.

Usage:
    python flux2_train.py \\
        --pretrained_model_name_or_path path/to/FLUX.2-dev.safetensors \\
        --dataset_config dataset.toml \\
        --output_dir ./output \\
        --network_dim 32
"""

import sys
import os
import argparse
import logging
import torch

# Setup paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

# Import Flux.2 support
from src.flux2_support import flux2_utils, flux2_models

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for Flux.2 training."""
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_flux2_model(model_path: str) -> bool:
    """
    Validate that model checkpoint is Flux.2 compatible.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        True if valid Flux.2 model
    """
    logger.info(f"[FLUX2] Validating model: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"[FLUX2] Model not found: {model_path}")
        return False
    
    # Use validator from flux2_utils
    return flux2_utils.validate_flux2_compatibility(model_path)


def load_flux2_models(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.bfloat16,
    enable_fp8: bool = False,
):
    """
    Load Flux.2 transformer and VAE models.
    
    Args:
        model_path: Path to Flux.2 checkpoint
        device: Device to load on
        dtype: Data type for model (bfloat16 recommended)
        enable_fp8: Enable FP8 quantization for VRAM saving
        
    Returns:
        Tuple of (transformer_model, vae_model)
    """
    logger.info("[FLUX2] Loading Flux.2 models...")
    
    # Load Flux.2 transformer
    loading_dtype = None if enable_fp8 else dtype
    is_schnell, transformer = flux2_utils.load_flow_model(
        model_path,
        loading_dtype,
        device,
        disable_mmap=False,
        model_type="flux2_dev"
    )
    
    if is_schnell:
        logger.warning("[FLUX2] Model detected as Schnell variant, expected Dev")
    
    # Apply FP8 quantization if enabled
    if enable_fp8:
        logger.info("[FLUX2] Applying FP8 quantization...")
        transformer = transformer.to(torch.float8_e4m3fn)
    
    logger.info(f"[FLUX2] Transformer loaded on {device}")
    
    # Note: VAE loading would go here if needed
    # For now, assuming VAE is handled separately or pre-loaded
    
    return transformer, None


def main():
    """
    Main training entry point for Flux.2 LoRA training.
    """
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("FLUX.2 LoRA Training Script")
    logger.info("=" * 80)
    
    parser = argparse.ArgumentParser(
        description="Train LoRA for Flux.2 Dev model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Core arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to Flux.2 Dev checkpoint (safetensors format)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to dataset configuration (TOML format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trained LoRA weights",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="flux2_lora",
        help="Output name for LoRA weights",
    )
    
    # Network (LoRA) arguments
    parser.add_argument(
        "--network_dim",
        type=int,
        default=32,
        help="LoRA dimension (rank). Recommended: 32 for Flux.2",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=None,
        help="LoRA alpha. If None, defaults to network_dim",
    )
    parser.add_argument(
        "--network_module",
        type=str,
        default="networks.lora",
        help="Network module to use",
    )
    
    # Training arguments
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    
    # Optimization arguments
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save VRAM",
    )
    parser.add_argument(
        "--cache_text_encoder_outputs",
        action="store_true",
        default=True,
        help="Cache text encoder outputs (required for Flux.2 without Mistral support)",
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        default=True,
        help="Cache text encoder outputs to disk",
    )
    parser.add_argument(
        "--fp8_base",
        action="store_true",
        help="Quantize base model to FP8 for VRAM saving",
    )
    
    # Data arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    parser.add_argument(
        "--num_cpu_threads_per_process",
        type=int,
        default=2,
        help="Number of CPU threads per process",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="Prompt for validation during training",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help="Steps between validations",
    )
    
    args = parser.parse_args()
    
    # Set defaults
    if args.network_alpha is None:
        args.network_alpha = args.network_dim
    
    # Validate model
    logger.info("[FLUX2] Starting training pipeline...")
    logger.info(f"[FLUX2] Model: {args.pretrained_model_name_or_path}")
    logger.info(f"[FLUX2] Dataset: {args.dataset_config}")
    logger.info(f"[FLUX2] Output: {args.output_dir}/{args.output_name}")
    logger.info(f"[FLUX2] LoRA Rank: {args.network_dim}")
    logger.info(f"[FLUX2] Max Steps: {args.max_train_steps}")
    logger.info(f"[FLUX2] Learning Rate: {args.learning_rate}")
    
    if not validate_flux2_model(args.pretrained_model_name_or_path):
        logger.error("[FLUX2] Model validation failed!")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log Flux.2 architecture info
    logger.info("[FLUX2] Flux.2 Architecture Information:")
    for key, value in flux2_models.FLUX2_ARCHITECTURE_SUMMARY.items():
        logger.info(f"[FLUX2]   {key}: {value}")
    
    # IMPORTANT: Validate cache settings
    if not args.cache_text_encoder_outputs:
        logger.error(
            "[FLUX2] ERROR: cache_text_encoder_outputs MUST be True for Flux.2!"
        )
        logger.error(
            "[FLUX2] Flux.2 uses Mistral encoder which is not supported yet."
        )
        logger.error(
            "[FLUX2] Please ensure your dataset has pre-cached text embeddings."
        )
        return 1
    
    logger.info("[FLUX2] Validation passed! Ready for training.")
    logger.info("[FLUX2] Note: Text encoder outputs MUST be cached in dataset!")
    
    # TODO: Integrate with actual sd-scripts training loop
    # For now, this is a skeleton showing how to load and validate Flux.2 models
    
    logger.info("[FLUX2] Training initialization complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
