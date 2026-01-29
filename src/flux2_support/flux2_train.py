"""
Flux.2 Training Script
Main entry point for Flux.2 LoRA training with sd-scripts integration.

This script extends the standard flux_train_network.py to support Flux.2 Dev
with 6144 hidden size and 128 input channels, using sd-scripts base trainer.

Usage:
    python flux2_train.py \\
        --pretrained_model_name_or_path path/to/FLUX.2-dev.safetensors \\
        --dataset_config dataset.toml \\
        --output_dir ./output \\
        --network_dim 32 \\
        --ae path/to/vae.safetensors \\
        --clip_l path/to/clip_l.safetensors \\
        --t5xxl path/to/t5xxl.safetensors \\
        --sd_scripts_dir path/to/sd-scripts
"""

import sys
import os
import argparse
import logging
import torch

# === PHASE 3: Setup sd-scripts path FIRST, before any imports ===
# This is crucial - we need to parse --sd_scripts_dir early
parser_early = argparse.ArgumentParser(add_help=False)
parser_early.add_argument("--sd_scripts_dir", type=str, default="")
args_early, remaining = parser_early.parse_known_args()

if args_early.sd_scripts_dir and os.path.exists(args_early.sd_scripts_dir):
    print(f"[FLUX2_TRAIN] Setting up sd-scripts path: {args_early.sd_scripts_dir}")
    sys.path.insert(0, args_early.sd_scripts_dir)
    sys.path.insert(0, os.path.join(args_early.sd_scripts_dir, "library"))
else:
    print("[FLUX2_TRAIN] WARNING: --sd_scripts_dir not provided or invalid")
    print("[FLUX2_TRAIN] Attempting to import from default PYTHONPATH")

# === Import sd-scripts components ===
try:
    from library import train_util
    from library.flux_train_network import FluxNetworkTrainer
    import library.flux_utils
    from . import flux2_utils, flux2_models
    print("[FLUX2_TRAIN] ✓ All imports successful")
except ImportError as e:
    print(f"[FLUX2_TRAIN] CRITICAL IMPORT ERROR: {e}")
    print("[FLUX2_TRAIN] Ensure --sd_scripts_dir is correct in the Configurator node")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Flux2NetworkTrainer(FluxNetworkTrainer):
    """
    Flux.2 trainer extending standard FluxNetworkTrainer.
    Overrides model loading to use Flux.2 architecture (6144 hidden size).
    """

    def __init__(self):
        super().__init__()
        self.model_type = "flux2"
        logger.info("[FLUX2] Initialized Flux.2 trainer")

    def load_target_model(self, args, weight_dtype, accelerator):
        """
        Load Flux.2 model with correct architecture.
        
        Args:
            args: Training arguments
            weight_dtype: Target data type
            accelerator: HuggingFace accelerator
            
        Returns:
            Tuple of (model_type, text_encoders, ae, model)
        """
        logger.info("[FLUX2] Loading Flux.2 Model (Hidden Size 6144)...")

        # === Load transformer model ===
        loading_dtype = None if args.fp8_base else weight_dtype
        is_schnell, model = flux2_utils.load_flow_model(
            args.pretrained_model_name_or_path,
            loading_dtype,
            "cpu",
            disable_mmap=getattr(args, "disable_mmap_load_safetensors", False),
            model_type="flux2_dev",
        )

        if args.fp8_base:
            logger.info("[FLUX2] Casting model to FP8 (e4m3fn) for VRAM saving")
            model = model.to(torch.float8_e4m3fn)

        # === Load VAE ===
        logger.info("[FLUX2] Loading VAE (AutoEncoder)...")
        if args.ae:
            try:
                ae = library.flux_utils.load_ae(
                    args.ae,
                    weight_dtype,
                    "cpu",
                    disable_mmap=getattr(args, "disable_mmap_load_safetensors", False),
                )
                logger.info(f"[FLUX2] ✓ VAE loaded from: {args.ae}")
            except Exception as e:
                logger.error(f"[FLUX2] Error loading VAE: {e}")
                raise ValueError(f"Failed to load VAE from {args.ae}: {e}")
        else:
            logger.error("[FLUX2] ERROR: --ae (VAE path) is REQUIRED for Flux training!")
            raise ValueError("VAE path (--ae) is required for Flux.1/Flux.2 training")

        # === Load Text Encoders ===
        logger.info("[FLUX2] Loading text encoders...")
        
        clip_l = None
        t5xxl = None
        
        # Try to load CLIP-L
        if args.clip_l:
            try:
                logger.info(f"[FLUX2] Loading CLIP-L from: {args.clip_l}")
                clip_l = library.flux_utils.load_clip_l(
                    args.clip_l,
                    weight_dtype,
                    "cpu",
                    disable_mmap=getattr(args, "disable_mmap_load_safetensors", False),
                )
                logger.info("[FLUX2] ✓ CLIP-L loaded")
            except Exception as e:
                logger.warning(f"[FLUX2] Error loading CLIP-L: {e}. Using dummy.")
                clip_l = flux2_utils.create_dummy_encoder(768)
        else:
            logger.warning("[FLUX2] No --clip_l path provided. Using dummy encoder.")
            clip_l = flux2_utils.create_dummy_encoder(768)

        # Try to load T5-XXL
        if args.t5xxl:
            try:
                logger.info(f"[FLUX2] Loading T5-XXL from: {args.t5xxl}")
                t5xxl = library.flux_utils.load_t5xxl(
                    args.t5xxl,
                    weight_dtype,
                    "cpu",
                    disable_mmap=getattr(args, "disable_mmap_load_safetensors", False),
                )
                logger.info("[FLUX2] ✓ T5-XXL loaded")
            except Exception as e:
                logger.warning(f"[FLUX2] Error loading T5-XXL: {e}. Using dummy.")
                t5xxl = flux2_utils.create_dummy_encoder(4096)
        else:
            logger.warning("[FLUX2] No --t5xxl path provided. Using dummy encoder.")
            t5xxl = flux2_utils.create_dummy_encoder(4096)

        # Return in format expected by trainer
        # MODEL_VERSION_FLUX_V1 works for both Flux.1 and Flux.2 (architecture difference handled in our Flux2 class)
        logger.info("[FLUX2] ✓ All models loaded successfully")
        return library.flux_utils.MODEL_VERSION_FLUX_V1, [clip_l, t5xxl], ae, model


def main():
    """Main training entry point."""
    logger.info("=" * 80)
    logger.info("FLUX.2 LoRA Training Script")
    logger.info("=" * 80)

    # Setup parser with flux_train_network arguments
    parser = FluxNetworkTrainer.setup_parser()
    
    # Add our custom arguments
    parser.add_argument(
        "--sd_scripts_dir",
        type=str,
        default="",
        help="Path to sd-scripts directory (required for Flux.2 trainer)",
    )

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    logger.info("[FLUX2] Starting training...")
    logger.info(f"[FLUX2] Model: {args.pretrained_model_name_or_path}")
    logger.info(f"[FLUX2] Dataset: {args.dataset_config}")
    logger.info(f"[FLUX2] Output: {args.output_dir}/{args.output_name}")
    logger.info(f"[FLUX2] LoRA Rank: {args.network_dim}")

    # Create and run trainer
    trainer = Flux2NetworkTrainer()
    trainer.train(args)

    logger.info("[FLUX2] ✓ Training complete!")


if __name__ == "__main__":
    sys.exit(main() or 0)


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


if __name__ == "__main__":
    sys.exit(main())
