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

# === КРИТИЧЕСКИ ВАЖНО: Добавить training_libs ДО всех импортов ===
# Получаем путь к текущему файлу и идем вверх к корню проекта
current_file = os.path.abspath(__file__)
# flux2_train.py -> flux2_support -> src -> ComfyUI-Flux2-LoRA-Manager
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
training_libs_path = os.path.join(project_root, "training_libs")

if os.path.exists(training_libs_path):
    # Добавляем в НАЧАЛО sys.path, чтобы перебить все конфликты
    sys.path.insert(0, training_libs_path)
    print(f"[FLUX2_TRAIN] ✓ training_libs added: {training_libs_path}")
else:
    print(f"[FLUX2_TRAIN] ⚠ training_libs not found at: {training_libs_path}")
    print("[FLUX2_TRAIN] Attempting fallback to system packages...")

# Проверка imagesize ПОСЛЕ добавления training_libs
try:
    import imagesize
    print(f"[FLUX2_TRAIN] ✓ imagesize found: {imagesize.__file__}")
except ImportError:
    print("[FLUX2_TRAIN] ⚠ imagesize module NOT found!")
    print("[FLUX2_TRAIN] This will cause dataset loading to fail")

# Парсим sd_scripts_dir
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--sd_scripts_dir", type=str, default="")
args, _ = parser.parse_known_args()

if args.sd_scripts_dir and os.path.exists(args.sd_scripts_dir):
    print(f"[FLUX2_TRAIN] Setting up sd-scripts path: {args.sd_scripts_dir}")
    # Добавляем sd-scripts в конец (после training_libs)
    if args.sd_scripts_dir not in sys.path:
        sys.path.append(args.sd_scripts_dir)
    library_path = os.path.join(args.sd_scripts_dir, "library")
    if os.path.exists(library_path) and library_path not in sys.path:
        sys.path.append(library_path)
else:
    print("[FLUX2_TRAIN] WARNING: --sd_scripts_dir not provided")

# Теперь можно импортировать библиотеки
import logging
import torch
try:
    from library import train_util, flux_train_utils
    import library.flux_train_network as base_trainer
    print("[FLUX2_TRAIN] ✓ All imports successful")
except ImportError as e:
    print(f"[FLUX2_TRAIN] CRITICAL IMPORT ERROR: {e}")
    print("[FLUX2_TRAIN] Ensure --sd_scripts_dir is correct in the Configurator node")
    sys.exit(1)

# Импортируем наши модули Flux.2
try:
    # Относительный импорт может не работать, используем абсолютный путь
    flux2_support_dir = os.path.dirname(current_file)
    if flux2_support_dir not in sys.path:
        sys.path.insert(0, flux2_support_dir)
    
    import flux2_utils
    import flux2_models
    print("[FLUX2_TRAIN] ✓ Flux.2 modules loaded")
except ImportError as e:
    print(f"[FLUX2_TRAIN] ERROR loading Flux.2 modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import FluxNetworkTrainer base class after all paths are set up
try:
    from library.flux_train_network import FluxNetworkTrainer
except ImportError as e:
    print(f"[FLUX2_TRAIN] CRITICAL: Cannot import FluxNetworkTrainer: {e}")
    sys.exit(1)


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
        vae_path = getattr(args, "ae", None)  # Safe attribute access
        if vae_path:
            try:
                ae = library.flux_utils.load_ae(
                    vae_path,
                    weight_dtype,
                    "cpu",
                    disable_mmap=getattr(args, "disable_mmap_load_safetensors", False),
                )
                logger.info(f"[FLUX2] ✓ VAE loaded from: {vae_path}")
            except Exception as e:
                logger.error(f"[FLUX2] Error loading VAE: {e}")
                raise ValueError(f"Failed to load VAE from {vae_path}: {e}")
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
