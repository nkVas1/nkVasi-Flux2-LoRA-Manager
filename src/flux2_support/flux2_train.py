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
import importlib.util

# === FIX 2: Fallback-импорт модулей sd-scripts ===
def _load_sd_scripts_modules(sd_scripts_dir: str):
    """
    Fallback-импорт модулей sd-scripts напрямую с диска.
    
    Используется, когда стандартный импорт от library.flux_train_network не работает
    из-за проблем с accelerate wrapper или конфликтов пути.
    
    Args:
        sd_scripts_dir: Path to sd-scripts root directory
        
    Returns:
        Tuple of (train_util, flux_train_utils, FluxNetworkTrainer)
    """
    print("[FLUX2] Attempting FIX 2: Direct sd-scripts module loading via importlib...")
    
    lib_dir = os.path.join(sd_scripts_dir, "library")
    if not os.path.exists(lib_dir):
        print(f"[FLUX2] ERROR: library dir not found at: {lib_dir}")
        return None, None, None
    
    # Add lib_dir to path
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    
    try:
        # Try standard imports first with the new path
        import train_util
        import flux_train_utils
    except ImportError:
        # Load directly from file
        print("[FLUX2] Standard import failed, using direct file loading...")
        
        train_util_path = os.path.join(lib_dir, "train_util.py")
        spec = importlib.util.spec_from_file_location("train_util", train_util_path)
        train_util = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_util)
        
        flux_train_utils_path = os.path.join(lib_dir, "flux_train_utils.py")
        spec = importlib.util.spec_from_file_location("flux_train_utils", flux_train_utils_path)
        flux_train_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(flux_train_utils)
    
    # Load flux_train_network.py for FluxNetworkTrainer
    flux_train_path = os.path.join(sd_scripts_dir, "flux_train_network.py")
    if not os.path.exists(flux_train_path):
        print(f"[FLUX2] ERROR: flux_train_network.py not found at: {flux_train_path}")
        return train_util, flux_train_utils, None
    
    spec = importlib.util.spec_from_file_location("flux_train_network_module", flux_train_path)
    flux_train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flux_train_module)
    
    # Extract FluxNetworkTrainer class
    FluxNetworkTrainer = getattr(flux_train_module, "FluxNetworkTrainer", None)
    if FluxNetworkTrainer is None:
        print("[FLUX2] ERROR: FluxNetworkTrainer class not found in flux_train_network.py")
        return train_util, flux_train_utils, None
    
    print("[FLUX2] ✓ FIX 2: Successfully loaded sd-scripts modules via importlib")
    return train_util, flux_train_utils, FluxNetworkTrainer


print(f"[FLUX2] Current sys.path[0]: {sys.path[0]}")
print(f"[FLUX2] Current working directory: {os.getcwd()}")

# 2. Попытка 1: Ищем training_libs в существующих путях
found_libs = False
for p in sys.path:
    if p.endswith("training_libs") and os.path.exists(p):
        print(f"[FLUX2] ✓ Found training_libs in path: {p}")
        # Перемещаем в начало для приоритета
        sys.path.remove(p)
        sys.path.insert(0, p)
        found_libs = True
        break

if not found_libs:
    # Попытка 2: Ищем относительно текущего рабочего каталога
    # process.py устанавливает cwd, поэтому можем попробовать найти плагин отсюда
    cwd = os.getcwd()
    
    # Вверху по дереву: обычно sd-scripts находится в custom_nodes или в kohya_ss
    # Попытаемся подняться на несколько уровней вверх и найти ComfyUI-Flux2-LoRA-Manager
    for levels_up in range(1, 6):
        possible_root = os.path.join(cwd, *[".."] * levels_up, 
                                     "custom_nodes", "ComfyUI-Flux2-LoRA-Manager", "training_libs")
        possible_root = os.path.normpath(os.path.abspath(possible_root))
        if os.path.exists(possible_root):
            sys.path.insert(0, possible_root)
            print(f"[FLUX2] ✓ Found training_libs via cwd relative path: {possible_root}")
            found_libs = True
            break
    
    if not found_libs and args.sd_scripts_dir:
        # Попытка 3: Пытаемся вывести путь на основе sd_scripts_dir
        # Обычно sd-scripts лежит в kohya_ss, который может быть в custom_nodes
        # G:\ComfyUI\custom_nodes\kohya_ss\sd-scripts -> ищем custom_nodes
        # или G:\ComfyUI\kohya_ss\sd-scripts -> ищем ComfyUI
        
        parts = args.sd_scripts_dir.split(os.sep)
        
        # Ищем индекс "custom_nodes" в пути
        if "custom_nodes" in parts:
            idx = parts.index("custom_nodes")
            root = os.sep.join(parts[:idx+1])
            libs = os.path.join(root, "ComfyUI-Flux2-LoRA-Manager", "training_libs")
            if os.path.exists(libs):
                sys.path.insert(0, libs)
                print(f"[FLUX2] ✓ Found training_libs via sd_scripts deduction: {libs}")
                found_libs = True
        
        # Попытка 3b: Если sd-scripts находится в kohya_ss рядом с плагином
        if not found_libs:
            # Ищем папку kohya_ss и предполагаем, что плагин рядом
            if "kohya_ss" in parts or "sd-scripts" in parts:
                # Попробуем найти корень ComfyUI
                # G:\ComfyUI\custom_nodes\kohya_ss\sd-scripts или G:\ComfyUI\kohya_ss\sd-scripts
                for i in range(len(parts)-1, -1, -1):
                    if parts[i] in ["kohya_ss", "sd-scripts"]:
                        # Поднимаемся выше kohya_ss/sd-scripts
                        root = os.sep.join(parts[:i])
                        libs = os.path.join(root, "custom_nodes", "ComfyUI-Flux2-LoRA-Manager", "training_libs")
                        if os.path.exists(libs):
                            sys.path.insert(0, libs)
                            print(f"[FLUX2] ✓ Found training_libs via kohya_ss deduction: {libs}")
                            found_libs = True
                            break

if not found_libs:
    print("[FLUX2] ⚠ WARNING: Could not locate training_libs automatically.")
    print("[FLUX2] This will likely cause 'imagesize' import to fail.")

# 3. Добавляем sd-scripts (как и раньше)
if args.sd_scripts_dir and os.path.exists(args.sd_scripts_dir):
    if args.sd_scripts_dir not in sys.path:
        sys.path.append(args.sd_scripts_dir)
    lib_path = os.path.join(args.sd_scripts_dir, "library")
    if lib_path not in sys.path:
        sys.path.append(lib_path)
    print(f"[FLUX2] Added sd-scripts: {args.sd_scripts_dir}")

# 4. Проверка imagesize (диагностика)
try:
    import imagesize
    print(f"[FLUX2] ✓ imagesize imported successfully: {imagesize.__file__}")
except ImportError as e:
    print(f"[FLUX2] ❌ ERROR: imagesize not found. Check sys.path or install in training_libs!")
    print(f"[FLUX2] Error details: {e}")

# === Остальные импорты ===
import logging
import torch

# === FIX 2: Try standard import first, use fallback if fails ===
train_util = None
flux_train_utils = None
FluxNetworkTrainer = None

# Попытка 1: Стандартный импорт
try:
    from library import train_util, flux_train_utils
    from library.flux_train_network import FluxNetworkTrainer
    print("[FLUX2_TRAIN] ✓ All sd-scripts imports successful (standard import)")
except ImportError as e:
    print(f"[FLUX2_TRAIN] ⚠ Standard import failed: {e}")
    print("[FLUX2_TRAIN] Attempting FIX 2: Fallback loader...")
    
    if args.sd_scripts_dir and os.path.exists(args.sd_scripts_dir):
        train_util, flux_train_utils, FluxNetworkTrainer = _load_sd_scripts_modules(args.sd_scripts_dir)
        
        if FluxNetworkTrainer is None:
            print("[FLUX2_TRAIN] CRITICAL IMPORT ERROR: FluxNetworkTrainer not available")
            print("[FLUX2_TRAIN] Ensure --sd_scripts_dir points to valid sd-scripts installation")
            sys.exit(1)
    else:
        print(f"[FLUX2_TRAIN] CRITICAL ERROR: Cannot use fallback without --sd_scripts_dir")
        print(f"[FLUX2_TRAIN] Received: {args.sd_scripts_dir}")
        sys.exit(1)

if train_util is None or flux_train_utils is None:
    print("[FLUX2_TRAIN] CRITICAL ERROR: train_util or flux_train_utils not available")
    sys.exit(1)

# Импортируем наши модули Flux.2
try:
    # Ищем flux2_support папку
    flux2_support_dir = os.path.dirname(os.path.abspath(__file__))
    if flux2_support_dir not in sys.path:
        sys.path.insert(1, flux2_support_dir)
    
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
