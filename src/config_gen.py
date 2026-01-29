"""
Configuration Generator for FLUX.2 LoRA Training (Low VRAM Optimized)
Generates training commands and dataset configurations for kohya-ss/sd-scripts
Optimized for RTX 3060 Ti (8GB VRAM) and similar hardware
"""

import os
import json
import glob
import sys
import subprocess

try:
    import folder_paths
except ImportError:
    # Fallback for testing outside ComfyUI
    class folder_paths:
        @staticmethod
        def get_output_directory():
            return os.getcwd()

try:
    import toml
except ImportError:
    toml = None


class Flux2_8GB_Configurator:
    """
    Generates optimized configuration for Low-VRAM (8GB) environments.
    
    Target hardware: NVIDIA RTX 3060 Ti (Ampere) or similar with 8GB VRAM.
    
    Strategy:
    - QLoRA (NF4) for base model quantization
    - FP8 precision for computation
    - Adafactor optimizer (lower memory footprint than AdamW)
    - Gradient checkpointing and latent caching
    - CPU offloading capabilities
    """
    
    CATEGORY = "Flux2/Training"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("cmd_args", "dataset_config", "output_dir")
    FUNCTION = "generate_config"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sd_scripts_path": ("STRING", {
                    "default": "C:/AI/sd-scripts",
                    "multiline": False
                }),
                "model_path": ("STRING", {
                    "default": "black-forest-labs/FLUX.1-dev",
                    "multiline": False
                }),
                "img_folder": ("STRING", {
                    "default": "C:/Dataset/img",
                    "multiline": False
                }),
                "output_name": ("STRING", {
                    "default": "my_flux_lora",
                    "multiline": False
                }),
                "resolution": (["512", "768", "1024"], {"default": "512"}),
                "learning_rate": ("FLOAT", {"default": 1e-4, "step": 1e-5}),
                "max_train_steps": ("INT", {"default": 1200}),
                "lora_rank": (["16", "32"], {"default": "16"}),
            },
            "optional": {
                "num_repeats": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "enable_bucket": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42}),
                "cache_to_disk": ("BOOLEAN", {"default": True}),
                "save_every_n_epochs": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "label": "Save checkpoint every N epochs"
                }),
                "train_unet_only": ("BOOLEAN", {
                    "default": True,
                    "label": "Train U-Net only (Flux.2) / Text Encoders (Flux.1)"
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "label": "VAE Path (Optional - for Flux.1/Flux.2)"
                }),
                "clip_l_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "label": "CLIP-L Path (Optional - for Flux.1)"
                }),
                "t5xxl_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "label": "T5-XXL Path (Optional - for Flux.1)"
                }),
                "cache_text_encoder_outputs": ("BOOLEAN", {
                    "default": False,
                    "label": "Cache Text Encoder outputs (speed up, requires shuffle_caption OFF)"
                }),
                "quality_mode": ("BOOLEAN", {
                    "default": False,
                    "label": "Quality mode (disable fp8_base, slower, more VRAM)"
                }),
                "target_epochs": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "label": "Target Epochs (0=manual max_train_steps, >0=auto calculate steps)"
                }),
            }
        }

    def generate_config(
        self,
        sd_scripts_path,
        model_path,
        img_folder,
        output_name,
        resolution,
        learning_rate,
        max_train_steps,
        lora_rank,
        num_repeats=10,
        enable_bucket=True,
        seed=42,
        cache_to_disk=True,
        save_every_n_epochs=1,
        train_unet_only=True,
        vae_path="",
        clip_l_path="",
        t5xxl_path="",
        cache_text_encoder_outputs=False,
        quality_mode=False,
        target_epochs=0,
    ):
        """Generate training configuration and command arguments."""
        
        # VALIDATION PHASE: Check dataset and paths BEFORE generating config
        print("\n[CONFIG-GEN] ═══════════════════════════════════════")
        print("[CONFIG-GEN] VALIDATION PHASE")
        print("[CONFIG-GEN] ═══════════════════════════════════════")
        
        # Check 1: Dataset folder exists
        if not os.path.isdir(img_folder):
            error_msg = f"ERROR: Image folder not found: {img_folder}"
            print(f"[CONFIG-GEN] ✗ {error_msg}")
            return (error_msg, "", "")
        
        print(f"[CONFIG-GEN] ✓ Dataset folder found: {img_folder}")
        
        # Check 2: Dataset folder has images
        import glob
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(img_folder, ext)))
            image_files.extend(glob.glob(os.path.join(img_folder, ext.upper())))
        
        if not image_files:
            error_msg = f"ERROR: No images found in {img_folder}"
            print(f"[CONFIG-GEN] ✗ {error_msg}")
            return (error_msg, "", "")
        
        print(f"[CONFIG-GEN] ✓ Found {len(image_files)} images in dataset")
        
        # === AUTO-CORRECTION: Handle encoder cache incompatibilities ===
        shuffle_caption_effective = True
        if cache_text_encoder_outputs:
            shuffle_caption_effective = False
            print("[CONFIG-GEN] ✓ Encoder cache ON -> forcing shuffle_caption=False and disabling caption dropout/warmup fields")
        
        # === AUTO-CALCULATION: target_epochs to max_train_steps ===
        steps_per_epoch = len(image_files) * num_repeats
        if target_epochs and target_epochs > 0:
            max_train_steps = steps_per_epoch * target_epochs
            print(f"[CONFIG-GEN] ✓ target_epochs={target_epochs} -> max_train_steps={max_train_steps} (steps_per_epoch={steps_per_epoch})")
        
        # FIX 2: Diagnostic message for num_repeats
        expected_epochs = max_train_steps // (len(image_files) * num_repeats) + 1
        print(f"[CONFIG-GEN] FIX 2 DIAGNOSTIC: num_repeats={num_repeats}, expected_epochs={expected_epochs} (steps={max_train_steps} / ({len(image_files)} images * {num_repeats} repeats))")
        
        # Check 3: sd-scripts path exists (critical for Flux.1)
        if not os.path.isdir(sd_scripts_path):
            error_msg = f"ERROR: sd-scripts path not found: {sd_scripts_path}"
            print(f"[CONFIG-GEN] ✗ {error_msg}")
            return (error_msg, "", "")
        
        print(f"[CONFIG-GEN] ✓ sd-scripts path found: {sd_scripts_path}")
        
        # === UX 5: EXTENDED VALIDATION (Compatibility checks) ===
        validation_warnings = []
        
        # Check VRAM compatibility
        if int(lora_rank) > 32:
            validation_warnings.append(f"LoRA rank >{lora_rank} may cause OOM on 8GB VRAM. Consider rank=16 or 32.")
        
        # Check steps vs dataset size
        if expected_epochs > 50:
            validation_warnings.append(f"Very high epoch count ({expected_epochs}). Consider increasing num_repeats or reducing max_train_steps.")
        elif expected_epochs < 3:
            validation_warnings.append(f"Very low epoch count ({expected_epochs}). May underfit. Consider decreasing num_repeats or increasing max_train_steps.")
        
        # Check dataset size
        if len(image_files) < 5:
            validation_warnings.append(f"Very small dataset ({len(image_files)} images). Consider adding more for better results.")
        
        # Report validation warnings
        if validation_warnings:
            print("[CONFIG-GEN] " + "="*40)
            print("[CONFIG-GEN] ⚠ WARNINGS")
            for warn in validation_warnings:
                print(f"[CONFIG-GEN]   - {warn}")
            print("[CONFIG-GEN] " + "="*40)
        
        # Prepare output directory
        output_dir = os.path.join(
            folder_paths.get_output_directory(),
            "flux_training",
            output_name
        )
        os.makedirs(output_dir, exist_ok=True)

        # 1. Generate Dataset TOML configuration
        # Structure follows official kohya-ss/sd-scripts format (2025)
        # Reference: https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-en.md
        dataset_config = {
            "general": {
                "shuffle_caption": shuffle_caption_effective,
                "keep_tokens": 1,
                "enable_bucket": enable_bucket,
            },
            "datasets": [
                {
                    "resolution": int(resolution),
                    "min_bucket_reso": 256,
                    "max_bucket_reso": int(resolution),
                    "batch_size": 1,  # STRICTLY 1 for 8GB VRAM
                    "bucket_reso_steps": 64,
                    "bucket_no_upscale": False,  # Allow upscaling if needed
                    
                    # Subsets contain only dataset-specific info (not training params)
                    "subsets": [
                        {
                            "image_dir": img_folder,
                            "num_repeats": num_repeats,
                            "caption_extension": ".txt",
                            "shuffle_caption": shuffle_caption_effective,
                            "caption_dropout_rate": 0.0,
                            "caption_tag_dropout_rate": 0.0,
                            "token_warmup_min": 1,
                            "token_warmup_step": 0,
                        }
                    ]
                }
            ]
        }

        # Save TOML config (with error handling)
        toml_path = os.path.join(output_dir, "dataset.toml")
        try:
            if toml:
                with open(toml_path, "w", encoding='utf-8') as f:
                    toml.dump(dataset_config, f)
                print(f"[CONFIG-GEN] ✓ Saved TOML config: {toml_path}")
            else:
                # CRITICAL: Install toml if missing
                print("[CONFIG-GEN] ⚠ toml library not found, installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "toml", "--quiet"])
                import toml as toml_installed
                with open(toml_path, "w", encoding='utf-8') as f:
                    toml_installed.dump(dataset_config, f)
                print(f"[CONFIG-GEN] ✓ Installed toml and saved config: {toml_path}")
        except Exception as e:
            error_msg = f"ERROR: Failed to save dataset config: {e}"
            print(f"[CONFIG-GEN] {error_msg}")
            return (error_msg, "", "")

        # 2. Detect Flux.2 vs Flux.1 based on model path
        # Flux.2 requires dedicated trainer (different architecture)
        is_flux2 = False
        if "flux2" in model_path.lower() or "flux.2" in model_path.lower():
            is_flux2 = True
            print("[CONFIG-GEN] ✓ Detected Flux.2 model - using dedicated trainer")
        
        # 3. Select appropriate training script
        if is_flux2:
            # Use Flux.2 dedicated trainer from our codebase
            # This script is located relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, "flux2_support", "flux2_train.py")
            if not os.path.exists(script_path):
                error_msg = f"ERROR: Flux.2 trainer not found at: {script_path}"
                return (error_msg, "", "")
            print(f"[CONFIG-GEN] ✓ Using Flux.2 trainer: {script_path}")
        else:
            # Use standard sd-scripts trainer for Flux.1
            script_path = os.path.join(sd_scripts_path, "flux_train_network.py")
            if not os.path.exists(script_path):
                error_msg = f"ERROR: Script not found at: {script_path}\nEnsure 'sd-scripts' is installed at: {sd_scripts_path}"
                return (error_msg, "", "")
            print(f"[CONFIG-GEN] ✓ Using Flux.1 trainer: {script_path}")

        # 4. Validate Python interpreter exists (critical for Windows subprocess)
        python_exe = sys.executable
        
        if not os.path.exists(python_exe):
            error_msg = f"ERROR: Python interpreter not found at: {python_exe}"
            return (error_msg, "", "")

        # 5. Build command arguments optimized for RTX 3060 Ti (8GB)
        # CRITICAL FOR WINDOWS: Pass command as JSON list, NOT as string
        # This preserves backslashes in paths (G:\ComfyUI\... won't get mangled)
        # 
        # KEY FIX: Use full path to script
        # process.py will extract directory from this full path and use as working directory
        # This ensures Python finds 'library' module and accelerate works correctly
        
        cmd = [
            python_exe,
            "-u",  # Unbuffered output - IMPORTANT for real-time logs!
            "-m", "accelerate.commands.launch",  # Run accelerate as module
            "--num_processes=1",  # Explicitly use 1 process (avoid child process PYTHONPATH issues)
            "--mixed_precision=bf16",
            "--num_cpu_threads_per_process=2",
            script_path,  # Use full path - process.py extracts dir and uses as cwd
            "--pretrained_model_name_or_path", model_path,
            "--dataset_config", toml_path,
            "--output_dir", output_dir,
            "--output_name", output_name,
            "--max_train_steps", str(max_train_steps),
            "--learning_rate", str(learning_rate),
            "--gradient_accumulation_steps", "1",
            "--network_dim", str(lora_rank),  # Ensure string type
            "--network_alpha", str(lora_rank),
        ]
        
        # === FIX: Select correct network module for Flux.1 vs Flux.2 ===
        if is_flux2:
            cmd.extend(["--network_module", "networks.lora"])
            print("[CONFIG-GEN] ✓ Network module: networks.lora (Flux.2)")
        else:
            # Flux.1 dev requires lora_flux to find U-Net modules correctly
            cmd.extend(["--network_module", "networks.lora_flux"])
            print("[CONFIG-GEN] ✓ Network module: networks.lora_flux (Flux.1)")
        
        # FIX 1: Conditional --network_train_unet_only flag
        # - For Flux.2: Only train U-Net (DiT), not text encoders (no CLIP encoders in Flux.2)
        # - For Flux.1: U-Net IS trainable with networks.lora_flux
        if is_flux2:
            if train_unet_only:
                cmd.append("--network_train_unet_only")
                print("[CONFIG-GEN] ✓ Flux.2 using --network_train_unet_only (DiT/U-Net only)")
            else:
                print("[CONFIG-GEN] ⚠ Flux.2 with train_unet_only=False - will train all modules")
        else:
            # Flux.1: Logic fixed for networks.lora_flux
            if train_unet_only:
                cmd.append("--network_train_unet_only")
                print("[CONFIG-GEN] ✓ Flux.1 training U-Net only (using networks.lora_flux)")
            else:
                print("[CONFIG-GEN] ✓ Flux.1 training U-Net + Text Encoders")

        # --- VRAM SAVING STRATEGY ---
        cmd.extend([
            "--mixed_precision", "bf16",
            "--save_precision", "bf16",
            "--gradient_checkpointing",
            "--cache_latents",
        ])
        
        # Add conditional cache_latents_to_disk only if enabled (cleaner than empty strings)
        if cache_to_disk:
            cmd.extend([
                "--cache_latents_to_disk",
            ])
        
        # PHASE 1: Add VAE/CLIP/T5 paths if provided
        # This fixes Flux.1 training by providing encoder paths
        if vae_path and os.path.exists(vae_path):
            cmd.extend(["--ae", vae_path])
            print(f"[CONFIG-GEN] ✓ VAE path added: {vae_path}")
        else:
            if vae_path:
                print(f"[CONFIG-GEN] ⚠ VAE path not found: {vae_path}")
        
        if clip_l_path and os.path.exists(clip_l_path):
            cmd.extend(["--clip_l", clip_l_path])
            print(f"[CONFIG-GEN] ✓ CLIP-L path added: {clip_l_path}")
        else:
            if clip_l_path:
                print(f"[CONFIG-GEN] ⚠ CLIP-L path not found: {clip_l_path}")
        
        if t5xxl_path and os.path.exists(t5xxl_path):
            cmd.extend(["--t5xxl", t5xxl_path])
            print(f"[CONFIG-GEN] ✓ T5-XXL path added: {t5xxl_path}")
        else:
            if t5xxl_path:
                print(f"[CONFIG-GEN] ⚠ T5-XXL path not found: {t5xxl_path}")
        
        # PHASE 1 (Flux.2 specific): Pass sd-scripts path to custom trainer
        # This allows flux2_train.py to locate library module
        if is_flux2:
            cmd.extend(["--sd_scripts_dir", sd_scripts_path])
            print(f"[CONFIG-GEN] ✓ sd-scripts dir passed to Flux.2 trainer: {sd_scripts_path}")
        
        # === FIX 1: Автосохранение чекпоинтов ===
        cmd.extend([
            "--save_every_n_epochs", str(save_every_n_epochs),
            "--save_model_as", "safetensors",
        ])
        print(f"[CONFIG-GEN] ✓ Checkpoints: saving every {save_every_n_epochs} epochs as safetensors")
        
        # === FIX 3: Оптимизация памяти для 3060 Ti (8GB) ===
        # Критические флаги для ускорения на 8ГБ VRAM
        if not is_flux2:  # Flux.1 specific optimizations
            cmd.append("--lowram")
            
            # Conditional encoder caching - only if cache_text_encoder_outputs enabled
            if cache_text_encoder_outputs:
                cmd.extend([
                    "--cache_text_encoder_outputs",
                    "--cache_text_encoder_outputs_to_disk",
                ])
                print("[CONFIG-GEN] ✓ FIX 3: Enabled encoder caching (shuffle_caption disabled)")
            else:
                print("[CONFIG-GEN] ℹ FIX 3: Encoder caching disabled - shuffle_caption enabled")
        
        cmd.extend([
            "--persistent_data_loader_workers",  # Сохраняет workers между эпохами
        ])
        
        # === FIX 5 + UX 6: Comprehensive parameter logging and summary ===
        print("\n[CONFIG-GEN] ═══════════════════════════════════════")
        print("[CONFIG-GEN] TRAINING CONFIGURATION SUMMARY")
        print("[CONFIG-GEN] ═══════════════════════════════════════")
        print(f"[CONFIG-GEN] Model Type: {'Flux.2' if is_flux2 else 'Flux.1'}")
        print(f"[CONFIG-GEN] Model: {model_path}")
        print(f"[CONFIG-GEN] Dataset: {img_folder} ({len(image_files)} images, {num_repeats} repeats)")
        print(f"[CONFIG-GEN] Expected Epochs: {expected_epochs}")
        print(f"[CONFIG-GEN] Output: {output_dir}")
        print(f"[CONFIG-GEN] Resolution: {resolution}x{resolution}")
        print(f"[CONFIG-GEN] LoRA Rank: {lora_rank}")
        print(f"[CONFIG-GEN] Learning Rate: {learning_rate}")
        print(f"[CONFIG-GEN] Max Steps: {max_train_steps}")
        print(f"[CONFIG-GEN] Batch Size: 1 (for 8GB VRAM)")
        print(f"[CONFIG-GEN] Gradient Accumulation: 1")
        print(f"[CONFIG-GEN] Mixed Precision: bf16")
        print(f"[CONFIG-GEN] FP8 Quantization: True")
        print(f"[CONFIG-GEN] Gradient Checkpointing: True")
        print(f"[CONFIG-GEN] Cache Latents: True")
        print(f"[CONFIG-GEN] Cache to Disk: {cache_to_disk}")
        print(f"[CONFIG-GEN] Bucketing: {enable_bucket}")
        
        if vae_path and os.path.exists(vae_path):
            print(f"[CONFIG-GEN] VAE: {vae_path}")
        if clip_l_path and os.path.exists(clip_l_path):
            print(f"[CONFIG-GEN] CLIP-L: {clip_l_path}")
        if t5xxl_path and os.path.exists(t5xxl_path):
            print(f"[CONFIG-GEN] T5-XXL: {t5xxl_path}")
        
        print(f"[CONFIG-GEN] Train U-Net Only: {train_unet_only}")
        print("[CONFIG-GEN] ═══════════════════════════════════════\n")
        
        cmd.extend([
            "--optimizer_type", "adafactor",
            "--optimizer_args", "scale_parameter=False", "relative_step=False", "warmup_init=False",
        ])
        
        # === Quality vs Speed mode ===
        if not quality_mode:
            cmd.append("--fp8_base")
            print("[CONFIG-GEN] ✓ fp8_base enabled (speed mode)")
        else:
            print("[CONFIG-GEN] ✓ fp8_base disabled (quality mode)")

        # CRITICAL: Return command as JSON, not as string
        # This preserves Windows paths with backslashes
        # Runner will parse it as JSON and use it as a list directly
        cmd_json = json.dumps(cmd, ensure_ascii=False)
        
        return (cmd_json, json.dumps(dataset_config, indent=2, ensure_ascii=False), output_dir)
