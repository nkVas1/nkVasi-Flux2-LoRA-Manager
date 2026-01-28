"""
Configuration Generator for FLUX.2 LoRA Training (Low VRAM Optimized)
Generates training commands and dataset configurations for kohya-ss/sd-scripts
Optimized for RTX 3060 Ti (8GB VRAM) and similar hardware
"""

import os
import json

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
    ):
        """Generate training configuration and command arguments."""
        
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
                "shuffle_caption": True,
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
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "toml", "--quiet"])
                import toml as toml_installed
                with open(toml_path, "w", encoding='utf-8') as f:
                    toml_installed.dump(dataset_config, f)
                print(f"[CONFIG-GEN] ✓ Installed toml and saved config: {toml_path}")
        except Exception as e:
            error_msg = f"ERROR: Failed to save dataset config: {e}"
            print(f"[CONFIG-GEN] {error_msg}")
            return (error_msg, "", "")

        # 2. Validate script path BEFORE attempting to build command
        script_path = os.path.join(sd_scripts_path, "flux_train_network.py")
        if not os.path.exists(script_path):
            error_msg = f"ERROR: Script not found at: {script_path}\nEnsure 'sd-scripts' is installed at: {sd_scripts_path}"
            return (error_msg, "", "")

        # 3. Validate Python interpreter exists (critical for Windows subprocess)
        import sys
        python_exe = sys.executable
        
        if not os.path.exists(python_exe):
            error_msg = f"ERROR: Python interpreter not found at: {python_exe}"
            return (error_msg, "", "")

        # 4. Build command arguments optimized for RTX 3060 Ti (8GB)
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

            # --- VRAM SAVING STRATEGY ---
            "--mixed_precision", "bf16",
            "--save_precision", "bf16",
            "--gradient_checkpointing",
            "--cache_latents",
        ]
        
        # Add conditional cache_latents_to_disk only if enabled (cleaner than empty strings)
        if cache_to_disk:
            cmd.extend([
                "--cache_latents_to_disk",
            ])
        
        cmd.extend([
            "--optimizer_type", "adafactor",
            "--optimizer_args", "scale_parameter=False", "relative_step=False", "warmup_init=False",
            "--fp8_base",  # Crucial for 8GB: quantizes base model to FP8
        ])

        # CRITICAL: Return command as JSON, not as string
        # This preserves Windows paths with backslashes
        # Runner will parse it as JSON and use it as a list directly
        cmd_json = json.dumps(cmd, ensure_ascii=False)
        
        return (cmd_json, json.dumps(dataset_config, indent=2, ensure_ascii=False), output_dir)
