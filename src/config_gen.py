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
        dataset_config = {
            "general": {
                "shuffle_caption": True,
                "keep_tokens": 1,
                "seed": seed
            },
            "datasets": [
                {
                    "resolution": int(resolution),
                    "min_bucket_reso": 256,
                    "max_bucket_reso": int(resolution),
                    "caption_extension": ".txt",
                    "batch_size": 1,  # STRICTLY 1 for 8GB VRAM
                    "enable_bucket_reso_steps": enable_bucket,
                    "bucket_reso_steps": 64,
                    "image_dir": img_folder
                }
            ]
        }

        # Save TOML config
        toml_path = os.path.join(output_dir, "dataset.toml")
        if toml:
            with open(toml_path, "w", encoding='utf-8') as f:
                toml.dump(dataset_config, f)
        else:
            # Fallback: save as JSON with .toml extension
            with open(toml_path, "w", encoding='utf-8') as f:
                json.dump(dataset_config, f, indent=2)

        # 2. Validate script path
        script_path = os.path.join(sd_scripts_path, "flux_train_network.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"Kohya script not found at: {script_path}\n"
                f"Please ensure 'sd-scripts' is installed at the specified path."
            )

        # 3. Build command arguments optimized for RTX 3060 Ti (8GB)
        # Get absolute path to current Python interpreter (critical for Windows)
        import sys
        python_exe = sys.executable

        # IMPORTANT: On Windows, calling accelerate as a module is more reliable
        # than calling it as a command-line tool directly
        cmd = [
            python_exe,
            "-m", "accelerate.commands.launch",  # Run accelerate as module
            "--mixed_precision=bf16",
            "--num_cpu_threads_per_process=2",
            script_path,
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
            "--cache_latents_to_disk" if cache_to_disk else "",
            "--optimizer_type", "adafactor",
            "--optimizer_args", "scale_parameter=False", "relative_step=False", "warmup_init=False",
            "--fp8_base",  # Crucial for 8GB: quantizes base model to FP8
            "--highvram",  # Enables smart offloading in Kohya
        ]

        # Remove empty strings from command
        cmd = [arg for arg in cmd if arg]

        return (" ".join(cmd), json.dumps(dataset_config, indent=2, ensure_ascii=False), output_dir)
