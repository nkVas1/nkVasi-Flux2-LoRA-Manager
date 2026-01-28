"""
Configuration Validator for kohya-ss/sd-scripts TOML configs
Validates dataset.toml structure against official schema
"""

import os
from typing import Tuple, List, Dict, Any

try:
    import toml
except ImportError:
    toml = None


class ConfigValidator:
    """
    Validates kohya-ss dataset configuration files.
    Based on official schema: https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-en.md
    """
    
    # Official valid parameters for each level (as of 2025)
    GENERAL_PARAMS = {
        "shuffle_caption", "keep_tokens", "enable_bucket", "caption_dropout_rate",
        "caption_dropout_every_n_epochs", "caption_tag_dropout_rate",
        "face_crop_aug_range", "color_aug", "token_warmup_min", "token_warmup_step",
    }
    
    DATASET_PARAMS = {
        "resolution", "batch_size", "min_bucket_reso", "max_bucket_reso",
        "bucket_reso_steps", "bucket_no_upscale", "enable_bucket",
    }
    
    SUBSET_PARAMS = {
        "image_dir", "num_repeats", "caption_extension", "keep_tokens",
        "class_tokens", "is_reg", "flip_aug", "color_aug", "face_crop_aug_range",
        "random_crop", "shuffle_caption", "caption_dropout_rate",
        "caption_dropout_every_n_epochs", "caption_tag_dropout_rate",
    }
    
    # Known deprecated or invalid parameters
    DEPRECATED_PARAMS = {
        "enable_bucket_reso_steps": "Use 'enable_bucket' in [general] or [[datasets]] instead",
        "seed": "Use command line --seed argument instead (causes validation issues in config)",
    }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate dataset configuration structure and parameters.
        
        Args:
            config: Parsed TOML configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required top-level keys
        if "datasets" not in config:
            errors.append("Missing required 'datasets' key")
            return False, errors
        
        if not isinstance(config["datasets"], list):
            errors.append("'datasets' must be a list")
            return False, errors
        
        # Validate [general] section
        if "general" in config:
            for param in config["general"]:
                if param in cls.DEPRECATED_PARAMS:
                    errors.append(
                        f"Deprecated parameter in [general]: '{param}' - "
                        f"{cls.DEPRECATED_PARAMS[param]}"
                    )
                elif param not in cls.GENERAL_PARAMS:
                    errors.append(f"Unknown parameter in [general]: '{param}'")
        
        # Validate each [[datasets]] block
        for idx, dataset in enumerate(config["datasets"]):
            # Check for subsets
            if "subsets" not in dataset:
                errors.append(f"Dataset {idx}: Missing required 'subsets' key")
                continue
            
            if not isinstance(dataset["subsets"], list):
                errors.append(f"Dataset {idx}: 'subsets' must be a list")
                continue
            
            # Validate dataset-level parameters
            for param in dataset:
                if param == "subsets":
                    continue
                    
                if param in cls.DEPRECATED_PARAMS:
                    errors.append(
                        f"Dataset {idx}: Deprecated parameter '{param}' - "
                        f"{cls.DEPRECATED_PARAMS[param]}"
                    )
                elif param not in cls.DATASET_PARAMS:
                    errors.append(f"Dataset {idx}: Unknown parameter '{param}'")
            
            # Validate each subset
            for sub_idx, subset in enumerate(dataset["subsets"]):
                # Check required fields
                if "image_dir" not in subset:
                    errors.append(
                        f"Dataset {idx}, Subset {sub_idx}: Missing required 'image_dir'"
                    )
                
                # Check for invalid parameters
                for param in subset:
                    if param in cls.DEPRECATED_PARAMS:
                        errors.append(
                            f"Dataset {idx}, Subset {sub_idx}: Deprecated parameter '{param}'"
                        )
                    elif param not in cls.SUBSET_PARAMS:
                        errors.append(
                            f"Dataset {idx}, Subset {sub_idx}: Unknown parameter '{param}'"
                        )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @classmethod
    def validate_file(cls, toml_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a TOML file.
        
        Args:
            toml_path: Path to dataset.toml file
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not toml:
            return False, ["toml library not installed"]
        
        if not os.path.exists(toml_path):
            return False, [f"File not found: {toml_path}"]
        
        try:
            with open(toml_path, 'r', encoding='utf-8') as f:
                config = toml.load(f)
            
            return cls.validate_config(config)
        except Exception as e:
            return False, [f"Failed to parse TOML: {e}"]
    
    @classmethod
    def auto_fix_config(cls, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Attempt to automatically fix common configuration errors.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (fixed_config, changes_made)
        """
        import copy
        fixed = copy.deepcopy(config)
        changes = []
        
        # Fix deprecated parameters in [general]
        if "general" in fixed:
            if "seed" in fixed["general"]:
                del fixed["general"]["seed"]
                changes.append("Removed 'seed' from [general] (use CLI --seed instead)")
        
        # Fix each dataset
        for idx, dataset in enumerate(fixed.get("datasets", [])):
            # Move enable_bucket_reso_steps to enable_bucket
            if "enable_bucket_reso_steps" in dataset:
                value = dataset.pop("enable_bucket_reso_steps")
                if "enable_bucket" not in dataset and "enable_bucket" not in fixed.get("general", {}):
                    if "general" not in fixed:
                        fixed["general"] = {}
                    fixed["general"]["enable_bucket"] = value
                    changes.append(
                        f"Dataset {idx}: Converted 'enable_bucket_reso_steps' to 'enable_bucket' in [general]"
                    )
                else:
                    changes.append(
                        f"Dataset {idx}: Removed deprecated 'enable_bucket_reso_steps'"
                    )
            
            # Remove deprecated parameters from subsets
            for sub_idx, subset in enumerate(dataset.get("subsets", [])):
                if "seed" in subset:
                    del subset["seed"]
                    changes.append(
                        f"Dataset {idx}, Subset {sub_idx}: Removed deprecated 'seed'"
                    )
        
        return fixed, changes
