"""
Automatic Dependency Checker and Installer
Ensures all required packages are available before training
"""

import subprocess
import sys
import os
from typing import List, Dict, Tuple

# ComfyUI imports (graceful fallback)
try:
    from server import PromptServer
except ImportError:
    PromptServer = None


class DependencyManager:
    """Manages dependency checking and installation."""
    
    # Required packages with version constraints
    # Senior-level approach: Comprehensive dependency list to prevent cascading errors
    REQUIRED_PACKAGES = {
        # Core Deep Learning
        'torch': ('torch', '2.0.0'),
        'transformers': ('transformers', '4.36.2'),
        'diffusers': ('diffusers', '0.25.1'),
        'accelerate': ('accelerate', '0.25.0'),
        'safetensors': ('safetensors', '0.4.2'),
        'huggingface_hub': ('huggingface_hub', '0.20.3'),
        'tokenizers': ('tokenizers', '0.15.2'),
        'peft': ('peft', '0.7.1'),
        
        # Configuration & Serialization
        'toml': ('toml', '0.10.2'),
        'omegaconf': ('omegaconf', '2.3.0'),
        'pyyaml': ('pyyaml', '6.0.1'),
        'packaging': ('packaging', '23.2'),
        
        # Utilities & Performance
        'einops': ('einops', '0.7.0'),
        'regex': ('regex', '2023.12.25'),
        'requests': ('requests', '2.31.0'),
        'tqdm': ('tqdm', '4.66.1'),
        'filelock': ('filelock', '3.13.1'),
        'fsspec': ('fsspec', '2023.12.2'),
        
        # Image Processing (CRITICAL to prevent crashes)
        'imagesize': ('imagesize', '1.4.1'),
        'albumentations': ('albumentations', '1.3.1'),
        'opencv-python-headless': ('opencv-python-headless', '4.9.0.80'),
        
        # Scientific Computing (needed for data processing & optimization)
        'scipy': ('scipy', '1.11.4'),
        'pandas': ('pandas', '2.1.4'),
        'numexpr': ('numexpr', '2.9.0'),
    }
    
    @staticmethod
    def log_message(message: str, level: str = "INFO"):
        """Send log message to console and ComfyUI UI."""
        formatted = f"[DEPENDENCY-{level}] {message}"
        print(formatted)
        
        if PromptServer:
            try:
                PromptServer.instance.send_sync("flux_train_log", {"line": formatted})
            except Exception:
                pass
    
    @classmethod
    def check_package(cls, python_exe: str, package_name: str, env: Dict) -> bool:
        """Check if a package is installed."""
        check_code = f"""
import sys
try:
    import {package_name}
    sys.exit(0)
except ImportError:
    sys.exit(1)
"""
        try:
            result = subprocess.run(
                [python_exe, "-c", check_code],
                capture_output=True,
                timeout=10,
                env=env
            )
            return result.returncode == 0
        except Exception as e:
            cls.log_message(f"Error checking {package_name}: {e}", "WARN")
            return False
    
    @classmethod
    def install_package(cls, python_exe: str, package_spec: str, env: Dict) -> bool:
        """Install a single package."""
        try:
            cls.log_message(f"Installing {package_spec}...")
            
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", package_spec, 
                 "--no-warn-script-location", "--quiet"],
                capture_output=True,
                env=env,
                timeout=300  # 5 minutes max per package
            )
            
            if result.returncode == 0:
                cls.log_message(f"✓ {package_spec} installed successfully", "OK")
                return True
            else:
                error = result.stderr.decode() if result.stderr else "Unknown error"
                cls.log_message(f"Failed to install {package_spec}: {error}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            cls.log_message(f"Timeout installing {package_spec}", "ERROR")
            return False
        except Exception as e:
            cls.log_message(f"Exception installing {package_spec}: {e}", "ERROR")
            return False
    
    @classmethod
    def verify_and_install(cls, python_exe: str, env: Dict) -> Tuple[bool, List[str]]:
        """
        Check all dependencies and install missing ones.
        Returns (success, list_of_errors)
        """
        cls.log_message("Starting dependency verification...")
        
        missing = []
        errors = []
        
        # Check each required package
        for module_name, (package_spec, min_version) in cls.REQUIRED_PACKAGES.items():
            if not cls.check_package(python_exe, module_name, env):
                missing.append(package_spec)
        
        if not missing:
            cls.log_message("All required dependencies are installed", "OK")
            return True, []
        
        # Install missing packages
        cls.log_message(f"Found {len(missing)} missing packages: {', '.join(missing)}")
        
        for package_spec in missing:
            if not cls.install_package(python_exe, package_spec, env):
                errors.append(f"Failed to install {package_spec}")
        
        if errors:
            cls.log_message(f"Completed with {len(errors)} errors", "WARN")
            return False, errors
        else:
            cls.log_message("All dependencies installed successfully", "OK")
            return True, []
