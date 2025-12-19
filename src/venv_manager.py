"""
Virtual Environment Manager for Training Isolation

Создает и управляет изолированным Python окружением для обучения,
предотвращая конфликты версий библиотек между ComfyUI и sd-scripts.
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple, List


class VirtualEnvManager:
    """
    Manages isolated Python environments for training.
    Prevents version conflicts between ComfyUI and sd-scripts dependencies.
    """
    
    # Exact versions that work with sd-scripts
    TRAINING_REQUIREMENTS = {
        'torch': '2.1.0',
        'torchvision': '0.16.0',
        'transformers': '4.36.2',  # Совместима с GenerationMixin
        'diffusers': '0.25.0',
        'accelerate': '0.25.0',
        'safetensors': '0.4.1',
        'toml': '0.10.2',
        'omegaconf': '2.3.0',
        'einops': '0.7.0',
        'prodigyopt': '1.0',
        'lycoris-lora': '1.9.0',
    }
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize VirtualEnvManager.
        
        Args:
            base_dir: Base directory for venvs (defaults to plugin directory)
        """
        if base_dir is None:
            # Use plugin directory by default
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.base_dir = Path(base_dir)
        self.venv_dir = self.base_dir / "training_venv"
        self.cache_file = self.venv_dir / ".venv_cache.json"
        
    def venv_exists(self) -> bool:
        """Check if training venv exists and is valid."""
        if not self.venv_dir.exists():
            return False
        
        # Check for Python executable
        python_exe = self._get_venv_python()
        if not os.path.exists(python_exe):
            return False
        
        # Check cache file
        if not self.cache_file.exists():
            return False
        
        return True
    
    def _get_venv_python(self) -> str:
        """Get path to Python executable in venv."""
        if sys.platform == "win32":
            return str(self.venv_dir / "Scripts" / "python.exe")
        else:
            return str(self.venv_dir / "bin" / "python")
    
    def _get_venv_pip(self) -> str:
        """Get path to pip executable in venv."""
        if sys.platform == "win32":
            return str(self.venv_dir / "Scripts" / "pip.exe")
        else:
            return str(self.venv_dir / "bin" / "pip")
    
    def create_venv(self, force: bool = False) -> Tuple[bool, str]:
        """
        Create training virtual environment.
        
        Args:
            force: Force recreation if venv exists
            
        Returns:
            (success, message)
        """
        if self.venv_exists() and not force:
            return True, "Virtual environment already exists"
        
        print("[VENV] Creating training virtual environment...")
        
        # Remove old venv if forcing
        if force and self.venv_dir.exists():
            print("[VENV] Removing old virtual environment...")
            try:
                shutil.rmtree(self.venv_dir)
            except Exception as e:
                return False, f"Failed to remove old venv: {e}"
        
        # Create new venv
        try:
            # Use current Python to create venv
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_dir)],
                check=True,
                capture_output=True,
                timeout=60
            )
        except subprocess.CalledProcessError as e:
            return False, f"Failed to create venv: {e.stderr.decode()}"
        except Exception as e:
            return False, f"Failed to create venv: {e}"
        
        print("[VENV] Virtual environment created successfully")
        return True, "Virtual environment created"
    
    def install_requirements(self, progress_callback=None) -> Tuple[bool, List[str]]:
        """
        Install training requirements in venv.
        
        Args:
            progress_callback: Optional callback(package_name, status) for progress updates
            
        Returns:
            (success, list_of_errors)
        """
        if not self.venv_exists():
            return False, ["Virtual environment does not exist"]
        
        python_exe = self._get_venv_python()
        pip_exe = self._get_venv_pip()
        
        errors = []
        installed = []
        
        # Upgrade pip first
        print("[VENV] Upgrading pip...")
        try:
            subprocess.run(
                [python_exe, "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
                timeout=120
            )
        except Exception as e:
            errors.append(f"Failed to upgrade pip: {e}")
        
        # Install PyTorch first (special handling for CUDA)
        print("[VENV] Installing PyTorch with CUDA support...")
        torch_version = self.TRAINING_REQUIREMENTS['torch']
        torchvision_version = self.TRAINING_REQUIREMENTS['torchvision']
        
        try:
            if progress_callback:
                progress_callback("torch", "installing")
            
            subprocess.run(
                [
                    pip_exe, "install",
                    f"torch=={torch_version}",
                    f"torchvision=={torchvision_version}",
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                ],
                check=True,
                capture_output=True,
                timeout=600  # 10 minutes for torch
            )
            
            installed.append("torch")
            installed.append("torchvision")
            
            if progress_callback:
                progress_callback("torch", "success")
        except subprocess.CalledProcessError as e:
            error_msg = f"torch: {e.stderr.decode()}"
            errors.append(error_msg)
            if progress_callback:
                progress_callback("torch", "failed")
        
        # Install other packages
        for package, version in self.TRAINING_REQUIREMENTS.items():
            if package in ['torch', 'torchvision']:
                continue  # Already installed
            
            print(f"[VENV] Installing {package}=={version}...")
            
            try:
                if progress_callback:
                    progress_callback(package, "installing")
                
                subprocess.run(
                    [pip_exe, "install", f"{package}=={version}"],
                    check=True,
                    capture_output=True,
                    timeout=300  # 5 minutes per package
                )
                
                installed.append(package)
                
                if progress_callback:
                    progress_callback(package, "success")
            except subprocess.CalledProcessError as e:
                error_msg = f"{package}: {e.stderr.decode()}"
                errors.append(error_msg)
                if progress_callback:
                    progress_callback(package, "failed")
        
        # Save cache
        cache_data = {
            "installed_packages": installed,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "creation_timestamp": str(Path(self.venv_dir).stat().st_ctime),
        }
        
        try:
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"[VENV] Warning: Failed to save cache: {e}")
        
        if errors:
            return False, errors
        else:
            print(f"[VENV] Successfully installed {len(installed)} packages")
            return True, []
    
    def get_venv_command(self, original_command: List[str]) -> List[str]:
        """
        Convert command to use venv Python instead of system Python.
        
        Args:
            original_command: Original command list (e.g., ['python', 'script.py', ...])
            
        Returns:
            Modified command list using venv Python
        """
        if not self.venv_exists():
            raise RuntimeError("Virtual environment does not exist")
        
        venv_python = self._get_venv_python()
        
        # Replace first element (Python executable) with venv Python
        if original_command and original_command[0].endswith('python.exe'):
            new_command = [venv_python] + original_command[1:]
            return new_command
        
        return original_command
    
    def verify_installation(self) -> Tuple[bool, List[str]]:
        """
        Verify all required packages are installed and working.
        
        Returns:
            (all_ok, list_of_messages)
        """
        if not self.venv_exists():
            return False, ["Virtual environment does not exist"]
        
        python_exe = self._get_venv_python()
        messages = []
        all_ok = True
        
        # Test each package
        for package in self.TRAINING_REQUIREMENTS.keys():
            test_code = f"import {package}; print({package}.__version__)"
            
            try:
                result = subprocess.run(
                    [python_exe, "-c", test_code],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    version = result.stdout.strip()
                    messages.append(f"✓ {package}: {version}")
                else:
                    messages.append(f"✗ {package}: import failed")
                    all_ok = False
            except Exception as e:
                messages.append(f"✗ {package}: {e}")
                all_ok = False
        
        return all_ok, messages
    
    def setup_training_env(self, force_recreate: bool = False) -> Tuple[bool, str]:
        """
        Complete setup: create venv and install all requirements.
        
        Args:
            force_recreate: Force recreation of venv
            
        Returns:
            (success, status_message)
        """
        # Step 1: Create venv
        if force_recreate or not self.venv_exists():
            success, msg = self.create_venv(force=force_recreate)
            if not success:
                return False, msg
        
        # Step 2: Install requirements
        print("[VENV] Installing training requirements...")
        success, errors = self.install_requirements()
        
        if not success:
            error_msg = "Failed to install some packages:\n" + "\n".join(errors)
            return False, error_msg
        
        # Step 3: Verify installation
        print("[VENV] Verifying installation...")
        all_ok, messages = self.verify_installation()
        
        if not all_ok:
            return False, "Installation verification failed:\n" + "\n".join(messages)
        
        return True, "Training environment ready"


def ensure_training_venv(base_dir: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
    """
    Utility function to ensure training venv exists and is ready.
    
    Args:
        base_dir: Base directory for venv
        
    Returns:
        (success, message, venv_python_path)
    """
    manager = VirtualEnvManager(base_dir)
    
    if not manager.venv_exists():
        print("[VENV] Training environment not found, creating...")
        success, msg = manager.setup_training_env()
        
        if not success:
            return False, msg, None
    
    # Verify it works
    all_ok, messages = manager.verify_installation()
    
    if not all_ok:
        print("[VENV] Verification failed, recreating environment...")
        success, msg = manager.setup_training_env(force_recreate=True)
        
        if not success:
            return False, msg, None
    
    venv_python = manager._get_venv_python()
    return True, "Training environment ready", venv_python
