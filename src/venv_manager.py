"""
Standalone Package Manager for Training Isolation (Embedded Python Compatible)

Since embedded Python lacks 'venv' module, we use direct pip install
into a separate directory and manipulate sys.path instead.
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict


class StandalonePackageManager:
    """
    Manages isolated package directory for training without using venv.
    Compatible with embedded Python.
    """
    
    # Exact versions that work with sd-scripts
    TRAINING_REQUIREMENTS = {
        'torch': '2.1.2+cu121',
        'torchvision': '0.16.2+cu121',
        'transformers': '4.36.2',
        'diffusers': '0.25.1',
        'accelerate': '0.25.0',
        'safetensors': '0.4.2',
        'toml': '0.10.2',
        'omegaconf': '2.3.0',
        'einops': '0.7.0',
        'peft': '0.7.1',
        'bitsandbytes': '0.41.3',  # Will be blocked by import_blocker
    }
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize StandalonePackageManager."""
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.base_dir = Path(base_dir)
        self.libs_dir = self.base_dir / "training_libs"
        self.cache_file = self.libs_dir / ".install_cache.json"
        
    def libs_exist(self) -> bool:
        """Check if training libs directory exists and has packages."""
        if not self.libs_dir.exists():
            return False
        
        # Check if cache file exists
        if not self.cache_file.exists():
            return False
        
        # Check if at least some packages are installed
        site_packages = self.libs_dir
        has_packages = any(site_packages.glob("*"))
        
        return has_packages
    
    def _get_python_exe(self) -> str:
        """Get current Python executable."""
        return sys.executable
    
    def create_libs_dir(self, force: bool = False) -> Tuple[bool, str]:
        """
        Create training libs directory.
        
        Args:
            force: Force recreation if exists
            
        Returns:
            (success, message)
        """
        if self.libs_exist() and not force:
            return True, "Package directory already exists"
        
        print("[PKG-MGR] Creating training package directory...")
        
        # Remove old directory if forcing
        if force and self.libs_dir.exists():
            print("[PKG-MGR] Removing old package directory...")
            try:
                shutil.rmtree(self.libs_dir)
            except Exception as e:
                return False, f"Failed to remove old directory: {e}"
        
        # Create new directory
        try:
            self.libs_dir.mkdir(parents=True, exist_ok=True)
            print(f"[PKG-MGR] Created directory: {self.libs_dir}")
        except Exception as e:
            return False, f"Failed to create directory: {e}"
        
        return True, "Package directory created"
    
    def install_packages(self, progress_callback=None) -> Tuple[bool, List[str]]:
        """
        Install training requirements directly into libs directory.
        
        Args:
            progress_callback: Optional callback(package_name, status)
            
        Returns:
            (success, list_of_errors)
        """
        if not self.libs_dir.exists():
            return False, ["Package directory does not exist"]
        
        python_exe = self._get_python_exe()
        errors = []
        installed = []
        
        # Upgrade pip first
        print("[PKG-MGR] Ensuring pip is up to date...")
        try:
            subprocess.run(
                [python_exe, "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
                timeout=120
            )
        except Exception as e:
            print(f"[PKG-MGR] Warning: Could not upgrade pip: {e}")
        
        # Install PyTorch first (with CUDA)
        print("[PKG-MGR] Installing PyTorch with CUDA support...")
        torch_spec = f"torch=={self.TRAINING_REQUIREMENTS['torch']}"
        torchvision_spec = f"torchvision=={self.TRAINING_REQUIREMENTS['torchvision']}"
        
        try:
            if progress_callback:
                progress_callback("torch", "installing")
            
            # Install to target directory
            result = subprocess.run(
                [
                    python_exe, "-m", "pip", "install",
                    torch_spec, torchvision_spec,
                    "--target", str(self.libs_dir),
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                    "--no-warn-script-location",
                ],
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes for torch
            )
            
            if result.returncode == 0:
                installed.extend(['torch', 'torchvision'])
                print("[PKG-MGR] ✓ PyTorch installed")
                if progress_callback:
                    progress_callback("torch", "success")
            else:
                error_msg = f"torch: {result.stderr}"
                errors.append(error_msg)
                print(f"[PKG-MGR] ✗ PyTorch failed: {result.stderr}")
                if progress_callback:
                    progress_callback("torch", "failed")
        except subprocess.TimeoutExpired:
            errors.append("torch: Installation timeout (15 min)")
            if progress_callback:
                progress_callback("torch", "timeout")
        except Exception as e:
            errors.append(f"torch: {e}")
            if progress_callback:
                progress_callback("torch", "failed")
        
        # Install other packages
        for package, version in self.TRAINING_REQUIREMENTS.items():
            if package in ['torch', 'torchvision']:
                continue  # Already installed
            
            # Skip bitsandbytes on Windows (causes issues)
            if package == 'bitsandbytes' and sys.platform == 'win32':
                print(f"[PKG-MGR] Skipping {package} (Windows compatibility)")
                continue
            
            print(f"[PKG-MGR] Installing {package}=={version}...")
            
            try:
                if progress_callback:
                    progress_callback(package, "installing")
                
                # Remove +cu121 suffix for regular packages
                clean_version = version.split('+')[0]
                package_spec = f"{package}=={clean_version}"
                
                result = subprocess.run(
                    [
                        python_exe, "-m", "pip", "install",
                        package_spec,
                        "--target", str(self.libs_dir),
                        "--no-warn-script-location",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes per package
                )
                
                if result.returncode == 0:
                    installed.append(package)
                    print(f"[PKG-MGR] ✓ {package} installed")
                    if progress_callback:
                        progress_callback(package, "success")
                else:
                    error_msg = f"{package}: {result.stderr}"
                    errors.append(error_msg)
                    print(f"[PKG-MGR] ✗ {package} failed")
                    if progress_callback:
                        progress_callback(package, "failed")
                        
            except subprocess.TimeoutExpired:
                errors.append(f"{package}: Installation timeout")
                if progress_callback:
                    progress_callback(package, "timeout")
            except Exception as e:
                errors.append(f"{package}: {e}")
                if progress_callback:
                    progress_callback(package, "failed")
        
        # Save cache
        cache_data = {
            "installed_packages": installed,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "python_executable": python_exe,
            "libs_directory": str(self.libs_dir),
        }
        
        try:
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"[PKG-MGR] Warning: Failed to save cache: {e}")
        
        if errors:
            print(f"[PKG-MGR] Completed with {len(errors)} errors")
            return False, errors
        else:
            print(f"[PKG-MGR] ✓ Successfully installed {len(installed)} packages")
            return True, []
    
    def get_modified_env(self, base_env: Optional[Dict] = None) -> Dict:
        """
        Get environment variables with modified PYTHONPATH for training libs.
        
        Args:
            base_env: Base environment dict (defaults to os.environ.copy())
            
        Returns:
            Modified environment dict
        """
        if base_env is None:
            env = os.environ.copy()
        else:
            env = base_env.copy()
        
        if not self.libs_dir.exists():
            return env
        
        # Add training_libs to PYTHONPATH (highest priority)
        libs_path = str(self.libs_dir.absolute())
        current_pythonpath = env.get("PYTHONPATH", "")
        
        if current_pythonpath:
            env["PYTHONPATH"] = f"{libs_path}{os.pathsep}{current_pythonpath}"
        else:
            env["PYTHONPATH"] = libs_path
        
        return env
    
    def verify_installation(self) -> Tuple[bool, List[str]]:
        """
        Verify packages are installed and importable.
        
        Returns:
            (all_ok, list_of_messages)
        """
        if not self.libs_dir.exists():
            return False, ["Package directory does not exist"]
        
        python_exe = self._get_python_exe()
        messages = []
        all_ok = True
        
        # Get modified environment
        env = self.get_modified_env()
        
        # Test critical packages
        test_packages = ['torch', 'transformers', 'diffusers', 'accelerate']
        
        for package in test_packages:
            test_code = f"""
import sys
sys.path.insert(0, r'{self.libs_dir}')
try:
    import {package}
    print(f'{package}:' + {package}.__version__)
except Exception as e:
    print(f'{package}:ERROR:' + str(e))
"""
            
            try:
                result = subprocess.run(
                    [python_exe, "-c", test_code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=env
                )
                
                output = result.stdout.strip()
                
                if 'ERROR' in output:
                    messages.append(f"✗ {output}")
                    all_ok = False
                elif output:
                    messages.append(f"✓ {output}")
                else:
                    messages.append(f"✗ {package}: no output")
                    all_ok = False
                    
            except Exception as e:
                messages.append(f"✗ {package}: {e}")
                all_ok = False
        
        return all_ok, messages
    
    def setup_training_packages(self, force_reinstall: bool = False) -> Tuple[bool, str]:
        """
        Complete setup: create directory and install packages.
        
        Args:
            force_reinstall: Force reinstallation
            
        Returns:
            (success, status_message)
        """
        # Step 1: Create directory
        if force_reinstall or not self.libs_dir.exists():
            success, msg = self.create_libs_dir(force=force_reinstall)
            if not success:
                return False, msg
        
        # Step 2: Install packages
        print("[PKG-MGR] Installing training packages...")
        print("[PKG-MGR] This may take 5-10 minutes on first run...")
        
        success, errors = self.install_packages()
        
        if not success:
            error_msg = "Failed to install some packages:\n" + "\n".join(errors[:5])  # Show first 5 errors
            return False, error_msg
        
        # Step 3: Verify
        print("[PKG-MGR] Verifying installation...")
        all_ok, messages = self.verify_installation()
        
        if not all_ok:
            return False, "Installation verification failed:\n" + "\n".join(messages)
        
        return True, "Training packages ready"


def ensure_training_packages(base_dir: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
    """
    Ensure training packages are installed and ready.
    
    Returns:
        (success, message, libs_directory_path)
    """
    manager = StandalonePackageManager(base_dir)
    
    if not manager.libs_exist():
        print("[PKG-MGR] Training packages not found, installing...")
        success, msg = manager.setup_training_packages()
        
        if not success:
            return False, msg, None
    
    # Verify installation
    all_ok, messages = manager.verify_installation()
    
    if not all_ok:
        print("[PKG-MGR] Verification failed, reinstalling...")
        success, msg = manager.setup_training_packages(force_reinstall=True)
        
        if not success:
            return False, msg, None
    
    libs_path = str(manager.libs_dir.absolute())
    return True, "Training packages ready", libs_path
