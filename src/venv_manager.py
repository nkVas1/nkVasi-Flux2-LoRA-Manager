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
    
    # Complete isolated dependency tree for sd-scripts training
    # IMPORTANT: We use system PyTorch (already in ComfyUI) to avoid:
    # - 15+ minute download times
    # - 2-3GB disk space duplication  
    # - Version conflict complexity with torchvision
    TRAINING_REQUIREMENTS = {
        # Core ML libraries (use system versions)
        'torch': 'SKIP',
        'torchvision': 'SKIP',
        
        # Training framework packages (MUST be isolated - critical versions)
        'transformers': '4.36.2',  # Specific GenerationMixin API version
        'tokenizers': '0.15.2',  # Must match transformers 4.36.2 exactly
        'diffusers': '0.25.1',
        'accelerate': '0.25.0',
        'safetensors': '0.4.2',
        'huggingface_hub': '0.20.3',  # Has cached_download, compatible with transformers
        'peft': '0.7.1',
        
        # Utilities with fixed versions (prevent system version issues)
        'toml': '0.10.2',
        'omegaconf': '2.3.0',
        'einops': '0.7.0',
        'regex': '2023.12.25',  # Fixed version for tokenizers compatibility
        'requests': '2.31.0',  # huggingface_hub dependency
        'tqdm': '4.66.1',  # progress bars
        'pyyaml': '6.0.1',  # config files
        'filelock': '3.13.1',  # huggingface_hub dependency
        'fsspec': '2023.12.2',  # file system abstraction
        'packaging': '23.2',  # version parsing
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
        """Install training requirements (excluding torch/torchvision - use system versions)."""
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
        
        # === SKIP TORCH/TORCHVISION (Use system versions) ===
        print("[PKG-MGR] Note: Using system PyTorch (already in ComfyUI)")
        print("[PKG-MGR] Installing packages with correct versions...")
        
        # Install other packages (the ones that actually conflict)
        for package, version in self.TRAINING_REQUIREMENTS.items():
            if version == 'SKIP':
                print(f"[PKG-MGR] Skipping {package} (using system version)")
                continue
            
            # Skip bitsandbytes on Windows
            if package == 'bitsandbytes' and sys.platform == 'win32':
                print(f"[PKG-MGR] Skipping {package} (Windows compatibility)")
                continue
            
            print(f"[PKG-MGR] Installing {package}=={version}...")
            
            try:
                if progress_callback:
                    progress_callback(package, "installing")
                
                result = subprocess.run(
                    [
                        python_exe, "-m", "pip", "install",
                        f"{package}=={version}",
                        "--target", str(self.libs_dir),
                        "--no-warn-script-location",
                        "--no-deps",  # Important: don't install dependencies automatically
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180  # 3 minutes per package
                )
                
                if result.returncode == 0:
                    installed.append(package)
                    print(f"[PKG-MGR] ✓ {package} installed")
                    if progress_callback:
                        progress_callback(package, "success")
                else:
                    error_msg = f"{package}: {result.stderr[:200]}"
                    errors.append(error_msg)
                    print(f"[PKG-MGR] ✗ {package} failed: {result.stderr[:200]}")
                    if progress_callback:
                        progress_callback(package, "failed")
                        
            except subprocess.TimeoutExpired:
                errors.append(f"{package}: Timeout")
                print(f"[PKG-MGR] ✗ {package} timeout")
                if progress_callback:
                    progress_callback(package, "timeout")
            except Exception as e:
                errors.append(f"{package}: {str(e)}")
                print(f"[PKG-MGR] ✗ {package} failed: {e}")
                if progress_callback:
                    progress_callback(package, "failed")
        
        # Save cache
        cache_data = {
            "installed_packages": installed,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "note": "torch/torchvision use system packages",
        }
        
        try:
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"[PKG-MGR] Warning: Cache save failed: {e}")
        
        if errors:
            print(f"[PKG-MGR] Completed with {len(errors)} errors")
            return False, errors
        else:
            print(f"[PKG-MGR] ✓ Successfully installed {len(installed)} packages")
            return True, []
        
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
    
    def install_packages_with_ui_progress(self) -> Tuple[bool, List[str]]:
        """
        Install packages with progress updates to ComfyUI UI.
        """
        # Import PromptServer if available
        try:
            from server import PromptServer
        except ImportError:
            PromptServer = None
        
        def progress_callback(package_name, status):
            """Send progress update to UI."""
            if PromptServer:
                try:
                    message = f"[PKG] {package_name}: {status}"
                    PromptServer.instance.send_sync("flux_train_log", {"line": message})
                except Exception:
                    pass
            print(f"[PKG-MGR] {package_name}: {status}")
        
        return self.install_packages(progress_callback=progress_callback)
    
    def verify_installation(self) -> Tuple[bool, List[str]]:
        """Verify critical packages can be imported from training_libs."""
        if not self.libs_dir.exists():
            return False, ["training_libs directory does not exist"]
        
        python_exe = self._get_python_exe()
        messages = []
        all_ok = True
        
        # Critical packages to test
        critical_packages = [
            'transformers',
            'tokenizers', 
            'diffusers',
            'accelerate',
            'huggingface_hub',
            'safetensors',
        ]
        
        for package in critical_packages:
            # Test import with training_libs prioritized
            test_code = f"""
import sys
sys.path.insert(0, r'{self.libs_dir}')

try:
    import {package}
    version = getattr({package}, '__version__', 'unknown')
    location = {package}.__file__
    print(f'{package}:{{version}}:{{location}}')
except Exception as e:
    print(f'{package}:ERROR:{{str(e)}}')
"""
            
            try:
                result = subprocess.run(
                    [python_exe, "-c", test_code],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                output = result.stdout.strip()
                
                if 'ERROR' in output:
                    parts = output.split(':', 2)
                    error_detail = parts[2] if len(parts) > 2 else "Unknown error"
                    messages.append(f"✗ {package}: {error_detail[:100]}")
                    all_ok = False
                elif output:
                    parts = output.split(':', 2)
                    pkg_name = parts[0]
                    version = parts[1] if len(parts) > 1 else "?"
                    location = parts[2] if len(parts) > 2 else "?"
                    
                    # Check if loaded from training_libs
                    if 'training_libs' in location:
                        messages.append(f"✓ {pkg_name}: {version} (isolated)")
                    else:
                        messages.append(f"⚠ {pkg_name}: {version} (system - not isolated!)")
                        # This is OK for now but not ideal
                else:
                    messages.append(f"✗ {package}: No output")
                    all_ok = False
                    
            except subprocess.TimeoutExpired:
                messages.append(f"✗ {package}: Verification timeout")
                all_ok = False
            except Exception as e:
                messages.append(f"✗ {package}: {str(e)}")
                all_ok = False
        
        # Check system torch
        try:
            result = subprocess.run(
                [python_exe, "-c", "import torch; print(f'torch:{{torch.__version__}}')"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                messages.append(f"✓ {result.stdout.strip()} (system)")
            else:
                messages.append("✗ torch: Not available in system")
                all_ok = False
        except Exception:
            messages.append("✗ torch: Import test failed")
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
        
        success, errors = self.install_packages_with_ui_progress()  # With UI progress
        
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
