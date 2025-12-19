"""
Environment Checker - Validates Python environment before training

Checks for:
- Python version compatibility
- Required packages
- GPU availability
- Embedded Python limitations
"""

import sys
import os
import subprocess
from typing import Dict, List, Tuple


class EnvironmentChecker:
    """Validates environment and provides actionable fixes."""
    
    MIN_PYTHON_VERSION = (3, 10)
    MAX_PYTHON_VERSION = (3, 12)
    
    @classmethod
    def check_python_version(cls) -> Tuple[bool, str]:
        """Check if Python version is compatible."""
        version = sys.version_info[:2]
        
        if version < cls.MIN_PYTHON_VERSION:
            return False, f"Python {version[0]}.{version[1]} too old (need {cls.MIN_PYTHON_VERSION[0]}.{cls.MIN_PYTHON_VERSION[1]}+)"
        
        if version > cls.MAX_PYTHON_VERSION:
            return False, f"Python {version[0]}.{version[1]} too new (max {cls.MAX_PYTHON_VERSION[0]}.{cls.MAX_PYTHON_VERSION[1]})"
        
        return True, f"Python {version[0]}.{version[1]} OK"
    
    @classmethod
    def is_embedded_python(cls) -> bool:
        """Detect if running in embedded Python (no dev headers)."""
        # Embedded Python lacks 'include' directory
        include_dir = os.path.join(sys.prefix, 'include')
        has_headers = os.path.exists(os.path.join(include_dir, 'Python.h'))
        
        return not has_headers
    
    @classmethod
    def check_gpu(cls) -> Tuple[bool, str]:
        """Check if CUDA GPU is available."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                return True, f"GPU: {gpu_name}"
            else:
                return False, "No CUDA GPU detected"
        except ImportError:
            return False, "PyTorch not installed"
        except Exception as e:
            return False, f"GPU check failed: {e}"
    
    @classmethod
    def check_required_packages(cls) -> Dict[str, bool]:
        """Check if required packages are installed."""
        required = {
            'torch': False,
            'transformers': False,
            'diffusers': False,
            'accelerate': False,
            'safetensors': False,
            'toml': False,
        }
        
        for package in required.keys():
            try:
                __import__(package)
                required[package] = True
            except ImportError:
                pass
        
        return required
    
    @classmethod
    def run_full_check(cls) -> Tuple[bool, List[str]]:
        """
        Run all environment checks.
        Returns: (all_ok, list_of_messages)
        """
        messages = []
        all_ok = True
        
        # Python version
        ok, msg = cls.check_python_version()
        messages.append(f"{'✓' if ok else '✗'} {msg}")
        if not ok:
            all_ok = False
        
        # Embedded Python detection
        if cls.is_embedded_python():
            messages.append("⚠ Embedded Python detected (import blocker will be used)")
        else:
            messages.append("✓ Full Python installation detected")
        
        # GPU check
        ok, msg = cls.check_gpu()
        messages.append(f"{'✓' if ok else '✗'} {msg}")
        if not ok:
            all_ok = False
        
        # Package check
        packages = cls.check_required_packages()
        missing = [pkg for pkg, installed in packages.items() if not installed]
        
        if missing:
            messages.append(f"✗ Missing packages: {', '.join(missing)}")
            all_ok = False
        else:
            messages.append("✓ All required packages installed")
        
        return all_ok, messages


def print_environment_report():
    """Print a detailed environment report."""
    print("=" * 60)
    print("FLUX TRAINING ENVIRONMENT CHECK")
    print("=" * 60)
    
    all_ok, messages = EnvironmentChecker.run_full_check()
    
    for msg in messages:
        print(msg)
    
    print("=" * 60)
    
    if all_ok:
        print("✓ Environment check PASSED")
    else:
        print("✗ Environment check FAILED - see messages above")
        print("\nRecommended fixes:")
        print("1. Install missing packages: pip install torch transformers diffusers accelerate")
        print("2. Verify CUDA installation: nvidia-smi")
        print("3. Check Python version: python --version")
    
    print("=" * 60)
    
    return all_ok
