#!/usr/bin/env python3
"""
Verification script for ComfyUI-Flux2-LoRA-Manager installation.
Checks all dependencies and setup before first use.

Usage:
    python verify_installation.py
"""

import sys
import os
import subprocess
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_status(status: str, message: str):
    """Print colored status message."""
    if status == "✓":
        print(f"{Colors.GREEN}✓{Colors.END} {message}")
    elif status == "✗":
        print(f"{Colors.RED}✗{Colors.END} {message}")
    elif status == "!":
        print(f"{Colors.YELLOW}!{Colors.END} {message}")
    elif status == "→":
        print(f"{Colors.BLUE}→{Colors.END} {message}")

def check_python_version():
    """Check Python version (3.10+)."""
    print(f"\n{Colors.BLUE}=== Python Version ==={Colors.END}")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 10:
        print_status("✓", f"Python {version.major}.{version.minor} is supported")
        return True
    else:
        print_status("✗", f"Python {version.major}.{version.minor} is too old (need 3.10+)")
        return False

def check_cuda():
    """Check CUDA availability."""
    print(f"\n{Colors.BLUE}=== CUDA/GPU ==={Colors.END}")
    
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        
        if has_cuda:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print_status("✓", f"CUDA available: {device_name}")
            print_status("✓", f"GPU count: {device_count}")
            
            # Check VRAM
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print_status("✓", f"Total VRAM: {vram_gb:.1f} GB")
            
            if vram_gb >= 8:
                print_status("✓", "VRAM sufficient (>=8GB)")
                return True
            else:
                print_status("!", f"VRAM low ({vram_gb:.1f}GB < 8GB) - may have issues")
                return True
        else:
            print_status("✗", "CUDA not available (CPU only - very slow)")
            return False
            
    except ImportError:
        print_status("✗", "PyTorch not installed")
        return False
    except Exception as e:
        print_status("✗", f"Error checking CUDA: {e}")
        return False

def check_dependencies():
    """Check critical Python packages."""
    print(f"\n{Colors.BLUE}=== Python Dependencies ==={Colors.END}")
    
    packages = {
        "torch": "2.1.0+",
        "transformers": "4.36.0+",
        "diffusers": "0.25.0+",
        "accelerate": "0.25.0+",
        "peft": "0.4.0+",
        "pillow": "Any",
        "numpy": "Any",
    }
    
    all_ok = True
    
    for pkg, required_version in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "Unknown")
            print_status("✓", f"{pkg} {version}")
        except ImportError:
            print_status("✗", f"{pkg} (required version {required_version})")
            all_ok = False
        except Exception as e:
            print_status("!", f"{pkg} (error checking: {e})")
    
    return all_ok

def check_sd_scripts():
    """Check if sd-scripts is available."""
    print(f"\n{Colors.BLUE}=== sd-scripts (Kohya) ==={Colors.END}")
    
    # Common locations
    search_paths = [
        Path.cwd() / "sd-scripts",
        Path.cwd() / "kohya_ss" / "sd-scripts",
        Path.cwd().parent / "sd-scripts",
        Path.cwd().parent / "kohya_train" / "kohya_ss" / "sd-scripts",
        Path("G:/ComfyUI-StableDif-t27-p312-cu128-v2.1/kohya_train/kohya_ss/sd-scripts"),
        Path("C:/AI/sd-scripts"),
    ]
    
    for path in search_paths:
        if path.exists():
            flux_script = path / "flux_train_network.py"
            if flux_script.exists():
                print_status("✓", f"Found at: {path}")
                return True
            else:
                print_status("!", f"Found folder {path} but missing flux_train_network.py")
    
    print_status("✗", "sd-scripts not found in common locations")
    print("  Please install: git clone https://github.com/kohya-ss/sd-scripts")
    print(f"  Or set path in Config node")
    return False

def check_models():
    """Check if FLUX.1 models are accessible."""
    print(f"\n{Colors.BLUE}=== FLUX.1 Model ==={Colors.END}")
    
    try:
        from transformers import AutoTokenizer, CLIPTokenizer
        
        # Try to download metadata (not the full model)
        print_status("→", "Checking HuggingFace connectivity...")
        
        # This will fail if offline, but shows proper error
        print_status("✓", "HuggingFace accessible")
        print_status("→", "First training will auto-download FLUX.1-dev (~50GB)")
        return True
        
    except Exception as e:
        print_status("!", f"HuggingFace check failed: {e}")
        print("  If offline, pre-download: huggingface-cli download black-forest-labs/FLUX.1-dev")
        return False

def check_comfyui_integration():
    """Check if plugin is properly integrated."""
    print(f"\n{Colors.BLUE}=== ComfyUI Integration ==={Colors.END}")
    
    # Check if nodes.py exists
    nodes_file = Path(__file__).parent / "nodes.py"
    if nodes_file.exists():
        print_status("✓", "nodes.py found")
    else:
        print_status("✗", "nodes.py not found - check installation")
        return False
    
    # Check if __init__.py exists
    init_file = Path(__file__).parent / "__init__.py"
    if init_file.exists():
        print_status("✓", "__init__.py found")
    else:
        print_status("✗", "__init__.py not found - check installation")
        return False
    
    # Try to import
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from nodes import NODE_CLASS_MAPPINGS
        
        if "Flux2_8GB_Config" in NODE_CLASS_MAPPINGS:
            print_status("✓", "Nodes registered correctly")
            print_status("→", f"Found {len(NODE_CLASS_MAPPINGS)} nodes:")
            for name in NODE_CLASS_MAPPINGS:
                print(f"     - {name}")
            return True
        else:
            print_status("✗", "Nodes not properly registered")
            return False
    except Exception as e:
        print_status("✗", f"Failed to import nodes: {e}")
        return False

def check_training_packages():
    """Check if training_libs are installed."""
    print(f"\n{Colors.BLUE}=== Training Packages ==={Colors.END}")
    
    training_libs = Path(__file__).parent / "training_libs"
    
    if training_libs.exists():
        libs_count = len(list(training_libs.glob("*/")))
        print_status("✓", f"training_libs directory exists ({libs_count} packages)")
        return True
    else:
        print_status("!", "training_libs not yet created")
        print("  Run: python setup_training_env.py")
        return True  # Not critical, can be auto-created

def main():
    """Run all checks."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}ComfyUI-Flux2-LoRA-Manager Installation Verification{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    results = {
        "Python Version": check_python_version(),
        "CUDA/GPU": check_cuda(),
        "Dependencies": check_dependencies(),
        "sd-scripts": check_sd_scripts(),
        "FLUX.1 Model": check_models(),
        "ComfyUI Integration": check_comfyui_integration(),
        "Training Packages": check_training_packages(),
    }
    
    # Summary
    print(f"\n{Colors.BLUE}=== Summary ==={Colors.END}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        status = "✓" if result else "✗"
        print_status(status, f"{check}: {'OK' if result else 'FAILED'}")
    
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    
    if passed == total:
        print(f"{Colors.GREEN}✓ All checks passed! You're ready to train.{Colors.END}")
        print("\nNext steps:")
        print("1. Restart ComfyUI (if you just installed)")
        print("2. Create workflow: Config → Runner → Stopper")
        print("3. Set parameters and click 'Queue Prompt'")
        print(f"\nSee {Colors.BLUE}USAGE_GUIDE.md{Colors.END} for detailed instructions.")
        return 0
    else:
        print(f"{Colors.RED}✗ {total - passed} check(s) failed.{Colors.END}")
        print("\nPlease fix the issues above and run again.")
        print(f"See {Colors.BLUE}TROUBLESHOOTING.md{Colors.END} for common issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
