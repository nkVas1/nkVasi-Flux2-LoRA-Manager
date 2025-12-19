"""
Utility functions for path handling, logging, and common operations
"""

import os
import logging
from pathlib import Path
from typing import Union


def setup_logging(name: str = "flux2_lora") -> logging.Logger:
    """
    Setup logger instance with basic configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Args:
        path: Directory path
        
    Returns:
        Resolved Path object
    """
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: Union[str, Path]) -> Path:
    """
    Resolve path with home directory expansion.
    
    Args:
        path: Path string or Path object
        
    Returns:
        Resolved absolute path
    """
    path_obj = Path(path).expanduser()
    return path_obj.resolve()


def path_exists(path: Union[str, Path]) -> bool:
    """Check if path exists."""
    return Path(path).exists()


def safe_filename(filename: str) -> str:
    """
    Create safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Replace problematic characters
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
    return safe_name.strip("._-") or "file"


def get_file_size_mb(path: Union[str, Path]) -> float:
    """Get file size in MB."""
    return Path(path).stat().st_size / (1024 * 1024)


def is_embedded_python():
    """Check if running in embedded Python environment."""
    import sys
    # Embedded Python typically lacks 'include' directory with headers
    python_include = os.path.join(sys.prefix, 'include')
    return not os.path.exists(python_include)


def get_safe_env_for_training():
    """
    Get environment variables that prevent compilation errors
    in embedded Python.
    """
    import sys
    env = os.environ.copy()
    
    if is_embedded_python():
        # Disable packages that try to compile C extensions
        env.update({
            "BITSANDBYTES_NOWELCOME": "1",
            "DISABLE_TRITON": "1",
            "DISABLE_BITSANDBYTES_WARN": "1",
            "DIFFUSERS_DISABLE_TELEMETRY": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            # Force CPU compilation for bitsandbytes if it must load
            "BNB_CUDA_VERSION": "0",
        })
    
    return env


def patch_diffusers_imports():
    """
    Monkey-patch diffusers to avoid importing bitsandbytes/triton.
    Call this before importing diffusers in training scripts.
    """
    import sys
    import types
    import warnings
    
    # Create dummy modules to prevent import errors
    for module_name in ['bitsandbytes', 'triton']:
        if module_name not in sys.modules:
            dummy = types.ModuleType(module_name)
            dummy.__path__ = []
            sys.modules[module_name] = dummy
            warnings.warn(f"Patched {module_name} with dummy module (requires full Python dev environment)")


logger = setup_logging()
