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


logger = setup_logging()
