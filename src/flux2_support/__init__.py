"""
Flux.2 Support Module
Dedicated training infrastructure for Flux.2 Dev with 6144 hidden size and 128 input channels.

This module isolates Flux.2-specific logic from the standard Flux.1 system to prevent
conflicts and maintain backward compatibility.

Components:
- flux2_models: Flux.2 architecture and configuration
- flux2_utils: Model loading and utilities adapted for Flux.2
- flux2_train: Main training entry point for Flux.2 LoRA training
"""

from . import flux2_models
from . import flux2_utils

__all__ = ['flux2_models', 'flux2_utils']
