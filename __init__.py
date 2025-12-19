"""
ComfyUI FLUX.2 LoRA Manager - Custom Node Pack Entry Point
Registers node classes and display names for ComfyUI integration
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
