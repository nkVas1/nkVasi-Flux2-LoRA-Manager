"""
ComfyUI FLUX.2 LoRA Manager - Custom Node Pack Entry Point
Registers node classes and display names for ComfyUI integration
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Specify JavaScript extension directory for Web UI
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

__version__ = "1.6.0"

# Log for debugging
print(f"[Flux2-LoRA-Manager] v{__version__} loaded")
print(f"[Flux2-LoRA-Manager] UI extensions: {WEB_DIRECTORY}")
