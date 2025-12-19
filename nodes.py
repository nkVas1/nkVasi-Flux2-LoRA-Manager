"""
ComfyUI Entry Point - Node Registration
Imports and registers all node classes from the src module
"""

from .src.config_gen import Flux2_8GB_Configurator
from .src.process import Flux2_Runner, Flux2_Stopper

# Node registration dictionary for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Flux2_8GB_Config": Flux2_8GB_Configurator,
    "Flux2_Run_External": Flux2_Runner,
    "Flux2_Stop": Flux2_Stopper
}

# Display names with emojis for UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2_8GB_Config": "🛠️ FLUX.2 Config (Low VRAM)",
    "Flux2_Run_External": "🚀 Start Training (External)",
    "Flux2_Stop": "🛑 Emergency Stop"
}
