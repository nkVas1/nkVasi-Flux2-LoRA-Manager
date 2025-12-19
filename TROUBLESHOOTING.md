# Troubleshooting Guide - ComfyUI FLUX.2 LoRA Manager

## Quick Diagnostics

Run this in ComfyUI console to check your environment:
```python
import sys
sys.path.append("G:/ComfyUI-StableDif-t27-p312-cu128-v2.1/ComfyUI/custom_nodes/ComfyUI-Flux2-LoRA-Manager/src")
from environment_checker import print_environment_report
print_environment_report()
```

---

## Error: "Python.h not found" / Triton compilation failure

### Root Cause
Embedded Python (portable) lacks development headers needed for C extension compilation.

### Solution: Import Blocker System (Automatic - v1.3+)

**v1.3+ automatically blocks problematic imports**. If you see this error, the blocker failed to activate.

#### Automatic Fix (Recommended)

The new Import Blocker System handles this automatically:
- ✓ Blocks triton/bitsandbytes before they try to compile
- ✓ Creates dummy modules so imports don't fail
- ✓ Patches diffusers to skip quantizer imports
- ✓ Works automatically on every training run

**What to do if you still see the error:**

1. **Verify blocker is installed:**
   ```python
   from src.import_blocker import verify_blockers_active
   verify_blockers_active()
   ```

2. **Check console output for "[IMPORT-BLOCKER]" messages:**
   - `[IMPORT-BLOCKER] Installed: triton, bitsandbytes imports will be blocked`
   - `[IMPORT-BLOCKER] Verification passed`

3. **If blocker failed, use full Python:**
   - Download: [python.org](https://www.python.org/downloads/windows/)
   - Version: Python 3.10 or 3.11 (not 3.12 yet)
   - During install: ✓ "Add Python to PATH"
   
4. **Update ComfyUI to use full Python:**
   Edit `run_nvidia_gpu.bat`:
   ```batch
   set PYTHON=C:\Python310\python.exe
   %PYTHON% main.py
   ```

### Prevention

The import blocker system uses Python's meta_path hooks to intercept imports **before** C compilation is attempted. This is the same technique used by:
- PyTorch Lightning
- Hugging Face Transformers
- Ray (distributed computing)

**How it works:**

1. Wrapper script imports `import_blocker` first
2. `install_import_blockers()` adds hooks to `sys.meta_path`
3. Any attempt to `import triton` or `import bitsandbytes` returns dummy module
4. Training continues without C compilation errors

---

## Error: "can't open file 'flux_train_network.py': No such file or directory"

### Problem Description
When attempting to start training, you see an error like:
```
[Errno 2] No such file or directory: 'flux_train_network.py'
CRITICAL ERROR: Training script not found
```

### Root Causes
1. `sd-scripts` repository is not installed or in the wrong location
2. Path configuration in the Configurator node is incorrect
3. ComfyUI's Python environment doesn't have access to the sd-scripts location

### Solution

#### Step 1: Verify sd-scripts Installation

Check if `flux_train_network.py` exists:
```powershell
# On Windows - check if sd-scripts exists in ComfyUI root
cd G:\ComfyUI-StableDif-t27-p312-cu128-v2.1
dir sd-scripts\flux_train_network.py
```

Expected output should show the file exists.

#### Step 2: Download sd-scripts (if not present)

If sd-scripts is not installed, clone it:

```powershell
cd G:\ComfyUI-StableDif-t27-p312-cu128-v2.1

# Clone the repository
git clone https://github.com/kohya-ss/sd-scripts.git

# Install dependencies (using your ComfyUI Python environment)
cd sd-scripts
python.exe -m pip install -r requirements.txt
```

#### Step 3: Verify Directory Structure

After installation, your directory structure should look like:
```
G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\
├── sd-scripts\
│   ├── flux_train_network.py          ← Must exist here
│   ├── library\                        ← Kohya's library module
│   │   ├── device_utils.py
│   │   ├── train_util.py
│   │   └── ...
│   ├── requirements.txt
│   └── ...
├── ComfyUI\
│   ├── main.py
│   └── custom_nodes\
│       └── ComfyUI-Flux2-LoRA-Manager\
│           └── (this node's files)
└── kohya_train\                       ← Alternative layout
    └── kohya_ss\
        └── sd-scripts\                ← Or here
```

#### Step 4: Configure the Configurator Node

In ComfyUI, use the **Flux2_8GB_Configurator** node:

1. Set `sd_scripts_path` to the **full path** to your sd-scripts folder:
   - Default: `G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\sd-scripts`
   - Or wherever you installed sd-scripts

2. Verify other paths:
   - `model_path`: Should be a valid FLUX.1 model path or HuggingFace model ID
   - `img_folder`: Should point to your training images
   - `output_name`: Name for your LoRA output

#### Step 5: Check Console Output

After configuration, look at the console for debug messages:
```
[DEBUG] Working Dir: G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\sd-scripts
[DEBUG] PYTHONPATH: G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\sd-scripts;...
[DEBUG] Looking for: G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\sd-scripts\library
```

If these paths don't match your actual sd-scripts location, update the configurator.

---

## Error: "Failed to find Python libs" or "Python.h not found" (Triton/bitsandbytes)

### Problem Description
When training starts, you see errors like:
```
RuntimeError: Failed to find Python libs (Python.h)
error: Microsoft Visual C++ 14.0 is required
fatal error C1083: Cannot open include file: 'Python.h'
```

### Root Cause
**Embedded Python** (portable version) doesn't include header files (`Python.h`) needed for compiling C extensions. Triton and bitsandbytes libraries try to compile during import, which fails.

### Solution

#### Automatic (v1.2.6+)
The system now automatically:
1. ✓ Detects embedded Python environment
2. ✓ Disables problematic modules (Triton, bitsandbytes)
3. ✓ Sets environment variables to prevent compilation attempts
4. ✓ Uses pure Python alternatives

**No action needed!** The wrapper script includes:
```python
os.environ["DISABLE_TRITON"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```

#### If Issue Persists: Manual Fix

**Option 1: Install Full Python (Recommended)**

1. Download full Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. Install with "Add Python to PATH" checked
3. Install dependencies:
   ```powershell
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   python -m pip install transformers diffusers accelerate safetensors peft
   ```
4. Update Configurator to use full Python

**Option 2: Skip Quantization**

If you don't need 8-bit quantization:
1. Edit `sd-scripts/library/train_util.py`
2. Comment out `BitsAndBytesConfig` imports
3. Training will use standard precision

**Option 3: Verify Environment**

Check if automatic fix worked:
```powershell
# Test automatic disabling
$env:DISABLE_TRITON = "1"
python.exe -c "import os; print(os.environ.get('DISABLE_TRITON'))"
```

---

## Error: "ModuleNotFoundError: No module named 'library'"

### Problem Description
Script launches but crashes on import:
```
ModuleNotFoundError: No module named 'library'
ModuleNotFoundError: No module named 'diffusers'
```

### Solutions

1. **Ensure sd-scripts is in the correct location** (see above)

2. **Install Kohya dependencies:**
   ```powershell
   cd G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\sd-scripts
   python.exe -m pip install -r requirements.txt
   ```

3. **Install missing specific packages:**
   ```powershell
   python.exe -m pip install diffusers peft bitsandbytes transformers accelerate
   ```

4. **Check Python version** (requires Python 3.10+):
   ```powershell
   python.exe --version
   ```

---

## Error: "No such file or directory: 'library'"

### Root Cause
Similar to above - Python isn't finding the Kohya library module because:
- Working directory is wrong
- PYTHONPATH not set correctly
- sd-scripts not installed

### Solution
1. Verify sd-scripts installation (Step 1 above)
2. Ensure full path to sd-scripts in configurator
3. Check that `library` folder exists:
   ```powershell
   dir G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\sd-scripts\library
   ```

---

## Error: "ACCELERATE_MIXED_PRECISION must be one of..."

### Problem
Accelerate configuration error.

### Solution
Update `ACCELERATE_MIXED_PRECISION` environment variable:
```powershell
# Set valid value (no, fp16, bf16, fp8)
$env:ACCELERATE_MIXED_PRECISION = "bf16"
```

The code already does this, but if errors persist, this confirms the setting.

---

## VRAM Out of Memory (OOM) Errors

### Optimization Checklist

1. **Lower resolution** in configurator:
   - Start with `512` or `768` instead of `1024`

2. **Lower batch size** (already set to 1, which is minimum)

3. **Enable caching**:
   - Set `cache_to_disk` to `True` in configurator
   - Reduces active VRAM usage

4. **Check lora_rank**:
   - Use `16` instead of `32` for tighter VRAM budget
   - Smaller rank = less trainable parameters = less memory

5. **Monitor during training**:
   - Open GPU monitor (NVIDIA Settings) to see actual VRAM usage
   - First iteration usually shows peak VRAM usage

---

## Diagnostic Commands

### Check Python and Environment
```powershell
python.exe --version
python.exe -c "import sys; print(sys.executable)"
python.exe -c "import torch; print(torch.cuda.is_available())"
python.exe -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Test Kohya Installation
```powershell
cd G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\sd-scripts
python.exe -c "import library.device_utils; print('✓ Kohya installed correctly')"
```

### Check sd-scripts Directory
```powershell
# List contents to verify structure
dir G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\sd-scripts

# Specifically check for key files
dir G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\sd-scripts\flux_train_network.py
dir G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\sd-scripts\library
```

### Test PYTHONPATH
```powershell
python.exe -c "import sys; print('\\n'.join(sys.path))"
```

---

## Getting Help

When reporting issues, provide:

1. **Full error message** from the console
2. **Directory structure** of your installation:
   ```powershell
   tree G:\ComfyUI-StableDif-t27-p312-cu128-v2.1 /L 3
   ```
3. **Configuration** from your Flux2_8GB_Configurator node
4. **Python version and GPU info**:
   ```powershell
   python.exe --version
   python.exe -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python.exe -c "import torch; print(f'CUDA: {torch.version.cuda}')"
   python.exe -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
   ```

---

## Common Installation Paths

Different systems may have sd-scripts in different locations. The node searches these paths automatically:

1. `{ComfyUI}/sd-scripts` ← **Recommended**
2. `{ComfyUI}/kohya_ss/sd-scripts`
3. `{ComfyUI}/kohya_train/kohya_ss/sd-scripts`
4. `{ComfyUI}/custom_nodes/sd-scripts`
5. One level up from ComfyUI: `{parent}/sd-scripts`

If your installation is elsewhere, **always specify the full path** in the `sd_scripts_path` parameter of the Configurator node.

---

## Advanced Debugging

Enable more verbose output:

```powershell
# Set environment variables for debug output
$env:PYTHONUNBUFFERED = "1"
$env:ACCELERATE_DEBUG = "1"

# Then start ComfyUI as normal
```

Check ComfyUI logs in:
- `{ComfyUI}/log/` directory
- Console output where you started ComfyUI

---

## Quick Fix Checklist

- [ ] sd-scripts installed in `{ComfyUI}/sd-scripts`?
- [ ] `flux_train_network.py` file exists?
- [ ] `library/` folder exists in sd-scripts?
- [ ] Configurator `sd_scripts_path` set to correct location?
- [ ] Dependencies installed: `pip install -r requirements.txt` in sd-scripts?
- [ ] Python version 3.10+?
- [ ] GPU available and recognized?
- [ ] VRAM sufficient (8GB minimum)?

If all checkboxes are marked, training should work!
