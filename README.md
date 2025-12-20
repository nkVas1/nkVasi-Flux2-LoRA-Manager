# ComfyUI FLUX.2 LoRA Manager (Low VRAM Edition)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Extension-456789)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional ComfyUI wrapper designed specifically for training **FLUX.2/FLUX.1 LoRAs** on hardware with limited VRAM (e.g., **RTX 3060 Ti 8GB**). 

Unlike standard training nodes that crash ComfyUI by sharing memory, this extension orchestrates `kohya-ss/sd-scripts` in a **detached subprocess**, utilizing **QLoRA (NF4)**, **FP8 Base**, and **CPU Offloading** to fit training into 8GB VRAM.

## � Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete step-by-step usage guide ⭐ START HERE
- **[INFINITE_LOOP_FIX.md](INFINITE_LOOP_FIX.md)** - Fixed critical infinite loop issue in v1.5.1
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common errors and solutions
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Advanced troubleshooting guide
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes
- **[verify_installation.py](verify_installation.py)** - Pre-flight system check script

## �🚀 Features

- **🛡️ Embedded Python Protection**: Automatic import blocker system prevents compilation errors
- **Process Isolation:** Runs training in a separate system process. ComfyUI stays responsive; VRAM is dedicated 100% to training.
- **8GB VRAM Optimized:** Pre-configured presets using `bf16`, `adafactor`, `fp8_base`, and latent caching.
- **🔍 Environment Validation**: Pre-flight checks detect issues before training starts
- **Real-time Monitoring:** Streams training logs directly to the ComfyUI console window.
- **Safety First:** Emergency Stop node included to kill background processes instantly.
- **Ampere Ready:** Optimized for RTX 3000 series (supports native BF16).
- **Production-Ready:** Clean architecture, comprehensive documentation, and version control ready.

## 🛠️ Prerequisites

### System Requirements

- **Python**: 3.10 or 3.11 (embedded or full)
- **GPU**: NVIDIA with 8GB+ VRAM (RTX 3060 Ti / RTX 4060+)
- **CUDA**: 11.8 or 12.1+
- **Storage**: 20GB+ free space for models

### Compatibility

✅ Windows 10/11  
✅ Embedded Python (portable ComfyUI)  
✅ Full Python installations  
✅ Kohya-ss sd-scripts (latest)  
✅ Flux.1-dev and Flux.1-schnell models

### Required Software

1.  **Kohya SD-Scripts**: You must have [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) installed locally.
    ```bash
    git clone https://github.com/kohya-ss/sd-scripts
    cd sd-scripts
    pip install -r requirements.txt
    ```

2.  **FLUX Model**: A valid FLUX.1-dev or FLUX.2-dev model available locally or via Hugging Face.

3.  **Hardware**: NVIDIA GPU with at least **8GB VRAM** (recommended: RTX 3060 Ti or better).

## 📦 Installation

### ⚡ Quick Setup (v1.6.0 - 3-5 Minutes)

**NEW: Hybrid package isolation strategy - 75% faster setup!**

1. **Install the plugin:**
   ```bash
   cd G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\ComfyUI\custom_nodes
   git clone https://github.com/nkVasi/ComfyUI-Flux2-LoRA-Manager.git
   ```

2. **Setup training packages (3-5 min - automated):**
   ```bash
   cd ComfyUI-Flux2-LoRA-Manager
   python setup_training_env.py
   ```
   
   **What happens:**
   - ✅ Uses ComfyUI's system PyTorch (no re-download)
   - ✅ Installs only 8 conflicting packages (transformers, diffusers, etc.)
   - ✅ Completes in 3-5 minutes (vs 20+ min previously)
   - ✅ Saves 2-3GB disk space (no PyTorch duplication)
   - ✅ Shows **mega progress panel** with live updates
   
   **Expected output:**
   ```
   [Flux2-LoRA-Manager] v1.6.0 loaded
   [PKG-MGR] Skipping torch (using system version)
   [PKG-MGR] Skipping torchvision (using system version)
   [PKG-MGR] transformers: installing... (30 sec)
   [PKG-MGR] diffusers: installing... (20 sec)
   ...
   ✅ Training environment ready! (completed in 4 min 23 sec)
   ```

3. **Place sd-scripts:**
   ```
   G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\
   └── ComfyUI\
       └── kohya_train\
           └── kohya_ss\
               └── sd-scripts\    ← Clone kohya-ss/sd-scripts here
   ```

4. **Restart ComfyUI:**
   - Restart ComfyUI process
   - Refresh browser: **Ctrl+Shift+R** (hard refresh for new JS)
   - Check console: Should see `[FLUX Progress] ✓ Ready and Listening!`

### What Changed in v1.6.0?

| Aspect | v1.5.2 | v1.6.0 | Benefit |
|--------|--------|--------|---------|
| PyTorch installation | Reinstall 2.5.1 (15 min) | SKIP (use system) | 75% faster |
| Packages to install | 10+ | 8 | Simpler deps |
| Setup time | 20+ min | 3-5 min | **Much faster** |
| Disk space | ~5GB | ~2.5GB | **50% less** |
| Version conflicts | Yes | No | **Eliminated** |
| Progress UI | None | **Mega panel** | **Visual feedback** |

### Manual Setup (Advanced Users)

If automatic setup fails:

1. Create package directory:
   ```bash
   mkdir training_libs
   ```

2. Verify system PyTorch:
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__} available')"
   ```
   Should print version (if missing, install from PyTorch.org).

3. Install isolated packages:
   ```bash
   python -m pip install transformers==4.36.2 diffusers==0.25.1 --target training_libs --no-deps
   python -m pip install accelerate==0.25.0 peft==0.7.1 safetensors==0.4.0 --target training_libs --no-deps
   ```

4. Verify installation:
   ```bash
   python -c "import sys; sys.path.insert(0, 'training_libs'); import transformers; print('OK')"
   ```

### Troubleshooting Setup

**"ModuleNotFoundError: No module named 'torch'":**
- System PyTorch not found (required by v1.6.0)
- Install: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- Or use v1.5.2 branch (slower but self-contained)

**"Installation timeout (15+ min)":**
- Old v1.5.2 behavior - update to v1.6.0
- Or increase timeout: `python setup_training_env.py --timeout 900`

**"ImportError: GenerationMixin from transformers":**
- Fixed in v1.6.0 by package isolation
- If persists: `python setup_training_env.py --force`

**Progress panel not showing:**
- Hard refresh browser: **Ctrl+Shift+R**
- Check console: `F12 → Console → should see [FLUX Progress] messages`
- Clear browser cache if still not visible

## 🧩 Usage Workflow

### Step 1: Prepare Dataset
Create a folder with training images and captions:
```
dataset/
├── image_001.jpg
├── image_001.txt  (caption for image_001.jpg)
├── image_002.jpg
├── image_002.txt
└── ...
```

### Step 2: Build ComfyUI Workflow

1. Add `🛠️ FLUX.2 Config (Low VRAM)` node:
   - Set `sd_scripts_path` to your `sd-scripts` folder (e.g., `C:/AI/sd-scripts`)
   - Set `model_path` to your FLUX model (e.g., `black-forest-labs/FLUX.1-dev`)
   - Set `img_folder` to your dataset folder
   - Set `output_name` to a descriptive name (e.g., `my_character_lora`)
   - Set `resolution` to **512** (recommended for 8GB) or **768**
   - Adjust `max_train_steps` (1200–1600 typically)
   - Set `lora_rank` to **16** (safe for 8GB)

2. Connect the output `cmd_args` to:
   - `🚀 Start Training (External)` node
   - Set `trigger` to **True**

3. (Optional) Add `🛑 Emergency Stop` node set to **True** if you need to cancel training.

### Step 3: Queue and Monitor

- Click **Queue Prompt** to start.
- **Watch the terminal/console window** where ComfyUI is running — training logs will stream there in real-time.
- Do **NOT** close the terminal or ComfyUI will lose the training subprocess.

## ⚙️ Performance Estimates (RTX 3060 Ti, 8GB VRAM)

| Resolution | Rank | Steps | Batch Size | Est. Time (20 imgs) | VRAM Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **512x512** | 16 | 1200 | 1 | ~35 mins | ~7.2 GB |
| **768x768** | 16 | 1200 | 1 | ~1h 20m | ~7.8 GB |
| **1024x1024** | 16 | 1200 | 1 | ⚠️ High Risk | ~8.5 GB (May OOM) |

**Recommendations:**
- Start with **512x512** and **rank=16**
- If stable, try **768x768**
- Avoid 1024x1024 on 8GB without further optimization

## 🔧 Advanced Configuration

### Custom Train Command
If using a different trainer (SimpleTuner, AI-Toolkit, etc.), modify the `train_command` in `src/config_gen.py`:

```python
# Example for SimpleTuner:
cmd = [
    "python", "-m", "simpletuner.train",
    "--config", toml_path,
    # ... other args
]
```

### Memory Optimization Flags
The following are applied automatically for 8GB VRAM:
- `--fp8_base`: Quantizes base model to FP8
- `--gradient_checkpointing`: Reduces activation memory
- `--cache_latents_to_disk`: Offloads computed latents to disk
- `--optimizer_type adafactor`: More memory-efficient than AdamW

## 📋 Architecture Overview

```
ComfyUI-Flux2-LoRA-Manager/
├── __init__.py              # Package entry point
├── nodes.py                 # ComfyUI node registration
├── src/
│   ├── __init__.py
│   ├── config_gen.py        # Config + command generation
│   ├── process.py           # Subprocess management & nodes
│   └── utils.py             # Helper utilities
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── README.md                # This file
└── .gitignore               # Git configuration
```

### Key Classes:
- **`Flux2_8GB_Configurator`** (Node): Generates training config and command
- **`TrainingProcessManager`** (Singleton): Manages subprocess lifecycle
- **`Flux2_Runner`** (Node): Starts training
- **`Flux2_Stopper`** (Node): Emergency stop button

## 🐛 Troubleshooting

### "Script not found at ..."
- Ensure `sd_scripts_path` points to the folder containing `flux_train_network.py`
- Example: If the script is at `C:/AI/sd-scripts/flux_train_network.py`, set the path to `C:/AI/sd-scripts`

### Training crashes with OOM
- Reduce `resolution` (try 512 instead of 768)
- Reduce `lora_rank` (try 8 instead of 16)
- Ensure no other applications are using significant VRAM

### No output from training
- Check that the terminal/console window where ComfyUI is running is still visible
- Logs stream to stdout; if the console is closed, you won't see them

### Process doesn't stop when clicking "Emergency Stop"
- On Windows, it may take a few seconds for the process group to terminate
- As a last resort, use Task Manager to kill `python.exe` processes

## 📚 References

- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) — Core training framework
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — Custom nodes documentation
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — Memory-efficient optimizers
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) — Quantized LoRA background

## 📜 License

MIT License. Free for commercial and non-commercial use. See [LICENSE](LICENSE) for details.

## 🐛 Known Issues & Solutions

### Issue: `bitsandbytes.__spec__ is None`

**Root Cause:** In PyTorch 2.9+ with triton, transformers checks module availability using `importlib.util.find_spec()`. If a fake module has `__spec__ = None`, this raises `ValueError`.

**Solution (v1.7.0+):** We now use `ProperFakeModule` class that includes:
- Valid `ModuleSpec` with `origin="blocked"`
- Proper `__file__`, `__path__`, `__package__` attributes
- Passes all `importlib` checks

**Status:** ✅ **Fixed in current version**

---

### Issue: Training still fails on Windows

If you're still seeing import errors after updating:

1. **Delete cached files:**
   ```bash
   rm -rf training_libs/
   rm -rf __pycache__/
   rm -rf src/__pycache__/
   ```

2. **Restart ComfyUI** (fresh Python process)

3. **Check Python version:**
   ```bash
   python --version  # Must be 3.10-3.12
   ```

4. **Run diagnostics:**
   ```bash
   cd custom_nodes/ComfyUI-Flux2-LoRA-Manager
   python tests/test_import_blocker.py
   ```
   Should output: `✅ ALL TESTS PASSED`

---

### Success Indicators

When training starts successfully, you'll see:
```
[WRAPPER] ⚡ Installing import blockers...
[IMPORT-BLOCKER] Installing production import blockers...
[IMPORT-BLOCKER]   ✓ Blocked triton
[IMPORT-BLOCKER]   ✓ Blocked bitsandbytes
[IMPORT-BLOCKER] ✓ All blockers installed with proper __spec__
[IMPORT-BLOCKER] ✓ Triton properly blocked (importlib-verified)
[WRAPPER] ✓ Import protection verified
[WRAPPER] ✓ Training libs prioritized
[WRAPPER] ✓ transformers 4.36.2 loaded
```

Then training begins without import errors.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This extension is provided "as-is" without warranty. Training large models on limited VRAM can result in:
- Out-of-memory crashes
- System freezes
- Loss of training progress

Always backup your dataset and keep backups of partially trained models. Test with small datasets first.

## 🎓 Learn More

For in-depth explanation of the architecture and low-VRAM strategies, see the inline documentation in `src/config_gen.py` and `src/process.py`.

---

**Made with ❤️ for the ComfyUI community. Happy training!**
