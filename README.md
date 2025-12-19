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

### Quick Setup (Recommended)

1. **Install the plugin:**
   ```bash
   cd G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\ComfyUI\custom_nodes
   git clone https://github.com/nkVasi/ComfyUI-Flux2-LoRA-Manager.git
   ```

2. **Setup training packages:**
   ```bash
   cd ComfyUI-Flux2-LoRA-Manager
   python setup_training_env.py
   ```
   
   This creates `training_libs/` with correct dependency versions.
   Works with embedded Python (no venv needed).
   Typical time: 5-10 minutes (downloads ~2GB).

3. **Place sd-scripts:**
   ```
   G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\
   └── ComfyUI\
       └── kohya_train\
           └── kohya_ss\
               └── sd-scripts\    ← Clone here
   ```

4. **Restart ComfyUI and refresh browser**

### Manual Setup (Advanced)

If automatic setup fails:

1. Create package directory manually:
   ```bash
   mkdir training_libs
   ```

2. Install requirements:
   ```bash
   python -m pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --target training_libs --index-url https://download.pytorch.org/whl/cu121
   python -m pip install transformers==4.36.2 diffusers==0.25.1 accelerate==0.25.0 --target training_libs
   ```

3. Verify installation:
   ```bash
   python -c "import sys; sys.path.insert(0, 'training_libs'); import torch; print(torch.cuda.is_available())"
````
   python -c "from transformers import CLIPTextModel; print('OK')"
   ```

### Troubleshooting Setup

**"Failed to create venv":**
- Ensure Python 3.10-3.11 installed
- Run as administrator (Windows)
- Check disk space (need 5GB+)

**"Package installation timeout":**
- Check internet connection
- Use VPN if PyPI blocked
- Try: `python setup_training_env.py` again

**"ImportError: GenerationMixin":**
- This is fixed by venv isolation
- If error persists: `python setup_training_env.py --force`

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
