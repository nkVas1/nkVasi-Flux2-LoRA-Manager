# 🚀 ComfyUI Restart & Training Guide

**After applying ETAP A & B updates**

---

## Step 1: Restart ComfyUI

Close ComfyUI completely and restart it. This is **critical** because:

1. Python needs to reload all modules
2. The `import_blocker.py` installation happens **before** torch imports
3. Old module cache will interfere with xformers blocking

```bash
# If using portable ComfyUI:
1. Close ComfyUI window
2. Run: run_nvidia_gpu.bat (or your startup script)
3. Wait for "Server started at" message

# If using ComfyUI Manager:
1. Restart → ComfyUI
```

### Verify Restart Success

Look for these messages in console:
```
[IMPORT-BLOCKER] Installing production import blockers (Package-Aware)...
[IMPORT-BLOCKER]   ✓ Blocked triton (package)
[IMPORT-BLOCKER]   ✓ Blocked xformers (package)
[IMPORT-BLOCKER] ✓ All blockers installed with package support
```

---

## Step 2: Verify in ComfyUI UI

After restart, you should see new nodes in ComfyUI:

**In "Add Node" menu → "FluxTrainer" category:**
- 🤖 Flux Model Selector
- 📁 Flux Dataset Config
- 🔍 Flux Validation Settings
- ⚙️ Init Flux LoRA Training

**Still available (legacy):**
- 🛠️ FLUX.2 Config (Low VRAM)
- 🚀 Start Training (External)
- 🛑 Emergency Stop

---

## Step 3: Run Training with OLD Node (Test)

Before using new nodes, test that the xformers fix works:

1. **Add the legacy "FLUX.2 Config" node**
2. **Configure it normally** (dataset path, learning rate, etc.)
3. **Click "Run Training"**

**What should happen:**
```
✅ No error: "ImportError: DLL load failed while importing _C_flashattention"
✅ Training starts normally
✅ diffusers uses native PyTorch SDPA attention (no xformers)
✅ Performance is identical to xformers version
```

**If you see warnings like:**
```
WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions...
```
✅ This is NORMAL! It means xformers is blocked (as intended)  
✅ diffusers detected the block and switched to SDPA  
✅ Training continues normally

---

## Step 4: Build NEW Node Architecture (Optional)

Once old node works, try the new modular nodes:

### Simple Pipeline

```
FluxTrainModelSelect
  ↓ (output: TRAIN_FLUX_MODELS)
  ├─→ InitFluxLoRATraining ←─ FluxTrainDatasetConfig
              ↓ (output: NETWORKTRAINER)
              └─→ Flux2_Run_External
```

### How to Build

1. **Right-click canvas → Add Node**
2. **Search "Flux Model Selector"** → Add it
   - Set transformer: "flux1-dev.safetensors"
   - Enable FP8: True (for VRAM saving)

3. **Add "Flux Dataset Config"** node
   - image_dir: "path/to/your/images"
   - resolution: 1024
   - batch_size: 1

4. **Add "Init Flux LoRA Training"** node
   - Connect "flux_models" from Model Selector
   - Connect "dataset" from Dataset Config
   - Set max_train_steps: 1000
   - Set learning_rate: 0.0001
   - Set lora_name: "my_style_lora"

5. **Connect to legacy "Flux2_Run_External"** (for now)

6. **Queue the workflow**

---

## 🔍 Troubleshooting

### Issue: Still seeing xformers errors

**Solution:**
1. Fully close ComfyUI (check Task Manager - no python.exe)
2. Reopen ComfyUI
3. Check console for `[IMPORT-BLOCKER]` messages
4. If no messages, import_blocker may not be loaded - check `process.py` wrapper

### Issue: New nodes not appearing

**Solution:**
1. Update ComfyUI (some versions don't register new nodes immediately)
2. Or: Restart ComfyUI Server → Restart API
3. Clear browser cache: `Ctrl+Shift+Delete` → Clear browser cache

### Issue: Training fails with "module not found"

**Solution:**
1. This might be diffusers trying to import xformers despite our block
2. Check that `src/import_blocker.py` contains xformers in targets list
3. Verify with: `python tests/test_import_blocker.py`

### Issue: Old workflow still uses old format

**Solution:**
- Old workflows will continue to work!
- No need to convert, but you can gradually migrate to new nodes

---

## ✅ Success Indicators

You'll know everything is working when:

1. ✅ ComfyUI restarts without xformers errors
2. ✅ Legacy training node completes a full training run
3. ✅ New nodes appear in "Add Node" menu
4. ✅ New nodes can be connected without type errors
5. ✅ Console shows `[IMPORT-BLOCKER] ✓ All blockers installed`

---

## 📚 Further Documentation

- **Full Technical Details:** `docs/XFORMERS_FIX_AND_OPENART_NODES.md`
- **API Documentation:** Look for `TRAIN_FLUX_MODELS`, `TRAIN_DATASET` types
- **Architecture Design:** See node docstrings in `nodes.py`

---

## 🎯 What's Next?

Once new nodes are working:

1. **For Immediate Use:** Use with legacy `Flux2_Run_External` node
2. **For Future:** We'll add `FluxTrainExecutor` that runs directly from new nodes
3. **For Advanced:** Additional nodes for merging, testing, checkpointing

---

*Ready to go? Restart ComfyUI and enjoy the improved architecture!* 🚀
