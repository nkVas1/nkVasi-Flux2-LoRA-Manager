# Migration Guide v1.8 - Configuration System Overhaul

## 🔴 Critical Issue Fixed

**Error:** `voluptuous.error.MultipleInvalid: extra keys not allowed @ data['datasets']['enable_bucket_reso_steps']`

This error occurred because the configuration structure was incorrect. The parameter `enable_bucket_reso_steps` does not exist in kohya-ss/sd-scripts. The correct parameter is `enable_bucket`, and it belongs in the `[general]` section, not in the `[[datasets]]` section.

## ✅ What Changed?

### Configuration Structure (dataset.toml)

**Before (❌ Invalid):**
```toml
[general]
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = 768
batch_size = 1
enable_bucket_reso_steps = true  # ❌ INVALID PARAMETER
caption_extension = ".txt"        # ❌ WRONG LOCATION
image_dir = "/path/images"        # ❌ WRONG LOCATION

[datasets.subsets]
# Missing required subsets structure
```

**After (✅ Valid):**
```toml
[general]
shuffle_caption = true
keep_tokens = 1
enable_bucket = true              # ✅ CORRECT PARAMETER

[[datasets]]
resolution = 768
batch_size = 1
bucket_reso_steps = 64
bucket_no_upscale = false

[[datasets.subsets]]
image_dir = "/path/images"        # ✅ CORRECT LOCATION
num_repeats = 10                  # ✅ CORRECT LOCATION
caption_extension = ".txt"        # ✅ CORRECT LOCATION
```

## 📋 Action Required

### Step 1: Delete Old Configuration Files

Navigate to your output directory and delete old `dataset.toml` files:

```
ComfyUI/output/flux_training/YOUR_PROJECT_NAME/dataset.toml
```

These files will be regenerated with the correct structure.

### Step 2: Regenerate Configuration

1. Open ComfyUI workflow with `FLUX.2 Config (Low VRAM)` node
2. Optionally set `num_repeats` in the optional inputs (default: 10)
3. Click **Queue Prompt** to regenerate `dataset.toml`
4. New config will have the correct structure

### Step 3: Validate Configuration (Optional)

To verify your configuration is correct:

1. Add new **✅ FLUX.2 Config Validator** node to workflow
2. Set `toml_path` to path of your `dataset.toml`
3. Optionally enable `auto_fix` for automatic repair
4. Click **Queue Prompt**

Expected output: `✅ Configuration is valid! Ready for training.`

## 🚀 New Features

### 1. Configuration Validator Node

**Path:** ComfyUI UI → Add Node → Flux2 → ✅ FLUX.2 Config Validator

**Features:**
- ✅ Validates configuration against official sd-scripts schema
- ❌ Detects deprecated parameters (enable_bucket_reso_steps, seed, etc.)
- 🔧 Auto-fix capability for common errors
- 💾 Creates backup of original config before fixing
- 📊 Detailed error messages with exact locations

**Usage:**
```
Set toml_path → Enable auto_fix (optional) → Queue Prompt
```

### 2. Configurable num_repeats

Previously `num_repeats` was hardcoded to 10 epochs.

Now it's configurable in `FLUX.2 Config (Low VRAM)` node:
- **Default:** 10 epochs
- **Range:** 1-100 epochs
- **Usage:** Set in optional inputs

### 3. Auto-Install of Missing Libraries

If `toml` library is missing, it will be automatically installed:
- Shows progress: `[CONFIG-GEN] ⚠ toml library not found, installing...`
- Completes silently
- Saves config with newly installed library

## 📊 Parameter Mapping

### Deprecated Parameters (Removed)

| Old Parameter | Location | Issue | Solution |
|---|---|---|---|
| `enable_bucket_reso_steps` | `[[datasets]]` | Invalid in sd-scripts | Use `enable_bucket` in `[general]` |
| `seed` | `[general]` | Causes validation errors | Use `--seed` command-line argument |
| `keep_tokens` | `[[datasets.subsets]]` | Wrong location | Remove (use in subsets only) |

### Moved Parameters

| Parameter | Old Location | New Location | Reason |
|---|---|---|---|
| `enable_bucket` | `[[datasets]]` | `[general]` | Training-wide setting |
| `image_dir` | `[[datasets]]` | `[[datasets.subsets]]` | Dataset-specific |
| `caption_extension` | `[[datasets]]` | `[[datasets.subsets]]` | Dataset-specific |
| `num_repeats` | Hardcoded (10) | `[[datasets.subsets]]` | Now configurable |

### New Parameters

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `bucket_no_upscale` | `[[datasets]]` | `false` | Allow upscaling for bucketing |

## 🔧 Troubleshooting

### Issue: Config Validator shows "Unknown parameter"

**Solution:** Run validator with `auto_fix = true` to repair automatically.

### Issue: After auto-fix, still getting validation error

**Solution:** Check the error message for the exact parameter and location. You may need to regenerate config from scratch:
1. Delete dataset.toml
2. Clear output directory
3. Regenerate with updated Config node

### Issue: Training fails with "No module named toml"

**Solution:** Library will auto-install on next config generation. If that fails:
```bash
pip install toml
```

## 📚 Technical Details

### Configuration Schema

Follows official kohya-ss/sd-scripts format:
- Reference: https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-en.md
- Validated using voluptuous library
- Supports multiple datasets and subsets
- Compatible with all sd-scripts training scripts

### Validator Implementation

**File:** `src/config_validator.py`

- Comprehensive parameter validation
- Automatic error detection and repair
- Support for multiple datasets/subsets
- Backward compatibility with old configs

## 🎯 What This Fixes

✅ **Resolves training startup error:** "extra keys not allowed @ data['datasets']['enable_bucket_reso_steps']"
✅ **Proper sd-scripts compatibility:** Follows official schema exactly
✅ **Better user experience:** Clear error messages and auto-fix
✅ **Configuration reusability:** Validator node makes troubleshooting easy

## ✨ Backward Compatibility

- ✅ Old workflows still work (just regenerate config)
- ✅ No breaking changes to existing APIs
- ✅ Optional auto-fix preserves original as `.backup` file
- ✅ Can manually update if preferred

## 📝 Summary

This update completes the configuration system, providing:
1. **Correct structure** matching official sd-scripts requirements
2. **Validation tools** for debugging configuration issues
3. **Auto-fix capability** for common errors
4. **Better user feedback** with detailed error messages
5. **Configurable parameters** like num_repeats

Your training should now start successfully without configuration errors! 🚀
