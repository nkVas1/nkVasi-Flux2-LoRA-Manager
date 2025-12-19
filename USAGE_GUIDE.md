# Руководство по использованию нод FLUX.2 Training / Usage Guide

## 📋 Быстрый старт (Quick Start)

### 1. Подготовка

Перед началом убедитесь:
- ✅ sd-scripts установлены (https://github.com/kohya-ss/sd-scripts)
- ✅ FLUX.2 модель скачана (https://huggingface.co/black-forest-labs/FLUX.1-dev)
- ✅ Dataset с изображениями подготовлен
- ✅ ComfyUI перезагружен (новые ноды должны появиться)

### 2. Три основные ноды

```
┌─────────────────────┐
│  Flux2_8GB_Config   │ ← Генерирует конфиг и команду
└──────────┬──────────┘
           │ (cmd_args, dataset_config, output_dir)
           ↓
┌─────────────────────┐
│  Flux2_Run_External │ ← Запускает тренировку
├─ cmd_args ◄────────┤
├─ trigger (Bool)    │
└─────────────────────┘
           
           
┌─────────────────────┐
│  Flux2_Stop         │ ← Останавливает процесс
├─ stop (Bool)       │
└─────────────────────┘
```

## 🚀 Пошаговое использование (Step-by-Step)

### Шаг 1: Создать ноду Config

1. Откройте ComfyUI
2. Добавьте ноду: Right-click → "🛠️ FLUX.2 Config (Low VRAM)"
3. Заполните параметры:

```
sd_scripts_path    → G:\ComfyUI-StableDif-t27-p312-cu128-v2.1\kohya_train\kohya_ss\sd-scripts
model_path         → black-forest-labs/FLUX.1-dev
img_folder         → G:\path\to\your\dataset\images
output_name        → my_first_lora
resolution         → 768 (рекомендуется для 8GB VRAM)
learning_rate      → 0.0001
max_train_steps    → 1200
lora_rank          → 16
enable_bucket      → True
seed               → 42
cache_to_disk      → True
```

### Шаг 2: Добавить ноду Runner

1. Добавьте ноду: Right-click → "🚀 Start Training (External)"
2. Подключите `cmd_args` с Config на входе Runner
3. **Не меняйте `trigger`, оставьте False** (по умолчанию)

### Шаг 3: Запустить тренировку

**Первый раз:**
1. Установите `trigger = True` на Runner
2. Нажмите "Queue Prompt" (Ctrl+Enter)
3. Ожидайте "✅ Training Started!"
4. **Установите `trigger = False`** (важно!)

**Во время тренировки:**
- `trigger = False` → Видите логи, process работает в фоне
- Нажимаете "Queue Prompt" снова → Видите обновленные логи
- Тренировка НЕ перезапускается ✅

### Шаг 4: Остановить (если нужна)

1. Добавьте ноду: Right-click → "🛑 Emergency Stop"
2. Установите `stop = True`
3. Нажмите "Queue Prompt"
4. Ожидайте "✅ Stop signal sent"
5. Установите `stop = False`

## 🛠️ Параметры Config Ноды

| Параметр | Значение | Описание |
|----------|----------|---------|
| `sd_scripts_path` | Path | Полный путь к sd-scripts |
| `model_path` | HF Model ID или Path | FLUX.1-dev или local path |
| `img_folder` | Path | Папка с training images |
| `output_name` | String | Имя выходного LoRA (без .safetensors) |
| `resolution` | 512, 768, 1024 | Размер изображения (768 рекомендуется) |
| `learning_rate` | Float (1e-5 to 1e-3) | Скорость обучения (1e-4 по умолчанию) |
| `max_train_steps` | Int (100-5000) | Количество шагов (1200 по умолчанию) |
| `lora_rank` | 16, 32 | Размер LoRA (16 = легче, 32 = лучше) |
| `enable_bucket` | True/False | Bucketing для разных разрешений |
| `seed` | Int | Seed для воспроизводимости |
| `cache_to_disk` | True/False | Кэшировать latents на диск (освобождает VRAM) |

## 📊 Оптимизация для RTX 3060 Ti (8GB)

### Стратегия памяти

```
Total VRAM: 8GB
- FLUX.2 model (FP8): 2.5 GB
- LoRA + optimizer state: 1.5 GB
- Latent cache: 1.5 GB
- PyTorch overhead: 1.5 GB
- Free margin: 1 GB (для stability)
————————————————
= 8 GB (tight fit!)
```

### Рекомендуемые значения

```python
# Для успешной тренировки на 8GB:
resolution = 768              # Не выше (512 если 4GB)
batch_size = 1                # НИКОГДА выше 1 (жестко в коде)
gradient_accumulation = 1     # 1 для 8GB
learning_rate = 0.0001        # 1e-4 стандарт
optimizer = "adafactor"       # Легче чем AdamW
lora_rank = 16                # 32 слишком тяжело для 8GB
cache_latents_to_disk = True  # КРИТИЧНО для 8GB
fp8_base = True               # Квантует базовую модель
```

## 📈 Мониторинг прогресса

### Логи в ComfyUI

```
[FLUX-TRAIN] Running environment check...
[FLUX-TRAIN] ✓ CUDA available
[FLUX-TRAIN] ✓ PyTorch version correct
[FLUX-TRAIN] Ensuring training packages...
[FLUX-TRAIN] ✓ All packages ready
[FLUX-TRAIN] --- TRAINING PROCESS STARTED ---
[FLUX-TRAIN] Loaded model: black-forest-labs/FLUX.1-dev
[FLUX-TRAIN] Dataset: 42 images found
[FLUX-TRAIN] Starting training loop...
[FLUX-TRAIN] Step 1/1200: loss=0.245, lr=0.0001
[FLUX-TRAIN] Step 2/1200: loss=0.241, lr=0.0001
...
```

### Файлы вывода

```
ComfyUI/output/flux_training/my_first_lora/
├── dataset.toml              # Конфиг dataset
├── last.safetensors          # Последний checkpoint
├── diffusion_pytorch_model.safetensors  # Final LoRA weights
└── logs/
    └── training_log.txt      # Полный лог тренировки
```

## ⚠️ Частые проблемы

### Проблема: "No module named 'torch'"

**Решение:**
```bash
python setup_training_env.py
# Или переустановить:
python setup_training_env.py --force
```

### Проблема: "CUDA out of memory"

**Решение:**
1. Уменьшите `resolution` (с 768 на 512)
2. Уменьшите `lora_rank` (с 32 на 16)
3. Включите `cache_latents_to_disk`

### Проблема: "cannot import name 'GenerationMixin'"

**Решение:**
```bash
# Delete old packages
rmdir /s training_libs
# Reinstall
python setup_training_env.py
```

### Проблема: Dataset.toml не генерируется

**Решение:**
```bash
# Установить toml пакет
pip install toml
# Или config нода сама сгенерирует JSON вместо TOML
```

## 🔍 Продвинутое использование

### Кастомные параметры

Если нужны специальные параметры, отредактируйте `src/config_gen.py` и добавьте:

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            # ... existing ...
            "custom_param": ("STRING", {"default": "value"}),
        }
    }

def generate_config(self, ..., custom_param, ...):
    cmd.append("--custom_param")
    cmd.append(custom_param)
```

### Несколько LoRA тренировок параллельно

**Внимание**: На одной RTX 3060 Ti возможна только 1 тренировка. На RTX 4090 можно попробовать 2, но не рекомендуется.

### Загрузка в ComfyUI

После тренировки LoRA находится в:
```
ComfyUI/output/flux_training/my_first_lora/diffusion_pytorch_model.safetensors
```

Используйте в Node: Load LoRA
```
model_name → my_first_lora.safetensors
```

## 📚 Дополнительные ресурсы

- **sd-scripts**: https://github.com/kohya-ss/sd-scripts
- **FLUX.1-dev**: https://huggingface.co/black-forest-labs/FLUX.1-dev
- **LoRA обучение**: https://civitai.com/articles/guide-to-training-loras
- **Troubleshooting**: Смотрите TROUBLESHOOTING.md этого проекта

## 💡 Лучшие практики

1. **Начните с малого**: Сначала 100 шагов на test dataset, затем масштабируйте
2. **Мониторьте потребление памяти**: Откройте Task Manager, смотрите VRAM usage
3. **Сохраняйте checkpoints**: sd-scripts сохраняет each 50/100 шагов автоматически
4. **Используйте разнообразный dataset**: >30 изображений для хорошего LoRA
5. **Экспериментируйте с LR**: 1e-4 стандарт, но 1e-5 может быть лучше для стабильности

## 🚀 Что дальше?

После успешной тренировки LoRA:

```
LoRA → Use in generation nodes
    → Refine with more data
    → Merge with other LoRAs
    → Upload to CivitAI
    → Share community!
```

---

**Happy Training! 🎉**
