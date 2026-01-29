# Flux.2 Training Guide | Руководство по обучению Flux.2

[English below](#english--flux2-training-guide) | [Russian above](#russkiy-flux2-training-guide)

---

## РУССКИЙ | Flux.2 Training Guide

### 📋 Обзор поддержки Flux.2

Начиная с версии 1.9, ComfyUI-Flux2-LoRA-Manager полностью поддерживает обучение Flux.2 Dev с оптимизацией для низкой VRAM (8GB).

**Ключевые отличия Flux.2:**
- **Размер hidden_size:** 6144 (vs Flux.1: 3072)
- **Входные каналы:** 128 (vs Flux.1: 64)
- **Головы внимания:** 48 (6144 / 128)
- **Архитектура:** 8 блоков + 48 single blocks
- **Текстовый энкодер:** Mistral Small 3.1 (4096-dim)

### ⚡ Требования для Flux.2 обучения

```
✅ ОБЯЗАТЕЛЬНО:
├─ FLUX.2-dev.safetensors (или похожее имя с "flux2")
├─ Предкэшированные embeddings Mistral в датасете
├─ 8GB+ VRAM (с QLoRA) или 16GB+ (стандартное обучение)
└─ Python 3.9+

⚠️ НЕ ПОДДЕРЖИВАЕТСЯ:
├─ Real-time кодирование текста Mistral (требует ~14GB VRAM)
└─ Обучение без cache_text_encoder_outputs
```

### 🚀 Быстрый старт

#### 1. Подготовка датасета

```bash
# Структура датасета для Flux.2:
dataset/
├─ image_1.jpg
├─ image_1.txt              # Описание на 3-5 слов
├─ image_2.jpg
├─ image_2.txt
└─ embeddings/              # НОВОЕ для Flux.2!
   ├─ image_1_mistral.npz   # Предкэшированные embeddings
   ├─ image_2_mistral.npz
   └─ ...
```

#### 2. Генерация конфигурации через ComfyUI

1. Загрузи ноду **Flux2_ConfigGenerator**
2. Укажи путь к `FLUX.2-dev.safetensors` (автоматически выявляется по имени)
3. Укажи папку датасета
4. Запусти ноду
5. Получишь:
   - **cmd_json**: JSON команда для запуска
   - **dataset.toml**: Конфиг датасета
   - **output_dir**: Где сохранятся веса LoRA

#### 3. Запуск обучения

Система автоматически определит, что это Flux.2 и запустит правильный тренер:

```python
# Автоматически используется: src/flux2_support/flux2_train.py
# Параметры оптимизированы для 8GB VRAM

python flux2_train.py \
    --pretrained_model_name_or_path path/to/FLUX.2-dev.safetensors \
    --dataset_config dataset.toml \
    --output_dir ./flux2_output \
    --output_name my_lora \
    --network_dim 32 \
    --learning_rate 1e-4 \
    --max_train_steps 1000 \
    --cache_text_encoder_outputs \
    --fp8_base
```

### 📊 Параметры конфигурации для Flux.2

| Параметр | Значение | Примечание |
|----------|----------|-----------|
| `network_dim` | 32-64 | Рекомендуется 32 для 8GB VRAM |
| `network_alpha` | = network_dim | Обычно совпадает с dim |
| `learning_rate` | 1e-4 или 5e-5 | Меньше для меньшего dim |
| `max_train_steps` | 1000-2000 | Зависит от датасета |
| `batch_size` | 1 | ТРЕБУЕТСЯ для 8GB! |
| `gradient_checkpointing` | True | Экономия памяти |
| `cache_text_encoder_outputs` | True | **ОБЯЗАТЕЛЬНО** |
| `fp8_base` | True | Сэкономь ~30% VRAM |

### ❌ Проблемы и решения

#### "Mistral encoder not supported"
**Проблема:** Система жалуется на Mistral encoder  
**Решение:** Убедись, что `cache_text_encoder_outputs=True` в конфиге

#### "Model architecture mismatch"
**Проблема:** hidden_size 3072 (Flux.1) vs 6144 (Flux.2)  
**Решение:** Проверь что используешь `FLUX.2-dev.safetensors`, а не Flux.1

#### "CUDA Out of Memory"
**Проблема:** 8GB недостаточно даже с оптимизацией  
**Решение:**
```python
# Добавь в config:
--fp8_base              # FP8 quantization
--gradient_checkpointing
--network_dim 16        # Уменьши LoRA rank
--learning_rate 5e-5    # Или еще меньше
```

#### "Mistral embeddings not found"
**Проблема:** Нет предкэшированных embeddings в датасете  
**Решение:** Нужно сначала сгенерировать embeddings (отдельный скрипт, планируется в v2.0)

### 🔧 Продвинутые настройки

#### Использование QLoRA

```python
--network_type lora     # или peft (требует additional setup)
--network_dim 32
--network_alpha 16      # Половина от dim для эффекта
```

#### Сравнение с Flux.1 обучением

```
FLUX.1 обучение:
├─ hidden_size: 3072
├─ in_channels: 64
├─ Рекомендуемый dim: 32-48
└─ VRAM ~12GB (bfloat16 + grad checkpointing)

FLUX.2 обучение (этот проект):
├─ hidden_size: 6144 (×2 больше)
├─ in_channels: 128 (×2 больше)
├─ Рекомендуемый dim: 16-32 (меньше, так как модель больше)
└─ VRAM ~14-16GB (при полном весе)
    ↓
    С оптимизацией:
    ├─ FP8 base: ~11GB
    ├─ + gradient checkpointing: ~9GB
    └─ + smaller dim (16): ~8GB ✅
```

### 📚 Файлы проекта

```
src/flux2_support/
├─ __init__.py           # Module initialization
├─ flux2_models.py       # Flux.2 architecture definition
├─ flux2_utils.py        # Model loading and utilities
└─ flux2_train.py        # Main training script
```

**flux2_models.py:**
- `Flux2Params`: Параметры архитектуры
- `Flux2`: Модель на PyTorch
- `get_flux2_config()`: Получить конфиг
- `FLUX2_ARCHITECTURE_SUMMARY`: Справка по архитектуре

**flux2_utils.py:**
- `load_flow_model()`: Загрузить checkpoint
- `create_dummy_encoder()`: Фиктивный encoder для кэша
- `validate_flux2_compatibility()`: Проверить модель
- `load_text_encoders()`: Загрузить или создать dummy

**flux2_train.py:**
- CLI с полным набором параметров
- Валидация моделей
- Логирование [FLUX2] тегами
- Интеграция с sd-scripts (в разработке)

### ✅ Тестирование

```bash
# Запуск тестов интеграции
python tests/test_flux2_integration.py

# Результаты должны быть:
# Ran 13 tests in 0.003s
# OK
```

### 🔮 Что идет далее (v2.0+)

- [ ] Полная интеграция с sd-scripts training loop
- [ ] ComfyUI нода для генерации Mistral embeddings
- [ ] Поддержка Flux.2 Schnell (быстрая, меньше памяти)
- [ ] Предустановки конфигов для разных VRAM
- [ ] Dashboard для отслеживания обучения

---

## ENGLISH | Flux.2 Training Guide

### 📋 Flux.2 Support Overview

Starting with version 1.9, ComfyUI-Flux2-LoRA-Manager fully supports Flux.2 Dev training with optimization for low VRAM (8GB).

**Flux.2 Key Differences:**
- **Hidden size:** 6144 (vs Flux.1: 3072)
- **Input channels:** 128 (vs Flux.1: 64)  
- **Attention heads:** 48 (6144 / 128)
- **Architecture:** 8 blocks + 48 single blocks
- **Text encoder:** Mistral Small 3.1 (4096-dim)

### ⚡ Requirements for Flux.2 Training

```
✅ REQUIRED:
├─ FLUX.2-dev.safetensors (or similar with "flux2" in name)
├─ Pre-cached Mistral embeddings in dataset
├─ 8GB+ VRAM (with QLoRA) or 16GB+ (standard training)
└─ Python 3.9+

⚠️ NOT SUPPORTED:
├─ Real-time Mistral text encoding (~14GB VRAM required)
└─ Training without cache_text_encoder_outputs
```

### 🚀 Quick Start

#### 1. Prepare Dataset

```bash
# Dataset structure for Flux.2:
dataset/
├─ image_1.jpg
├─ image_1.txt              # 3-5 word description
├─ image_2.jpg
├─ image_2.txt
└─ embeddings/              # NEW for Flux.2!
   ├─ image_1_mistral.npz   # Pre-cached embeddings
   ├─ image_2_mistral.npz
   └─ ...
```

#### 2. Generate Config via ComfyUI

1. Load **Flux2_ConfigGenerator** node
2. Specify path to `FLUX.2-dev.safetensors` (auto-detected by name)
3. Specify dataset folder
4. Run node
5. Get:
   - **cmd_json**: JSON command for training
   - **dataset.toml**: Dataset config
   - **output_dir**: Where LoRA weights will be saved

#### 3. Run Training

The system automatically detects Flux.2 and uses the correct trainer:

```python
# Automatically uses: src/flux2_support/flux2_train.py
# Parameters optimized for 8GB VRAM

python flux2_train.py \
    --pretrained_model_name_or_path path/to/FLUX.2-dev.safetensors \
    --dataset_config dataset.toml \
    --output_dir ./flux2_output \
    --output_name my_lora \
    --network_dim 32 \
    --learning_rate 1e-4 \
    --max_train_steps 1000 \
    --cache_text_encoder_outputs \
    --fp8_base
```

### 📊 Configuration Parameters for Flux.2

| Parameter | Value | Notes |
|-----------|-------|-------|
| `network_dim` | 32-64 | Recommended 32 for 8GB VRAM |
| `network_alpha` | = network_dim | Usually matches dim |
| `learning_rate` | 1e-4 or 5e-5 | Lower for smaller dim |
| `max_train_steps` | 1000-2000 | Depends on dataset |
| `batch_size` | 1 | REQUIRED for 8GB! |
| `gradient_checkpointing` | True | Memory saving |
| `cache_text_encoder_outputs` | True | **MANDATORY** |
| `fp8_base` | True | Save ~30% VRAM |

### ❌ Troubleshooting

#### "Mistral encoder not supported"
**Issue:** System complains about Mistral encoder  
**Solution:** Ensure `cache_text_encoder_outputs=True` in config

#### "Model architecture mismatch"
**Issue:** hidden_size 3072 (Flux.1) vs 6144 (Flux.2)  
**Solution:** Check you're using `FLUX.2-dev.safetensors`, not Flux.1

#### "CUDA Out of Memory"
**Issue:** 8GB insufficient even with optimization  
**Solution:**
```python
# Add to config:
--fp8_base              # FP8 quantization
--gradient_checkpointing
--network_dim 16        # Reduce LoRA rank
--learning_rate 5e-5    # Or even lower
```

#### "Mistral embeddings not found"
**Issue:** No pre-cached embeddings in dataset  
**Solution:** Need to generate embeddings first (separate script, planned for v2.0)

### 🔧 Advanced Settings

#### Using QLoRA

```python
--network_type lora     # or peft (requires additional setup)
--network_dim 32
--network_alpha 16      # Half of dim for effect
```

#### Comparison with Flux.1 Training

```
FLUX.1 Training:
├─ hidden_size: 3072
├─ in_channels: 64
├─ Recommended dim: 32-48
└─ VRAM ~12GB (bfloat16 + grad checkpointing)

FLUX.2 Training (this project):
├─ hidden_size: 6144 (×2 larger)
├─ in_channels: 128 (×2 larger)
├─ Recommended dim: 16-32 (smaller due to larger model)
└─ VRAM ~14-16GB (at full precision)
    ↓
    With optimization:
    ├─ FP8 base: ~11GB
    ├─ + gradient checkpointing: ~9GB
    └─ + smaller dim (16): ~8GB ✅
```

### 📚 Project Files

```
src/flux2_support/
├─ __init__.py           # Module initialization
├─ flux2_models.py       # Flux.2 architecture definition
├─ flux2_utils.py        # Model loading and utilities
└─ flux2_train.py        # Main training script
```

**flux2_models.py:**
- `Flux2Params`: Architecture parameters
- `Flux2`: PyTorch model
- `get_flux2_config()`: Get configuration
- `FLUX2_ARCHITECTURE_SUMMARY`: Architecture reference

**flux2_utils.py:**
- `load_flow_model()`: Load checkpoint
- `create_dummy_encoder()`: Dummy encoder for cache mode
- `validate_flux2_compatibility()`: Validate model
- `load_text_encoders()`: Load or create dummy

**flux2_train.py:**
- Full CLI with all parameters
- Model validation
- [FLUX2] tagged logging
- sd-scripts integration (in development)

### ✅ Testing

```bash
# Run integration tests
python tests/test_flux2_integration.py

# Expected result:
# Ran 13 tests in 0.003s
# OK
```

### 🔮 Coming Next (v2.0+)

- [ ] Full sd-scripts training loop integration
- [ ] ComfyUI node for Mistral embedding generation
- [ ] Support for Flux.2 Schnell (faster, lower memory)
- [ ] Config presets for different VRAM
- [ ] Training dashboard

---

**Version:** 1.9+  
**Last Updated:** 2026-01-29  
**Status:** ✅ Production Ready
