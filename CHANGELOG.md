# 🚀 Улучшения v1.1: Веб-интерфейс логирования и исправления Windows

## ✨ Что изменилось

### 1️⃣ Добавлен веб-интерфейс для мониторинга логов (JavaScript)
**Файл:** `js/flux_monitor.js` (новый)

- 🟢 Плавающая панель с зеленым текстом на черном фоне (классический hacker-стиль)
- 📊 Реальное время: логи появляются в браузере по мере выполнения
- 🎨 Цветное выделение:
  - 🔴 Красный: ошибки, CUDA errors
  - 🟡 Жёлтый: warnings
  - 🔵 Голубой: steps, loss, metrics
  - 🟠 Оранжевый: loading, preparing
  - 🟢 Зелёный: success, finished
- 🖱️ Draggable: можно двигать панель по экрану
- ⏸️ Click-to-close: нажимаем на заголовок для закрытия

### 2️⃣ Обновлены Python файлы для WebSocket интеграции

#### `__init__.py` (изменён)
```python
WEB_DIRECTORY = "./js"  # ← Новая строка
```
ComfyUI теперь знает, где искать JavaScript расширения.

#### `src/process.py` (изменён)
```python
# Старо: только print()
print(f"[FLUX-TRAIN] {clean_line}")

# Ново: print() + WebSocket
print(f"[FLUX-TRAIN] {clean_line}")
PromptServer.instance.send_sync("flux_train_log", {"line": clean_line})
```
Теперь логи отправляются в браузер через WebSocket.

#### `src/config_gen.py` (исправлено для Windows)
```python
# Старо:
cmd = ["accelerate", "launch", ...]  # ← На Windows часто не работает

# Ново:
import sys
python_exe = sys.executable
cmd = [python_exe, "-m", "accelerate.commands.launch", ...]  # ← Надёжнее на Windows
```

---

## 📋 Структура проекта (обновлена)

```
ComfyUI-Flux2-LoRA-Manager/
├── __init__.py                    # ✏️ Добавлен WEB_DIRECTORY
├── nodes.py                       # (без изменений)
├── requirements.txt               # (без изменений)
├── LICENSE                        # (без изменений)
├── README.md                      # (без изменений)
├── GITHUB_SETUP.md               # (без изменений)
├── .gitignore                     # (без изменений)
│
├── js/                           # 🆕 НОВАЯ ПАПКА
│   └── flux_monitor.js          # 🆕 Веб-интерфейс для логов (180 строк)
│
└── src/                          # (существующие файлы)
    ├── __init__.py               # (без изменений)
    ├── config_gen.py             # ✏️ Windows fix для accelerate
    ├── process.py                # ✏️ WebSocket логирование
    └── utils.py                  # (без изменений)
```

---

## 🎯 Как это работает

### Архитектура WebSocket

```
┌─────────────────────────────────────────────────────────────┐
│                         WEB BROWSER                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 🟢 FLUX.2 Training Monitor (js/flux_monitor.js)     │  │
│  │ [████████████████████] 45% (Loss: 0.123)             │  │
│  │ [FLUX-TRAIN] Step 450/1200: loss=0.123              │  │
│  │ [FLUX-TRAIN] Saving checkpoint...                    │  │
│  │                                 (Click to close)      │  │
│  └──────────────────────────────────────────────────────┘  │
│              ↑ WebSocket: api.addEventListener()           │
└─────────────────────────────────────────────────────────────┘
                            ↑↓ (bidirectional)
┌─────────────────────────────────────────────────────────────┐
│                    ComfyUI SERVER (Python)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ PromptServer.send_sync("flux_train_log", {...})     │  │
│  └──────────────────────────────────────────────────────┘  │
│              ↑ Отправка логов из потока                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ _log_reader() → читает stdout процесса             │  │
│  └──────────────────────────────────────────────────────┘  │
│              ↑ Читает из subprocess                          │
└─────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────┐
│                 EXTERNAL PROCESS (subprocess)               │
│  python -m accelerate.commands.launch flux_train_network.py │
│         → STDOUT: [kohya_ss training logs]                 │
└─────────────────────────────────────────────────────────────┘
```

### Поток выполнения

1. **Пользователь** в ComfyUI устанавливает параметры и нажимает "Queue"
2. **Нода Config** генерирует команду с python.exe вместо просто "accelerate"
3. **Нода Runner** запускает процесс в отдельном потоке
4. **_log_reader()** читает stdout построчно и:
   - Печатает в консоль сервера (чёрное окно)
   - Отправляет через WebSocket в браузер
5. **JavaScript** (flux_monitor.js) ловит событие `flux_train_log` и:
   - Показывает панель (если скрыта)
   - Добавляет строку в лог-панель
   - Подсвечивает цветом
   - Авто-скроллит вниз
6. **Пользователь** видит логи **в реальном времени** в браузере 🎉

---

## 🔧 Настройка и использование

### Требования после обновления
Никаких новых зависимостей не нужно! WebSocket - встроенный в ComfyUI.

```bash
# requirements.txt остаётся без изменений
pip install -r requirements.txt
```

### Перезагрузка ComfyUI (ОБЯЗАТЕЛЬНА!)
```powershell
# 1. Закройте ComfyUI (Ctrl+C в консоли)
# 2. Перезапустите ComfyUI
# 3. Откройте браузер на http://localhost:8188
```

### Использование

1. **Добавьте ноды** в workflow обычным образом:
   - 🛠️ FLUX.2 Config (Low VRAM)
   - 🚀 Start Training (External)

2. **Заполните параметры** и нажмите "Queue Prompt"

3. **Смотрите логи**:
   - В **консоли сервера** (черное окно): `[FLUX-TRAIN] Step 1/1200...`
   - В **браузере** (右下 / bottom-right): 🟢 зелёная панель с логами

4. **Закройте панель**: кликните на заголовок "🚀 FLUX.2 Training Monitor"

---

## 🐛 Исправленные ошибки

### ❌ Проблема 1: Логи пропадают в никуда
**Было:** `print()` пишет только в консоль сервера  
**Исправлено:** WebSocket отправляет логи прямо в браузер

### ❌ Проблема 2: "accelerate: command not found" на Windows
**Было:** `cmd = ["accelerate", "launch", ...]`  
**Исправлено:** `cmd = [sys.executable, "-m", "accelerate.commands.launch", ...]`

### ❌ Проблема 3: Тип данных в команде
**Было:** `"--network_dim", lora_rank` (может быть int)  
**Исправлено:** `"--network_dim", str(lora_rank)` (гарантированно string)

---

## 📊 Примеры логов в браузере

```
=== FLUX TRAIN LOG STARTED ===
[FLUX-TRAIN] Loading model: black-forest-labs/FLUX.1-dev
[FLUX-TRAIN] Model loaded successfully (memory: 7.2GB)
[FLUX-TRAIN] Loading dataset from C:/Dataset/img
[FLUX-TRAIN] Found 42 images with captions
[FLUX-TRAIN] Initializing training...
[FLUX-TRAIN] Starting training loop
[FLUX-TRAIN] Step 1/1200: loss=2.543 (lr=0.0001)
[FLUX-TRAIN] Step 2/1200: loss=2.341
[FLUX-TRAIN] Step 3/1200: loss=2.125
...
[FLUX-TRAIN] Saving checkpoint at step 600
[FLUX-TRAIN] LoRA saved to: outputs/my_lora_20250115_143022/lora_model.safetensors
[FLUX-TRAIN] Training finished successfully!

✅ TRAINING COMPLETED
```

**Цветовое выделение:**
- 🔴 `CUDA out of memory` → Красный
- 🟡 `WARNING: High gradient norm detected` → Жёлтый
- 🔵 `Step 450/1200: loss=0.234` → Голубой
- 🟢 `Training finished successfully` → Зелёный

---

## 🔄 Git коммиты

Обновлённый проект готов к пушу:

```bash
git add .
git commit -m "v1.1: Add web UI monitoring, WebSocket logging, Windows fixes

- Add flux_monitor.js: Real-time training logs in browser
- Update process.py: WebSocket integration for live streaming
- Update config_gen.py: Fix accelerate execution on Windows
- Update __init__.py: Register JS extension directory
- Improve error handling and log formatting"

git push origin main
```

---

## ✅ Чек-лист для проверки

После обновления убедитесь:

- [ ] Папка `js/` создана с файлом `flux_monitor.js`
- [ ] `__init__.py` содержит `WEB_DIRECTORY = "./js"`
- [ ] `src/process.py` содержит `PromptServer.instance.send_sync(...)`
- [ ] `src/config_gen.py` содержит `sys.executable, "-m", "accelerate.commands.launch"`
- [ ] **ComfyUI перезагружен** (важно!)
- [ ] При запуске видна 🟢 зелёная панель справа снизу

---

## 📞 Troubleshooting

### Панель не появляется при тренировке
1. Откройте DevTools браузера (F12 → Console)
2. Проверьте наличие ошибок JavaScript
3. Убедитесь, что `WEB_DIRECTORY = "./js"` в `__init__.py`
4. Перезагрузите ComfyUI (Ctrl+Shift+R в браузере)

### "AttributeError: 'NoneType' object has no attribute 'send_sync'"
- Это нормально, если тренировка запустилась через CLI без браузера
- PromptServer.instance может быть None в тестах
- Код имеет обработку исключений, тренировка продолжится

### Процесс не запускается на Windows
- Убедитесь, что `accelerate` установлен: `pip list | grep accelerate`
- Проверьте путь к Python: `python -m accelerate.commands.launch --help`
- Смотрите логи в консоли сервера для деталей

---

## 🎓 Техническое объяснение

### Почему WebSocket, а не Server-Sent Events (SSE)?
- WebSocket уже встроен в ComfyUI (используется для взаимодействия UI ↔ Backend)
- Меньше overhead чем SSE для частых обновлений
- Bidirectional (можно потом добавить управление из браузера)

### Почему `sys.executable` вместо просто `"accelerate"`?
На Windows есть несколько питонов:
- Python из conda
- Python из виртуального окружения
- Python из PATH

`sys.executable` гарантирует, что мы используем **тот же Python**, в котором установлена accelerate.

### Почему отдельный поток для _log_reader?
- `readline()` — блокирующая операция
- Если блокировать основной поток ComfyUI, UI зависнет
- Отдельный daemon-поток позволяет читать логи без блокировки

---

**Версия:** 1.1  
**Дата обновления:** 2025-01-15  
**Совместимость:** ComfyUI stable + dev  
**Python:** 3.10+  
**ОС:** Windows, Linux, macOS
