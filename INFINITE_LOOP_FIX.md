# Исправление бесконечного цикла / Infinite Loop Fix

## Проблема (Problem)

При запуске workflow в ComfyUI ноды с `OUTPUT_NODE = True` выполняются автоматически при каждом изменении input'ов. Это привело к бесконечному циклу:

```
trigger=False → Flux2_Runner возвращает статус
ComfyUI подумает, что результат изменился → Перезапустит workflow
trigger остается False → Снова выполнится Flux2_Runner
... бесконечный цикл
```

## Решение (Solution)

Изменена логика исполнения обеих нод (`Flux2_Runner` и `Flux2_Stopper`) для предотвращения автоматического повторного исполнения:

### Flux2_Runner (Новая логика)

```python
if not trigger:
    # trigger=False → Вернуть статус БЕЗ изменений
    # Это сигнал ComfyUI не перезапускать workflow
    if manager.is_running():
        return "⏳ Training in progress..."
    else:
        return "⏸️ Ready. Set trigger=True to start"

if trigger:
    # trigger=True → Проверить, запущен ли процесс
    if manager.is_running():
        # Да → Вернуть текущий статус (без перезапуска)
        return "⏳ Training running"
    else:
        # Нет → Запустить новый процесс
        manager.start_training(...)
        return "✅ Training Started!"
```

### Flux2_Stopper (Новая логика)

```python
if not stop:
    # stop=False → Вернуть статус без остановки
    if manager.is_running():
        return "⏳ Training running. Set stop=True to stop"
    else:
        return "ℹ️ No training running"

if stop:
    # stop=True → Остановить процесс
    manager.stop_training()
    return "✅ Stop signal sent"
```

## Как это работает / How it Works

### Состояние 1: Никакая тренировка не запущена

```
Status: No training
trigger=False → "⏸️ Ready. Set trigger=True to start"
trigger=True  → "✅ Training Started!"
```

### Состояние 2: Тренировка запущена

```
Status: Training in progress
trigger=False → "⏳ Training in progress..." (без повторного запуска)
trigger=True  → "⏳ Training running:\n[last logs]" (статус, без перезапуска)
```

## Ключевые изменения / Key Changes

| Аспект | Было | Стало |
|--------|------|-------|
| trigger=False | Возвращает сообщение | Возвращает статус (без side effects) |
| trigger=True + Running | Ошибка "уже запущено" | Возвращает текущий статус |
| Повторные запуски | Бесконечный цикл | Одноразовое исполнение |
| OUTPUT_NODE | Всегда переexecution | Только при реальных изменениях |

## Пример использования / Usage Example

### Сценарий: Запуск тренировки

1. **Подготовка конфига**:
   - Config нода автоматически генерирует команду
   - Результат подается на вход Runner

2. **Запуск тренировки**:
   - Устанавливаем `trigger=True` на Runner
   - Нажимаем "Queue Prompt"
   - Видим "✅ Training Started!"
   - Процесс запускается в фоне

3. **Мониторинг**:
   - Устанавливаем `trigger=False` на Runner
   - Нажимаем "Queue Prompt"
   - Видим "⏳ Training in progress..." (логи обновляются)
   - **Workflow НЕ зацикливается** ✅

4. **Остановка** (если нужна):
   - Устанавливаем `stop=True` на Stopper
   - Нажимаем "Queue Prompt"
   - Видим "✅ Stop signal sent"
   - Процесс завершается

## Технические детали / Technical Details

### Почему это работает

1. **Immutable outputs** - Если нода вернет ОДНО И ТО ЖЕ значение, ComfyUI не перезапустит зависимые ноды
2. **State management** - Состояние процесса (running/stopped) хранится в `TrainingProcessManager` (singleton)
3. **No side effects on False** - Когда `trigger=False`, нода просто читает состояние, не изменяя его

### Что изменилось в коде

**File: `src/process.py`**

1. **Flux2_Runner.run_training()** (lines ~543-640)
   - Добавлена проверка `if not trigger` в начале
   - Возвращает статус вместо пустого значения при trigger=False
   - Не перезапускает уже работающий процесс

2. **Flux2_Stopper.stop_process()** (lines ~650-690)
   - Переписана логика на основе `stop` флага
   - Возвращает статус при `stop=False` (без остановки)
   - Выполняет остановку только при `stop=True`

## Тестирование / Testing

### Проверить исправление

1. Откройте ComfyUI
2. Создайте workflow с Config → Runner → Stopper
3. **Тест 1**: Установите `trigger=False`, нажмите Queue несколько раз
   - ✅ Должны видеть "Ready" статус, НЕ бесконечный цикл
4. **Тест 2**: Установите `trigger=True`, нажмите Queue
   - ✅ Должны видеть "Training Started", процесс запускается
5. **Тест 3**: Оставьте `trigger=False`, нажмите Queue
   - ✅ Должны видеть логи прогресса, НЕ перезапуск тренировки

### Признаки успеха

- ✅ Логи в ComfyUI обновляются в реальном времени
- ✅ Нет "Prompt executed" спама каждые 0.01 сек
- ✅ Можно остановить процесс без ошибок
- ✅ Workflow стабилен (не крашится)

## Версия / Version

- **Fixed in**: v1.5.0
- **Affected nodes**: Flux2_Runner, Flux2_Stopper
- **Breaking changes**: Нет (только внутренняя логика)

## Дальнейшие улучшения / Future Improvements

- [ ] Добавить кнопку "Refresh Status" для явного обновления логов
- [ ] Сохранять логи в файл для долгосрочных тренировок
- [ ] Добавить ETA (estimated time to completion)
- [ ] Поддерживать несколько параллельных тренировок (на отдельных GPUs)
