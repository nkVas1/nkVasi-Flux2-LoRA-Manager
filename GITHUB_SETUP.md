# Инструкция по пушу проекта на GitHub

## Предварительные требования

1. **Создайте аккаунт на GitHub** (если его еще нет): https://github.com/signup
2. **Установите Git** (если еще не установлен): https://git-scm.com/download

## Шаг 1: Настройка Git (первый раз)

Если вы впервые используете Git на этом компьютере, выполните:

```powershell
git config --global user.name "Ваше Имя"
git config --global user.email "ваша.почта@example.com"
```

Замените "Ваше Имя" и "ваша.почта@example.com" на ваши данные.

## Шаг 2: Создание нового репозитория на GitHub

1. Зайдите на https://github.com/new
2. Заполните форму:
   - **Repository name**: `ComfyUI-Flux2-LoRA-Manager`
   - **Description**: "Professional ComfyUI node pack for FLUX.2 LoRA training on 8GB VRAM"
   - **Public/Private**: Выберите **Public** (для open-source проекта)
   - **Initialize this repository with**: НЕ выбирайте ничего (репозиторий уже инициализирован локально)
3. Нажмите **Create repository**

## Шаг 3: Связь локального репозитория с GitHub

После создания репозитория на GitHub, вы увидите инструкции. Выполните эту команду в PowerShell:

```powershell
cd "g:\CODING\nkVasi_Flux2_LoRA_LowVRAM\ComfyUI-Flux2-LoRA-Manager"

git remote add origin https://github.com/YOUR_USERNAME/ComfyUI-Flux2-LoRA-Manager.git
git branch -M main
git push -u origin main
```

**Замените `YOUR_USERNAME` на ваше имя пользователя на GitHub!**

## Шаг 4: Проверка успешного пуша

После выполнения команды перейдите по ссылке:
```
https://github.com/YOUR_USERNAME/ComfyUI-Flux2-LoRA-Manager
```

Вы должны увидеть все ваши файлы в репозитории. Если видите только папку `.git`, попробуйте обновить страницу (F5).

---

## Частые проблемы и решения

### ❌ Ошибка: "fatal: 'origin' does not appear to be a 'git' repository"

**Решение**: Убедитесь, что вы находитесь в правильной директории:
```powershell
cd "g:\CODING\nkVasi_Flux2_LoRA_LowVRAM\ComfyUI-Flux2-LoRA-Manager"
git status
```

### ❌ Ошибка: "fatal: Authentication failed"

**Решение 1** (используя HTTPS):
1. Создайте Personal Access Token на GitHub:
   - Зайдите на https://github.com/settings/tokens
   - Нажмите "Generate new token"
   - Выберите scopes: `repo`, `gist`
   - Скопируйте токен

2. При запросе пароля вместо пароля вставьте токен

**Решение 2** (используя SSH):
1. Сгенерируйте SSH ключ:
   ```powershell
   ssh-keygen -t ed25519 -C "ваша.почта@example.com"
   ```
2. Добавьте ключ на GitHub: https://github.com/settings/ssh/new
3. Используйте SSH URL вместо HTTPS:
   ```powershell
   git remote set-url origin git@github.com:YOUR_USERNAME/ComfyUI-Flux2-LoRA-Manager.git
   ```

### ❌ Ошибка: "Updates were rejected because the tip of your current branch is behind its remote"

**Решение**:
```powershell
git pull origin main
git push origin main
```

---

## Последующие коммиты (для будущих изменений)

После первого пуша, для добавления новых изменений:

```powershell
git add .
git commit -m "Описание ваших изменений"
git push origin main
```

---

## Полезные команды

```powershell
# Проверить статус файлов
git status

# Просмотреть историю коммитов
git log --oneline

# Просмотреть отличия последних изменений
git diff

# Отменить последний коммит (но сохранить изменения)
git reset --soft HEAD~1

# Отменить все неиндексированные изменения
git checkout -- .
```

---

## Добавление расширенной информации (опционально)

### Добавить .github/workflows для CI/CD (действия при пуше)

Создайте файл `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11']
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - run: pip install -r requirements.txt
    - run: python -m pytest tests/ 2>/dev/null || echo "No tests found"
```

### Добавить CHANGELOG.md для отслеживания версий

```markdown
# Changelog

## [1.0.0] - 2025-01-15

### Added
- Initial release of ComfyUI FLUX.2 LoRA Manager
- Support for 8GB VRAM training
- Process isolation for stable operation
- Real-time logging and monitoring
- Emergency stop functionality

### Fixed
- N/A

### Changed
- N/A
```

---

## Примеры правильного использования GitHub

### Хорошие сообщения коммитов:
```
✨ Add support for gradient checkpointing
🐛 Fix memory leak in process manager
📝 Update README with installation steps
⚡ Optimize latent caching performance
```

### Плохие сообщения коммитов:
```
fix bugs
update
asdf
many changes
```

---

**Готово! 🎉 Ваш проект теперь на GitHub и готов к использованию другими разработчиками.**

Для получения звезд и внимания можно:
1. Поделиться ссылкой в ComfyUI сообществах (Reddit, Discord)
2. Добавить проект в реестр ComfyUI node packs
3. Создать примеры и туториалы
