cd src && uv run python server.py

# 📚 Инструкция по запуску Parlor Vision

## Компоненты для установки:

Python 3.10+ — Скачать

Git — Скачать

Ollama — Скачать

### 📦 Шаг 1: Установка Ollama

Windows:

Скачайте установщик с ollama.com

Запустите OllamaSetup.exe и следуйте инструкциям

После установки Ollama запустится автоматически (иконка в трее)

```
ollama --version

```

### 🤖 Шаг 2: Скачивание языковой модели

```
# Скачать модель Gemma 4 E2B (~5 ГБ)
ollama pull gemma4:e2b

# Или E4B (более качественная, ~8 ГБ)
ollama pull gemma4:e4b

# Проверка установленных моделей:
ollama list

```

### 📦 Шаг 5: Установка зависимостей

```
# Установить uv (быстрый менеджер пакетов)
pip install uv

# Установить зависимости проекта
uv pip install -e .

```

### ⚙️ Шаг 6: Настройка .env файла

Создайте файл .env в папке src/:

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e2b

# STT (распознавание речи)
STT_BACKEND=whisper
WHISPER_MODEL=large-v3-turbo
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

# Сервер
PORT=8000

```

### 🚀 Шаг 7: Запуск сервера

```
cd src

uv run python server.py
```

### Шаг 8: Открытие в браузере

Откройте браузер и перейдите по адресу:

```
http://localhost:8000
```
