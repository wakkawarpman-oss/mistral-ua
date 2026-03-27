#!/bin/bash
# Запуск Містраль (Groq primary | Ollama fallback)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv"
PYTHON="$VENV/bin/python3"
OLLAMA_PID=""

echo "=== Містраль R&D ==="

# 1. venv
if [ ! -f "$PYTHON" ]; then
    echo "Створення venv..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
fi

# 2. Читаємо .env
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# 3. Перевіряємо Groq
if [ -n "$GROQ_API_KEY" ]; then
    echo "Бекенд: Groq LPU (~500 tok/s)"
else
    echo "GROQ_API_KEY не знайдено — fallback на Ollama local"
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Запуск Ollama..."
        ollama serve > /tmp/ollama_mistral.log 2>&1 &
        OLLAMA_PID=$!
        sleep 3
    fi
    MODEL="${OLLAMA_MODEL:-mistral-ua}"
    if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
        echo "Завантаження $MODEL..."
        ollama pull "$MODEL"
    fi
fi

# 4. Запуск
echo "Запуск..."
"$PYTHON" "$SCRIPT_DIR/mistral_chat.py"

# 5. Зупинка Ollama якщо ми запустили
if [ -n "$OLLAMA_PID" ]; then
    kill "$OLLAMA_PID" 2>/dev/null
fi


# 1. Перевірка venv
if [ ! -f "$PYTHON" ]; then
    echo "Створення virtual environment..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
    echo "Готово."
fi

# 2. Запуск Ollama якщо не запущений
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Запуск Ollama..."
    ollama serve > /tmp/ollama_mistral.log 2>&1 &
    OLLAMA_PID=$!
    sleep 3
    echo "Ollama запущено (PID: $OLLAMA_PID)"
fi

# 3. Перевірка наявності моделі
MODEL="mistral-ua"
if ! ollama list 2>/dev/null | grep -q "mistral"; then
    echo "Завантаження моделі $MODEL (~4.1 GB)..."
    ollama pull "$MODEL"
fi

# 4. Запуск асистента
echo "Запуск асистента..."
"$PYTHON" "$SCRIPT_DIR/mistral_chat.py"

# 5. Зупинка Ollama якщо ми його запустили
if [ -n "$OLLAMA_PID" ]; then
    echo "Зупинка Ollama..."
    kill "$OLLAMA_PID" 2>/dev/null
fi
