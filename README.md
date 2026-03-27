# Mistral ML Engineer R&D — Локальна установка

## Що це?
Локальний AI-асистент на базі **Mistral 7B** (4-bit квантизація), оптимізований для **MacBook M2 з 8 GB RAM**.

Підхід: **асиметричне мислення** — пошук простих, але неочевидних рішень.

---

## Структура проєкту
```
MISTRAL/
├── mistral_chat.py      # Інтерактивний чат з Mistral
├── mistral_api.py       # Python API для інтеграції
├── check_status.py      # Перевірка статусу системи
├── start.sh             # Автоматичний запуск
├── requirements.txt     # Python залежності
└── venv/                # Virtual environment
```

---

## Швидкий старт

### 1. Перевірка статусу
```bash
python check_status.py
```

### 2. Запуск (автоматично)
```bash
chmod +x start.sh
./start.sh
```

### 3. Запуск вручну
```bash
# Термінал 1 — запустити Ollama
ollama serve

# Термінал 2 — запустити чат
source venv/bin/activate
python mistral_chat.py
```

---

## Команди в чаті
| Команда | Дія |
|---------|-----|
| `/clear` | Очистити контекст розмови |
| `/info` | Показати статус моделі |
| `/exit` | Вийти |

---

## Python API
```python
from mistral_api import MistralML

ml = MistralML()

# Звичайний запит
answer = ml.ask("Як оптимізувати трансформер для edge-пристроїв?")

# Асиметричний аналіз
analysis = ml.analyze_asymmetric("Проблема малих даних у ML")

# Генерація коду
code = ml.generate_code("детектор аномалій для часових рядів")

# Стрімінг
for chunk in ml.stream("Поясни LoRA фінтюнінг"):
    print(chunk, end="", flush=True)
```

---

## Технічні параметри
| Параметр | Значення |
|----------|----------|
| Модель | Mistral 7B Instruct Q4_0 |
| RAM (модель) | ~4.1 GB |
| Контекст | 4096 токенів |
| Backend | Apple Metal (MPS) через Ollama |
| Швидкість | ~15-25 токенів/сек на M2 |

---

## Загрузка моделі (якщо не завантажена)
```bash
ollama pull mistral:7b-instruct-q4_0
```

## Альтернативні моделі для 8 GB RAM
```bash
ollama pull mistral:latest           # Те ж саме
ollama pull phi3:mini                # Менша, швидша (3.8B)
ollama pull gemma2:2b                # Дуже швидка (2B)
```
