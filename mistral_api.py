#!/usr/bin/env python3
"""
Містраль — API модуль (Groq primary | Ollama fallback)
"""

import os
import json
import requests
from typing import Generator
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq as _Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral-ua")

ML_ENGINEER_SYSTEM = """Ти — Містраль, асиметричний інженер-аналітик R&D.
Спілкуєшся українською. Відповідаєш технічно, конкретно, без води."""


class MistralML:
    """Клієнт Містраль: Groq (~500 tok/s) або Ollama (local M2)"""

    def __init__(self):
        self.history: list = []
        self.backend = self._detect()

    def _detect(self) -> str:
        if _GROQ_AVAILABLE and GROQ_API_KEY:
            return "groq"
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                return "ollama"
        except Exception:
            pass
        return "none"

    def is_ready(self) -> bool:
        return self.backend != "none"

    def ask(
        self,
        prompt: str,
        system: str = ML_ENGINEER_SYSTEM,
        temperature: float = 0.7,
        keep_history: bool = False,
    ) -> str:
        messages = [{"role": "system", "content": system}]
        if keep_history:
            messages += self.history
        messages.append({"role": "user", "content": prompt})

        result = ""
        if self.backend == "groq":
            result = self._ask_groq(messages, temperature)
        else:
            result = self._ask_ollama(messages, temperature)

        if keep_history:
            self.history += [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": result},
            ]
        return result

    def _ask_groq(self, messages: list, temperature: float) -> str:
        client = _Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
            stream=False,
        )
        return resp.choices[0].message.content or ""

    def _ask_ollama(self, messages: list, temperature: float) -> str:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 8192},
        }
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
            return r.json()["message"]["content"]
        except Exception as e:
            return f"Помилка: {e}"

    def stream(
        self,
        prompt: str,
        system: str = ML_ENGINEER_SYSTEM,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if self.backend == "groq":
            yield from self._stream_groq(messages, temperature)
        else:
            yield from self._stream_ollama(messages, temperature)

    def _stream_groq(self, messages: list, temperature: float) -> Generator[str, None, None]:
        client = _Groq(api_key=GROQ_API_KEY)
        with client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
            stream=True,
        ) as s:
            for chunk in s:
                yield chunk.choices[0].delta.content or ""

    def _stream_ollama(self, messages: list, temperature: float) -> Generator[str, None, None]:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_ctx": 8192},
        }
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if chunk := data.get("message", {}).get("content", ""):
                            yield chunk
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    def analyze_asymmetric(self, problem: str) -> str:
        return self.ask(
            f"Проаналізуй асиметрично:\n{problem}\n\n"
            "1. СТАНДАРТНЕ РІШЕННЯ і його слабкості\n"
            "2. АСИМЕТРИЧНЕ РІШЕННЯ (простіше, неочевидне)\n"
            "3. РЕАЛІЗАЦІЯ (код або кроки)\n"
            "4. РИЗИКИ",
            temperature=0.8,
        )

    def generate_code(self, task: str) -> str:
        return self.ask(
            f"Напиши повний робочий Python код для: {task}\n"
            "Type hints, docstrings, обробка помилок, оптимізовано під M2/MPS.",
            temperature=0.3,
        )

    def clear_history(self):
        self.history = []


if __name__ == "__main__":
    ml = MistralML()
    print(f"Бекенд: {ml.backend}")
    if ml.is_ready():
        print(ml.ask("Привіт! Що ти вмієш?"))


ML_ENGINEER_SYSTEM = """Ти — асиметричний ML-інженер R&D.
Аналізуй проблеми з неочевидних кутів. Шукай прості рішення там де всі шукають складні.
Відповідай конкретно. Надавай робочий Python код. Вказуй на слабкі місця підходів."""


class MistralML:
    """Клієнт для Mistral через Ollama"""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url
        self.history = []

    def is_ready(self) -> bool:
        """Перевірка готовності сервісу"""
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=3)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model.split(":")[0] in m for m in models)
        except Exception:
            return False

    def list_models(self) -> list:
        """Список доступних моделей"""
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=3)
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def ask(
        self,
        prompt: str,
        system: str = ML_ENGINEER_SYSTEM,
        temperature: float = 0.7,
        keep_history: bool = False
    ) -> str:
        """Одиночний запит без стрімінгу"""
        messages = [{"role": "system", "content": system}]

        if keep_history:
            messages += self.history

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 4096,
                "num_predict": 2048,
            }
        }

        try:
            resp = requests.post(
                f"{self.url}/api/chat",
                json=payload,
                timeout=120
            )
            result = resp.json()["message"]["content"]

            if keep_history:
                self.history.append({"role": "user", "content": prompt})
                self.history.append({"role": "assistant", "content": result})

            return result
        except Exception as e:
            return f"Помилка: {e}"

    def stream(
        self,
        prompt: str,
        system: str = ML_ENGINEER_SYSTEM,
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """Стрімінг відповіді"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_ctx": 4096}
        }

        with requests.post(
            f"{self.url}/api/chat",
            json=payload,
            stream=True,
            timeout=120
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if chunk := data.get("message", {}).get("content", ""):
                            yield chunk
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    def analyze_asymmetric(self, problem: str) -> str:
        """Асиметричний аналіз проблеми"""
        prompt = f"""Проаналізуй цю проблему з асиметричної точки зору:

ПРОБЛЕМА: {problem}

Дай відповідь у форматі:
1. СТАНДАРТНЕ РІШЕННЯ (яке всі використовують і чому воно може бути неефективним)
2. СЛАБКІ МІСЦЯ стандартного підходу
3. АСИМЕТРИЧНЕ РІШЕННЯ (просте, але неочевидне)
4. РЕАЛІЗАЦІЯ (конкретні кроки або код)
5. РИЗИКИ та як їх мінімізувати"""

        return self.ask(prompt, temperature=0.8)

    def generate_code(self, task: str) -> str:
        """Генерація Python коду"""
        prompt = f"""Напиши робочий Python код для: {task}

Вимоги:
- Оптимізовано для Apple M2 (8GB RAM)
- Використовуй MPS backend для PyTorch якщо потрібно
- Мінімальні залежності
- Коментарі українською мовою
- Передбач обробку помилок"""

        return self.ask(prompt, temperature=0.3)

    def clear_history(self):
        """Очистити історію розмови"""
        self.history = []


# Приклад використання
if __name__ == "__main__":
    ml = MistralML()

    if not ml.is_ready():
        print(f"Сервіс недоступний. Доступні моделі: {ml.list_models()}")
        exit(1)

    # Тест асиметричного аналізу
    result = ml.analyze_asymmetric(
        "Як навчити ML-модель з мінімумом розміченних даних?"
    )
    print(result)
