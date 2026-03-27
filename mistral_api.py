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

    def doublecheck(self, question: str, answer: str) -> str:
        """Верифікує відповідь: точність, логіка, пропуски, вердикт."""
        from modules import MODULES
        system = MODULES["doublecheck"]["system"]
        prompt = f"ПИТАННЯ:\n{question}\n\nВІДПОВІДЬ ДО ПЕРЕВІРКИ:\n{answer}"
        return self.ask(prompt, system=system, temperature=0.3)

    # ── Function calling (Groq tool_calls API) ────────────────────────────────

    def call_tool(
        self,
        prompt: str,
        tools: list[dict],
        system: str = ML_ENGINEER_SYSTEM,
        temperature: float = 0.3,
    ) -> dict:
        """
        Викликає Groq tool_calls API.
        tools — список OpenAI-style function schemas:
          [{"type":"function","function":{"name":...,"description":...,"parameters":{...}}}]
        Повертає:
          {"tool": str, "args": dict}  — коли модель обрала інструмент
          {"content": str}             — коли звичайна відповідь
          {"error": str}               — при помилці
        """
        if self.backend != "groq":
            return {"error": "Function calling доступне лише через Groq API"}

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
        try:
            client = _Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=1024,
            )
            choice = resp.choices[0]
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                tc = choice.message.tool_calls[0]
                return {
                    "tool": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                }
            return {"content": choice.message.content or ""}
        except Exception as e:
            return {"error": str(e)}

    def structured_output(
        self,
        prompt: str,
        schema: dict,
        name: str = "respond",
        description: str = "Structured response",
        system: str = ML_ENGINEER_SYSTEM,
    ) -> dict:
        """
        Витягує структуровані дані за JSON-схемою через function calling.
        Повертає розпарсений dict або {"error": str}.
        """
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema,
            },
        }
        result = self.call_tool(prompt, tools=[tool], system=system)
        if "args" in result:
            return result["args"]
        if "error" in result:
            return result
        # fallback: намагаємось розпарсити JSON з вільного тексту
        try:
            content = result.get("content", "")
            start = content.find("{")
            end   = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except Exception:
            pass
        return {"error": "Не вдалося витягти структуровані дані", "raw": result.get("content", "")}

    def clear_history(self):
        self.history = []


if __name__ == "__main__":
    ml = MistralML()
    print(f"Бекенд: {ml.backend}")
    if ml.is_ready():
        print(ml.ask("Привіт! Що ти вмієш?"))
