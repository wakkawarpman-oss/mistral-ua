#!/usr/bin/env python3
"""
Містраль — Agent Pipeline
Ланцюжок: Planner → Coder → DoubleCheck → RedTeam
Кожен крок використовує відповідний системний промпт з modules.py
"""

import json
import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator

from modules import MODULES

try:
    from groq import AsyncGroq as _AsyncGroq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral-ua")


# ── AI call ───────────────────────────────────────────────────────────────────

async def _ai(system: str, user: str, temperature: float = 0.5) -> str:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    if _GROQ_AVAILABLE and GROQ_API_KEY:
        client = _AsyncGroq(api_key=GROQ_API_KEY)
        resp = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=msgs,
            temperature=temperature,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""
    else:
        import httpx
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{OLLAMA_URL}/api/chat", json={
                "model": OLLAMA_MODEL,
                "messages": msgs,
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": 8192},
            })
            return r.json()["message"]["content"]


# ── Pipeline Step ─────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    step:    str
    output:  str
    ok:      bool = True
    error:   str  = ""


@dataclass
class PipelineResult:
    task:   str
    steps:  list[StepResult] = field(default_factory=list)

    def final(self) -> str:
        """Останній успішний вивід."""
        for s in reversed(self.steps):
            if s.ok:
                return s.output
        return ""

    def summary(self) -> str:
        lines = [f"# Agent Pipeline: {self.task}\n"]
        for i, s in enumerate(self.steps, 1):
            icon = "✅" if s.ok else "❌"
            lines.append(f"## {icon} Крок {i}: {s.step}")
            lines.append(s.output if s.ok else f"Помилка: {s.error}")
            lines.append("")
        return "\n".join(lines)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class AgentPipeline:
    """
    Послідовний ланцюжок агентів:
      1. Planner  — розкладає задачу на кроки
      2. Coder    — пише код/рішення
      3. DoubleCheck — верифікує правильність
      4. RedTeam  — знаходить слабкі місця

    Кожен крок отримує вивід попереднього як контекст.
    """

    STEPS = [
        ("planner",     "Стратегічний план",  0.6),
        ("coder",       "Реалізація",         0.3),
        ("doublecheck", "Верифікація",        0.2),
        ("redteam",     "Red Team аналіз",    0.5),
    ]

    async def run(self, task: str) -> PipelineResult:
        result  = PipelineResult(task=task)
        context = task

        for module_key, step_name, temp in self.STEPS:
            system = MODULES[module_key]["system"]
            prompt = (
                f"ЗАВДАННЯ:\n{task}\n\n"
                f"КОНТЕКСТ (вивід попереднього кроку):\n{context}"
                if context != task else
                f"ЗАВДАННЯ:\n{task}"
            )
            try:
                output = await _ai(system, prompt, temperature=temp)
                sr = StepResult(step=step_name, output=output)
                result.steps.append(sr)
                context = output  # передаємо далі
            except Exception as e:
                result.steps.append(StepResult(
                    step=step_name, output="", ok=False, error=str(e)
                ))
                break  # зупиняємось при помилці

        return result

    async def stream(self, task: str) -> AsyncGenerator[dict, None]:
        """
        Стрімінг по кроках. Кожен yield:
          {"step": str, "chunk": str} під час генерації
          {"step": str, "done": True, "output": str} по завершенню кроку
        """
        context = task

        for module_key, step_name, temp in self.STEPS:
            system = MODULES[module_key]["system"]
            prompt = (
                f"ЗАВДАННЯ:\n{task}\n\nКОНТЕКСТ:\n{context}"
                if context != task else
                f"ЗАВДАННЯ:\n{task}"
            )
            msgs = [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ]
            full = ""

            try:
                if _GROQ_AVAILABLE and GROQ_API_KEY:
                    client = _AsyncGroq(api_key=GROQ_API_KEY)
                    async with client.chat.completions.stream(
                        model=GROQ_MODEL, messages=msgs,
                        temperature=temp, max_tokens=2048,
                    ) as s:
                        async for chunk in s:
                            delta = chunk.choices[0].delta.content or ""
                            if delta:
                                full += delta
                                yield {"step": step_name, "chunk": delta}
                else:
                    import httpx
                    async with httpx.AsyncClient(timeout=120) as c:
                        async with c.stream("POST", f"{OLLAMA_URL}/api/chat", json={
                            "model": OLLAMA_MODEL, "messages": msgs,
                            "stream": True,
                            "options": {"temperature": temp, "num_ctx": 8192},
                        }) as r:
                            async for line in r.aiter_lines():
                                if line:
                                    d = json.loads(line)
                                    ch = d.get("message", {}).get("content", "")
                                    if ch:
                                        full += ch
                                        yield {"step": step_name, "chunk": ch}

                context = full
                yield {"step": step_name, "done": True, "output": full}

            except Exception as e:
                yield {"step": step_name, "done": True, "output": "", "error": str(e)}
                break


# ── Singleton ─────────────────────────────────────────────────────────────────
_pipeline: AgentPipeline | None = None


def get_pipeline() -> AgentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AgentPipeline()
    return _pipeline


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    async def demo():
        pipeline = AgentPipeline()
        task = "Розробити мінімальний детектор аномалій для часового ряду навантаження CPU на Python"
        print(f"Pipeline task: {task}\n{'='*60}")
        result = await pipeline.run(task)
        print(result.summary())

    asyncio.run(demo())
