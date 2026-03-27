#!/usr/bin/env python3
"""
Містраль — Groq-powered чат (fallback: Ollama local)
~500 tok/s через Groq LPU | M2 local як резерв
"""

import os
import sys
import json
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

console = Console()

# ─── Конфігурація ───────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # ~500 tok/s
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral-ua")

SYSTEM_PROMPT = """Ти — Містраль, асиметричний інженер-аналітик R&D.
Спілкуєшся українською. Відповідаєш технічно, конкретно, без води.
Використовуй вбудовані інженерні модулі: логіка, каузальний аналіз,
red team, стратегія, код, аудит, DevOps, QA, DSP, CV, оптимізація."""


# ─── Groq backend ───────────────────────────────────────────
def stream_groq(messages: list) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    full = ""
    with client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full += delta
    print()
    return full


# ─── Ollama fallback ────────────────────────────────────────
def stream_ollama(messages: list) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.7, "num_ctx": 8192, "num_predict": 4096},
    }
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120
        ) as resp:
            full = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            print(chunk, end="", flush=True)
                            full += chunk
                        if data.get("done"):
                            print()
                            break
                    except json.JSONDecodeError:
                        continue
            return full
    except Exception as e:
        console.print(f"\n[red]Ollama помилка: {e}[/red]")
        return ""


# ─── Авто-вибір бекенду ─────────────────────────────────────
def detect_backend() -> str:
    if GROQ_AVAILABLE and GROQ_API_KEY:
        return "groq"
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if r.status_code == 200:
            return "ollama"
    except Exception:
        pass
    return "none"


def stream_response(messages: list, backend: str) -> str:
    if backend == "groq":
        try:
            return stream_groq(messages)
        except Exception as e:
            console.print(f"[yellow]Groq недоступний ({e}), fallback → Ollama[/yellow]")
            return stream_ollama(messages)
    return stream_ollama(messages)


# ─── Main ───────────────────────────────────────────────────
def main():
    backend = detect_backend()

    backend_label = {
        "groq":   "[bold green]Groq LPU[/bold green] (~500 tok/s)",
        "ollama": "[bold yellow]Ollama local[/bold yellow] (M2)",
        "none":   "[bold red]бекенд не знайдено![/bold red]",
    }[backend]

    console.print(Panel.fit(
        f"[bold cyan]Містраль[/bold cyan] | {backend_label}\n"
        "[dim]Команди: /exit  /clear  /info  /backend[/dim]",
        border_style="cyan",
    ))

    if backend == "none":
        console.print("[red]Встанови GROQ_API_KEY у .env або запусти ollama serve[/red]")
        sys.exit(1)

    if backend == "groq":
        console.print(f"[dim]Модель: {GROQ_MODEL}[/dim]\n")
    else:
        console.print(f"[dim]Модель: {OLLAMA_MODEL}[/dim]\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = Prompt.ask("[bold blue]Ти[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]До побачення![/dim]")
            break

        if not user_input:
            continue

        match user_input.lower():
            case "/exit" | "/quit" | "exit" | "quit":
                console.print("[dim]До побачення![/dim]")
                break
            case "/clear":
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                console.print("[green]Контекст очищено[/green]")
                continue
            case "/info":
                console.print(f"[cyan]Бекенд:[/cyan] {backend}")
                console.print(f"[cyan]Повідомлень:[/cyan] {len(messages) - 1}")
                continue
            case "/backend":
                backend = detect_backend()
                console.print(f"[cyan]Поточний бекенд: {backend}[/cyan]")
                continue

        messages.append({"role": "user", "content": user_input})
        console.print("\n[bold green]Містраль:[/bold green]")

        response = stream_response(messages, backend)
        if response:
            messages.append({"role": "assistant", "content": response})
            console.print()


if __name__ == "__main__":
    main()


console = Console()

OLLAMA_URL = "http://localhost:11434"
MODEL = "mistral-ua"

# Системний промпт — ML-інженер R&D з асиметричним мисленням
SYSTEM_PROMPT = """Ти — професійний ML-інженер R&D на ім'я Містраль.
Твій підхід — асиметричний: ти завжди шукаєш прості, але неочевидні рішення там, де інші шукають складні.

Твої принципи:
1. Простота над складністю — найкраще рішення часто найпростіше
2. Асиметричне мислення — виявляй слабкі місця стандартних підходів
3. Практичність — пропонуй рішення, які можна реалізувати зараз
4. Критичний аналіз — вмій критикувати власні рішення (red teaming)
5. Ефективність ресурсів — оптимізуй під доступні ресурси (M2, 8GB RAM)

Ти спілкуєшся українською і надаєш конкретні, технічні відповіді з кодом коли потрібно.
Ти розбираєшся в: ML/DL, Python, PyTorch, обробці даних, OSINT, автоматизації."""


def check_ollama_running():
    """Перевірка чи запущений Ollama"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def check_model_available():
    """Перевірка наявності моделі"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(MODEL.split(":")[0] in m for m in models)
    except Exception:
        pass
    return False


def stream_response(messages):
    """Стрімінг відповіді від Mistral"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "num_ctx": 4096,
            "num_predict": 2048,
        }
    }

    try:
        with requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=120
        ) as resp:
            full_response = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "message" in data and "content" in data["message"]:
                            chunk = data["message"]["content"]
                            full_response += chunk
                            print(chunk, end="", flush=True)
                        if data.get("done", False):
                            print()
                            break
                    except json.JSONDecodeError:
                        continue
            return full_response
    except requests.exceptions.Timeout:
        console.print("\n[red]Timeout: модель відповідає занадто довго[/red]")
        return ""
    except Exception as e:
        console.print(f"\n[red]Помилка: {e}[/red]")
        return ""


def main():
    console.print(Panel.fit(
        "[bold cyan]Mistral ML Engineer R&D[/bold cyan]\n"
        "[dim]Асиметричний підхід до ML задач | M2 Mac оптимізовано[/dim]\n"
        "[dim]Команди: /exit — вихід | /clear — очистити | /info — статус[/dim]",
        border_style="cyan"
    ))

    # Перевірка Ollama
    if not check_ollama_running():
        console.print("[red]Ollama не запущений![/red]")
        console.print("[yellow]Запусти: ollama serve[/yellow]")
        sys.exit(1)

    # Перевірка моделі
    if not check_model_available():
        console.print(f"[red]Модель {MODEL} не знайдена![/red]")
        console.print(f"[yellow]Завантаж: ollama pull {MODEL}[/yellow]")

        # Перевірка яка модель є
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            models = [m["name"] for m in resp.json().get("models", [])]
            if models:
                console.print(f"[green]Доступні моделі: {', '.join(models)}[/green]")
                console.print("[yellow]Використай одну з них або дочекайся завантаження Mistral[/yellow]")
        except Exception:
            pass
        sys.exit(1)

    console.print(f"[green]Модель {MODEL} готова[/green]\n")

    # Контекст розмови
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Головний цикл
    while True:
        try:
            user_input = Prompt.ask("[bold blue]Ти[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]До побачення![/dim]")
            break

        if not user_input:
            continue

        # Команди
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]До побачення![/dim]")
            break
        elif user_input.lower() == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            console.print("[green]Контекст очищено[/green]")
            continue
        elif user_input.lower() == "/info":
            console.print(f"[cyan]Модель:[/cyan] {MODEL}")
            console.print(f"[cyan]Повідомлень у контексті:[/cyan] {len(messages) - 1}")
            console.print(f"[cyan]Ollama:[/cyan] {OLLAMA_URL}")
            continue

        messages.append({"role": "user", "content": user_input})

        console.print(f"\n[bold green]Містраль:[/bold green]")
        response = stream_response(messages)

        if response:
            messages.append({"role": "assistant", "content": response})
            console.print()


if __name__ == "__main__":
    main()
