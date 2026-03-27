#!/usr/bin/env python3
"""
Перевірка статусу Mistral та системи
"""

import subprocess
import requests
import platform
import sys

OLLAMA_URL = "http://localhost:11434"
MODEL = "mistral:7b-instruct-q4_0"


def check(label, ok, info=""):
    status = "[OK]" if ok else "[X]"
    print(f"  {status} {label}" + (f" — {info}" if info else ""))
    return ok


def main():
    print("\n=== Статус системи Mistral ML Engineer ===\n")

    # Система
    print("Система:")
    import platform
    mac_ver = platform.mac_ver()[0]
    check("macOS", True, mac_ver)
    check("Python", True, sys.version.split()[0])
    check("Apple Silicon", platform.machine() == "arm64", platform.machine())

    # RAM
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
        ram_gb = int(result.stdout.strip()) / (1024**3)
        ok = ram_gb >= 8
        check("RAM", ok, f"{ram_gb:.0f} GB {'(мінімум для Mistral 7B Q4)' if ok else '(мало!)'}")
    except Exception:
        pass

    # Вільний диск
    try:
        result = subprocess.run(
            ["df", "-g", "/"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            free_gb = int(lines[1].split()[3])
            check("Вільне місце", free_gb >= 5, f"{free_gb} GB")
    except Exception:
        pass

    print("\nOllama:")

    # Ollama встановлений
    try:
        result = subprocess.run(
            ["which", "ollama"],
            capture_output=True, text=True
        )
        check("Ollama встановлений", result.returncode == 0, result.stdout.strip())
    except Exception:
        check("Ollama встановлений", False)

    # Ollama запущений
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        running = resp.status_code == 200
        check("Ollama сервер запущений", running, OLLAMA_URL if running else "запусти: ollama serve")
    except Exception:
        check("Ollama сервер запущений", False, "запусти: ollama serve")
        print("\n  Для запуску: ollama serve")
        return

    print("\nМодель:")

    # Список моделей
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = resp.json().get("models", [])
        if models:
            for m in models:
                size_gb = m.get("size", 0) / (1024**3)
                check(m["name"], True, f"{size_gb:.1f} GB")
        else:
            check("Моделі", False, f"Немає. Завантаж: ollama pull {MODEL}")
    except Exception as e:
        check("Моделі", False, str(e))

    print("\nPython бібліотеки:")
    libs = ["ollama", "requests", "rich"]
    for lib in libs:
        try:
            __import__(lib)
            check(lib, True)
        except ImportError:
            check(lib, False, f"pip install {lib}")

    print("\n" + "="*40)
    print("Запуск асистента: python mistral_chat.py")
    print("або: ./start.sh")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()
