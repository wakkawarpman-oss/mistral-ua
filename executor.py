#!/usr/bin/env python3
"""
Містраль — Python Sandbox Executor
Безпечне виконання коду: subprocess + timeout + whitelist фільтр
"""

import subprocess
import tempfile
import textwrap
import re
from pathlib import Path

# Патерни які БЛОКУЄМО (деструктивні / мережеві / системні)
# Примітка: trailing \b не використовується — багато патернів закінчуються
# на non-word символ (напр. `eval\s*\(`), тому leading \b достатній.
_BLOCKED = re.compile(
    r"\b(os\.system|subprocess\.Popen|subprocess\.call|subprocess\.run"
    r"|__import__\s*\(\s*['\"]os['\"]"
    r"|open\s*\(.*['\"]w['\"]"
    r"|shutil\.(rmtree|move|copy)"
    r"|requests\.|httpx\.|urllib\.request"
    r"|socket\."
    r"|exec\s*\(|eval\s*\("
    r"|import\s+os\b"
    r"|from\s+os\s+import"
    r")",
    re.MULTILINE,
)

_MAX_CODE_LEN = 8192
_TIMEOUT_SEC  = 15


def is_safe(code: str) -> tuple[bool, str]:
    """Повертає (safe, reason). Мінімальна перевірка — не замінює sandbox."""
    if len(code) > _MAX_CODE_LEN:
        return False, f"Код занадто великий ({len(code)} > {_MAX_CODE_LEN} символів)"
    if m := _BLOCKED.search(code):
        return False, f"Заблокована операція: `{m.group(0).strip()}`"
    return True, ""


def run_code(code: str) -> dict:
    """
    Виконує Python код у ізольованому subprocess.

    Returns:
        {
            "stdout": str,
            "stderr": str,
            "returncode": int,
            "error": str | None,   # safety / timeout error
        }
    """
    ok, reason = is_safe(code)
    if not ok:
        return {"stdout": "", "stderr": "", "returncode": -1, "error": reason}

    # Записуємо у тимчасовий файл (безпечніше ніж -c для довгих скриптів)
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, dir="/tmp"
    ) as f:
        f.write(textwrap.dedent(code))
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SEC,
            cwd="/tmp",
            # Без доступу до домашньої директорії
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        return {
            "stdout":     result.stdout[:4096],
            "stderr":     result.stderr[:2048],
            "returncode": result.returncode,
            "error":      None,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout":     "",
            "stderr":     "",
            "returncode": -1,
            "error":      f"Timeout: виконання перевищило {_TIMEOUT_SEC}s",
        }
    except Exception as e:
        return {
            "stdout":     "",
            "stderr":     "",
            "returncode": -1,
            "error":      str(e),
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def format_result(r: dict) -> str:
    """Форматує результат для виводу в чат."""
    lines = []
    if r.get("error"):
        lines.append(f"⛔ {r['error']}")
    if r["stdout"]:
        lines.append(f"```\n{r['stdout'].rstrip()}\n```")
    if r["stderr"]:
        lines.append(f"⚠️ stderr:\n```\n{r['stderr'].rstrip()}\n```")
    if not lines:
        code = r.get("returncode", -1)
        lines.append(f"✅ Виконано (код {code}, без виводу)")
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test
    tests = [
        "print('Hello from Містраль sandbox')\nprint(2**10)",
        "import math\nprint([round(math.sin(i*0.5),3) for i in range(8)])",
        "os.system('rm -rf /')",   # має бути заблоковано
    ]
    for code in tests:
        print(f"\n{'─'*40}\nКод: {code[:60]!r}")
        r = run_code(code)
        print(format_result(r))
