"""
Система знань Містраль
─────────────────────
knowledge/          ← постійні знання (.md файли)
memory.json         ← динамічна пам'ять (/remember)

Використання:
    from knowledge import load_context, remember, forget_all, list_memory
"""

import json
import os
from pathlib import Path
from datetime import datetime

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
MEMORY_FILE   = Path(__file__).parent / "memory.json"


# ─── Постійні знання (knowledge/*.md) ──────────────────────────────────────────

def load_knowledge() -> str:
    """Читає всі .md файли з knowledge/ і повертає їх вміст."""
    if not KNOWLEDGE_DIR.exists():
        return ""

    parts = []
    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8").strip()
        if text:
            parts.append(f"### [{md_file.stem}]\n{text}")

    return "\n\n".join(parts)


def knowledge_files() -> list[str]:
    """Список файлів у knowledge/."""
    if not KNOWLEDGE_DIR.exists():
        return []
    return [f.name for f in sorted(KNOWLEDGE_DIR.glob("*.md"))]


def add_knowledge_file(name: str, content: str) -> str:
    """Створює новий .md файл у knowledge/."""
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    fname = name if name.endswith(".md") else f"{name}.md"
    path = KNOWLEDGE_DIR / fname
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return str(path)


# ─── Динамічна пам'ять (memory.json) ───────────────────────────────────────────

def _load_raw() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_raw(entries: list[dict]) -> None:
    MEMORY_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def remember(text: str, tag: str = "") -> None:
    """Зберегти запис у пам'ять."""
    entries = _load_raw()
    entries.append({
        "text": text.strip(),
        "tag": tag.strip(),
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    _save_raw(entries)


def forget_last(n: int = 1) -> int:
    """Видалити останні n записів. Повертає скільки видалено."""
    entries = _load_raw()
    removed = min(n, len(entries))
    _save_raw(entries[:-removed] if removed else entries)
    return removed


def forget_all() -> None:
    """Очистити всю пам'ять."""
    _save_raw([])


def list_memory() -> list[dict]:
    """Список усіх записів."""
    return _load_raw()


def load_memory() -> str:
    """Повертає пам'ять як текст для системного промпту."""
    entries = _load_raw()
    if not entries:
        return ""
    lines = []
    for e in entries:
        tag = f" #{e['tag']}" if e.get("tag") else ""
        lines.append(f"- [{e['ts']}]{tag} {e['text']}")
    return "\n".join(lines)


# ─── Об'єднаний контекст ────────────────────────────────────────────────────────

def load_context() -> str:
    """
    Повертає рядок для додавання в системний промпт.
    Порожній рядок — якщо немає нічого зберігати.
    """
    parts = []

    knowledge = load_knowledge()
    if knowledge:
        parts.append("## БАЗА ЗНАНЬ\n" + knowledge)

    memory = load_memory()
    if memory:
        parts.append("## ПАМ'ЯТЬ\n" + memory)

    if not parts:
        return ""

    return "\n\n" + "\n\n".join(parts)
