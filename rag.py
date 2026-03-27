#!/usr/bin/env python3
"""
Містраль — RAG (Retrieval-Augmented Generation)
TF-IDF пошук по knowledge/ без важких залежностей.
Працює offline, без векторних БД.

Використання:
    from rag import RAG
    rag = RAG()                       # завантажить knowledge/
    hits = rag.search("дрон FPV")     # повертає список { text, score, source }
    context = rag.context("дрон FPV") # готовий блок для системного промпту
"""

import re
from pathlib import Path
from typing import TypedDict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_CHUNK_SIZE   = 400   # слів на чанк
_OVERLAP      = 80    # слів перекриття між чанками
_TOP_K        = 4     # скільки фрагментів повертати


class Hit(TypedDict):
    text:   str
    score:  float
    source: str


def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _OVERLAP) -> list[str]:
    """Ділить текст на чанки зі скраєм перекриття."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks


def _load_md(path: Path) -> str:
    """Читає markdown, прибирає заголовки # і зайві пробіли."""
    raw = path.read_text(encoding="utf-8")
    # Прибираємо markdown розмітку для кращого TF-IDF
    raw = re.sub(r"^#+\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\*\*(.+?)\*\*", r"\1", raw)
    raw = re.sub(r"`[^`]+`", "", raw)
    return raw.strip()


class RAG:
    """
    TF-IDF Knowledge Retriever.
    При init — сканує knowledge/ і будує індекс.
    """

    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR):
        self._dir    = knowledge_dir
        self._chunks: list[str] = []
        self._metas:  list[str] = []   # source filename per chunk
        self._vec:    TfidfVectorizer | None = None
        self._matrix = None
        self._build_index()

    # ── Index ─────────────────────────────────────────────────────────────────

    def _build_index(self) -> None:
        """Завантажує всі .md файли з knowledge/ і будує TF-IDF матрицю."""
        self._chunks = []
        self._metas  = []

        if not self._dir.exists():
            self._dir.mkdir(parents=True)
            return

        for path in sorted(self._dir.glob("**/*.md")):
            text   = _load_md(path)
            chunks = _chunk_text(text)
            for ch in chunks:
                self._chunks.append(ch)
                self._metas.append(path.name)

        if not self._chunks:
            return

        self._vec    = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=10_000,
            sublinear_tf=True,
        )
        self._matrix = self._vec.fit_transform(self._chunks)

    def reload(self) -> int:
        """Перебудовує індекс після додавання нових файлів. Повертає кількість чанків."""
        self._build_index()
        return len(self._chunks)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = _TOP_K) -> list[Hit]:
        """Семантичний TF-IDF пошук. Повертає list[Hit]."""
        if not self._chunks or self._vec is None:
            return []

        q_vec   = self._vec.transform([query])
        scores  = cosine_similarity(q_vec, self._matrix).flatten()
        indices = np.argsort(scores)[::-1][:top_k]

        hits: list[Hit] = []
        for i in indices:
            s = float(scores[i])
            if s < 0.01:
                continue
            hits.append(Hit(text=self._chunks[i], score=round(s, 4), source=self._metas[i]))
        return hits

    def context(self, query: str, top_k: int = _TOP_K) -> str:
        """
        Повертає готовий блок тексту для вставки в системний промпт.
        Порожній рядок якщо знань не знайдено.
        """
        hits = self.search(query, top_k)
        if not hits:
            return ""
        parts = [f"[{h['source']} | relevance={h['score']}]\n{h['text']}" for h in hits]
        return "\n\n---\n".join(parts)

    # ── Write ─────────────────────────────────────────────────────────────────

    def remember(self, text: str, filename: str = "notes.md") -> Path:
        """
        Зберігає новий блок знань у knowledge/<filename>.
        Автоматично перебудовує індекс.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / filename
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n\n{text.strip()}\n")
        self.reload()
        return path

    def stats(self) -> dict:
        """Статистика індексу."""
        return {
            "chunks":    len(self._chunks),
            "files":     len(set(self._metas)),
            "dir":       str(self._dir),
        }


# ── Singleton для використання в web_server і telegram_bot ───────────────────
_rag: RAG | None = None


def get_rag() -> RAG:
    global _rag
    if _rag is None:
        _rag = RAG()
    return _rag


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = get_rag()
    print(f"RAG stats: {rag.stats()}")

    queries = ["дрон", "MLX", "проект Містраль", "асиметричний підхід"]
    for q in queries:
        hits = rag.search(q, top_k=2)
        print(f"\nЗапит: '{q}'  ({len(hits)} hits)")
        for h in hits:
            print(f"  [{h['source']} | {h['score']}] {h['text'][:120]}...")
