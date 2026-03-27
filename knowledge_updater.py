#!/usr/bin/env python3
"""
Містраль — Military Knowledge Updater
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Завантажує ТІЛЬКИ верифіковані відкриті дані з дозволених джерел:
  - Wikipedia (через MediaWiki API)
  - Wikimedia Commons metadata
  - OSINT open-source data (публічні PDF через посилання з whitelist)

НЕ генерує дані! НЕ використовує AI для "вигадування" фактів.
Все збережене — реальні тексти з верифікованих джерел.

Використання:
    python3 knowledge_updater.py --topic "FPV drone" --source wikipedia
    python3 knowledge_updater.py --topic "артилерія" --lang uk
    python3 knowledge_updater.py --list-topics
"""

import re
import sys
import json
import time
import hashlib
import argparse
import textwrap
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

# ── Конфіг ───────────────────────────────────────────────────────────────────

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge" / "military"

# WHITELIST джерел — ТІЛЬКИ вони дозволені
ALLOWED_SOURCES = {
    "wikipedia",          # Wikipedia API (uk/en)
    "wikidata",           # Wikidata structured facts
}

# Теми, які ДОЗВОЛЕНІ для завантаження (military-relevant)
TOPIC_WHITELIST_PATTERNS = [
    r"drone|бпла|fpv|uav|uas",
    r"artiller|артилері",
    r"rocket|ракет",
    r"missile|ракет",
    r"armor|броне|tank|танк",
    r"infantry|піхот",
    r"electronic.*war|радіоелектронн",
    r"counter.*battery|контрбатарей",
    r"sniper|снайпер",
    r"mine|міна|мінн",
    r"radar|радар|radar",
    r"satellite|супутник",
    r"camouflage|маскуванн",
    r"logistics|логістик",
    r"nato|нато",
    r"military|військ",
    r"combat|бойов",
    r"weapon|зброя|weapon",
    r"grenade|граната",
    r"mortar|мінометн|mortir",
    r"fortif|укріпленн|fortif",
    r"intelligence|розвідк",
    r"special.*force|спецназ|спецпризначен",
]

TOPIC_RE = re.compile(
    "|".join(TOPIC_WHITELIST_PATTERNS), re.IGNORECASE
)

# ── Верифікація теми ──────────────────────────────────────────────────────────

def is_topic_allowed(topic: str) -> bool:
    return bool(TOPIC_RE.search(topic))


# ── Wikipedia API ─────────────────────────────────────────────────────────────

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
HEADERS  = {"User-Agent": "MistralMilitaryKB/1.0 (educational; non-commercial)"}


def _wiki_request(lang: str, params: dict) -> dict:
    url = WIKI_API.format(lang=lang)
    params["format"] = "json"
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    req = urllib.request.Request(full_url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_wikipedia(topic: str, lang: str = "uk", limit: int = 5) -> list[dict]:
    """Пошук статей Wikipedia за темою."""
    data = _wiki_request(lang, {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": limit,
        "srnamespace": 0,
    })
    return data.get("query", {}).get("search", [])


def fetch_wikipedia_article(page_id: int, lang: str = "uk") -> dict | None:
    """
    Завантажує повний текст статті за page_id.
    Повертає {'title': str, 'text': str, 'url': str, 'timestamp': str} або None.
    """
    data = _wiki_request(lang, {
        "action": "query",
        "pageids": page_id,
        "prop": "extracts|info",
        "exintro": False,       # вся стаття, не тільки вступ
        "explaintext": True,    # plain text, без HTML
        "inprop": "url",
    })
    pages = data.get("query", {}).get("pages", {})
    page  = pages.get(str(page_id), {})
    if "extract" not in page:
        return None
    return {
        "title":     page.get("title", ""),
        "text":      page["extract"],
        "url":       page.get("fullurl", f"https://{lang}.wikipedia.org/?curid={page_id}"),
        "timestamp": page.get("touched", ""),
        "lang":      lang,
        "source":    "wikipedia",
        "page_id":   page_id,
    }


# ── Очищення тексту ───────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """Прибирає зайве, залишає інформативний текст."""
    # Видаляємо порожні секції-заголовки
    text = re.sub(r"\n{3,}", "\n\n", raw)
    # Видаляємо рядки-"диви" що складаються тільки з пробілів
    text = "\n".join(l for l in text.splitlines() if l.strip() or l == "")
    return text.strip()


# ── Збереження ────────────────────────────────────────────────────────────────

def _safe_filename(title: str, lang: str) -> str:
    slug = re.sub(r"[^\w\-]", "_", title.lower())[:60]
    slug = re.sub(r"_+", "_", slug).strip("_")
    return f"{lang}_{slug}.md"


def save_article(article: dict, overwrite: bool = False) -> Path:
    """
    Зберігає статтю у knowledge/military/ у форматі Markdown.
    Повертає шлях до збереженого файлу.
    """
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    fname  = _safe_filename(article["title"], article["lang"])
    fpath  = KNOWLEDGE_DIR / fname

    if fpath.exists() and not overwrite:
        print(f"  [вже існує] {fpath.name}")
        return fpath

    # Контрольна сума для запобігання дублям
    checksum = hashlib.sha256(article["text"].encode()).hexdigest()[:12]

    content = textwrap.dedent(f"""\
        ---
        title: {article['title']}
        source: {article['source']}
        url: {article['url']}
        lang: {article['lang']}
        updated: {article.get('timestamp', '')}
        saved_at: {datetime.now(timezone.utc).isoformat()}
        checksum: {checksum}
        verified: wikipedia_api
        ---

        # {article['title']}

        > Джерело: [{article['url']}]({article['url']})
        > Верифіковано: Wikipedia API (автоматично, не AI-генерація)

        {clean_text(article['text'])}
    """)

    fpath.write_text(content, encoding="utf-8")
    print(f"  [збережено] {fpath.name} ({len(article['text'])} символів)")
    return fpath


# ── RAG інтеграція ────────────────────────────────────────────────────────────

def update_rag_index():
    """Перебудовує RAG індекс після завантаження нових статей."""
    try:
        from rag import get_rag
        rag = get_rag()
        rag.reload()
        s = rag.stats()
        print(f"\n  RAG index: {s['chunks']} chunks із {s['files']} файлів")
    except ImportError:
        print("\n  [попередження] RAG модуль недоступний, пропускаємо індексацію")
    except Exception as e:
        print(f"\n  [попередження] RAG reload: {e}")


# ── Головна функція ────────────────────────────────────────────────────────────

def fetch_topic(
    topic: str,
    lang: str = "uk",
    limit: int = 3,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """
    Основна функція: шукає і зберігає статті за темою.
    Повертає список збережених файлів.
    """
    if not is_topic_allowed(topic):
        print(
            f"❌ Тема '{topic}' не входить у whitelist military-тематики.\n"
            f"   Дозволені теми: дрони, артилерія, ракети, РЕБ, НАТо, etc."
        )
        return []

    print(f"\n🔍 Пошук: '{topic}' [{lang}.wikipedia.org] (ліміт: {limit})")
    try:
        results = search_wikipedia(topic, lang=lang, limit=limit)
    except urllib.error.URLError as e:
        print(f"❌ Помилка мережі: {e}")
        return []

    if not results:
        print("   Нічого не знайдено.")
        return []

    saved = []
    for r in results:
        title   = r["title"]
        page_id = r["pageid"]
        snippet = re.sub(r"<[^>]+>", "", r.get("snippet", ""))[:80]
        print(f"\n  → {title} (id={page_id})")
        print(f"    {snippet}…")

        if dry_run:
            print("    [dry-run] пропускаємо завантаження")
            continue

        try:
            article = fetch_wikipedia_article(page_id, lang=lang)
            if article and len(article["text"]) > 200:
                path = save_article(article, overwrite=overwrite)
                saved.append(path)
            else:
                print(f"    [пропущено] стаття порожня або занадто коротка")
            time.sleep(0.5)  # поважаємо Wikipedia rate limit
        except Exception as e:
            print(f"    [помилка] {e}")

    return saved


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Завантажує перевірені military знання з Wikipedia в knowledge/military/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Приклади:
              python3 knowledge_updater.py --topic "FPV drone" --lang en
              python3 knowledge_updater.py --topic "артилерія" --lang uk --limit 5
              python3 knowledge_updater.py --topic "РЕБ" --dry-run
              python3 knowledge_updater.py --batch   (базовий набір тем)
        """),
    )
    p.add_argument("--topic",     type=str, help="Тема для пошуку")
    p.add_argument("--lang",      type=str, default="uk", choices=["uk", "en"], help="Мова (uk/en)")
    p.add_argument("--limit",     type=int, default=3, help="Макс. кількість статей (default: 3)")
    p.add_argument("--overwrite", action="store_true", help="Перезаписати існуючі файли")
    p.add_argument("--dry-run",   action="store_true", help="Пошук без збереження")
    p.add_argument("--batch",     action="store_true", help="Завантажити базовий набір military тем")
    p.add_argument("--list-topics", action="store_true", help="Показати whitelist тем")
    p.add_argument("--stats",     action="store_true", help="Статистика knowledge/military/")
    return p


BATCH_TOPICS = [
    ("FPV drone",              "en"),
    ("БПЛА",                   "uk"),
    ("артилерія",              "uk"),
    ("реактивна система залпового вогню", "uk"),
    ("протитанкова ракета",    "uk"),
    ("радіоелектронна боротьба", "uk"),
    ("маскування військ",      "uk"),
    ("drone warfare",          "en"),
    ("counter-battery fire",   "en"),
    ("military logistics",     "en"),
]


def cmd_stats():
    files = list(KNOWLEDGE_DIR.glob("*.md")) if KNOWLEDGE_DIR.exists() else []
    total_chars = sum(f.stat().st_size for f in files)
    print(f"\n📚 knowledge/military/")
    print(f"   Файлів:    {len(files)}")
    print(f"   Розмір:    {total_chars / 1024:.1f} KB")
    if files:
        print("\n   Останні файли:")
        for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            print(f"   · {f.name}")


def main():
    args = build_parser().parse_args()

    if args.list_topics:
        print("Whitelist military-тематики (regex patterns):")
        for p in TOPIC_WHITELIST_PATTERNS:
            print(f"  · {p}")
        return

    if args.stats:
        cmd_stats()
        return

    if args.batch:
        print("📥 Batch завантаження базового military набору...")
        all_saved = []
        for topic, lang in BATCH_TOPICS:
            saved = fetch_topic(topic, lang=lang, limit=2, overwrite=args.overwrite, dry_run=args.dry_run)
            all_saved.extend(saved)
            time.sleep(1)
        print(f"\n✅ Збережено: {len(all_saved)} файлів")
        if all_saved and not args.dry_run:
            update_rag_index()
        return

    if not args.topic:
        build_parser().print_help()
        sys.exit(1)

    saved = fetch_topic(
        args.topic,
        lang=args.lang,
        limit=args.limit,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    if saved and not args.dry_run:
        update_rag_index()
        print(f"\n✅ Готово: {len(saved)} нових файлів у knowledge/military/")


if __name__ == "__main__":
    main()
