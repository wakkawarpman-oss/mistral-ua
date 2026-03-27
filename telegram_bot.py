#!/usr/bin/env python3
"""
Містраль — Telegram бот
Команди:
  /start          — привітання
  /help           — список команд
  /mode <назва>   — змінити режим (ml, logic, redteam, coder, planner...)
  /mode           — показати поточний режим
  /clear          — очистити контекст розмови
  /check          — перевірити останню відповідь (DoubleCheck)
  будь-який текст — відповідь Містраля
"""

import asyncio
import io
import os
import json
import logging

from dotenv import load_dotenv
load_dotenv()

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, BotCommand, Voice
from aiogram.filters import Command, CommandStart
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

try:
    from groq import AsyncGroq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "mistral-ua")

logging.basicConfig(level=logging.WARNING)

from modules import MODULES as _MODULES
from executor import run_code, format_result as fmt_exec

# Синхронізуємо MODES з modules.py (єдине джерело правди)
MODES = {k: {"label": v["label"], "system": v["system"]} for k, v in _MODULES.items()}

DEFAULT_MODE = "ml"

# ── Стан користувачів ────────────────────────────────────────────────────────────
# { user_id: { "mode": str, "history": list, "last_q": str, "last_a": str } }
user_state: dict[int, dict] = {}


def get_state(uid: int) -> dict:
    if uid not in user_state:
        user_state[uid] = {"mode": DEFAULT_MODE, "history": [], "last_q": "", "last_a": ""}
    return user_state[uid]


# ── AI запит ────────────────────────────────────────────────────────────────────
async def ask_ai(messages: list) -> str:
    if _GROQ_AVAILABLE and GROQ_API_KEY:
        client = AsyncGroq(api_key=GROQ_API_KEY)
        resp = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""
    else:
        import httpx
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7, "num_ctx": 8192},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            return r.json()["message"]["content"]


def escape_md(text: str) -> str:
    """Екранує спецсимволи для MarkdownV2."""
    chars = r"\_*[]()~`>#+-=|{}.!"
    for c in chars:
        text = text.replace(c, f"\\{c}")
    return text


# ── Handlers ─────────────────────────────────────────────────────────────────────
async def cmd_start(msg: Message):
    state = get_state(msg.from_user.id)
    mode_info = MODES[state["mode"]]
    text = (
        f"Привіт\\! Я *Містраль* — асиметричний інженер\\-аналітик R\\&D\\.\n\n"
        f"Поточний режим: *{escape_md(mode_info['label'])}*\n\n"
        f"Команди:\n"
        f"`/mode` — показати всі режими\n"
        f"`/mode coder` — переключити режим\n"
        f"`/clear` — очистити контекст\n"
        f"`/check` — DoubleCheck останньої відповіді\n\n"
        f"Пиши будь\\-яке питання — відповім\\."
    )
    await msg.answer(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_help(msg: Message):
    modes_list = "\n".join(
        f"`{k}` — {v['label']}" for k, v in MODES.items() if k != "doublecheck"
    )
    text = (
        f"*Режими Містраль:*\n{escape_md(modes_list)}\n\n"
        f"Перемикання: `/mode coder`\n"
        f"Очистити контекст: `/clear`\n"
        f"Верифікація відповіді: `/check`"
    )
    await msg.answer(text, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_mode(msg: Message):
    state = get_state(msg.from_user.id)
    args = (msg.text or "").split(maxsplit=1)

    if len(args) < 2:
        # Показати поточний та доступні
        current = MODES[state["mode"]]["label"]
        available = ", ".join(k for k in MODES if k != "doublecheck")
        await msg.answer(
            f"Режим: *{escape_md(current)}*\n\nДоступні: `{available}`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    new_mode = args[1].strip().lower()
    if new_mode not in MODES or new_mode == "doublecheck":
        available = ", ".join(k for k in MODES if k != "doublecheck")
        await msg.answer(f"Невідомий режим\\. Доступні: `{available}`", parse_mode=ParseMode.MARKDOWN_V2)
        return

    state["mode"] = new_mode
    state["history"] = []  # скидаємо контекст при зміні режиму
    label = escape_md(MODES[new_mode]["label"])
    await msg.answer(f"✓ Режим: *{label}*\nКонтекст очищено\\.", parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_clear(msg: Message):
    state = get_state(msg.from_user.id)
    state["history"] = []
    state["last_q"] = ""
    state["last_a"] = ""
    await msg.answer("Контекст очищено\\.", parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_run(msg: Message):
    """Виконати Python код у sandbox. Формат: /run\n```python\nкод\n```"""
    raw = (msg.text or "").split(maxsplit=1)
    if len(raw) < 2:
        await msg.answer(
            "Формат:\n`/run print('hello')`\n\nАбо багаторядковий:\n```\n/run\nx = 2**10\nprint(x)\n```",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return
    code = raw[1].strip().strip("`").strip()
    # Прибираємо блок ```python ... ```
    if code.startswith("python"):
        code = code[6:].strip()
    thinking = await msg.answer("⚙️ Виконую...")
    r = run_code(code)
    result = fmt_exec(r)
    await thinking.delete()
    await msg.answer(result)


async def cmd_check(msg: Message):
    state = get_state(msg.from_user.id)
    if not state["last_a"]:
        await msg.answer("Немає відповіді для перевірки\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return

    thinking = await msg.answer("⟳ DoubleCheck\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    messages = [
        {"role": "system", "content": MODES["doublecheck"]["system"]},
        {"role": "user", "content": f"ПИТАННЯ:\n{state['last_q']}\n\nВІДПОВІДЬ ДО ПЕРЕВІРКИ:\n{state['last_a']}"},
    ]
    try:
        result = await ask_ai(messages)
        await thinking.delete()
        # Надсилаємо як звичайний текст щоб уникнути проблем з екрануванням
        await msg.answer(f"⟳ DoubleCheck:\n\n{result}")
    except Exception as e:
        await thinking.delete()
        await msg.answer(f"Помилка DoubleCheck: {e}")


async def handle_voice(msg: Message, bot: Bot):
    """Голосове повідомлення → Whisper → AI відповідь."""
    state = get_state(msg.from_user.id)

    thinking = await msg.answer("🎤 Розпізнаю...")
    try:
        # Завантажуємо OGG аудіо в пам'ять
        buf = io.BytesIO()
        await bot.download(msg.voice, destination=buf)
        buf.seek(0)

        if not (_GROQ_AVAILABLE and GROQ_API_KEY):
            await thinking.edit_text("Голосовий ввід потребує Groq API.")
            return

        client = AsyncGroq(api_key=GROQ_API_KEY)
        transcription = await client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=("voice.ogg", buf, "audio/ogg"),
            language="uk",
        )
        text = transcription.text.strip()
        if not text:
            await thinking.edit_text("Не вдалося розпізнати мовлення.")
            return

        await thinking.edit_text(f"🎤 {text}")

    except Exception as e:
        await thinking.edit_text(f"Помилка STT: {e}")
        return

    # Передаємо розпізнаний текст до AI як звичайне повідомлення
    mode = state["mode"]
    system = MODES[mode]["system"]
    history = state["history"][-20:]
    messages = [{"role": "system", "content": system}] + history + [
        {"role": "user", "content": text}
    ]
    ai_msg = await msg.answer("⋯")
    try:
        answer = await ask_ai(messages)
        await ai_msg.edit_text(answer)

        state["history"].append({"role": "user", "content": text})
        state["history"].append({"role": "assistant", "content": answer})
        if len(state["history"]) > 40:
            state["history"] = state["history"][-40:]
        state["last_q"] = text
        state["last_a"] = answer
    except Exception as e:
        await ai_msg.edit_text(f"Помилка AI: {e}")


async def handle_message(msg: Message):
    if not msg.text:
        return
    state = get_state(msg.from_user.id)
    mode = state["mode"]
    system = MODES[mode]["system"]

    # Формуємо повідомлення з контекстом (останні 10 пар)
    history = state["history"][-20:]
    messages = [{"role": "system", "content": system}] + history + [
        {"role": "user", "content": msg.text}
    ]

    thinking = await msg.answer("⋯")
    try:
        answer = await ask_ai(messages)
        await thinking.delete()
        await msg.answer(answer)

        # Зберігаємо в контекст
        state["history"].append({"role": "user", "content": msg.text})
        state["history"].append({"role": "assistant", "content": answer})
        # Обрізаємо до 40 повідомлень
        if len(state["history"]) > 40:
            state["history"] = state["history"][-40:]
        state["last_q"] = msg.text
        state["last_a"] = answer

    except Exception as e:
        await thinking.delete()
        await msg.answer(f"Помилка: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────────
async def main():
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN не знайдено в .env")
        return

    bot = Bot(
        token=TELEGRAM_TOKEN,
        default=DefaultBotProperties(parse_mode=None),
    )
    dp = Dispatcher()

    # Реєстрація команд
    dp.message.register(cmd_start, CommandStart())
    dp.message.register(cmd_help, Command("help"))
    dp.message.register(cmd_mode, Command("mode"))
    dp.message.register(cmd_clear, Command("clear"))
    dp.message.register(cmd_run, Command("run"))
    dp.message.register(cmd_check, Command("check"))
    dp.message.register(handle_voice, F.voice)
    dp.message.register(handle_message, F.text)

    await bot.set_my_commands([
        BotCommand(command="start",  description="Привітання"),
        BotCommand(command="mode",   description="Змінити режим мислення"),
        BotCommand(command="run",    description="Виконати Python код"),
        BotCommand(command="check",  description="DoubleCheck останньої відповіді"),
        BotCommand(command="clear",  description="Очистити контекст"),
        BotCommand(command="help",   description="Список команд"),
    ])

    backend = "Groq" if (_GROQ_AVAILABLE and GROQ_API_KEY) else "Ollama"
    print(f"\n{'='*40}")
    print(f"  Містраль Telegram бот")
    print(f"  Бекенд: {backend}")
    print(f"  Режими: {', '.join(k for k in MODES if k != 'doublecheck')}")
    print(f"{'='*40}\n")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
