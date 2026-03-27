#!/usr/bin/env python3
"""
Mistral ML Engineer — Desktop App
Десктоп чат без браузера
"""

import tkinter as tk
from tkinter import scrolledtext
import threading
import requests
import json

OLLAMA_URL = "http://localhost:11434"
MODEL = "mistral-ua"

SYSTEM = """Ти — Містраль, досвідчений військовий інженер.
Твоя спеціалізація: оборонні технології, робототехніка (дрони, наземні платформи), системи зв'язку та розвідки, інженерне забезпечення.

Твої принципи:
1. Технічна точність — стандарти та специфікації важливіші за загальні слова
2. Асиметричний підхід — прості рішення часто ефективніші за складні
3. Практичність — що можна реалізувати з доступними ресурсами
4. Критичний аналіз — завжди вказуй на слабкі місця рішення
5. Безпека — враховуй ризики та способи їх мінімізації

Відповідай українською. Надавай конкретні технічні відповіді, схеми, код або специфікації коли потрібно.
Ти розбираєшся в: робототехніці (ROS, SLAM, дрони), Python, C++, електроніці, системах зв'язку, OSINT, автоматизації."""

BG       = "#0d1117"
BG2      = "#161b22"
BORDER   = "#30363d"
USER_CLR = "#1f6feb"
BOT_CLR  = "#3fb950"
TEXT_CLR = "#e6edf3"
MUTED    = "#8b949e"
FONT     = ("Helvetica", 13)
FONT_B   = ("Helvetica", 13, "bold")
FONT_S   = ("Helvetica", 11)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mistral — Військовий Інженер")
        self.geometry("800x600")
        self.minsize(600, 400)
        self.configure(bg=BG)
        self.history = []
        self._build()
        self._check_ollama()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=BG2, height=44)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⬡  Mistral — Військовий Інженер", bg=BG2,
                 fg=TEXT_CLR, font=FONT_B, padx=14).pack(side="left", pady=8)
        self.status_lbl = tk.Label(hdr, text="● перевірка...", bg=BG2,
                                   fg=MUTED, font=FONT_S, padx=14)
        self.status_lbl.pack(side="right", pady=8)

        # Chat area
        self.chat = scrolledtext.ScrolledText(
            self, bg=BG, fg=TEXT_CLR, font=FONT,
            relief="flat", bd=0, padx=16, pady=12,
            wrap="word", state="disabled",
            insertbackground=TEXT_CLR,
        )
        self.chat.pack(fill="both", expand=True, padx=0, pady=0)

        # Tags for coloring
        self.chat.tag_config("you",    foreground=USER_CLR, font=FONT_B)
        self.chat.tag_config("bot",    foreground=BOT_CLR,  font=FONT_B)
        self.chat.tag_config("msg",    foreground=TEXT_CLR, font=FONT)
        self.chat.tag_config("muted",  foreground=MUTED,    font=FONT_S)

        # Footer
        footer = tk.Frame(self, bg=BG2, pady=10, padx=12)
        footer.pack(fill="x")

        self.input = tk.Text(
            footer, height=3, bg=BG, fg=TEXT_CLR, font=FONT,
            relief="flat", bd=1, padx=10, pady=8,
            wrap="word", insertbackground=TEXT_CLR,
            highlightbackground=BORDER, highlightcolor=USER_CLR,
            highlightthickness=1,
        )
        self.input.pack(side="left", fill="x", expand=True)
        self.input.bind("<Return>",       self._on_enter)
        self.input.bind("<Shift-Return>", self._newline)

        self.btn = tk.Button(
            footer, text="Надіслати", bg=USER_CLR, fg="white",
            font=FONT_B, relief="flat", padx=14, pady=8,
            activebackground="#388bfd", activeforeground="white",
            cursor="hand2", command=self._send,
        )
        self.btn.pack(side="right", padx=(10, 0))

        hint = tk.Label(self, text="Enter — надіслати  |  Shift+Enter — нова лінія",
                        bg=BG, fg=MUTED, font=("Helvetica", 10))
        hint.pack(pady=(0, 6))

    def _on_enter(self, event):
        self._send()
        return "break"

    def _newline(self, event):
        return None  # default newline

    def _append(self, text, tag="msg"):
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _send(self):
        text = self.input.get("1.0", "end").strip()
        if not text:
            return
        self.input.delete("1.0", "end")
        self.btn.configure(state="disabled")

        self._append("\nВи\n", "you")
        self._append(text + "\n", "msg")
        self.history.append({"role": "user", "content": text})

        threading.Thread(target=self._stream, daemon=True).start()

    def _stream(self):
        self._append("\nМістраль\n", "bot")
        payload = {
            "model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM}] + self.history,
            "stream": True,
        }
        full = ""
        try:
            with requests.post(
                f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120
            ) as resp:
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            full += token
                            self._append(token, "msg")
                        if chunk.get("done"):
                            break
        except Exception as e:
            self._append(f"\n[Помилка: {e}]\n", "muted")

        self._append("\n", "msg")
        if full:
            self.history.append({"role": "assistant", "content": full})
        self.btn.configure(state="normal")
        self.input.focus()

    def _check_ollama(self):
        def check():
            try:
                r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
                if r.status_code == 200:
                    models = [m["name"] for m in r.json().get("models", [])]
                    if any("mistral" in m for m in models):
                        self.status_lbl.configure(text="● готово", fg=BOT_CLR)
                        self._append("Mistral готовий. Задай питання!\n", "muted")
                        return
            except Exception:
                pass
            self.status_lbl.configure(text="● Ollama не запущений", fg="#f85149")
            self._append("Помилка: запусти Ollama (ollama serve)\n", "muted")
        threading.Thread(target=check, daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
