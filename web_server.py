#!/usr/bin/env python3
"""
Містраль — мобільний веб-чат
Відкривай на iPhone: http://<IP-Mac>:8000
"""

import os
import json
import asyncio
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from modules import MODULES
from executor import run_code, format_result as fmt_exec

load_dotenv()

try:
    from groq import AsyncGroq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """Ти — Містраль, асиметричний інженер-аналітик R&D.
Спілкуєшся українською. Відповідаєш технічно, конкретно, без води.
Використовуй вбудовані інженерні модулі: логіка, причинно-наслідковий аналіз,
red team, стратегія, код, аудит, DevOps, QA, DSP, CV, оптимізація, системна інженерія."""

DOUBLECHECK_SYSTEM = """Ти — Містраль у режимі DoubleCheck. Твоя єдина роль: незалежна верифікація.

Отримаєш ПИТАННЯ та ВІДПОВІДЬ. Виконай аналіз за структурою:

1. ТОЧНІСТЬ — перевір кожне фактичне твердження (чи відповідає реальності)
2. ЛОГІКА — є внутрішні суперечності або хибні умовиводи?
3. ПРОПУСКИ — що важливе не згадано, проігноровано або спрощено надмірно?
4. ВПЕВНЕНІСТЬ — оціни достовірність відповіді у відсотках (0–100%)
5. ВЕРДИКТ:
   ✅ Підтверджено — відповідь точна
   ⚠️ Частково — є неточності, дай уточнення
   ❌ Помилка — дай виправлену версію

Будь критичним, але справедливим. Не повторюй відповідь без потреби.
Відповідай українською."""

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.get("/modules")
async def get_modules():
    return JSONResponse([{"key": k, "label": v["label"], "desc": v["desc"]}
                         for k, v in MODULES.items()])


@app.post("/execute")
async def execute_code(request: Request):
    body = await request.json()
    code = body.get("code", "").strip()
    if not code:
        return JSONResponse({"error": "no code"})
    r = run_code(code)
    return JSONResponse({
        "stdout":     r["stdout"],
        "stderr":     r["stderr"],
        "returncode": r["returncode"],
        "error":      r.get("error"),
        "formatted":  fmt_exec(r),
    })


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    module_key = body.get("module", "ml")

    if not messages:
        return {"error": "no messages"}

    system = MODULES.get(module_key, MODULES["ml"])["system"]
    full_messages = [{"role": "system", "content": system}] + messages

    async def generate():
        if _GROQ_AVAILABLE and GROQ_API_KEY:
            client = AsyncGroq(api_key=GROQ_API_KEY)
            async with client.chat.completions.stream(
                model=GROQ_MODEL,
                messages=full_messages,
                temperature=0.7,
                max_tokens=4096,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield f"data: {json.dumps({'content': delta})}\n\n"
        else:
            import httpx
            payload = {
                "model": os.getenv("OLLAMA_MODEL", "mistral-ua"),
                "messages": full_messages,
                "stream": True,
                "options": {"temperature": 0.7, "num_ctx": 8192},
            }
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", "http://localhost:11434/api/chat", json=payload) as r:
                    async for line in r.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                chunk = data.get("message", {}).get("content", "")
                                if chunk:
                                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                            except Exception:
                                continue
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/doublecheck")
async def doublecheck_endpoint(request: Request):
    body = await request.json()
    question = body.get("question", "")
    answer = body.get("answer", "")
    msgs = [
        {"role": "system", "content": DOUBLECHECK_SYSTEM},
        {"role": "user", "content": f"ПИТАННЯ:\n{question}\n\nВІДПОВІДЬ ДО ПЕРЕВІРКИ:\n{answer}"},
    ]

    async def generate():
        if _GROQ_AVAILABLE and GROQ_API_KEY:
            client = AsyncGroq(api_key=GROQ_API_KEY)
            async with client.chat.completions.stream(
                model=GROQ_MODEL,
                messages=msgs,
                temperature=0.3,
                max_tokens=2048,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield f"data: {json.dumps({'content': delta})}\n\n"
        else:
            import httpx
            payload = {
                "model": os.getenv("OLLAMA_MODEL", "mistral-ua"),
                "messages": msgs,
                "stream": True,
                "options": {"temperature": 0.3, "num_ctx": 8192},
            }
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", "http://localhost:11434/api/chat", json=payload) as r:
                    async for line in r.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                chunk = data.get("message", {}).get("content", "")
                                if chunk:
                                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                            except Exception:
                                continue
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ─── Mobile Chat UI ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Містраль</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  :root {
    --bg: #0d1117; --bg2: #161b22; --border: #30363d;
    --user: #1f6feb; --bot: #3fb950; --text: #e6edf3; --muted: #8b949e;
    --font: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
  }
  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--font); }
  body { display: flex; flex-direction: column; height: 100dvh; }

  header {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 8px 12px;
    padding-top: max(8px, env(safe-area-inset-top));
    display: flex; align-items: center; gap: 8px;
    position: sticky; top: 0; z-index: 10; flex-wrap: wrap;
  }
  .logo { width: 30px; height: 30px; background: var(--user); border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 15px; flex-shrink: 0; }
  .header-info h1 { font-size: 15px; font-weight: 600; }
  .header-info span { font-size: 10px; color: var(--bot); }
  #module-select {
    background: var(--bg); border: 1px solid var(--border); color: var(--text);
    border-radius: 8px; font-size: 12px; padding: 4px 8px; cursor: pointer; max-width: 160px;
  }
  #metrics-label { font-size: 10px; color: var(--muted); white-space: nowrap; }

  #messages {
    flex: 1; overflow-y: auto; padding: 16px; display: flex;
    flex-direction: column; gap: 12px; scroll-behavior: smooth;
  }

  .msg { display: flex; flex-direction: column; max-width: 85%; gap: 4px; }
  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.bot  { align-self: flex-start; align-items: flex-start; }

  .msg-label { font-size: 11px; color: var(--muted); padding: 0 4px; }
  .msg.user .msg-label { color: var(--user); }
  .msg.bot  .msg-label { color: var(--bot); }

  .bubble {
    padding: 10px 14px; border-radius: 16px; font-size: 15px;
    line-height: 1.5; word-break: break-word; white-space: pre-wrap;
  }
  .msg.user .bubble { background: var(--user); color: #fff; border-bottom-right-radius: 4px; }
  .msg.bot  .bubble { background: var(--bg2); border: 1px solid var(--border); border-bottom-left-radius: 4px; }

  code { background: #0d1117; border: 1px solid var(--border); border-radius: 4px;
    padding: 1px 5px; font-family: 'SF Mono', monospace; font-size: 13px; }
  pre { background: #0d1117; border: 1px solid var(--border); border-radius: 8px;
    padding: 12px; overflow-x: auto; margin: 8px 0; }
  pre code { background: none; border: none; padding: 0; font-size: 12px; }

  .typing { display: flex; gap: 4px; padding: 10px 14px; }
  .dot { width: 6px; height: 6px; background: var(--muted); border-radius: 50%; animation: bounce 1.2s infinite; }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-6px)} }

  footer {
    background: var(--bg2); border-top: 1px solid var(--border);
    padding: 12px 16px;
    padding-bottom: max(12px, env(safe-area-inset-bottom));
    display: flex; gap: 10px; align-items: flex-end;
  }
  textarea {
    flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 12px;
    color: var(--text); font-family: var(--font); font-size: 15px;
    padding: 10px 14px; resize: none; outline: none; max-height: 120px;
    line-height: 1.4; transition: border-color 0.2s;
  }
  textarea:focus { border-color: var(--user); }
  textarea::placeholder { color: var(--muted); }
  button {
    width: 40px; height: 40px; background: var(--user); border: none; border-radius: 10px;
    color: white; font-size: 18px; cursor: pointer; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; transition: opacity 0.2s;
    touch-action: manipulation; -webkit-user-select: none; user-select: none;
    -webkit-tap-highlight-color: transparent;
  }
  button:disabled { opacity: 0.4; pointer-events: none; }
  button:active { opacity: 0.7; }

  .clear-btn {
    background: none; border: 1px solid var(--border); color: var(--muted);
    font-size: 14px; width: auto; padding: 0 10px; border-radius: 8px;
  }
  .dc-btn {
    background: none; border: 1px solid #3fb950; color: #3fb950;
    font-size: 12px; width: auto; padding: 3px 10px; border-radius: 8px;
    margin-top: 4px; cursor: pointer; align-self: flex-start; transition: opacity 0.2s;
  }
  .dc-btn:disabled { opacity: 0.4; pointer-events: none; }
  .dc-btn:active { opacity: 0.6; }
  .msg.dc-result .bubble { border-color: #f0883e; }
  .msg.dc-result .msg-label { color: #f0883e !important; }

  /* iOS Safari: footer тримається над клавіатурою */
  footer { position: fixed; left: 0; right: 0; bottom: 0; }
  #messages { padding-bottom: 80px; }
</style>
</head>
<body>
<header>
  <div class="logo">⬡</div>
  <div class="header-info">
    <h1>Містраль</h1>
    <span id="backend-label">Groq LPU</span>
  </div>
  <select id="module-select" onchange="onModuleChange(this.value)"><option value="">завантаження...</option></select>
  <span id="metrics-label">—</span>
  <button type="button" class="clear-btn" id="clear-btn" style="margin-left:auto">Очистити</button>
</header>

<div id="messages">
  <div class="msg bot">
    <span class="msg-label">Містраль</span>
    <div class="bubble">Привіт. Я готовий — пиши будь-яке технічне питання.</div>
  </div>
</div>

<footer>
  <textarea id="input" placeholder="Напиши запит..." rows="1"
    oninput="autoResize(this)"
    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
  <button type="button" id="send-btn">↑</button>
</footer>

<script>
const messages = [];
let busy = false;
let currentModule = 'ml';
let tStart = 0, tokenCount = 0;

// ── Завантаження модулів ──────────────────────────────────────────
async function loadModules() {
  try {
    const res = await fetch('/modules');
    const mods = await res.json();
    const sel = document.getElementById('module-select');
    sel.innerHTML = mods.map(m =>
      `<option value="${m.key}" title="${m.desc}">${m.label}</option>`
    ).join('');
    sel.value = currentModule;
  } catch(e) { console.error('modules load error', e); }
}

function onModuleChange(key) {
  currentModule = key;
  messages.length = 0;
  const sel = document.getElementById('module-select');
  const label = sel.options[sel.selectedIndex]?.text || key;
  const wrap = document.getElementById('messages');
  wrap.innerHTML = `<div class="msg bot">
    <span class="msg-label">Містраль</span>
    <div class="bubble">Режим: <b>${label}</b>. Контекст очищено.</div>
  </div>`;
}

// iOS Safari: footer тримається над клавіатурою через visualViewport API
if (window.visualViewport) {
  const footer = document.querySelector('footer');
  const adjust = () => {
    const offset = window.innerHeight - window.visualViewport.height - window.visualViewport.offsetTop;
    footer.style.bottom = Math.max(0, offset) + 'px';
    scrollDown();
  };
  window.visualViewport.addEventListener('resize', adjust);
  window.visualViewport.addEventListener('scroll', adjust);
}

document.getElementById('send-btn').addEventListener('click', send);
document.getElementById('clear-btn').addEventListener('click', clearChat);
loadModules();

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function scrollDown() {
  const m = document.getElementById('messages');
  m.scrollTop = m.scrollHeight;
}

function addMsg(role, text) {
  const wrap = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `
    <span class="msg-label">${role === 'user' ? 'Ти' : 'Містраль'}</span>
    <div class="bubble">${escHtml(text)}</div>`;
  wrap.appendChild(div);
  scrollDown();
  return div.querySelector('.bubble');
}

function addTyping() {
  const wrap = document.getElementById('messages');
  const div = document.createElement('div');
  div.id = 'typing';
  div.className = 'msg bot';
  div.innerHTML = `<span class="msg-label">Містраль</span>
    <div class="bubble"><div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>`;
  wrap.appendChild(div);
  scrollDown();
}

function removeTyping() {
  document.getElementById('typing')?.remove();
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/`([^`]+)`/g,'<code>$1</code>');
}

function renderBubble(text) {
  // Проста підсвітка коду
  let r = escHtml(text);
  r = r.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code>${code.trim()}</code></pre>`);
  return r;
}

async function send() {
  if (busy) return;
  const input = document.getElementById('input');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  input.style.height = 'auto';
  busy = true;
  document.getElementById('send-btn').disabled = true;

  messages.push({role: 'user', content: text});
  addMsg('user', text);
  addTyping();

  let full = '';
  let bubble = null;
  tStart = Date.now(); tokenCount = 0;

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({messages, module: currentModule}),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const lines = decoder.decode(value).split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') break;
        try {
          const {content} = JSON.parse(data);
          if (!bubble) { removeTyping(); bubble = addMsg('bot', ''); }
          full += content;
          tokenCount += content.split(/\s+/).length;
          bubble.innerHTML = renderBubble(full);
          scrollDown();
        } catch {}
      }
    }
  } catch(e) {
    removeTyping();
    addMsg('bot', 'Помилка з\'єднання: ' + e.message);
  }

  if (full) {
    const elapsed = (Date.now() - tStart) / 1000;
    const tps = elapsed > 0 ? Math.round(tokenCount / elapsed) : 0;
    document.getElementById('metrics-label').textContent = `${tps} tok/s · ${elapsed.toFixed(1)}s`;
    messages.push({role: 'assistant', content: full});
    const dcBtn = document.createElement('button');
    dcBtn.className = 'dc-btn';
    dcBtn.textContent = '⟳ DoubleCheck';
    const q = text;
    const a = full;
    dcBtn.addEventListener('click', () => doDoubleCheck(dcBtn, q, a));
    bubble.parentElement.appendChild(dcBtn);
  }
  busy = false;
  document.getElementById('send-btn').disabled = false;
  // не викликаємо input.focus() — на iOS це знову підіймає клавіатуру і зсуває layout
}

function clearChat() {
  messages.length = 0;
  const wrap = document.getElementById('messages');
  wrap.innerHTML = `<div class="msg bot">
    <span class="msg-label">Містраль</span>
    <div class="bubble">Контекст очищено. Пиши.</div>
  </div>`;
}

function addDCMsg() {
  const wrap = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg bot dc-result';
  div.innerHTML = '<span class="msg-label">⟳ DoubleCheck</span><div class="bubble"></div>';
  wrap.appendChild(div);
  scrollDown();
  return div.querySelector('.bubble');
}

async function doDoubleCheck(btn, question, answer) {
  if (busy) return;
  btn.disabled = true;
  busy = true;
  document.getElementById('send-btn').disabled = true;
  addTyping();
  let full = '';
  let bubble = null;
  try {
    const res = await fetch('/doublecheck', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, answer}),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const lines = decoder.decode(value).split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') break;
        try {
          const {content} = JSON.parse(data);
          if (!bubble) { removeTyping(); bubble = addDCMsg(); }
          full += content;
          bubble.innerHTML = renderBubble(full);
          scrollDown();
        } catch {}
      }
    }
  } catch(e) {
    removeTyping();
    addMsg('bot', 'DoubleCheck помилка: ' + e.message);
  }
  busy = false;
  document.getElementById('send-btn').disabled = false;
  btn.remove();
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    import socket

    # Показати IP для iPhone
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*45}")
    print(f"  Містраль веб-чат")
    print(f"  Локально:  http://localhost:8000")
    print(f"  iPhone:    http://{local_ip}:8000")
    print(f"{'='*45}\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
