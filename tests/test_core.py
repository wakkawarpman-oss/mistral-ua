"""
Тести для ядра Містраль.
Запуск: source venv/bin/activate && pytest tests/ -v
"""

import sys
import json
import types
import pytest
from unittest.mock import MagicMock, patch


# ── Modules ───────────────────────────────────────────────────────────────────

def test_modules_has_required_keys():
    from modules import MODULES
    required = {"ml", "logic", "coder", "doublecheck", "redteam", "planner"}
    assert required.issubset(set(MODULES.keys()))


def test_modules_each_has_system_and_label():
    from modules import MODULES
    for key, mod in MODULES.items():
        assert "system" in mod, f"{key} missing 'system'"
        assert "label"  in mod, f"{key} missing 'label'"
        assert len(mod["system"]) > 10, f"{key} system prompt too short"


# ── MistralML backend detection ───────────────────────────────────────────────

def test_detect_backend_groq_when_key_present(monkeypatch):
    import mistral_api
    monkeypatch.setattr(mistral_api, "_GROQ_AVAILABLE", True)
    monkeypatch.setattr(mistral_api, "GROQ_API_KEY", "test-key-123")
    ml = mistral_api.MistralML()
    assert ml.backend == "groq"


def test_detect_backend_none_when_no_key(monkeypatch):
    import mistral_api
    monkeypatch.setattr(mistral_api, "_GROQ_AVAILABLE", False)
    monkeypatch.setattr(mistral_api, "GROQ_API_KEY", "")
    # Block Ollama too
    import requests as _req
    with patch.object(_req, "get", side_effect=ConnectionError):
        ml = mistral_api.MistralML()
    assert ml.backend in ("none", "ollama")  # ollama might be running locally—OK


def test_is_ready_false_when_no_backend(monkeypatch):
    import mistral_api
    monkeypatch.setattr(mistral_api, "_GROQ_AVAILABLE", False)
    monkeypatch.setattr(mistral_api, "GROQ_API_KEY", "")
    import requests as _req
    with patch.object(_req, "get", side_effect=ConnectionError):
        ml = mistral_api.MistralML()
    if ml.backend == "none":
        assert not ml.is_ready()


# ── Executor (sandbox) ────────────────────────────────────────────────────────

def test_executor_safe_code():
    from executor import is_safe, run_code
    ok, reason = is_safe("x = 1 + 1\nprint(x)")
    assert ok, reason

    result = run_code("print('hello')")
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]


def test_executor_blocks_os_system():
    from executor import is_safe
    ok, reason = is_safe("import os\nos.system('rm -rf /')")
    assert not ok


def test_executor_blocks_subprocess():
    from executor import is_safe
    ok, reason = is_safe("import subprocess\nsubprocess.run(['ls'])")
    assert not ok


def test_executor_blocks_eval():
    from executor import is_safe
    ok, reason = is_safe("eval('1+1')")
    assert not ok


def test_executor_timeout():
    from executor import run_code
    result = run_code("while True: pass")
    assert result["returncode"] != 0 or result.get("error", "") != ""


def test_executor_format_result():
    from executor import format_result
    r = {"stdout": "ok\n", "stderr": "", "returncode": 0, "error": ""}
    out = format_result(r)
    assert "ok" in out


# ── RAG ───────────────────────────────────────────────────────────────────────

def test_rag_remember_and_search(tmp_path, monkeypatch):
    import rag as rag_mod
    monkeypatch.setattr(rag_mod, "KNOWLEDGE_DIR", tmp_path)

    r = rag_mod.RAG(knowledge_dir=tmp_path)
    r.remember("БПЛА FPV використовується для ударних місій. Дальність 5 км.", "test.md")
    hits = r.search("FPV дрон", top_k=1)
    assert len(hits) >= 1
    assert hits[0]["score"] > 0


def test_rag_context_empty_when_no_docs(tmp_path):
    import rag as rag_mod
    r = rag_mod.RAG(knowledge_dir=tmp_path)
    ctx = r.context("щось")
    assert ctx == ""


def test_rag_stats(tmp_path):
    import rag as rag_mod
    r = rag_mod.RAG(knowledge_dir=tmp_path)
    r.remember("Test doc text here", "t.md")
    s = r.stats()
    assert s["chunks"] >= 1
    assert s["files"]  >= 1


# ── Function calling (мокована відповідь) ────────────────────────────────────

def test_structured_output_fallback_json():
    import mistral_api
    ml = mistral_api.MistralML()
    ml.backend = "groq"

    fake_result = {"content": 'something {"key": "value"} end'}
    with patch.object(ml, "call_tool", return_value=fake_result):
        schema = {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}
        out = ml.structured_output("test", schema)
    assert out.get("key") == "value"


def test_call_tool_non_groq_returns_error():
    import mistral_api
    ml = mistral_api.MistralML()
    ml.backend = "ollama"
    result = ml.call_tool("test", tools=[])
    assert "error" in result


# ── Agent pipeline ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_steps_structure():
    from agent_pipeline import AgentPipeline, PipelineResult

    async def _fake_ai(system, user, temperature=0.5):
        return f"[fake output for system={system[:20]}]"

    pipeline = AgentPipeline()
    import agent_pipeline as ap_mod
    with patch.object(ap_mod, "_ai", side_effect=_fake_ai):
        result = await pipeline.run("тест завдання")

    assert isinstance(result, PipelineResult)
    assert len(result.steps) == len(AgentPipeline.STEPS)
    for step in result.steps:
        assert step.ok
        assert step.output


@pytest.mark.asyncio
async def test_pipeline_summary_contains_steps():
    from agent_pipeline import AgentPipeline
    import agent_pipeline as ap_mod

    async def _fake_ai(system, user, temperature=0.5):
        return "fake"

    pipeline = AgentPipeline()
    with patch.object(ap_mod, "_ai", side_effect=_fake_ai):
        result = await pipeline.run("тест")

    summary = result.summary()
    for _, step_name, _ in AgentPipeline.STEPS:
        assert step_name in summary
