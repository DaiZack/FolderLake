"""Tests for v3 phrase tags, the LLM provider layer, and the app command."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import duckdb
import pytest

from folderlake import llm
from folderlake.catalog import scan
from folderlake.cli import main as cli_main


def _tags(db, name):
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute(
        """
        SELECT t.tag FROM file_topics t
        JOIN files f ON f.id = t.file_id WHERE f.name = ?
        ORDER BY t.weight DESC
        """,
        [name],
    ).fetchall()
    con.close()
    return [t for (t,) in rows]


def test_english_phrase_tag_suppresses_component(tmp_path):
    body = (
        "Personal finance basics: a personal finance plan starts with a "
        "budget. Good personal finance habits grow savings over time. "
    )
    (tmp_path / "money.txt").write_text(body * 3, encoding="utf-8")
    (tmp_path / "other.txt").write_text(
        "Garden notes about tomatoes, watering and compost. " * 5,
        encoding="utf-8")
    db = tmp_path / "catalog.duckdb"
    scan(tmp_path, db)
    tags = _tags(db, "money.txt")
    assert "personal finance" in tags
    assert "finance" not in tags  # component suppressed by the phrase


def test_chinese_trigram_tag_suppresses_bigram(tmp_path):
    (tmp_path / "a.txt").write_text(
        "自行车维修记录：自行车刹车需要调整，自行车链条需要润滑。"
        "这辆自行车是通勤用的自行车，每周骑行三次。", encoding="utf-8")
    (tmp_path / "b.txt").write_text(
        "花园种植计划：春天播种番茄和黄瓜，定期浇水施肥。", encoding="utf-8")
    db = tmp_path / "catalog.duckdb"
    scan(tmp_path, db)
    tags = _tags(db, "a.txt")
    assert "自行车" in tags
    assert "自行" not in tags and "行车" not in tags


def test_data_column_phrase_tag(tmp_path):
    (tmp_path / "sales.csv").write_text(
        "total_bill,tip_amount\n10,2\n20,4\n30,6\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text(
        "Meeting notes about the weekly schedule and office supplies. " * 3,
        encoding="utf-8")
    db = tmp_path / "catalog.duckdb"
    scan(tmp_path, db)
    tags = _tags(db, "sales.csv")
    assert "total bill" in tags
    assert "bill" not in tags


# ----------------------------------------------------------------------
# LLM provider layer
# ----------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_fake(monkeypatch, payload, sink):
    def fake_urlopen(req, timeout=None):
        sink["url"] = req.full_url
        sink["headers"] = dict(req.header_items())
        sink["body"] = json.loads(req.data)
        return _FakeResp(payload)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)


def test_openai_compatible_provider(monkeypatch):
    sink = {}
    _install_fake(monkeypatch, {"choices": [{"message": {"content": '{"tags": ["x"]}'}}]}, sink)
    monkeypatch.setenv("FOLDERLAKE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("FOLDERLAKE_LLM_API_KEY", "sk-test")
    monkeypatch.delenv("FOLDERLAKE_LLM_BASE_URL", raising=False)
    out = llm.chat_json("hello")
    assert out == {"tags": ["x"]}
    assert sink["url"] == "https://api.openai.com/v1/chat/completions"
    assert sink["headers"].get("Authorization") == "Bearer sk-test"
    assert sink["body"]["model"] == "gpt-4o-mini"


def test_anthropic_provider(monkeypatch):
    sink = {}
    _install_fake(monkeypatch, {"content": [{"type": "text", "text": '{"tags": []}'}]}, sink)
    monkeypatch.setenv("FOLDERLAKE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    monkeypatch.delenv("FOLDERLAKE_LLM_BASE_URL", raising=False)
    out = llm.chat_json("hello")
    assert out == {"tags": []}
    assert sink["url"] == "https://api.anthropic.com/v1/messages"
    assert sink["headers"].get("X-api-key") == "ak-test"


def test_huggingface_uses_router(monkeypatch):
    sink = {}
    _install_fake(monkeypatch, {"choices": [{"message": {"content": "{}"}}]}, sink)
    monkeypatch.setenv("FOLDERLAKE_LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.delenv("FOLDERLAKE_LLM_BASE_URL", raising=False)
    assert llm.chat_json("hello") == {}
    assert sink["url"].startswith("https://router.huggingface.co/v1/")


def test_ollama_default_and_fenced_json(monkeypatch):
    sink = {}
    _install_fake(monkeypatch, {"response": "```json\n{\"tags\": [\"a\"]}\n```"}, sink)
    monkeypatch.delenv("FOLDERLAKE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("FOLDERLAKE_LLM_BASE_URL", raising=False)
    assert llm.chat_json("hello") == {"tags": ["a"]}
    assert sink["url"] == "http://localhost:11434/api/generate"


def test_provider_failure_returns_none(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(llm.urllib.request, "urlopen", boom)
    monkeypatch.setenv("FOLDERLAKE_LLM_PROVIDER", "openai")
    assert llm.chat("hello") is None
    assert llm.chat_json("hello") is None


def _fake_litellm(monkeypatch, content=None, raises=False):
    import types

    sink = {}
    mod = types.ModuleType("litellm")

    def completion(**kwargs):
        sink.update(kwargs)
        if raises:
            raise RuntimeError("provider down")
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg)])

    mod.completion = completion
    mod.suppress_debug_info = False
    monkeypatch.setitem(sys.modules, "litellm", mod)
    return sink


def test_litellm_preferred_when_installed(monkeypatch):
    sink = _fake_litellm(monkeypatch, content='{"tags": ["via-litellm"]}')
    monkeypatch.setenv("FOLDERLAKE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("FOLDERLAKE_LLM_API_KEY", "sk-x")
    assert llm.chat_json("hi") == {"tags": ["via-litellm"]}
    assert sink["model"] == "openai/gpt-4o-mini"
    assert sink["api_key"] == "sk-x"


def test_litellm_failure_falls_back_to_builtin(monkeypatch):
    _fake_litellm(monkeypatch, raises=True)
    http = {}
    _install_fake(monkeypatch, {"response": '{"tags": ["builtin"]}'}, http)
    monkeypatch.delenv("FOLDERLAKE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("FOLDERLAKE_LLM_BASE_URL", raising=False)
    assert llm.chat_json("hi") == {"tags": ["builtin"]}
    assert http["url"] == "http://localhost:11434/api/generate"


def test_app_starts_home_server(monkeypatch):
    # v0.2: `app` no longer picks a folder and errors out — it starts
    # the home web app (history / new scan / settings) immediately.
    calls = {}

    def fake_run_app(host, port, open_browser=True):
        calls.update(host=host, port=port, open_browser=open_browser)
        return 0

    monkeypatch.setattr("folderlake.appserver.run_app", fake_run_app)
    assert cli_main(["app", "--port", "8791"]) == 0
    assert calls == {"host": "127.0.0.1", "port": 8791,
                     "open_browser": True}
