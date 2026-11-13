"""LLM provider layer — one prompt in, one reply out.

Transport: LiteLLM when it is installed (`pip install folderlake[llm]`)
— provider quirks, retries and response shapes are a solved problem
and we do not reinvent them. Without LiteLLM, a small built-in client
below speaks to the same providers with the same env configuration,
so the base install stays dependency-light. Both paths return None on
any failure; callers fall back to rule-based topics.

Providers (FOLDERLAKE_LLM_PROVIDER):
  ollama       Local Ollama server (default). OLLAMA_HOST or
               FOLDERLAKE_LLM_BASE_URL, model default "qwen3".
  openai       Any OpenAI-compatible chat endpoint: OpenAI itself,
               DeepSeek, Qwen/DashScope compatible mode, Groq, Together,
               LM Studio, vLLM, text-generation-webui — set
               FOLDERLAKE_LLM_BASE_URL + FOLDERLAKE_LLM_API_KEY.
               Default base https://api.openai.com/v1, model gpt-4o-mini.
  anthropic    Anthropic Messages API. Key from FOLDERLAKE_LLM_API_KEY
               or ANTHROPIC_API_KEY; model default claude-haiku-4-5.
  huggingface  Hugging Face's OpenAI-compatible router
               (https://router.huggingface.co/v1) — hosted inference for
               open models without running them locally. Key from
               FOLDERLAKE_LLM_API_KEY or HF_TOKEN. (Running transformers
               locally is deliberately not bundled: torch alone would
               dwarf the rest of the app; a local HF model is reachable
               anyway through the openai provider + a local server.)

Every failure returns None — callers fall back to rule-based topics.
"""

from __future__ import annotations

import json
import os
import urllib.request

TIMEOUT = 30


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def provider_name() -> str:
    p = _env("FOLDERLAKE_LLM_PROVIDER", default="ollama").lower()
    return {"hf": "huggingface"}.get(p, p)


def _model(default: str) -> str:
    return _env("FOLDERLAKE_LLM_MODEL", default=default)


def _post(url: str, payload: dict, headers: dict | None = None) -> dict | None:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _chat_completions(base: str, key: str, model: str, prompt: str) -> str | None:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    payload = _post(
        f"{base.rstrip('/')}/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}]},
        headers,
    )
    if not payload:
        return None
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _litellm_target(provider: str) -> tuple[str, str | None, str | None]:
    """(model, api_base, api_key) for LiteLLM, per provider."""
    if provider == "ollama":
        host = _env("FOLDERLAKE_LLM_BASE_URL", "OLLAMA_HOST",
                    default="http://localhost:11434")
        return f"ollama/{_model('qwen3')}", host, None
    if provider == "openai":
        return (
            f"openai/{_model('gpt-4o-mini')}",
            _env("FOLDERLAKE_LLM_BASE_URL", default="https://api.openai.com/v1"),
            _env("FOLDERLAKE_LLM_API_KEY", "OPENAI_API_KEY") or None,
        )
    if provider == "anthropic":
        return (
            f"anthropic/{_model('claude-haiku-4-5')}",
            None,
            _env("FOLDERLAKE_LLM_API_KEY", "ANTHROPIC_API_KEY") or None,
        )
    # Hugging Face's router is OpenAI-compatible: reach it through
    # LiteLLM's openai route with the router as api_base.
    return (
        f"openai/{_model('meta-llama/Llama-3.1-70B-Instruct')}",
        _env("FOLDERLAKE_LLM_BASE_URL",
             default="https://router.huggingface.co/v1"),
        _env("FOLDERLAKE_LLM_API_KEY", "HF_TOKEN", "HUGGINGFACE_API_KEY") or None,
    )


def _chat_litellm(prompt: str, provider: str) -> str | None:
    """Via LiteLLM; None when unavailable or on any error."""
    try:
        import litellm
    except Exception:
        return None
    litellm.suppress_debug_info = True
    model, api_base, api_key = _litellm_target(provider)
    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_base=api_base,
            api_key=api_key,
            timeout=TIMEOUT,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def chat(prompt: str) -> str | None:
    """Send one prompt to the configured provider; raw reply text or None.

    LiteLLM first when installed; the built-in client below otherwise
    (and as a fallback if the LiteLLM call itself fails).
    """
    provider = provider_name()
    via_lib = _chat_litellm(prompt, provider)
    if via_lib is not None:
        return via_lib
    if provider == "ollama":
        host = _env("FOLDERLAKE_LLM_BASE_URL", "OLLAMA_HOST",
                    default="http://localhost:11434")
        payload = _post(
            f"{host.rstrip('/')}/api/generate",
            {"model": _model("qwen3"), "prompt": prompt, "stream": False},
        )
        return payload.get("response") if payload else None
    if provider == "openai":
        return _chat_completions(
            _env("FOLDERLAKE_LLM_BASE_URL", default="https://api.openai.com/v1"),
            _env("FOLDERLAKE_LLM_API_KEY", "OPENAI_API_KEY"),
            _model("gpt-4o-mini"),
            prompt,
        )
    if provider == "huggingface":
        return _chat_completions(
            _env("FOLDERLAKE_LLM_BASE_URL",
                 default="https://router.huggingface.co/v1"),
            _env("FOLDERLAKE_LLM_API_KEY", "HF_TOKEN", "HUGGINGFACE_API_KEY"),
            _model("meta-llama/Llama-3.1-70B-Instruct"),
            prompt,
        )
    if provider == "anthropic":
        base = _env("FOLDERLAKE_LLM_BASE_URL",
                    default="https://api.anthropic.com")
        payload = _post(
            f"{base.rstrip('/')}/v1/messages",
            {
                "model": _model("claude-haiku-4-5"),
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
            {
                "x-api-key": _env("FOLDERLAKE_LLM_API_KEY", "ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
            },
        )
        if not payload:
            return None
        try:
            return "".join(
                block.get("text", "")
                for block in payload["content"]
                if block.get("type") == "text"
            ) or None
        except (KeyError, TypeError):
            return None
    return None


def parse_json_response(text: str | None) -> dict | None:
    """Tolerant JSON extraction: strips code fences, finds the first object."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def chat_json(prompt: str) -> dict | None:
    return parse_json_response(chat(prompt))
