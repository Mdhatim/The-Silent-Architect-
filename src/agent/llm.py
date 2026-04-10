from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests
from pydantic import ValidationError

from .schemas import IntentResult


@dataclass(frozen=True)
class LLMResult:
    parsed: IntentResult | None
    raw_text: str
    provider: str
    detail: str = ""


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if (v is not None and v.strip() != "") else default


SYSTEM_PROMPT = """You are an intent classifier and action planner for a local agent.
Return ONLY valid JSON for this schema:
{
  "intent": "create_file" | "write_code" | "summarize_text" | "general_chat",
  "confidence": 0.0-1.0,
  "rationale": "short reason",
  "path": "relative path under output/ if needed",
  "content": "file/code content if needed",
  "text": "text to summarize if needed"
}

Rules:
- If user asks to create a file (even without code), use intent=create_file and set path.
- If user asks to write code, use intent=write_code, set path (pick a sensible filename) and content (the code).
- If user asks to summarize something, use intent=summarize_text and set text to summarize (use the user's message if it's the text).
- Otherwise use general_chat.
- All paths must be relative and must not start with '/' or '..'. Prefer "agent_output.txt", "script.py", etc.
"""


def classify_and_plan(user_text: str) -> LLMResult:
    provider = (_env("LLM_PROVIDER", "ollama") or "ollama").lower()
    if provider in {"ollama", "local"}:
        return _classify_ollama(user_text)
    return _classify_openai_compatible(user_text)


def _try_parse_json(s: str) -> tuple[IntentResult | None, str]:
    raw = s.strip()
    # strip common wrappers
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        # try to find first/last brace
        try:
            i = raw.find("{")
            j = raw.rfind("}")
            if i != -1 and j != -1 and j > i:
                obj = json.loads(raw[i : j + 1])
            else:
                return None, s
        except Exception:
            return None, s
    try:
        return IntentResult.model_validate(obj), raw
    except ValidationError:
        return None, raw


def _classify_ollama(user_text: str) -> LLMResult:
    host = _env("OLLAMA_HOST", "http://localhost:11434") or "http://localhost:11434"
    model = _env("OLLAMA_MODEL", "llama3.1:8b") or "llama3.1:8b"

    # Prefer ollama python package if available; otherwise HTTP.
    try:
        import ollama  # type: ignore

        resp: Any = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            options={"temperature": 0.2},
        )
        raw = (resp.get("message", {}) or {}).get("content", "") if isinstance(resp, dict) else str(resp)
        parsed, raw2 = _try_parse_json(raw)
        return LLMResult(parsed=parsed, raw_text=raw2, provider="ollama", detail=f"model={model}")
    except Exception:
        pass

    # HTTP fallback
    try:
        url = host.rstrip("/") + "/api/chat"
        r = requests.post(
            url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                "options": {"temperature": 0.2},
                "stream": False,
            },
            timeout=120,
        )
        if r.status_code >= 300:
            return LLMResult(None, r.text, "ollama", detail=f"HTTP {r.status_code}")
        j = r.json()
        raw = ((j.get("message") or {}).get("content") or "").strip()
        parsed, raw2 = _try_parse_json(raw)
        return LLMResult(parsed=parsed, raw_text=raw2, provider="ollama", detail=f"model={model}")
    except Exception as e:
        return LLMResult(None, "", "ollama", detail=f"{type(e).__name__}: {e}")


def _classify_openai_compatible(user_text: str) -> LLMResult:
    api_key = _env("OPENAI_API_KEY") or _env("GROQ_API_KEY")
    if not api_key:
        return LLMResult(None, "", "api", detail="Missing OPENAI_API_KEY or GROQ_API_KEY")

    base_url = _env("OPENAI_BASE_URL") or _env("GROQ_BASE_URL") or "https://api.openai.com/v1"
    model = _env("OPENAI_CHAT_MODEL") or _env("GROQ_CHAT_MODEL") or "gpt-4.1-mini"

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.2,
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=120)
        if r.status_code >= 300:
            return LLMResult(None, r.text, "api", detail=f"HTTP {r.status_code}")
        j = r.json()
        raw = (((j.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        parsed, raw2 = _try_parse_json(raw)
        return LLMResult(parsed=parsed, raw_text=raw2, provider="api", detail=f"model={model}")
    except Exception as e:
        return LLMResult(None, "", "api", detail=f"{type(e).__name__}: {e}")

