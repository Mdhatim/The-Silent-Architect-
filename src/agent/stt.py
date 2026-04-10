from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class STTResult:
    text: str
    provider: str
    detail: str = ""


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if (v is not None and v.strip() != "") else default


def transcribe(audio_path: str) -> STTResult:
    provider = (_env("STT_PROVIDER", "local") or "local").lower()
    if provider == "api":
        return _transcribe_api(audio_path)
    return _transcribe_local(audio_path)


def _transcribe_local(audio_path: str) -> STTResult:
    # Prefer faster-whisper if installed (good on CPU).
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model_size = _env("WHISPER_MODEL", "small") or "small"
        device = _env("WHISPER_DEVICE", "cpu") or "cpu"
        compute_type = _env("WHISPER_COMPUTE_TYPE", "int8") or "int8"

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(audio_path, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return STTResult(
            text=text,
            provider="local:faster-whisper",
            detail=f"lang={getattr(info, 'language', None)}",
        )
    except Exception:
        pass

    # Fallback to transformers pipeline.
    try:
        from transformers import pipeline  # type: ignore

        model_id = _env("HF_STT_MODEL", "openai/whisper-small") or "openai/whisper-small"
        asr = pipeline("automatic-speech-recognition", model=model_id)
        out: Any = asr(audio_path)
        text = (out.get("text") if isinstance(out, dict) else str(out)).strip()
        return STTResult(text=text, provider="local:transformers", detail=f"model={model_id}")
    except Exception as e:
        return STTResult(
            text="",
            provider="local",
            detail=f"Local STT failed: {type(e).__name__}: {e}",
        )


def _transcribe_api(audio_path: str) -> STTResult:
    # Minimal OpenAI-compatible transcription using requests.
    # Users can also set GROQ_* env vars (Groq is OpenAI-compatible for STT).
    import requests

    api_key = _env("OPENAI_API_KEY") or _env("GROQ_API_KEY")
    if not api_key:
        return STTResult(text="", provider="api", detail="Missing OPENAI_API_KEY or GROQ_API_KEY")

    base_url = _env("OPENAI_BASE_URL") or _env("GROQ_BASE_URL") or "https://api.openai.com/v1"
    model = _env("OPENAI_MODEL") or _env("GROQ_STT_MODEL") or "gpt-4o-mini-transcribe"

    url = base_url.rstrip("/") + "/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f)}
        data = {"model": model}
        r = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        if r.status_code >= 300:
            return STTResult(text="", provider="api", detail=f"STT error {r.status_code}: {r.text[:500]}")
        j = r.json()
        return STTResult(text=(j.get("text") or "").strip(), provider="api", detail=f"model={model}")

