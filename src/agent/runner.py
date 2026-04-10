from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .llm import classify_and_plan
from .safe_tools import ToolAction, create_file, write_text
from .schemas import Intent, IntentResult


@dataclass(frozen=True)
class PipelineResult:
    transcription: str
    intent: str
    confidence: float
    action_taken: str
    output: str
    raw_model_output: str


def run_from_text(repo_root: Path, text: str) -> PipelineResult:
    llm_res = classify_and_plan(text)
    parsed: IntentResult | None = llm_res.parsed

    if parsed is None:
        return PipelineResult(
            transcription=text,
            intent="unknown",
            confidence=0.0,
            action_taken="No action (failed to parse model output as JSON).",
            output="",
            raw_model_output=llm_res.raw_text or llm_res.detail,
        )

    action: ToolAction | None = None
    final_out = ""

    if parsed.intent == Intent.create_file:
        path = parsed.path or "new_file.txt"
        action = create_file(repo_root, path, overwrite=False)
        final_out = action.detail

    elif parsed.intent == Intent.write_code:
        path = parsed.path or "generated_code.py"
        content = parsed.content or ""
        if not content.strip():
            content = "# (No code generated)\n"
        action = write_text(repo_root, path, content, append=False)
        final_out = action.detail

    elif parsed.intent == Intent.summarize_text:
        txt = parsed.text or ""
        if not txt.strip():
            txt = text
        # simple local summarization via LLM (reuse classifier provider but different prompt)
        summary = _summarize_with_llm(txt)
        action = ToolAction(name="summarize_text", detail="Summarized text.", output=summary)
        final_out = summary

    else:
        action = ToolAction(name="general_chat", detail="Chat response.", output=_chat_with_llm(text))
        final_out = action.output

    return PipelineResult(
        transcription=text,
        intent=parsed.intent.value,
        confidence=float(parsed.confidence),
        action_taken=action.detail if action else "No action",
        output=final_out,
        raw_model_output=llm_res.raw_text,
    )


def _summarize_with_llm(text: str) -> str:
    from .llm import _classify_ollama, _classify_openai_compatible, _env

    provider = (_env("LLM_PROVIDER", "ollama") or "ollama").lower()
    prompt = (
        "Summarize the following text in 5-8 bullet points. "
        "Be faithful and concise.\n\nTEXT:\n" + text
    )
    if provider in {"ollama", "local"}:
        r = _classify_ollama(prompt)
    else:
        r = _classify_openai_compatible(prompt)

    # For summarization we want raw_text; the classifier returns JSON usually, so handle both.
    if r.parsed and r.parsed.intent == Intent.summarize_text and (r.parsed.content or r.parsed.text):
        return (r.parsed.content or r.parsed.text or "").strip()
    return (r.raw_text or "").strip() or "Summary unavailable (LLM error)."


def _chat_with_llm(text: str) -> str:
    from .llm import _classify_ollama, _classify_openai_compatible, _env

    provider = (_env("LLM_PROVIDER", "ollama") or "ollama").lower()
    prompt = "You are a helpful assistant. Reply to the user:\n\n" + text
    if provider in {"ollama", "local"}:
        r = _classify_ollama(prompt)
    else:
        r = _classify_openai_compatible(prompt)

    if r.parsed and r.parsed.content:
        return r.parsed.content.strip()
    return (r.raw_text or "").strip() or "No response (LLM error)."

