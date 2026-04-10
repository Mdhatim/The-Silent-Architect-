from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from src.agent.runner import run_from_text
from src.agent.stt import transcribe


REPO_ROOT = Path(__file__).resolve().parent


def _save_uploaded_to_temp(uploaded_file: str) -> str:
    # Gradio gives a filepath string for Audio(type="filepath")
    return uploaded_file


def run_pipeline(audio_filepath: str | None) -> tuple[str, str, str, str]:
    if not audio_filepath:
        return "", "", "No action", "Please record or upload audio."

    stt = transcribe(_save_uploaded_to_temp(audio_filepath))
    transcription = stt.text
    if not transcription.strip():
        return "", "", "No action", f"Transcription failed. ({stt.provider}) {stt.detail}".strip()

    res = run_from_text(REPO_ROOT, transcription)
    intent = f"{res.intent} (confidence={res.confidence:.2f})"
    action = res.action_taken
    output = res.output
    return transcription, intent, action, output


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Local Audio Agent") as demo:
        gr.Markdown(
            "### Local Audio Agent\n"
            "Record or upload audio → Speech-to-Text → Intent → Tool execution (restricted to `output/`)."
        )

        with gr.Row():
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio input")

        run_btn = gr.Button("Run", variant="primary")

        with gr.Row():
            transcription = gr.Textbox(label="Transcribed text", lines=4)
        with gr.Row():
            intent = gr.Textbox(label="Detected intent", lines=1)
        with gr.Row():
            action = gr.Textbox(label="Action taken", lines=2)
        with gr.Row():
            output = gr.Textbox(label="Final output / result", lines=10)

        run_btn.click(fn=run_pipeline, inputs=[audio], outputs=[transcription, intent, action, output])

        gr.Markdown(
            "### Notes\n"
            "- All file writes are restricted to the `output/` folder.\n"
            "- Configure providers in `.env` (copy from `.env.example`)."
        )
    return demo


if __name__ == "__main__":
    load_dotenv()
    ui = build_ui()
    ui.launch()

