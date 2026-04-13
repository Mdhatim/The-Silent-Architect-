import gradio as gr
from src.agent.runner import run_from_text
from src.agent.stt import transcribe
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

def run_pipeline(audio_file):
    if not audio_file:
        return "", "", "No action", "Please provide audio."

    # STT Requirement [cite: 10]
    stt = transcribe(audio_file)
    
    # Intent & Tool Execution 
    res = run_from_text(REPO_ROOT, stt.text)
    
    return (
        stt.text, 
        f"{res.intent} ({res.confidence:.2f})", 
        res.action_taken, 
        res.output
    )

with gr.Blocks() as demo:
    gr.Markdown("# The Silent Architect")
    audio = gr.Audio(sources=["microphone", "upload"], type="filepath") # Requirement [cite: 7]
    btn = gr.Button("Run Agent")
    
    t_text = gr.Textbox(label="1. Transcribed Text")
    t_intent = gr.Textbox(label="2. Detected Intent")
    t_action = gr.Textbox(label="3. Action Taken")
    t_result = gr.Textbox(label="4. Final output / result")

    btn.click(run_pipeline, inputs=[audio], outputs=[t_text, t_intent, t_action, t_result])

demo.launch()