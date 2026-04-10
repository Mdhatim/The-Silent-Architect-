# Local Audio AI Agent (STT → Intent → Tools → UI)

This project runs a **local agent** that:

- Accepts audio via **microphone** or **file upload**
- Converts audio to text via **Speech-to-Text (STT)**
- Uses an LLM to **classify intent** (and plan an action)
- Executes **local tools** with a hard safety constraint: **writes only inside `output/`**
- Displays the full pipeline in a clean **web UI**

## Supported intents (minimum)

- **Create a file**
- **Write code** to a new or existing file
- **Summarize text**
- **General chat**

## Safety

All file creation and writing is restricted to the repository’s `output/` directory. Any attempt to escape that directory is blocked.

## Setup (Windows / PowerShell)

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy environment configuration:

```bash
copy .env.example .env
```

## Run

```bash
python app.py
```

Open the local URL shown in the terminal.

## STT (Speech-to-Text)

### Local (default)

Default `STT_PROVIDER=local`.

The app tries:

1. `faster-whisper` (recommended for CPU)
2. `transformers` Whisper pipeline fallback

You can configure:

- `WHISPER_MODEL` (e.g. `small`, `base`, `medium`)
- `WHISPER_DEVICE` (`cpu`)
- `WHISPER_COMPUTE_TYPE` (`int8`, `int8_float16`, etc.)
- `HF_STT_MODEL` (e.g. `openai/whisper-small`)

### API fallback (optional)

If your machine is too slow for local STT, set `STT_PROVIDER=api` in `.env` and provide:

- `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`, `OPENAI_MODEL`)
  - or -
- `GROQ_API_KEY` (+ optional `GROQ_BASE_URL`, `GROQ_STT_MODEL`)

If you use API STT, document your reason in this README (e.g., “CPU-only laptop too slow for Whisper medium; using API for responsiveness”).

## LLM intent understanding

### Local (default): Ollama

Set in `.env`:

- `LLM_PROVIDER=ollama`
- `OLLAMA_HOST=http://localhost:11434`
- `OLLAMA_MODEL=llama3.1:8b`

Make sure Ollama is running and you have pulled the model:

```bash
ollama pull llama3.1:8b
```

### API fallback

Set `LLM_PROVIDER=api` and provide:

- `OPENAI_API_KEY` / `OPENAI_CHAT_MODEL`
  - or -
- `GROQ_API_KEY` / `GROQ_CHAT_MODEL`

## Output folder

Generated files are written to:

- `output/`

Example: user asks “Create a Python file with a retry function.” → the agent writes a `.py` file inside `output/`.

