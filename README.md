# Jarvis Voice Assistant

A fully-local, Siri/Jarvis-style voice assistant running on Apple Silicon — record your voice, get an AI response spoken back instantly.

## Stack

| Component | Technology |
|-----------|-----------|
| **TTS** | [Qwen3-TTS 1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) (local) · [Cartesia Sonic-3](https://cartesia.ai) (cloud) |
| **STT** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) small (CPU, int8) |
| **AI brain** | Anthropic Claude (`claude-opus-4-6`) |
| **UI** | [Gradio](https://gradio.app) 6.x — 3-tab interface |
| **Hardware** | Mac Apple Silicon — MPS, float16, SDPA attention |

## Features

- **Jarvis Mode** — push-to-talk voice conversation with AI
- **TTS provider toggle** — switch between fast cloud (Cartesia ~300ms) and fully-local (Qwen3-TTS)
- **TTS Studio** — type any text, pick a voice, generate & download audio
- **Voice Clone Lab** — upload a reference recording, clone that voice, synthesise new speech
- **9 preset voices** — English, Chinese, Japanese, Korean

## Setup

### 1. System dependency

```bash
brew install sox
```

### 2. Install Python packages

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
CARTESIA_API_KEY=sk_car_...   # optional — only needed for Cartesia TTS
ASSISTANT_NAME=Jarvis          # name shown in the UI
CLAUDE_MODEL=claude-opus-4-6
```

### 4. Run

```bash
python app_gradio.py
# → http://127.0.0.1:7860
```

## Project Structure

```
qwen-tts/
├── app_gradio.py           # Gradio entrypoint — 3-tab layout
├── src/
│   ├── tts_engine.py       # Qwen3-TTS local engine (module-level singleton)
│   ├── stt_engine.py       # faster-whisper wrapper (module-level singleton)
│   ├── cartesia_engine.py  # Cartesia cloud TTS client
│   ├── assistant.py        # Claude SDK conversation manager
│   └── audio_utils.py      # Temp file helpers
├── config/
│   └── settings.py         # Device detection, model IDs, voice lists
├── requirements.txt
└── .env.example
```

## Preset Voices (Qwen3-TTS)

| Voice | Description |
|-------|-------------|
| Vivian | Bright young female (Chinese) |
| Serena | Warm gentle female (Chinese) |
| Ryan | Dynamic male (English) |
| Aiden | Sunny American male (English) |
| Uncle_Fu | Mellow seasoned male (Chinese) |
| Dylan | Youthful Beijing male |
| Eric | Lively Chengdu male |
| Ono_Anna | Playful Japanese female |
| Sohee | Warm Korean female |

## Adding Cartesia Voices

Find voice IDs at [play.cartesia.ai/voices](https://play.cartesia.ai/voices), then add to `config/settings.py`:

```python
CARTESIA_VOICES = {
    "Cathe": "6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
    "My Voice": "<your-voice-id>",
}
```

## Requirements

- Python 3.10+
- macOS Apple Silicon (MPS) — also runs on CUDA or CPU
- ~3.6 GB VRAM for local Qwen3-TTS (float16)
