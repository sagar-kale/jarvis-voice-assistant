# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Jarvis Voice Assistant

A fully-local voice assistant (Siri / Jarvis style) running on Apple Silicon.

| Component | Technology |
|-----------|-----------|
| **TTS** | Qwen3-TTS 1.7B (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`) |
| **STT** | faster-whisper small (CPU, int8) |
| **AI brain** | Anthropic Claude SDK (`claude-opus-4-6`) |
| **UI** | Gradio ≥ 4.0 (3 tabs) |
| **Hardware** | Mac Apple Silicon — MPS, float16, SDPA attention |

---

## Setup

```bash
brew install sox             # system dependency (audio processing)
cd qwen-tts
pip install -r requirements.txt
cp .env.example .env        # then add your ANTHROPIC_API_KEY
python app_gradio.py        # → http://127.0.0.1:7860
```

---

## Project Structure

```
qwen-tts/
├── app_gradio.py           # Gradio entrypoint — 3-tab layout
├── src/
│   ├── __init__.py
│   ├── tts_engine.py       # Qwen3-TTS 1.7B (preset voices + zero-shot cloning)
│   ├── stt_engine.py       # faster-whisper small wrapper
│   ├── assistant.py        # Claude SDK conversation manager
│   └── audio_utils.py      # BytesIO helpers, temp file management
├── config/
│   ├── __init__.py
│   └── settings.py         # Env loading, device auto-detect, model IDs
├── requirements.txt
└── .env.example
```

---

## Environment Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `ANTHROPIC_API_KEY` | — | Required |
| `CLAUDE_MODEL` | `claude-opus-4-6` | |
| `ASSISTANT_NAME` | `Jarvis` | Used throughout UI |
| `SYSTEM_PROMPT` | Built-in concise assistant prompt | |

---

## Key Architecture Notes

- **Device detection** (`config/settings.py`): auto-selects `mps` → `cuda` → `cpu`
- **Dtype selection**: `float16` on MPS (~3.6 GB), `bfloat16` on CUDA, `float32` on CPU. `attn_implementation="sdpa"` (no FlashAttention2)
- **Model loading** (`src/tts_engine.py`): uses `device_map=DEVICE` + `dtype=TORCH_DTYPE` + `low_cpu_mem_usage=True` in `from_pretrained()` to load directly onto the target device without double-memory peaks
- **Model caching**: Both TTS and Whisper models use module-level singletons (`_TTS_MODEL`, `_WHISPER_MODEL`) — loaded once on first call, reused for the process lifetime
- **Audio pipeline** (`app_gradio.py`): Jarvis Mode uses `gr.Audio(sources=["microphone"])` → `stop_recording` event → STT → Claude → TTS → `gr.Audio(autoplay=True)`. Mic is cleared after each turn via `.then()` chaining.
- **No re-render collisions**: Gradio uses WebSockets — only the explicitly updated components change; recording never replays old TTS audio
- **Voice cloning**: Pass `ref_audio` + `ref_text` to `Qwen3TTSModel.generate()`
- **Conversation history**: stored in `gr.Chatbot` value (UI) and `ClaudeAssistant._history` (API calls)

---

## Preset Voices

| Speaker | Description |
|---------|-------------|
| Vivian | Bright young female (Chinese) — **default** |
| Serena | Warm gentle female (Chinese) |
| Ryan | Dynamic male (English) |
| Aiden | Sunny American male (English) |
| Uncle_Fu | Mellow seasoned male (Chinese) |
| Dylan | Youthful Beijing male |
| Eric | Lively Chengdu male |
| Ono_Anna | Playful Japanese female |
| Sohee | Warm Korean female |

---

## Claude Flow V3 Integration

This project uses **claude-flow** for multi-agent orchestration. Reference the full config in `../cloude-flow/`.

### Setup

```bash
claude mcp add claude-flow -- npx -y @claude-flow/cli@latest
npx @claude-flow/cli@latest daemon start
npx @claude-flow/cli@latest doctor --fix
```

### Swarm Initialization

```bash
npx @claude-flow/cli@latest swarm init --topology hierarchical --max-agents 8 --strategy specialized
```

### Key CLI Commands

```bash
npx @claude-flow/cli@latest agent spawn -t coder --name my-coder
npx @claude-flow/cli@latest memory search --query "tts patterns"
npx @claude-flow/cli@latest memory store --key "key" --value "value" --namespace qwen-tts
npx @claude-flow/cli@latest task list
```

### Concurrency Rules

- All related operations go in ONE message — batch file reads, writes, and Bash commands together
- Spawn ALL agents in ONE message using the Task tool with `run_in_background: true`
- After spawning agents, stop — do not poll or check status; wait for results to arrive

### Model Routing

| Tier | Model | Use |
|------|-------|-----|
| 1 | Agent Booster (WASM) | Simple transforms, <1ms, $0 |
| 2 | Haiku | Low-complexity tasks |
| 3 | Sonnet/Opus | Architecture, complex reasoning |

Check for `[AGENT_BOOSTER_AVAILABLE]` before spawning — use Edit tool directly when available.

### Useful Agent Types

- `coder`, `reviewer`, `tester`, `planner`, `researcher`
- `sparc-coord`, `specification`, `architecture`
- `security-auditor`, `performance-engineer`

### File Organization

- `/src` — source code
- `/tests` — test files
- `/config` — configuration
- `/scripts` — utility scripts
- `/docs` — documentation
- Never save working files to the project root
