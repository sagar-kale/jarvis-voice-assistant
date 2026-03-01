import os
import torch
from dotenv import load_dotenv

load_dotenv()

# ── API / assistant ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Jarvis")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    f"You are {ASSISTANT_NAME}, a helpful AI voice assistant. "
    "You are concise, intelligent, and friendly. "
    "Keep responses under 3 sentences unless detail is explicitly needed.",
)

# ── Device auto-detect ─────────────────────────────────────────────────────────
def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _detect_device()

# float16 halves model memory (7.14GB → ~3.6GB) and is well supported on MPS since PyTorch 2.x
TORCH_DTYPE = torch.float16 if DEVICE == "mps" else (torch.bfloat16 if DEVICE == "cuda" else torch.float32)

# No FlashAttention2 on MPS; use SDPA
ATTN_IMPLEMENTATION = "sdpa"

# ── Model IDs ─────────────────────────────────────────────────────────────────
QWEN_TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
WHISPER_MODEL_SIZE = "small"

# ── Preset voices ─────────────────────────────────────────────────────────────
PRESET_VOICES = [
    "Vivian",      # Bright young female (Chinese)
    "Serena",      # Warm gentle female (Chinese)
    "Ryan",        # Dynamic male (English)
    "Aiden",       # Sunny American male (English)
    "Uncle_Fu",    # Mellow seasoned male (Chinese)
    "Dylan",       # Youthful Beijing male
    "Eric",        # Lively Chengdu male
    "Ono_Anna",    # Playful Japanese female
    "Sohee",       # Warm Korean female
]

DEFAULT_VOICE = "Sohee"

# ── Cartesia TTS ───────────────────────────────────────────────────────────────
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_MODEL = "sonic-3"

# Voice name → Cartesia voice ID. Add more from https://play.cartesia.ai/voices
CARTESIA_VOICES: dict[str, str] = {
    "Cathe":          "6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
}

CARTESIA_DEFAULT_VOICE = "Cathe"
