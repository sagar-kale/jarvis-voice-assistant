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

# Voice name → Cartesia voice ID
CARTESIA_VOICES: dict[str, str] = {
    # Custom
    "Cathe":                      "6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
    # English
    "Blake - Helpful Agent":      "a167e0f3-df7e-4d52-a9c3-f949145efdab",
    "Brooke - Big Sister":        "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
    "Caroline - Southern Guide":  "f9836c6e-a0bd-460e-9d3c-f7299fa60f94",
    "Cathy - Coworker":           "e8e5fffb-252c-436d-b842-8879b84445b6",
    "Jacqueline - Reassuring":    "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    "Katie - Friendly Fixer":     "f786b574-daa5-4673-aa0c-cbe3e8534c02",
    "Ronald - Thinker":           "5ee9feff-1265-424a-9d7f-8e4d431a12c7",
    "Theo - Modern Narrator":     "79f8b5fb-2cc8-479a-80df-29f7a7cf1a3e",
    # Hindi
    "Arushi - Hinglish Speaker":  "95d51f79-c397-46f9-b49a-23763d3eaa2d",
    "Riya - College Roommate":    "faf0731e-dfb9-4cfc-8119-259a79b27e12",
}

CARTESIA_DEFAULT_VOICE = "Cathe"
