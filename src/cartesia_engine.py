"""
Cartesia TTS engine — cloud-hosted, low-latency alternative to local Qwen3-TTS.

API docs: https://docs.cartesia.ai/api-reference/tts/bytes
"""
from __future__ import annotations

import requests

from config.settings import CARTESIA_API_KEY, CARTESIA_MODEL

_HEADERS = {
    "Cartesia-Version": "2025-04-16",
    "Content-Type": "application/json",
}


def generate_speech(text: str, voice_id: str, speed: float = 1.0) -> bytes:
    """
    Generate speech via Cartesia API.

    Returns WAV bytes (PCM f32le, 44100 Hz) ready for soundfile / Gradio.
    """
    if not CARTESIA_API_KEY:
        raise ValueError("CARTESIA_API_KEY is not set in .env")

    response = requests.post(
        "https://api.cartesia.ai/tts/bytes",
        headers={**_HEADERS, "X-API-Key": CARTESIA_API_KEY},
        json={
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "output_format": {
                "container": "wav",
                "encoding": "pcm_f32le",
                "sample_rate": 44100,
            },
            "language": "en",
            "generation_config": {"speed": speed, "volume": 1},
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.content
