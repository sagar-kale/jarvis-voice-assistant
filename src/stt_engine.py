import io
from faster_whisper import WhisperModel

from config.settings import WHISPER_MODEL_SIZE
from src.audio_utils import write_temp_audio, cleanup_temp

_WHISPER_MODEL: WhisperModel | None = None


def _load_whisper() -> WhisperModel:
    """Load faster-whisper once and cache as module singleton."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        print("Loading Whisper STT model…")
        _WHISPER_MODEL = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def transcribe(audio_bytes: bytes) -> dict:
    """
    Transcribe audio bytes to text.

    Returns:
        {
            "text": str,
            "language": str,
            "language_probability": float,
        }
    """
    model = _load_whisper()
    tmp_path = write_temp_audio(audio_bytes, suffix=".wav")

    try:
        segments, info = model.transcribe(
            str(tmp_path),
            beam_size=5,
            language="en",  # force English; set to None for auto-detect
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return {
            "text": text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
        }
    finally:
        cleanup_temp()
