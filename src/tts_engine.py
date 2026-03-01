import io
import numpy as np
import soundfile as sf

from config.settings import (
    DEVICE,
    TORCH_DTYPE,
    ATTN_IMPLEMENTATION,
    QWEN_TTS_MODEL_ID,
)
from src.audio_utils import write_temp_audio

_TTS_MODEL = None


def _load_tts():
    """Load Qwen3-TTS once and cache as module singleton."""
    global _TTS_MODEL
    if _TTS_MODEL is None:
        print("Loading Qwen3-TTS model…")
        from qwen_tts import Qwen3TTSModel  # type: ignore

        _TTS_MODEL = Qwen3TTSModel.from_pretrained(
            QWEN_TTS_MODEL_ID,
            dtype=TORCH_DTYPE,
            device_map=DEVICE,
            attn_implementation=ATTN_IMPLEMENTATION,
            low_cpu_mem_usage=True,
        )
    return _TTS_MODEL


def _wav_to_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    """Convert a numpy waveform to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def generate_speech(
    text: str,
    speaker: str = "Ryan",
    instruct: str = "",
) -> bytes:
    """
    Generate speech using a preset CustomVoice speaker.

    Returns WAV bytes.
    """
    model = _load_tts()
    wavs, sample_rate = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        instruct=instruct if instruct else None,
        max_new_tokens=400,   # ~33s at 12Hz — enough for any short reply
    )
    return _wav_to_bytes(wavs[0], sample_rate)


def clone_voice(
    text: str,
    ref_audio_bytes: bytes,
    ref_text: str,
) -> bytes:
    """
    Zero-shot voice clone using the Base model variant.

    Note: Requires loading a Base model (not CustomVoice). If the loaded model
    is CustomVoice, this will raise a descriptive ValueError from the library.

    Returns WAV bytes.
    """
    model = _load_tts()
    ref_path = write_temp_audio(ref_audio_bytes, suffix=".wav")

    wavs, sample_rate = model.generate_voice_clone(
        text=text,
        ref_audio=str(ref_path),
        ref_text=ref_text,
        max_new_tokens=400,
    )
    return _wav_to_bytes(wavs[0], sample_rate)
