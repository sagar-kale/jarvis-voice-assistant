import io
import tempfile
from pathlib import Path

_temp_files: list[Path] = []


def bytes_to_audio_input(audio_bytes: bytes) -> io.BytesIO:
    """Wrap raw audio bytes in a BytesIO buffer for st.audio()."""
    buf = io.BytesIO(audio_bytes)
    buf.seek(0)
    return buf


def get_temp_path(suffix: str = ".wav") -> Path:
    """Create a named temp file and track it for later cleanup."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    path = Path(tmp.name)
    _temp_files.append(path)
    return path


def write_temp_audio(audio_bytes: bytes, suffix: str = ".wav") -> Path:
    """Write audio bytes to a temp file and return the path."""
    path = get_temp_path(suffix)
    path.write_bytes(audio_bytes)
    return path


def cleanup_temp() -> None:
    """Remove all tracked temp files."""
    for path in _temp_files:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    _temp_files.clear()
