"""
Jarvis — local voice assistant powered by Qwen3-TTS + faster-whisper + Claude.

Run:
    python app_gradio.py
"""
from __future__ import annotations

import io
import numpy as np
import soundfile as sf
import gradio as gr

from config.settings import (
    PRESET_VOICES, DEFAULT_VOICE, ASSISTANT_NAME,
    CARTESIA_VOICES, CARTESIA_DEFAULT_VOICE,
)


# ── Audio conversion helpers ───────────────────────────────────────────────────

def _numpy_to_wav_bytes(arr: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, arr.astype(np.float32), sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _wav_bytes_to_numpy(wav_bytes: bytes) -> tuple[int, np.ndarray]:
    buf = io.BytesIO(wav_bytes)
    arr, sr = sf.read(buf, dtype="int16")
    return sr, arr


def _prep_mic_arr(arr: np.ndarray) -> np.ndarray:
    """Normalise microphone int16 input and collapse stereo → mono."""
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr


# ── Lazy singletons ────────────────────────────────────────────────────────────

_assistant = None


def _get_assistant():
    global _assistant
    if _assistant is None:
        from src.assistant import ClaudeAssistant
        _assistant = ClaudeAssistant()
    return _assistant


# ── Pipeline functions ─────────────────────────────────────────────────────────

def process_voice_message(
    audio_input: tuple[int, np.ndarray] | None,
    provider: str,
    local_voice: str,
    cartesia_voice: str,
    emotion: str,
    history: list[dict],
) -> tuple[tuple[int, np.ndarray] | None, list[dict]]:
    """Mic → STT → Claude → TTS pipeline for Jarvis Mode."""
    if audio_input is None:
        return None, history

    sr, arr = audio_input
    if arr is None or arr.size == 0:
        return None, history

    arr = _prep_mic_arr(arr)
    wav_bytes = _numpy_to_wav_bytes(arr, sr)

    # STT
    from src.stt_engine import transcribe
    result = transcribe(wav_bytes)
    user_text = result["text"].strip()

    if not user_text:
        gr.Warning("No speech detected — please try again.")
        return None, history

    history = history + [{"role": "user", "content": user_text}]

    # Claude
    reply = _get_assistant().chat(user_text)
    history = history + [{"role": "assistant", "content": reply}]

    # TTS — route to selected provider
    if provider == "Cartesia":
        from src.cartesia_engine import generate_speech as cartesia_tts
        voice_id = CARTESIA_VOICES[cartesia_voice]
        tts_bytes = cartesia_tts(reply, voice_id=voice_id)
    else:
        from src.tts_engine import generate_speech
        tts_bytes = generate_speech(reply, speaker=local_voice, instruct=emotion)

    sr_out, arr_out = _wav_bytes_to_numpy(tts_bytes)
    return (sr_out, arr_out), history


def reset_conversation() -> list:
    _get_assistant().reset()
    return []


def generate_tts_studio(text: str, voice: str, emotion: str):
    if not text.strip():
        gr.Warning("Please enter some text.")
        return None
    from src.tts_engine import generate_speech
    wav_bytes = generate_speech(text, speaker=voice, instruct=emotion)
    return _wav_bytes_to_numpy(wav_bytes)


def clone_voice_fn(
    ref_audio: tuple[int, np.ndarray] | None,
    ref_text: str,
    clone_text: str,
):
    if ref_audio is None:
        gr.Warning("Please upload a reference audio file.")
        return None
    if not ref_text.strip():
        gr.Warning("Please enter the reference transcript.")
        return None
    if not clone_text.strip():
        gr.Warning("Please enter the text to synthesise.")
        return None

    sr, arr = ref_audio
    arr = _prep_mic_arr(arr)
    ref_bytes = _numpy_to_wav_bytes(arr, sr)

    from src.tts_engine import clone_voice
    wav_bytes = clone_voice(text=clone_text, ref_audio_bytes=ref_bytes, ref_text=ref_text)
    return _wav_bytes_to_numpy(wav_bytes)


def _toggle_voice_panels(provider: str):
    """Show the correct voice dropdown based on selected TTS provider."""
    return (
        gr.update(visible=provider == "Local (Qwen3-TTS)"),
        gr.update(visible=provider == "Cartesia"),
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title=f"{ASSISTANT_NAME} Voice Assistant") as demo:
    gr.Markdown(f"# 🤖 {ASSISTANT_NAME} Voice Assistant")

    with gr.Tabs():

        # ── Tab 1: Jarvis Mode ─────────────────────────────────────────────────
        with gr.Tab("🤖 Jarvis Mode"):
            with gr.Row():
                with gr.Column(scale=1, min_width=240):
                    gr.Markdown("### TTS Provider")
                    tts_provider = gr.Radio(
                        choices=["Local (Qwen3-TTS)", "Cartesia"],
                        value="Cartesia",
                        label="Engine",
                    )

                    # Local voice options (hidden when Cartesia selected)
                    with gr.Group(visible=False) as local_group:
                        voice_select = gr.Dropdown(
                            choices=PRESET_VOICES,
                            value=DEFAULT_VOICE,
                            label="Local voice",
                        )
                        emotion_input = gr.Textbox(
                            label="Emotion / style",
                            placeholder='e.g. "Speak warmly and slowly"',
                        )

                    # Cartesia voice options (visible by default)
                    with gr.Group(visible=True) as cartesia_group:
                        cartesia_voice_select = gr.Dropdown(
                            choices=list(CARTESIA_VOICES.keys()),
                            value=CARTESIA_DEFAULT_VOICE,
                            label="Cartesia voice",
                        )

                    reset_btn = gr.Button("🔄 Reset conversation", variant="secondary")

                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label=ASSISTANT_NAME,
                        height=420,
                    )
                    tts_audio_out = gr.Audio(
                        label=f"{ASSISTANT_NAME}'s voice",
                        autoplay=True,
                        interactive=False,
                    )
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="🎤 Click to record, click again to stop",
                    )

            # Toggle visibility when provider changes
            tts_provider.change(
                fn=_toggle_voice_panels,
                inputs=[tts_provider],
                outputs=[local_group, cartesia_group],
            )

            mic_input.stop_recording(
                fn=process_voice_message,
                inputs=[mic_input, tts_provider, voice_select, cartesia_voice_select, emotion_input, chatbot],
                outputs=[tts_audio_out, chatbot],
                show_progress="minimal",
            ).then(
                fn=lambda: gr.update(value=None),
                outputs=[mic_input],
            )

            reset_btn.click(
                fn=reset_conversation,
                outputs=[chatbot],
            )

        # ── Tab 2: TTS Studio ──────────────────────────────────────────────────
        with gr.Tab("🔊 TTS Studio"):
            gr.Markdown("Type text, choose a voice, generate and download.")
            with gr.Row():
                with gr.Column():
                    studio_text = gr.Textbox(
                        label="Text to speak",
                        placeholder="Enter text here…",
                        lines=5,
                    )
                    with gr.Row():
                        studio_voice = gr.Dropdown(
                            choices=PRESET_VOICES,
                            value=DEFAULT_VOICE,
                            label="Voice",
                        )
                        studio_emotion = gr.Textbox(
                            label="Emotion / style",
                            placeholder='e.g. "Fast and excited"',
                        )
                    studio_btn = gr.Button("🔊 Generate", variant="primary")

                with gr.Column():
                    studio_audio_out = gr.Audio(label="Output", interactive=False)

            studio_btn.click(
                fn=generate_tts_studio,
                inputs=[studio_text, studio_voice, studio_emotion],
                outputs=[studio_audio_out],
            )

        # ── Tab 3: Voice Clone ─────────────────────────────────────────────────
        with gr.Tab("🎭 Voice Clone"):
            gr.Markdown(
                "Upload a reference recording → provide transcript → "
                "synthesise new speech in that voice."
            )
            with gr.Row():
                with gr.Column():
                    ref_audio_input = gr.Audio(
                        label="Reference audio (.wav / .mp3)",
                        type="numpy",
                    )
                    ref_transcript = gr.Textbox(
                        label="Reference transcript",
                        placeholder="e.g. Hello, my name is Alex…",
                    )
                    clone_text_input = gr.Textbox(
                        label="Text to synthesise in cloned voice",
                        placeholder="Enter text to speak…",
                        lines=3,
                    )
                    clone_btn = gr.Button("🎭 Clone & Generate", variant="primary")

                with gr.Column():
                    clone_audio_out = gr.Audio(label="Cloned output", interactive=False)

            clone_btn.click(
                fn=clone_voice_fn,
                inputs=[ref_audio_input, ref_transcript, clone_text_input],
                outputs=[clone_audio_out],
            )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=True)
