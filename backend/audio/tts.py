from __future__ import annotations

from io import BytesIO


def synthesize_speech(text: str, lang: str = "en") -> bytes:

    from gtts import gTTS

    buffer = BytesIO()
    tts = gTTS(text=text, lang=lang)
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer.read()