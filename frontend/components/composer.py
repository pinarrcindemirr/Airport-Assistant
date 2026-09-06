"""
Unified composer: text + photo + voice on ONE screen, all available at once
(no tabs, no mode switching).
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
_AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg"]


def _save_temp(uploaded_file, allowed_ext) -> str:
    """Persist an in-memory Streamlit upload to a temp file and return its path
    (process_query expects a file PATH, not an in-memory buffer)."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext.lstrip(".") not in allowed_ext:
        ext = "." + allowed_ext[0]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as fh:
        fh.write(uploaded_file.getbuffer())
        return fh.name


def _popover(label):
    """st.popover when available (Streamlit >= 1.32), else an expander."""
    if hasattr(st, "popover"):
        return st.popover(label, use_container_width=True)
    return st.expander(label)


def _render_attachment_controls() -> None:
    """The 'Add photo' / 'Add voice' buttons + the preview chips."""
    col_photo, col_voice, _ = st.columns([1, 1, 3])

    # ---- Add photo ------ #
    with col_photo:
        with _popover("\U0001F4CE  Add photo"):
            img = st.file_uploader(
                "Upload a photo of an airport sign, symbol, or service icon",
                type=_IMAGE_TYPES,
                key=f"img_uploader_{st.session_state.img_nonce}",
            )
            if img is not None:
                st.session_state.img_path = _save_temp(img, _IMAGE_TYPES)
                st.session_state.img_name = img.name

    # ---- Add voice -------- #
    with col_voice:
        with _popover("\U0001F3A4  Add voice"):
            st.caption("Record your question, or upload a WAV/MP3.")
            rec = None
            if hasattr(st, "audio_input"):
                rec = st.audio_input(
                    "Record",
                    key=f"aud_recorder_{st.session_state.aud_nonce}",
                )
            up = st.file_uploader(
                "Upload audio",
                type=_AUDIO_TYPES,
                key=f"aud_uploader_{st.session_state.aud_nonce}",
            )
            chosen = rec or up
            if chosen is not None:
                st.session_state.aud_bytes = chosen.getvalue()
                st.session_state.aud_path = _save_temp(chosen, _AUDIO_TYPES)
                st.session_state.aud_name = chosen.name if up is not None else "recording.wav"

    # ---- Preview chips ----------- #
    if st.session_state.img_path or st.session_state.aud_path:
        st.markdown('<div class="via-attachments">', unsafe_allow_html=True)
        pcols = st.columns(2)
        if st.session_state.img_path:
            with pcols[0]:
                st.image(st.session_state.img_path, width=120)
                st.caption(st.session_state.img_name or "image")
                if st.button("\u00d7 Remove photo", key="rm_img"):
                    st.session_state.img_path = None
                    st.session_state.img_name = None
                    st.session_state.img_nonce += 1
                    st.rerun()
        if st.session_state.aud_path:
            with pcols[1]:
                if st.session_state.aud_bytes:
                    st.audio(st.session_state.aud_bytes)
                st.caption(st.session_state.aud_name or "voice note")
                if st.button("\u00d7 Remove voice", key="rm_aud"):
                    st.session_state.aud_path = None
                    st.session_state.aud_name = None
                    st.session_state.aud_bytes = None
                    st.session_state.aud_nonce += 1
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_composer(on_submit) -> None:
    st.markdown('<h1 class="via-title">VIA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="via-subtitle">Ask by text, voice, or photo</p>',
                unsafe_allow_html=True)

    with st.container(border=True):
        text = st.text_input(
            "Your question",
            value=st.session_state.prefill,
            placeholder="e.g. Where is gate B12?",
            label_visibility="collapsed",
        )

        _render_attachment_controls()

        send = st.button("Send  \u2192", type="primary", key="composer_send")

    if send:
        has_input = bool(text.strip()) or st.session_state.img_path or st.session_state.aud_path
        if not has_input:
            st.warning("Type a question, or add a photo or voice note first.")
            return
        on_submit(
            text.strip() or None,
            st.session_state.img_path,
            st.session_state.aud_path,
        )
        st.rerun()

    # Welcome card
    st.markdown(
        """
        <div class="via-card via-welcome">
            <div class="via-badge via-badge-ready">[ SYSTEM READY ]</div>
            <h3>Welcome to VIA. How can I assist you today?</h3>
            <p>I help you find gates, lounges, baggage claim, and other
            airport services.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
