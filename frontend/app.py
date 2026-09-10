from __future__ import annotations

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from components.sidebar import render_sidebar
from components.composer import render_composer
from components.response import render_response, render_error


# --------------- #
# One-time setup
# --------------- #
st.set_page_config(
    page_title="VIA - Vion International Airport",
    page_icon="\u2708\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_css() -> None:
    css_path = os.path.join(_HERE, "styles", "custom.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as fh:
            st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)


def _init_state() -> None:
    defaults = {
        "view": "composer",       
        "response": None,         
        "query_summary": None,     
        "prefill": "",            
        "img_nonce": 0,          
        "aud_nonce": 0,        
        "img_path": None,
        "img_name": None,
        "aud_path": None,
        "aud_name": None,
        "aud_bytes": None,
        "error_message": None,
        "pending_quick_query": None,
        "models_warmed": False,
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


def _warm_up_models() -> None:

    if st.session_state.models_warmed:
        return
    placeholder = st.empty()
    placeholder.markdown(
        """
        <div class="via-loading-box via-loading-warmup">
            <div class="via-loading-spinner"></div>
            <span>Initializing VIA systems for the first time
            — this may take up to 20 seconds…</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    from backend.assistant import warm_up
    warm_up()
    placeholder.empty()
    st.session_state.models_warmed = True


def _cleanup_temp_files(*paths) -> None:

    for path in paths:
        if not path:
            continue
        try:
            os.remove(path)
        except OSError:
            pass


def reset_to_composer() -> None:
    """Full reset back to a clean composer (the 'New Search' action)."""
    _cleanup_temp_files(st.session_state.img_path, st.session_state.aud_path)
    st.session_state.view = "composer"
    st.session_state.response = None
    st.session_state.query_summary = None
    st.session_state.prefill = ""
    st.session_state.img_path = None
    st.session_state.img_name = None
    st.session_state.aud_path = None
    st.session_state.aud_name = None
    st.session_state.aud_bytes = None
    st.session_state.error_message = None
    st.session_state.img_nonce += 1
    st.session_state.aud_nonce += 1


# ----------------------------------------------------------- #
# Backend call (the ONLY place the UI talks to the assistant)
# ----------------------------------------------------------- #
def run_query(text: str | None, image_path: str | None, audio_path: str | None):
    from backend.assistant import process_query

    placeholder = st.empty()
    placeholder.markdown(
        """
        <div class="via-loading-box">
            <div class="via-loading-spinner"></div>
            <span>Searching the airport knowledge base\u2026</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        response = process_query(
            text=text or None,
            image_path=image_path,
            audio_path=audio_path,
        )
    except Exception as e:
        placeholder.empty()
        _cleanup_temp_files(image_path, audio_path)
        st.session_state.error_message = str(e)
        st.session_state.view = "error"
        return
    placeholder.empty()
    _cleanup_temp_files(image_path, audio_path)

    st.session_state.response = response
    st.session_state.query_summary = {
        "text": text or None,
        "image_name": st.session_state.img_name,
        "audio_name": st.session_state.aud_name,
    }
    st.session_state.view = "result"


# ------------- #
# Render
# ------------- #
_load_css()
_init_state()
_warm_up_models()

render_sidebar(on_new_search=reset_to_composer)

if st.session_state.pending_quick_query:
    _quick_q = st.session_state.pending_quick_query
    st.session_state.pending_quick_query = None
    _cleanup_temp_files(st.session_state.img_path, st.session_state.aud_path)
    st.session_state.img_path = None
    st.session_state.img_name = None
    st.session_state.aud_path = None
    st.session_state.aud_name = None
    st.session_state.aud_bytes = None
    run_query(_quick_q, None, None)

if st.session_state.view == "result" and st.session_state.response is not None:
    render_response(
        st.session_state.response,
        st.session_state.query_summary,
        on_new_search=reset_to_composer,
    )
elif st.session_state.view == "error":
    render_error(
        st.session_state.error_message,
        on_new_search=reset_to_composer,
    )
else:
    render_composer(on_submit=run_query)