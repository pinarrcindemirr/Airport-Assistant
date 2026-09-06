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
from components.response import render_response


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
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


def reset_to_composer() -> None:
    """Full reset back to a clean composer (the 'New Search' action)."""
    st.session_state.view = "composer"
    st.session_state.response = None
    st.session_state.query_summary = None
    st.session_state.prefill = ""
    st.session_state.img_path = None
    st.session_state.img_name = None
    st.session_state.aud_path = None
    st.session_state.aud_name = None
    st.session_state.aud_bytes = None
    st.session_state.img_nonce += 1
    st.session_state.aud_nonce += 1


# ----------------------------------------------------------- #
# Backend call (the ONLY place the UI talks to the assistant)
# ----------------------------------------------------------- #
def run_query(text: str | None, image_path: str | None, audio_path: str | None):
    from backend.assistant import process_query

    with st.spinner("Searching the airport knowledge base\u2026"):
        response = process_query(
            text=text or None,
            image_path=image_path,
            audio_path=audio_path,
        )

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

render_sidebar(on_new_search=reset_to_composer)

if st.session_state.view == "result" and st.session_state.response is not None:
    render_response(
        st.session_state.response,
        st.session_state.query_summary,
        on_new_search=reset_to_composer,
    )
else:
    render_composer(on_submit=run_query)
