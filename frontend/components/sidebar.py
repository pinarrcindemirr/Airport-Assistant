from __future__ import annotations

import streamlit as st

SAMPLE_QUERIES = [
    "Where is gate B12?",
    "How do I get to baggage claim?",
    "Where is the nearest prayer room?",
    "Where can I find a restaurant?",
]

TAGLINE = "Your guide to terminals, gates, and airport services."


def render_sidebar(on_new_search) -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="via-brand">
                <div class="via-brand-mark">\u2708\ufe0f</div>
                <div>
                    <div class="via-brand-title">Vion International Airport</div>
                    <div class="via-brand-sub">VIA AI Concierge</div>
                </div>
            </div>
            <div class="via-tagline">{TAGLINE}</div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("\U0001F50D  New Search", use_container_width=True, key="sb_new"):
            on_new_search()
            st.rerun()

        st.markdown('<div class="via-section-label">SAMPLE QUERIES</div>',
                    unsafe_allow_html=True)
        for i, q in enumerate(SAMPLE_QUERIES):
            if st.button(q, key=f"sb_sample_{i}", use_container_width=True):
                on_new_search()
                st.session_state.prefill = q
                st.session_state.view = "composer"
                st.rerun()

        st.markdown('<div class="via-section-label">HOW IT WORKS</div>',
                    unsafe_allow_html=True)
        st.markdown(
            """
            <ol class="via-steps">
                <li>Type, speak, or upload a photo \u2014 or combine them.</li>
                <li>Ask about a gate, service, or an airport sign.</li>
                <li>Get clear directions and service information.</li>
            </ol>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="via-sidebar-footer">', unsafe_allow_html=True)
        st.caption("Help Center \u00b7 Settings")
        st.markdown("</div>", unsafe_allow_html=True)
