from __future__ import annotations

import html

import streamlit as st

def _speech_text(response) -> str:
    if response.answered and response.record:
        record = response.record
        parts = [f"{record.get('name', '')}."]
        if record.get("directions"):
            parts.append(f"Directions: {record['directions']}.")
        if record.get("opening_hours"):
            parts.append(f"Opening hours: {record['opening_hours']}.")
        return " ".join(parts)
    return response.message


def _speech_button(response) -> None:
    if st.button("\U0001F50A  Listen to this", key="result_listen"):
        from backend.audio.tts import synthesize_speech
        with st.spinner("Generating audio\u2026"):
            audio_bytes = synthesize_speech(_speech_text(response))
        st.audio(audio_bytes, format="audio/mp3")

CATEGORY_LABELS = {
    "gate": "Gate Info",
    "check_in": "Check-In",
    "baggage_claim": "Baggage Claim",
    "security": "Security",
    "information_desk": "Information Desk",
    "lounge": "Lounge",
    "restaurant": "Restaurant",
    "restroom": "Restroom",
    "prayer_room": "Prayer Room",
    "lost_and_found": "Lost & Found",
    "customs": "Customs",
    "currency_exchange": "Currency Exchange",
    "special_assistance": "Special Assistance",
    "transport": "Transport",
    "pharmacy": "Pharmacy",
    "smoking_area": "Smoking Area",
}


def _esc(value) -> str:
    return html.escape(str(value)) if value is not None else ""


def _confidence_bar(confidence: float, tone: str) -> str:
    pct = max(0, min(100, round(confidence * 100)))
    return (
        f'<div class="via-conf via-conf-{tone}">'
        f'  <div class="via-conf-head">'
        f'    <span>CONFIDENCE</span><span>{pct}%</span>'
        f'  </div>'
        f'  <div class="via-conf-track"><div class="via-conf-fill" '
        f'style="width:{pct}%"></div></div>'
        f'</div>'
    )


def _query_summary(summary: dict | None) -> None:
    if not summary:
        return
    parts = []
    if summary.get("text"):
        parts.append(f'\U0001F4AC {_esc(summary["text"])}')
    if summary.get("image_name"):
        parts.append(f'\U0001F5BC {_esc(summary["image_name"])}')
    if summary.get("audio_name"):
        parts.append(f'\U0001F3A4 {_esc(summary["audio_name"])}')
    separator = "&nbsp;&nbsp;\u00b7&nbsp;&nbsp;"
    joined = separator.join(parts)
    st.markdown(
        f'<div class="via-query-summary">{joined}</div>',
        unsafe_allow_html=True,
    )


def _success_card(record: dict, confidence: float) -> None:
    from backend.fusion.router import FUSION_CONFIDENCE_THRESHOLD

    category = record.get("category", "")
    label = CATEGORY_LABELS.get(category, category.replace("_", " ").title())

    terminal = record.get("terminal", "")
    zone = record.get("floor_or_zone", "")
    zone_bits = " \u00b7 ".join([b for b in (terminal, zone) if b])

    if confidence >= 0.75:
        tone = "high"
    elif confidence >= FUSION_CONFIDENCE_THRESHOLD:
        tone = "mid"
    else:
        tone = "low"

    rows = ""
    for icon, title, key in (
        ("\U0001F9ED", "Directions", "directions"),
        ("\U0001F551", "Opening Hours", "opening_hours"),
        ("\u267F", "Accessibility", "accessibility"),
    ):
        val = record.get(key)
        if val:
            rows += (
                f'<div class="via-row">'
                f'  <div class="via-row-icon">{icon}</div>'
                f'  <div><div class="via-row-title">{title}</div>'
                f'  <div class="via-row-text">{_esc(val)}</div></div>'
                f'</div>'
            )

    contact = record.get("contact")
    contact_html = (
        f'<div class="via-contact">\u260E\ufe0f {_esc(contact)}</div>' if contact else ""
    )

    zone_chip_html = (
        f'<div class="via-zone-chip">\U0001F4CD {_esc(zone_bits)}</div>' if zone_bits else ""
    )

    st.markdown(
        f"""
        <div class="via-card via-answer">
            <div class="via-answer-top">
                <div class="via-badge">[ {label.upper()} ]</div>
                {_confidence_bar(confidence, tone)}
            </div>
            <h2 class="via-answer-name">{_esc(record.get("name", ""))}</h2>
            {zone_chip_html}
            <div class="via-rows">{rows}</div>
            {contact_html}<div class="via-disclaimer">Information may change \u2014 please confirm with airport staff.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _no_answer_card(response, confidence: float) -> None:
    st.markdown(
        f"""
        <div class="via-card via-lowconf">
            <div class="via-badge via-badge-warn">[ NOT CONFIDENT ]</div>
            <p class="via-lowconf-msg">{_esc(response.message)}</p>
            {_confidence_bar(confidence, "low")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _alt_record_confidence(response) -> float:
    if not response.alternate_record:
        return 0.0
    alt_id = response.alternate_record.get("id")
    for m in response.modality_results:
        if m.record_id == alt_id:
            return m.confidence
    return 0.0


def _alternate_suggestion(response) -> None:
    if response.agreement is False and response.alternate_record:
        alt = response.alternate_record
        st.markdown(
            '<div class="via-alt-label">Inputs disagreed \u2014 did you mean this instead?</div>',
            unsafe_allow_html=True,
        )
        if st.button(f"\U0001F504  {alt['name']}", key="alt_record_btn", use_container_width=True):
            from backend.schemas import FusionResponse
            st.session_state.response = FusionResponse(
                answered=True,
                record=alt,
                confidence=_alt_record_confidence(response),
                modality_results=response.modality_results,
                agreement=None,
                message=f"Here's what I found: {alt['name']}.",
                alternate_record=None,
            )
            st.rerun()


UNUSED_MODALITY_MESSAGES = {
    "text": "your typed question didn't match anything in our records",
    "image": "your photo didn't match any known airport sign or location",
    "audio": "we couldn't clearly understand your voice input",
    "ocr": "we couldn't read clear text on your photo",
}


def _unused_modalities_note(response) -> None:
    unused = getattr(response, "unused_modalities", None)
    if not (response.answered and response.agreement is not False and unused):
        return
    parts = [UNUSED_MODALITY_MESSAGES.get(m, f"your {m} input") for m in unused]
    note = "; ".join(parts)
    st.caption(f"\u2139\ufe0f Note: {note}.")


def _decision_expander(response) -> None:

    results = getattr(response, "modality_results", None)
    if not results:
        return
    with st.expander("How I decided (per-modality breakdown)"):
        for m in results:
            line = (
                f"**{m.modality}** \u2014 "
                f"{'answered' if m.answered else 'abstained'} \u00b7 "
                f"confidence {m.confidence:.2f}"
            )
            if m.record_id:
                line += f" \u00b7 record `{m.record_id}`"
            st.markdown(line)
            transcript = (m.extra or {}).get("transcribed_text")
            if transcript:
                st.caption(f"Transcribed: \u201c{transcript}\u201d")
        if response.agreement is True:
            st.success("Multiple inputs agreed on the same record.")
        elif response.agreement is False:
            st.info("Inputs pointed to different records; the more confident one was used.")

def render_error(message: str, on_new_search) -> None:
    st.markdown(
        """
        <div class="via-card via-lowconf">
            <div class="via-badge via-badge-warn">[ SOMETHING WENT WRONG ]</div>
            <p class="via-lowconf-msg">We couldn't process that request right now.
            Please try again, or check with airport staff if the problem continues.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Technical details"):
        st.code(message)
    if st.button("\U0001F50D  New Search", use_container_width=True,
                 type="primary", key="error_new"):
        on_new_search()
        st.rerun()

def render_response(response, summary, on_new_search) -> None:
    _query_summary(summary)

    if response.answered and response.record:
        _success_card(response.record, response.confidence)
    else:
        _no_answer_card(response, response.confidence)

    _alternate_suggestion(response)
    _unused_modalities_note(response)
    _decision_expander(response)
    _speech_button(response)    

    left, mid, right = st.columns([1, 1, 1])
    with mid:
        if st.button("\U0001F50D  New Search", use_container_width=True,
                     type="primary", key="result_new"):
            on_new_search()
            st.rerun()

    if not (response.answered and response.record):
        with right:
            if st.button("\U0001F4CD  Find Info Desk", use_container_width=True,
                         key="result_infodesk"):
                st.session_state.pending_quick_query = "Where is the nearest information desk?"
                st.rerun()