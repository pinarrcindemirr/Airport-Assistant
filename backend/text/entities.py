from __future__ import annotations

import re

GATE_PATTERN = re.compile(r"\b([A-Ca-c])\s?-?\s?(\d{1,2})\b")

TERMINAL_PATTERN = re.compile(r"\bterminal\s?(\d)\b", re.IGNORECASE)


FLIGHT_NUMBER_PATTERN = re.compile(r"\b([A-Z]{2})\s?(\d{2,4})\b")


FLIGHT_TYPE_KEYWORDS = {
    "international": ["international", "abroad", "overseas"],
    "domestic": ["domestic", "internal"],
}

SERVICE_KEYWORDS = {
    "gate": ["gate"],
    "baggage_claim": ["baggage", "luggage", "suitcase", "bags"],
    "check_in": ["check in", "check-in", "checkin", "bag drop", "counter"],
    "security": ["security"],
    "lounge": ["lounge"],
    "restaurant": ["restaurant", "food", "eat", "coffee", "cafe", "bite"],
    "restroom": ["restroom", "toilet", "bathroom"],
    "prayer_room": ["prayer", "pray"],
    "lost_and_found": ["lost", "found", "missing"],
    "customs": ["customs", "passport control"],
    "currency_exchange": ["currency", "cash", "money", "exchange", "atm", "withdraw"],
    "special_assistance": ["wheelchair", "assistance", "disability", "mobility"],
    "transport": ["taxi", "train", "bus", "transport", "metro", "city centre", "city center", "downtown"],
    "information_desk": ["information desk", "info desk", "help desk", "information"],
    "pharmacy": ["pharmacy", "chemist", "medicine", "medication", "painkillers"],
    "smoking_area": ["smoking", "smoke", "cigarette", "vape"],
}


def extract_gate(text: str) -> str | None:
    """Extract a gate code like 'B12' or 'A05' from free text, or None if absent."""
    m = GATE_PATTERN.search(text)
    return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else None


def extract_terminal(text: str) -> int | None:
    """Extract a terminal number (1 or 2) from free text, or None if absent."""
    m = TERMINAL_PATTERN.search(text)
    return int(m.group(1)) if m else None


def extract_flight_number(text: str) -> str | None:
    """Extract an airline flight code like 'TK1980' from free text, or None if absent."""
    m = FLIGHT_NUMBER_PATTERN.search(text)
    return f"{m.group(1)}{m.group(2)}" if m else None


def extract_flight_type(text: str) -> str | None:

    text_lower = text.lower()
    for ftype, keywords in FLIGHT_TYPE_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                return ftype
    return None


def extract_service_categories(text: str) -> list[str]:

    text_lower = text.lower()
    matches = []
    for cat, keywords in SERVICE_KEYWORDS.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"s?\b"
            if re.search(pattern, text_lower):
                matches.append(cat)
                break
    return matches


def extract_entities(text: str) -> dict:

    return {
        "gate": extract_gate(text),
        "terminal": extract_terminal(text),
        "flight_number": extract_flight_number(text),
        "flight_type": extract_flight_type(text),
        "service_categories": extract_service_categories(text),
    }


if __name__ == "__main__":
    # Demo / sanity check against real queries from data/text/airport_queries.csv
    import csv
    from pathlib import Path

    manifest = Path(__file__).resolve().parents[2] / "data" / "text" / "airport_queries.csv"
    if manifest.exists():
        rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    else:
        rows = [{"query": q} for q in [
            "where is gate B12", "is there a lounge near terminal 2",
            "gate for TK1980", "where can I get cash",
            "how do I get to baggage claim", "coffee near gate A",
        ]]

    for row in rows[:20]:
        q = row["query"]
        print(f"{q!r:55} -> {extract_entities(q)}")