"""
Lightweight entity extraction for passenger text queries.

Covers the brief's "entity annotation" subtask (Preprocessing > Text
Pipeline): identifying entities such as gate number, terminal number,
flight number, service name/category, and flight type (international vs
domestic).

Implemented with regex rather than a trained NER model (e.g. spaCy),
because these are highly-structured, domain-specific tokens (airport gate
codes, terminal numbers) that a general-purpose NER model isn't trained to
recognise reliably anyway - it would need fine-tuning on airport data to do
this well, which contradicts the frozen-model philosophy applied everywhere
else in this project. Regex is precise, dependency-light (no extra model
download), and fully explainable - every match can be traced to an exact
pattern rather than an opaque model decision.

This does NOT feed into retrieval (which already reaches 92.5% hit@1 via
pure semantic embedding matching - see backend/text/evaluation output). It
is a complementary annotation layer: useful for vocabulary/intent analysis
in the data exploration notebook, and for surfacing extracted entities in
the UI response (e.g. highlighting "Gate B12" or "Terminal 2" explicitly).

Note: airline names are intentionally NOT extracted. The knowledge base
distinguishes check-in areas by flight type (International vs Domestic),
not by carrier, so an airline entity would have no KB record to resolve to.
Flight type is extracted instead, because it maps directly onto the KB's
"Terminal 1 International Check-in" / "Terminal 1 Domestic Check-in" records.
"""

from __future__ import annotations

import re

# Gate codes: a letter (A/B/C, matching the KB's actual gates B12/A05/C22)
# followed by 1-2 digits, with an optional space/hyphen ("B12", "B 12", "B-12").
GATE_PATTERN = re.compile(r"\b([A-Ca-c])\s?-?\s?(\d{1,2})\b")

# "Terminal 1" / "terminal 2" (the two terminals that exist in the KB).
TERMINAL_PATTERN = re.compile(r"\bterminal\s?(\d)\b", re.IGNORECASE)

# Airline flight codes like "TK1980": two letters + 2-4 digits.
FLIGHT_NUMBER_PATTERN = re.compile(r"\b([A-Z]{2})\s?(\d{2,4})\b")

# Flight type keywords -> maps onto the KB's International/Domestic check-in
# records. "arrivals/departures" are common near-synonyms passengers use.
FLIGHT_TYPE_KEYWORDS = {
    "international": ["international", "abroad", "overseas"],
    "domestic": ["domestic", "internal"],
}

# Maps each KB category to the everyday words a passenger might use for it-
# reuses the same category vocabulary as backend/kb/kb.py's `category` field,
# so a match here can be looked up directly via get_records_by_category().
SERVICE_KEYWORDS = {
    "gate": ["gate"],
    # NOTE: singular "bag" removed - it fired incorrectly on "drop my bags
    # before security" (a check-in / bag-drop action, not baggage claim).
    # "bags" (plural) is kept because it reliably means collecting luggage.
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
    """
    Extract flight type ('international' or 'domestic') from free text, or None.
    Maps directly onto the KB's International/Domestic check-in records.
    """
    text_lower = text.lower()
    for ftype, keywords in FLIGHT_TYPE_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                return ftype
    return None


def extract_service_categories(text: str) -> list[str]:
    """
    Return every KB category whose keyword list matches this text (possibly
    more than one). Uses word-boundary matching, not plain substring search:
    a naive `"bus" in text` would wrongly fire on "business" (as in "business
    lounge"), since "bus" is literally a substring of "business".
    """
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
    """
    Run all extractors on a single query and return a structured dict.
    Any field an extractor finds nothing for is None/[] - never guessed.
    """
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