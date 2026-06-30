"""
Knowledge base access layer.

This module is the single gateway between the airport knowledge base
(JSON file) and the rest of the application. No other module should read
the JSON directly; everything goes through here. If the storage format
ever changes (e.g. to SQLite), only this file needs updating.

Responsibilities:
  1. Load the KB JSON (cached, loaded once).
  2. Provide simple access to records (all / by id / by category / related).
  3. Build the text used for embedding, at runtime, so the source data
     stays the single source of truth:
       - text pipeline  -> name + description + directions + location
                           (+ keywords, optional - see note below)
       - image pipeline -> visual_description only (for CLIP)
"""

import json
from functools import lru_cache
from typing import Optional

from backend.utils.config import KB_PATH


@lru_cache(maxsize=1)
def load_kb() -> dict:
    """Load and cache the full knowledge base JSON (loaded only once)."""
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_records() -> list[dict]:
    """Return the list of all airport records."""
    return load_kb()["records"]


def get_record_by_id(record_id: str) -> Optional[dict]:
    """Return a single record by its id, or None if not found."""
    for record in get_all_records():
        if record["id"] == record_id:
            return record
    return None


def get_records_by_category(category: str) -> list[dict]:
    """
    Return all records in a given category.

    Used later for intent-to-category matching (e.g. a 'baggage' intent maps
    to the baggage_claim category) and for class-distribution analysis in the
    data exploration notebook.
    """
    return [r for r in get_all_records() if r["category"] == category]


def get_related(record_id: str) -> list[dict]:
    """Return the records listed in a record's related_facilities."""
    record = get_record_by_id(record_id)
    if record is None:
        return []
    related = []
    for rel_id in record.get("related_facilities", []):
        rel = get_record_by_id(rel_id)
        if rel is not None:
            related.append(rel)
    return related


def build_text_for_embedding(record: dict, include_keywords: bool = False) -> str:
    """
    Build the combined text used by the TEXT pipeline (SentenceTransformer).

    Combines the fields a passenger query is likely to match against:
    name, description, directions, and location. Generated at runtime so it
    always reflects the current record (no duplicated/stale field in the JSON).

    Args:
        include_keywords: If True, append the record's keywords as an extra
            line. This is left OFF by default and exposed as a flag so the two
            variants can be compared in an ablation study (see notebook 02).
            SentenceTransformer is trained on natural sentences, so appending a
            comma-separated keyword list may help OR hurt retrieval - it must be
            measured, not assumed.
    """
    text = (
        f"{record['name']}. "
        f"{record['description']} "
        f"{record['directions']} "
        f"Located in {record['terminal']}, {record['floor_or_zone']}."
    )
    if include_keywords:
        text += f" Keywords: {', '.join(record.get('keywords', []))}."
    return text


def build_visual_text(record: dict) -> str:
    """
    Build the text used by the IMAGE pipeline (CLIP).

    Uses only visual_description, so CLIP matches images against how a place
    LOOKS, not against opening hours or directions (which would hurt matching).
    """
    return record["visual_description"]


# Quick self-check when run directly: python -m backend.kb.kb
if __name__ == "__main__":
    kb = load_kb()
    records = get_all_records()
    print(f"Airport: {kb['airport_name']} ({kb['airport_code']})")
    print(f"Records loaded: {len(records)}")

    sample = get_record_by_id("gate_b12")
    print("\nSample record:", sample["name"])
    print("Text-embedding (no keywords):")
    print("  ", build_text_for_embedding(sample))
    print("Text-embedding (with keywords):")
    print("  ", build_text_for_embedding(sample, include_keywords=True))
    print("Visual-embedding string:")
    print("  ", build_visual_text(sample))

    print("\nCategory check - all 'gate' records:")
    for r in get_records_by_category("gate"):
        print("  -", r["name"])

    print("\nRelated facilities of gate_b12:")
    for r in get_related("gate_b12"):
        print("  -", r["name"])