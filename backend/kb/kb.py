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

    return record["visual_description"]
