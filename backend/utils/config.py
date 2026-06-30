"""
Central configuration for the Airport Assistant.

All file paths are defined here, relative to the project root, so that
the project runs unchanged on any machine (no hard-coded user paths).
"""

from pathlib import Path

# Project root = two levels up from this file:
#   backend/utils/config.py  ->  backend/utils  ->  backend  ->  <ROOT>
ROOT_DIR = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = ROOT_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"

# Knowledge base files
KB_PATH = KB_DIR / "airport_kb.json"
KB_SCHEMA_PATH = KB_DIR / "airport_kb_schema.json"

# Retrieval / confidence settings (used later by the text & fusion pipelines)
CONFIDENCE_THRESHOLD = 0.35   # below this similarity, the system abstains / hands over
TOP_K = 3                     # how many candidate records to return