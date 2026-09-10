from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = ROOT_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"

# Knowledge base files
KB_PATH = KB_DIR / "airport_kb.json"
KB_SCHEMA_PATH = KB_DIR / "airport_kb_schema.json"

# Retrieval / confidence settings (used later by the text & fusion pipelines)
CONFIDENCE_THRESHOLD = 0.35   
TOP_K = 3                    