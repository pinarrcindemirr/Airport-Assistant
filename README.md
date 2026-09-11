# VIA — Vion International Airport Passenger Assistant

A multimodal (image + voice + text) AI assistant for airport passenger assistance.
MSc Multi-Modal Chatbots coursework — BSBI / University for the Creative Arts.

## Architecture

[Architecture diagram to be added — see report for the full pipeline diagram: input modalities → CLIP/Whisper/NLP encoders → fusion/routing → FAISS retrieval → confidence gate → Streamlit response panel]

## Installation

```bash
git clone <repo-url>
cd Airport-Assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Python 3.10 is recommended (matches the development environment).

## Running

```bash
streamlit run frontend/app.py
```

The app opens automatically in your browser (usually http://localhost:8501).
On first use, models are loaded into memory in parallel (~18 seconds) — an
"Initializing VIA systems…" box will show during this warm-up. This only
happens once per server process.

Try any of the sidebar's sample queries to get started, or use the composer
to enter text, upload/record audio, and upload an image — individually or
combined.

## Project Structure

```
Airport-Assistant/
├── backend/
│   ├── assistant.py           # lazy singleton entry point, warm_up()
│   ├── schemas.py              # ModalityResult, FusionResponse
│   ├── fusion/
│   │   ├── router.py           # rule-based fusion/routing, disagreement handling
│   │   └── confidence.py       # per-modality confidence ranges & thresholds
│   ├── text/
│   │   ├── intent_classifier.py
│   │   ├── entities.py
│   │   └── retrieval.py        # MiniLM + FAISS (IndexFlatIP)
│   ├── image/
│   │   ├── retrieval.py        # CLIP ViT-B/32 retrieval
│   │   └── ocr_reader.py       # EasyOCR sign reading
│   └── audio/
│       └── whisper_model.py    # Whisper base transcription
├── frontend/
│   ├── app.py                  # Streamlit entry point
│   ├── components/
│   │   ├── composer.py
│   │   ├── response.py
│   │   └── sidebar.py
│   └── styles/
│       └── custom.css
├── data/
│   ├── knowledge_base/
│   │   └── airport_kb.json     # 22 records, 16 categories
│   ├── images/                 # 88 mockup images across 16 categories
│   ├── audio/                  # sample passenger voice queries
│   └── text/
│       └── airport_queries.csv
├── evaluation/                 # evaluation outputs referenced in the report
│   ├── clip_vs_qwen_comparison.csv
│   ├── clip_vs_qwen_per_category.csv
│   ├── clip_vs_qwen_summary.csv
│   ├── system_metrics_summary.csv
│   ├── whisper_base_vs_small_by_condition.csv
│   ├── whisper_base_vs_small_per_file.csv
│   └── outputs_*.png            # charts used in the report
├── tests/
│   └── test_fusion.py
├── experimental/                # Colab-only GPU experiments (Qwen2.5-VL), not in the production pipeline
├── scripts/
├── requirements.txt
└── README.md
```

## Components

- **Image:** CLIP ViT-B/32 (frozen) + FAISS retrieval over 88 mockup images, plus EasyOCR for direct sign-text reading
- **Audio:** Whisper base (frozen) → text
- **Text:** all-MiniLM-L6-v2 sentence-transformer + FAISS (IndexFlatIP) retrieval, NearestCentroid intent classifier, rule-based entity extraction
- **Fusion:** rule-based routing across text/image/audio/OCR modalities, with a 0.5 confidence threshold, disagreement penalty, and "Did you mean X?" alternate suggestions when modalities conflict
- **Knowledge base:** JSON, 22 airport records across 16 categories (`data/knowledge_base/airport_kb.json`)
- **Accessibility:** gTTS text-to-speech for responses

All models are used frozen, with no fine-tuning, consistent with the assignment's proof-of-concept scope.

## Note on Colab / experimental work

`experimental/` contains GPU-only work (Qwen2.5-VL-3B-Instruct comparison against CLIP) run in Google Colab
on a T4 GPU. This is not part of the production Streamlit pipeline, which is fully CPU-deployable —
see the report's Model Design section for the accuracy-vs-deployability trade-off discussion.

## License

MIT
