# VIA — Vion International Airport Passenger Assistant

## Installation

```bash
git clone <repo-url>
cd Airport-Assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Running

```bash
streamlit run frontend/app.py
```

## Note on Colab / experimental work

`experimental/` contains GPU-only work (Qwen2.5-VL-3B-Instruct comparison against CLIP) run in Google Colab
on a T4 GPU. This is not part of the production Streamlit pipeline, which is fully CPU-deployable —
see the report's Model Design section for the accuracy-vs-deployability trade-off discussion.

