# Airport Assistant

A multimodal (image + voice + text) AI assistant for airport passenger assistance.
MSc Multi-Modal Chatbots coursework.

## Architecture

[Architecture diagram to be added]

## Installation

\`\`\`bash
git clone <repo-url>
cd Airport-Assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
\`\`\`

## Running

\`\`\`bash
streamlit run app/streamlit_app.py
\`\`\`

## Project Structure

[Folder tree to be added]

## Components

- **Image:** CLIP (frozen) + FAISS retrieval
- **Audio:** Whisper (frozen) → text
- **Text:** SentenceTransformer + intent/entity extraction
- **Fusion:** rule-based routing + confidence threshold
- **Knowledge base:** JSON, 15-20 airport records

## License

MIT