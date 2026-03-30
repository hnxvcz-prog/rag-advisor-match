# Quickstart

1. Place the 10 Word documents into `data/raw/`.
2. Set your `OPENAI_API_KEY` environment variable.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the data ingestion script: `python -m src.core.document_parser` (Initializes JSON and embeddings).
5. Start the frontend server: `streamlit run src/ui/app.py`.
