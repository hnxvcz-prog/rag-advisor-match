# Quickstart

1. Place the 10 Word documents into `data/raw/`.
2. Set your `OPENAI_API_KEY` environment variable.
3. Install dependencies: `pip install -r requirements.txt`.
4. (Optional) Run the data ingestion script explicitly: `python -m src.core.document_parser`. (Note: The Streamlit app will automatically process raw files if `data/processed` is missing).
5. Start the frontend server: `streamlit run src/ui/app.py`.
