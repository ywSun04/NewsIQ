# NewsIQ

News classification, retrieval, and extraction on the BBC corpus.

Three tasks on one page:

1. Classify an article into business, entertainment, politics, sport, or tech.
2. Ask a question against the archive and receive an answer with citations.
3. Extract structured fields from one article.

## Data

The corpus is the public BBC News classification dataset (2,225 articles, 2004–2005). The full file is not stored in this repository.

```bash
python scripts/download_bbc.py
```

This writes `data/bbc-text.csv`. That file is gitignored.

An OpenRouter API key is not required for this download.

## Page

Train the classifiers once, then start the page:

```bash
python src/classify.py
streamlit run app.py
```

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` before using archive answers or extraction. Classification does not call the generator. Model files in `models/` and the embedding cache in `data/embeddings/` stay on the machine that built them.
