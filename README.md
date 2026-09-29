# NewsIQ

News classification, retrieval, and extraction on the BBC corpus.

Three tasks on one page:

1. Classify an article into business, entertainment, politics, sport, or tech.
2. Ask a question against the archive and receive an answer with citations.
3. Extract structured fields from one article.

## Data

The corpus is the public BBC News classification dataset released with TensorFlow (2,225 articles, 2004-2005). It contains no personal data. The full file is not stored in this repository. This repository is a proof of concept on that historical archive, not a product for current news.

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

Saved scores are in `results/`. Re-running `src/rag_answer.py` or `src/extract.py` calls the generator and spends money. `src/keyword_baseline.py`, `src/classify.py`, and `src/retrieve.py` do not.

## Submit

Record the screen with sound, following `writeup/demo_script.md`. The course materials do not state a video length.

Before pushing, `git status` must not list `.env` or `data/bbc-text.csv`.

```bash
git push origin HEAD
```

The public repository is https://github.com/ywSun04/NewsIQ.

On NTULearn, submit the analysis PDF at `writeup/NewsIQ_tradeoff_analysis.pdf`, the repository URL, and the video. The problem statement was already submitted. Follow the course page if it asks for an extra file.
