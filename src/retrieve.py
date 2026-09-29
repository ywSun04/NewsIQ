"""Local chunk retrieval for the frozen RAG questions. No generation model is called."""

import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 200
OVERLAP_WORDS = 40
TOP_K = 3
# Frozen before generation. 0.35 abstained on 0 of 10 adversarial questions.
# 0.50 abstains on 8 of 10 adversarial questions and 5 of 40 grounded ones.
ABSTAIN_COSINE = 0.50
ABSTAIN_RULE = (
    "Frozen before generation. Abstain when the best chunk cosine is below 0.50. "
    "Chosen because 0.35 stopped 0 of 10 adversarial questions, while 0.50 stops "
    "8 of 10 and abstains on 5 of 40 grounded questions. Not retuned after answers "
    "are generated."
)
CACHE = ROOT / "data" / "embeddings"


def chunk_text(text):
    words = text.split()
    if not words:
        return []
    step = CHUNK_WORDS - OVERLAP_WORDS
    pieces = []
    start = 0
    while start < len(words):
        piece = words[start : start + CHUNK_WORDS]
        pieces.append(" ".join(piece))
        if start + CHUNK_WORDS >= len(words):
            break
        start += step
    return pieces


def load_corpus():
    with (ROOT / "data" / "bbc-text.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    chunks = []
    owners = []
    for article_id, row in enumerate(rows):
        for text in chunk_text(row["text"]):
            chunks.append(text)
            owners.append(article_id)
    return chunks, np.asarray(owners, dtype=np.int32)


def embed(model, texts):
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def build_index(model):
    CACHE.mkdir(parents=True, exist_ok=True)
    vector_path = CACHE / "chunk_vectors.npy"
    owner_path = CACHE / "chunk_owners.npy"
    if vector_path.exists() and owner_path.exists():
        return np.load(vector_path), np.load(owner_path)
    chunks, owners = load_corpus()
    vectors = embed(model, chunks)
    np.save(vector_path, vectors)
    np.save(owner_path, owners)
    return vectors, owners


def top_hits(query_vector, vectors, owners):
    scores = vectors @ query_vector
    order = np.argpartition(-scores, TOP_K - 1)[:TOP_K]
    order = order[np.argsort(-scores[order])]
    hits = []
    for index in order:
        hits.append({
            "article_id": int(owners[index]),
            "cosine": round(float(scores[index]), 4),
        })
    return hits


def main():
    questions = json.loads((ROOT / "eval" / "rag_50.json").read_text(encoding="utf-8"))["questions"]
    model = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE / "model"))
    vectors, owners = build_index(model)
    query_vectors = embed(model, [item["question"] for item in questions])

    rows = []
    for item, query_vector in zip(questions, query_vectors):
        hits = top_hits(query_vector, vectors, owners)
        retrieved_ids = [hit["article_id"] for hit in hits]
        gold_ids = item["gold_article_ids"]
        hit = any(article_id in retrieved_ids for article_id in gold_ids)
        best = hits[0]["cosine"]
        rows.append({
            "id": item["id"],
            "type": item["type"],
            "gold_article_ids": gold_ids,
            "top3": hits,
            "hit_at_3": hit if item["type"] == "grounded" else None,
            "best_cosine": best,
            "would_abstain": best < ABSTAIN_COSINE,
        })

    grounded = [row for row in rows if row["type"] == "grounded"]
    adversarial = [row for row in rows if row["type"] == "adversarial"]
    hits = sum(1 for row in grounded if row["hit_at_3"])
    result = {
        "model": MODEL_NAME,
        "chunk_words": CHUNK_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "top_k": TOP_K,
        "abstain_cosine": ABSTAIN_COSINE,
        "abstain_rule": ABSTAIN_RULE,
        "n_chunks": int(len(owners)),
        "recall_at_3": round(hits / len(grounded), 4),
        "grounded_hits": hits,
        "grounded_n": len(grounded),
        "would_abstain_all": sum(1 for row in rows if row["would_abstain"]),
        "would_abstain_grounded": sum(1 for row in grounded if row["would_abstain"]),
        "would_abstain_adversarial": sum(1 for row in adversarial if row["would_abstain"]),
        "questions": rows,
    }
    destination = ROOT / "results" / "retrieval.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "n_chunks",
        "recall_at_3",
        "grounded_hits",
        "grounded_n",
        "would_abstain_all",
        "would_abstain_grounded",
        "would_abstain_adversarial",
    )}, indent=2))


if __name__ == "__main__":
    main()
