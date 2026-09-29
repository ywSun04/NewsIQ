"""Run one classification, one archive question, and one extraction through the page functions."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import serve

GROUNDED = "Which country awarded Burren Energy two new oil exploration contracts?"


def trim(result):
    passages = []
    for item in result.get("passages") or []:
        passages.append({
            "passage": item["passage"],
            "article_id": item["article_id"],
            "category": item["category"],
            "cosine": item["cosine"],
            "text": item["text"][:240],
        })
    kept = {key: value for key, value in result.items() if key != "passages"}
    kept["passages"] = passages
    return kept


def main():
    article = serve.sample_article()
    classification = serve.classify_article(article, "calibrated_linear_svc")
    calls = 0
    usd = 0.0
    rag = serve.answer_question(GROUNDED, calls, usd)
    if rag.get("called_model"):
        calls += 1
        usd += rag.get("usd") or 0
    extraction = serve.extract_article(article, calls, usd)
    if extraction.get("called_model"):
        calls += 1
        usd += extraction.get("usd") or 0
    payload = {
        "banner": "BBC News 2004-2005",
        "classification": classification,
        "rag": trim(rag),
        "extraction": extraction,
        "session_calls": calls,
        "session_usd": round(usd, 6),
    }
    destination = ROOT / "results" / "ui_smoke.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "label": classification.get("label"),
        "abstain": classification.get("abstain"),
        "probability": classification.get("top_probability"),
        "rag_answer": rag.get("answer"),
        "rag_abstain": rag.get("abstain"),
        "rag_usd": rag.get("usd"),
        "extract_l1": extraction.get("l1_pass"),
        "extract_topic": (extraction.get("fields") or {}).get("topic"),
        "extract_usd": extraction.get("usd"),
        "session_usd": round(usd, 6),
    }, indent=2))


if __name__ == "__main__":
    main()