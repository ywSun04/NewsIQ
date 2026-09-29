"""Answer the frozen RAG questions from retrieved chunks. Threshold stays at 0.50."""

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieve

ROOT = retrieve.ROOT
MODEL = "openai/gpt-4o-mini"
INPUT_USD_PER_TOKEN = 0.15 / 1_000_000
OUTPUT_USD_PER_TOKEN = 0.60 / 1_000_000
MAX_TOKENS = 400
SPEND_CEILING_USD = 1.0
INSUFFICIENT = "The corpus does not contain sufficient evidence."
SYSTEM = """You answer questions about a 2004-2005 BBC news archive.
Use only the evidence passages. Do not use outside knowledge.
If the passages do not state the answer, abstain.
When you answer, copy the shortest phrase that answers the question. Do not write a sentence.
Return only JSON with keys abstain (boolean), answer (string), and citation_indexes (array of passage numbers).
If you abstain, answer is empty and citation_indexes is empty."""


def normalise(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_key():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            key = line.split("=", 1)[1].strip()
            if key:
                return key
    raise SystemExit("OPENROUTER_API_KEY is missing from .env")


def load_categories():
    import csv
    with (ROOT / "data" / "bbc-text.csv").open(newline="", encoding="utf-8") as handle:
        return [row["category"] for row in csv.DictReader(handle)]


def evidence_block(hits, chunks, categories):
    lines = []
    for number, hit in enumerate(hits, start=1):
        text = chunks[hit["chunk_index"]]
        lines.append(
            f"Passage {number} (article {hit['article_id']}, {categories[hit['article_id']]}):\n{text}"
        )
    return "\n\n".join(lines)


def call_model(key, question, evidence):
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\n{evidence}"},
        ],
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            last_error = f"HTTP {error.code}: {detail}"
            if error.code not in (429, 500, 502, 503):
                raise SystemExit(last_error) from error
            time.sleep(2 * (attempt + 1))
        except urllib.error.URLError as error:
            last_error = str(error.reason)
            time.sleep(2 * (attempt + 1))
    else:
        raise SystemExit(last_error)
    usage = payload.get("usage") or {}
    raw = payload["choices"][0]["message"].get("content") or ""
    return raw, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def parse_payload(raw):
    data = json.loads(raw)
    abstain = bool(data.get("abstain"))
    answer = "" if abstain else str(data.get("answer") or "").strip()
    citations = data.get("citation_indexes") or []
    return abstain, answer, citations


def is_correct(item, abstain, answer):
    if item["must_abstain"]:
        return abstain
    return (not abstain) and normalise(answer) == normalise(item["gold_answer"])


def citation_records(hits, chunks, categories, indexes):
    chosen = []
    for number in indexes:
        if not isinstance(number, int) or not 1 <= number <= len(hits):
            continue
        hit = hits[number - 1]
        text = chunks[hit["chunk_index"]]
        chosen.append({
            "passage": number,
            "article_id": hit["article_id"],
            "category": categories[hit["article_id"]],
            "cosine": hit["cosine"],
            "snippet": text[:240],
        })
    return chosen


def summarise(rows, spent):
    grounded = [row for row in rows if row["type"] == "grounded"]
    adversarial = [row for row in rows if row["type"] == "adversarial"]
    called = [row for row in rows if row["called_model"]]
    return {
        "generator": MODEL,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "abstain_cosine": retrieve.ABSTAIN_COSINE,
        "price_card_usd_per_million": {"input": 0.15, "output": 0.60},
        "n": len(rows),
        "answer_correctness": round(sum(row["correct"] for row in rows) / len(rows), 4),
        "grounded_correct": sum(row["correct"] for row in grounded),
        "grounded_n": len(grounded),
        "adversarial_correct": sum(row["correct"] for row in adversarial),
        "adversarial_n": len(adversarial),
        "retrieval_abstained": sum(row["abstain_reason"] == "retrieval" for row in rows),
        "model_abstained": sum(row["abstain_reason"] == "model" for row in rows),
        "calls": len(called),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "usd": round(spent, 6),
        "mean_usd_per_call": round(spent / len(called), 6) if called else 0,
        "questions": rows,
    }


def main():
    questions = json.loads((ROOT / "eval" / "rag_50.json").read_text(encoding="utf-8"))["questions"]
    destination = ROOT / "results" / "rag.json"
    done = {}
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        done = {row["id"]: row for row in previous.get("questions", [])}

    key = load_key()
    categories = load_categories()
    model = retrieve.SentenceTransformer(retrieve.MODEL_NAME, cache_folder=str(retrieve.CACHE / "model"))
    chunks, owners = retrieve.load_corpus()
    vectors, cached_owners = retrieve.build_index(model)
    if len(chunks) != len(cached_owners) or not (owners == cached_owners).all():
        raise SystemExit("Embedding cache does not match the current chunking.")
    query_vectors = retrieve.embed(model, [item["question"] for item in questions])
    cost_path = ROOT / "results" / "cost_log.jsonl"
    spent = sum(row.get("usd", 0) for row in done.values())
    rows = []

    for item, query_vector in zip(questions, query_vectors):
        if item["id"] in done:
            rows.append(done[item["id"]])
            continue
        hits = retrieve.top_hits(query_vector, vectors, owners)
        best = hits[0]["cosine"]
        record = {
            "id": item["id"],
            "type": item["type"],
            "gold_answer": item["gold_answer"],
            "gold_article_ids": item["gold_article_ids"],
            "best_cosine": best,
            "top3": [
                {"article_id": hit["article_id"], "cosine": hit["cosine"]}
                for hit in hits
            ],
        }
        if best < retrieve.ABSTAIN_COSINE:
            record.update({
                "called_model": False,
                "abstain": True,
                "abstain_reason": "retrieval",
                "answer": INSUFFICIENT,
                "citations": [],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "usd": 0,
            })
        else:
            if spent >= SPEND_CEILING_USD:
                raise SystemExit(f"Spend ceiling reached at ${spent:.4f}")
            evidence = evidence_block(hits, chunks, categories)
            raw, prompt_tokens, completion_tokens = call_model(key, item["question"], evidence)
            try:
                abstain, answer, indexes = parse_payload(raw)
            except json.JSONDecodeError:
                raw, prompt_tokens_2, completion_tokens_2 = call_model(key, item["question"], evidence)
                prompt_tokens += prompt_tokens_2
                completion_tokens += completion_tokens_2
                abstain, answer, indexes = parse_payload(raw)
            usd = prompt_tokens * INPUT_USD_PER_TOKEN + completion_tokens * OUTPUT_USD_PER_TOKEN
            spent += usd
            record.update({
                "called_model": True,
                "abstain": abstain,
                "abstain_reason": "model" if abstain else None,
                "answer": INSUFFICIENT if abstain else answer,
                "citations": [] if abstain else citation_records(hits, chunks, categories, indexes),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "usd": round(usd, 6),
                "raw": raw,
            })
            with cost_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "task": "rag",
                    "id": item["id"],
                    "model": MODEL,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "usd": round(usd, 6),
                }) + "\n")
        record["correct"] = is_correct(item, record["abstain"], record["answer"])
        rows.append(record)
        destination.write_text(
            json.dumps(summarise(rows, sum(row["usd"] for row in rows)), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{record['id']} correct={record['correct']} abstain={record['abstain_reason']} spent={spent:.4f}", flush=True)
        time.sleep(0.2)

    final = summarise(rows, sum(row["usd"] for row in rows))
    destination.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: final[key] for key in (
        "answer_correctness",
        "grounded_correct",
        "grounded_n",
        "adversarial_correct",
        "adversarial_n",
        "retrieval_abstained",
        "model_abstained",
        "calls",
        "usd",
    )}, indent=2))


if __name__ == "__main__":
    main()
