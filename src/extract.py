"""Extract five fields from the frozen 50 articles. Rules are not changed after scoring."""

import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "openai/gpt-4o-mini"
INPUT_USD_PER_TOKEN = 0.15 / 1_000_000
OUTPUT_USD_PER_TOKEN = 0.60 / 1_000_000
MAX_TOKENS = 500
EVAL_SPEND_CEILING_USD = 1.0
SESSION_MAX_CALLS = 30
SESSION_MAX_USD = 0.05
CORPUS_ROWS = 2225
FIELDS = ["people", "organisations", "locations", "dates", "topic"]
SYSTEM = """Extract fields from one BBC news article. Copy names and dates as written.
people: named persons only, fullest form in the article. Omit bare titles. If a full name is present, do not also list mr or ms plus the surname.
organisations: named companies, parties, clubs, broadcasters, hospitals, agencies, and bands. One entry each. Omit product brands.
locations: named countries, cities, regions, and venues. Omit nationality adjectives such as british or saudi.
dates: absolute dates only, such as 2004, 23 march, or june 2004. Omit last year, tuesday, and last month. If a year appears only inside a longer date, keep the longer date and not the bare year.
topic: three to eight English words.
Return only JSON with keys people, organisations, locations, dates, and topic.
people, organisations, locations, and dates are arrays of strings. topic is a string. Use an empty array when a list has no items."""


def normalise(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_key():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            key = line.split("=", 1)[1].strip().lower()
            if key:
                return key
    raise SystemExit("OPENROUTER_API_KEY is missing from .env")


def load_articles():
    with (ROOT / "data" / "bbc-text.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def session_blocked(calls, usd):
    """Page session cap. The offline 50-case run uses a separate spend ceiling."""
    return calls >= SESSION_MAX_CALLS or usd >= SESSION_MAX_USD


def call_model(key, article):
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": article},
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


def l1_errors(data):
    if not isinstance(data, dict):
        return ["not an object"]
    errors = []
    for field in ("people", "organisations", "locations", "dates"):
        value = data.get(field)
        if not isinstance(value, list):
            errors.append(f"{field} is not a list")
        elif not all(isinstance(item, str) for item in value):
            errors.append(f"{field} contains a non-string")
    if not isinstance(data.get("topic"), str):
        errors.append("topic is not a string")
    missing = [field for field in FIELDS if field not in data]
    if missing:
        errors.append("missing " + ", ".join(missing))
    return errors


def as_set(values):
    return {normalise(item) for item in values if normalise(item)}


def score_fields(gold, predicted):
    scores = {}
    for field in ("people", "organisations", "locations", "dates"):
        scores[field] = int(as_set(predicted.get(field, [])) == as_set(gold[field]))
    scores["topic"] = int(normalise(predicted.get("topic", "")) == normalise(gold["topic"]))
    return scores


def summarise(rows):
    scored = [row for row in rows if "field_scores" in row]
    calls = [row for row in rows if row.get("called_model")]
    spent = sum(row.get("usd", 0) for row in rows)
    mean_case = (
        sum(row["case_score"] for row in scored) / len(scored) if scored else 0
    )
    per_field = {}
    for field in FIELDS:
        per_field[field] = round(
            sum(row["field_scores"][field] for row in scored) / len(scored), 4
        ) if scored else 0
    mean_usd = spent / len(calls) if calls else 0
    return {
        "generator": MODEL,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "price_card_usd_per_million": {"input": 0.15, "output": 0.60},
        "session_max_calls": SESSION_MAX_CALLS,
        "session_max_usd": SESSION_MAX_USD,
        "n": len(rows),
        "l1_pass_rate": round(sum(row["l1_pass"] for row in rows) / len(rows), 4) if rows else 0,
        "field_correctness": round(mean_case, 4),
        "per_field": per_field,
        "calls": len(calls),
        "prompt_tokens": sum(row.get("prompt_tokens", 0) for row in rows),
        "completion_tokens": sum(row.get("completion_tokens", 0) for row in rows),
        "usd": round(spent, 6),
        "mean_usd_per_extraction": round(mean_usd, 6),
        "full_corpus_usd": round(mean_usd * CORPUS_ROWS, 4),
        "cases": rows,
    }


def main():
    gold = json.loads((ROOT / "eval" / "extraction_50.json").read_text(encoding="utf-8"))
    articles = load_articles()
    destination = ROOT / "results" / "extraction.json"
    done = {}
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        done = {row["id"]: row for row in previous.get("cases", [])}
    key = load_key()
    cost_path = ROOT / "results" / "cost_log.jsonl"
    spent = sum(row.get("usd", 0) for row in done.values())
    rows = []

    for case in gold["cases"]:
        if case["id"] in done:
            rows.append(done[case["id"]])
            continue
        if spent >= EVAL_SPEND_CEILING_USD:
            raise SystemExit(f"Spend ceiling reached at ${spent:.4f}")
        article = articles[case["article_id"]]["text"]
        raw, prompt_tokens, completion_tokens = call_model(key, article)
        usd = prompt_tokens * INPUT_USD_PER_TOKEN + completion_tokens * OUTPUT_USD_PER_TOKEN
        spent += usd
        try:
            predicted = json.loads(raw)
        except json.JSONDecodeError:
            predicted = None
        errors = ["invalid JSON"] if predicted is None else l1_errors(predicted)
        passed = not errors
        scores = (
            score_fields(case["gold"], predicted)
            if passed
            else {field: 0 for field in FIELDS}
        )
        record = {
            "id": case["id"],
            "article_id": case["article_id"],
            "category": case["category"],
            "called_model": True,
            "l1_pass": passed,
            "l1_errors": errors,
            "field_scores": scores,
            "case_score": sum(scores.values()) / len(FIELDS),
            "predicted": predicted,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "usd": round(usd, 6),
        }
        rows.append(record)
        with cost_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "task": "extraction",
                "id": case["id"],
                "model": MODEL,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "usd": round(usd, 6),
            }) + "\n")
        destination.write_text(json.dumps(summarise(rows), indent=2) + "\n", encoding="utf-8")
        print(
            f"{record['id']} l1={passed} score={record['case_score']:.2f} spent={spent:.4f}",
            flush=True,
        )
        time.sleep(0.2)

    final = summarise(rows)
    destination.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: final[key] for key in (
        "l1_pass_rate",
        "field_correctness",
        "per_field",
        "mean_usd_per_extraction",
        "full_corpus_usd",
        "usd",
    )}, indent=2))


if __name__ == "__main__":
    main()
