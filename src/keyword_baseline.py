"""Keyword category baseline. No learned model is trained or imported."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["business", "entertainment", "politics", "sport", "tech"]


def normalise(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def count_hits(text, term):
    return len(re.findall(rf"\b{re.escape(normalise(term))}\b", text))


def predict(text, lexicon):
    scores = {
        category: sum(count_hits(text, term) for term in terms)
        for category, terms in lexicon["categories"].items()
    }
    best = max(scores.values())
    winners = [category for category, score in scores.items() if score == best]
    if best == 0 or len(winners) != 1:
        return None, scores
    return winners[0], scores


def weighted_f1(pairs):
    """Abstention counts as a miss for the true class and as no prediction."""
    total = 0.0
    support_sum = 0
    per_label = {}
    for label in LABELS:
        support = sum(1 for truth, _ in pairs if truth == label)
        tp = sum(1 for truth, pred in pairs if truth == label and pred == label)
        fp = sum(1 for truth, pred in pairs if pred == label and truth != label)
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        total += f1 * support
        support_sum += support
    return round(total / support_sum, 4), per_label


def main():
    lexicon = json.loads((ROOT / "eval" / "keyword_lexicon.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "eval" / "clf_split.json").read_text(encoding="utf-8"))
    with (ROOT / "data" / "bbc-text.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    pairs = []
    mistakes = []
    for article_id in split["test"]:
        truth = rows[article_id]["category"]
        prediction, scores = predict(normalise(rows[article_id]["text"]), lexicon)
        pairs.append((truth, prediction))
        if prediction != truth:
            mistakes.append({
                "article_id": article_id,
                "truth": truth,
                "prediction": prediction,
                "scores": scores,
            })

    abstained = sum(1 for _, prediction in pairs if prediction is None)
    f1, per_label = weighted_f1(pairs)
    accepted = [(truth, prediction) for truth, prediction in pairs if prediction is not None]
    accepted_f1, _ = weighted_f1(accepted) if accepted else (None, {})
    result = {
        "method": "hand keyword counts",
        "split": "test",
        "n": len(pairs),
        "weighted_f1_abstention_counts_as_miss": f1,
        "weighted_f1_on_accepted_only": accepted_f1,
        "abstention_rate": round(abstained / len(pairs), 4),
        "abstained": abstained,
        "per_label": per_label,
        "errors": mistakes,
    }
    destination = ROOT / "results" / "keyword_baseline.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "n",
        "weighted_f1_abstention_counts_as_miss",
        "weighted_f1_on_accepted_only",
        "abstention_rate",
        "abstained",
        "per_label",
    )}, indent=2))


if __name__ == "__main__":
    main()
