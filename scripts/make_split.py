"""Profile the BBC CSV, write a stratified split, and save one sample per category.

Article ids are 0-based row indexes in data/bbc-text.csv. No model is trained.
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "bbc-text.csv"
SOURCE = "https://storage.googleapis.com/download.tensorflow.org/data/bbc-text.csv"
SEED = 42
N_TEST = 100
N_VAL = 200


def load_rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return reader.fieldnames, rows


def stratified_take(groups, n_target, total):
    picked = {}
    for category in sorted(groups):
        picked[category] = round(len(groups[category]) * n_target / total)
    gap = n_target - sum(picked.values())
    order = sorted(groups, key=lambda category: len(groups[category]), reverse=True)
    index = 0
    while gap != 0 and order:
        category = order[index % len(order)]
        if gap > 0 and picked[category] < len(groups[category]):
            picked[category] += 1
            gap -= 1
        elif gap < 0 and picked[category] > 0:
            picked[category] -= 1
            gap += 1
        index += 1
        if index > len(order) * 5:
            break
    return picked


def main():
    fieldnames, rows = load_rows()
    counts = Counter(row["category"] for row in rows)
    profile = {
        "source": SOURCE,
        "path": "data/bbc-text.csv",
        "rows": len(rows),
        "columns": list(fieldnames),
        "counts": dict(sorted(counts.items())),
    }
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "data_profile.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )

    groups = {category: [] for category in sorted(counts)}
    for index, row in enumerate(rows):
        groups[row["category"]].append(index)

    rng = random.Random(SEED)
    for category in sorted(groups):
        rng.shuffle(groups[category])

    test_n = stratified_take(groups, N_TEST, len(rows))
    val_n = stratified_take(groups, N_VAL, len(rows))
    test, val, train = [], [], []
    allocation = {}
    for category in sorted(groups):
        ids = groups[category]
        n_test = test_n[category]
        n_val = val_n[category]
        test.extend(ids[:n_test])
        val.extend(ids[n_test : n_test + n_val])
        train.extend(ids[n_test + n_val :])
        allocation[category] = {
            "test": n_test,
            "val": n_val,
            "train": len(ids) - n_test - n_val,
        }

    split = {
        "seed": SEED,
        "id_scheme": "0-based row index in data/bbc-text.csv",
        "rule": (
            "Within each category, shuffle with random.Random(42), categories "
            "visited in alphabetical order. Take a stratified test of 100 and "
            "validation of 200. The validation set is only for the later "
            "abstention threshold. This file does not train a model."
        ),
        "counts": allocation,
        "test": sorted(test),
        "val": sorted(val),
        "train": sorted(train),
    }
    (ROOT / "eval" / "clf_split.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )

    sample_dir = ROOT / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    gitkeep = sample_dir / ".gitkeep"
    if gitkeep.exists():
        gitkeep.unlink()
    seen = set()
    for index, row in enumerate(rows):
        category = row["category"]
        if category in seen:
            continue
        seen.add(category)
        text = f"article_id: {index}\ncategory: {category}\n\n{row['text']}\n"
        (sample_dir / f"{category}.txt").write_text(text, encoding="utf-8")

    print(json.dumps(profile, indent=2))
    print("split", len(test), len(val), len(train))


if __name__ == "__main__":
    main()
