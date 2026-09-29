"""TF-IDF classifiers, validation-only abstention, and a leakage check.

The keyword lexicon is not read or edited. Hyperparameters below are fixed
before the test articles are scored. The test split is not used to fit a
model or to choose a threshold.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["business", "entertainment", "politics", "sport", "tech"]
LABEL_WORD = re.compile(r"\b(?:business|entertainment|politics|sport|tech)\b", re.I)
THRESHOLDS = [round(float(value), 2) for value in np.arange(0.20, 0.96, 0.01)]
NEAR_DUP_JACCARD = 0.90


def normalise(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_nb():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
        )),
        ("clf", MultinomialNB()),
    ])


def make_svc():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
        )),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(random_state=42, dual="auto", max_iter=5000),
            method="sigmoid",
            cv=5,
        )),
    ])


def load_rows():
    with (ROOT / "data" / "bbc-text.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def texts_labels(rows, ids):
    texts = [rows[article_id]["text"] for article_id in ids]
    labels = [rows[article_id]["category"] for article_id in ids]
    return texts, labels


def weighted_f1_with_abstention(pairs):
    """Same rule as the keyword baseline: an abstention is a miss, not a class."""
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
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(score, 4),
        }
        total += score * support
        support_sum += support
    return round(total / support_sum, 4) if support_sum else 0.0, per_label


def forced_report(y_true, y_pred):
    per_label_scores = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    per_label = {}
    for label, score in zip(LABELS, per_label_scores):
        support = sum(1 for truth in y_true if truth == label)
        per_label[label] = {"support": support, "f1": round(float(score), 4)}
    weighted = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    accuracy = sum(truth == pred for truth, pred in zip(y_true, y_pred)) / len(y_true)
    return {
        "weighted_f1": round(float(weighted), 4),
        "accuracy": round(accuracy, 4),
        "per_label": per_label,
    }


def sweep_thresholds(y_true, proba):
    pred_idx = proba.argmax(axis=1)
    max_p = proba.max(axis=1)
    y_idx = np.array([LABELS.index(label) for label in y_true])
    rows = []
    for threshold in THRESHOLDS:
        abstain = max_p < threshold
        rate = float(abstain.mean())
        accepted = ~abstain
        if accepted.any():
            accuracy = float((pred_idx[accepted] == y_idx[accepted]).mean())
        else:
            accuracy = None
        rows.append({
            "threshold": threshold,
            "abstention_rate": rate,
            "accuracy_accepted": accuracy,
            "n_accepted": int(accepted.sum()),
            "n_abstained": int(abstain.sum()),
        })
    band = [
        row for row in rows
        if row["accuracy_accepted"] is not None and 0.10 <= row["abstention_rate"] <= 0.25
    ]
    if band:
        best = max(row["accuracy_accepted"] for row in band)
        tied = [row for row in band if row["accuracy_accepted"] == best]
        tied.sort(key=lambda row: (abs(row["abstention_rate"] - 0.15), row["threshold"]))
        chosen = tied[0]
        reason = (
            "highest accuracy on accepted validation predictions among thresholds "
            "with abstention rate from 10% to 25%"
        )
        if len(tied) > 1:
            reason += "; ties broken by closeness to 15% abstention, then the lower threshold"
        band_empty = False
    else:
        viable = [row for row in rows if row["accuracy_accepted"] is not None]
        viable.sort(key=lambda row: (abs(row["abstention_rate"] - 0.15), row["threshold"]))
        chosen = viable[0]
        reason = (
            "no threshold had a validation abstention rate between 10% and 25%; "
            "chose the threshold whose abstention rate is closest to 15%"
        )
        band_empty = True
    return rows, chosen, reason, band_empty


def apply_threshold(y_true, proba, threshold):
    pred_idx = proba.argmax(axis=1)
    max_p = proba.max(axis=1)
    forced = [LABELS[index] for index in pred_idx]
    accepted_pairs = []
    abstained_pairs = []
    scored_pairs = []
    for truth, prediction, probability in zip(y_true, forced, max_p):
        if float(probability) < threshold:
            scored_pairs.append((truth, None))
            abstained_pairs.append((truth, prediction))
        else:
            scored_pairs.append((truth, prediction))
            accepted_pairs.append((truth, prediction))
    n = len(y_true)
    abstained = len(abstained_pairs)
    accepted_accuracy = (
        sum(truth == pred for truth, pred in accepted_pairs) / len(accepted_pairs)
        if accepted_pairs else None
    )
    forced_abstain_accuracy = (
        sum(truth == pred for truth, pred in abstained_pairs) / len(abstained_pairs)
        if abstained_pairs else None
    )
    abstain_f1, _ = weighted_f1_with_abstention(scored_pairs)
    return {
        "threshold": threshold,
        "n": n,
        "abstained": abstained,
        "abstention_rate": round(abstained / n, 4),
        "accuracy_accepted": None if accepted_accuracy is None else round(accepted_accuracy, 4),
        "accuracy_abstained_if_forced": (
            None if forced_abstain_accuracy is None else round(forced_abstain_accuracy, 4)
        ),
        "weighted_f1_abstention_counts_as_miss": abstain_f1,
    }


def round_sweep(rows):
    rounded = []
    for row in rows:
        accuracy = row["accuracy_accepted"]
        rounded.append({
            "threshold": row["threshold"],
            "abstention_rate": round(row["abstention_rate"], 4),
            "accuracy_accepted": None if accuracy is None else round(accuracy, 4),
            "n_accepted": row["n_accepted"],
            "n_abstained": row["n_abstained"],
        })
    return rounded


def label_word_counts(rows, ids):
    own = 0
    any_label = 0
    by_word = Counter()
    for article_id in ids:
        text = rows[article_id]["text"]
        found = {match.group(0).lower() for match in LABEL_WORD.finditer(text)}
        by_word.update(found)
        if found:
            any_label += 1
        if rows[article_id]["category"] in found:
            own += 1
    return {
        "test_articles": len(ids),
        "articles_containing_any_label_word": any_label,
        "articles_containing_their_own_label_word": own,
        "label_word_occurrences": dict(by_word),
        "words_stripped": LABELS,
    }


def near_duplicates(rows, train_ids, test_ids):
    train_sets = [(article_id, set(normalise(rows[article_id]["text"]).split())) for article_id in train_ids]
    pairs = []
    max_score = 0.0
    max_pair = None
    for test_id in test_ids:
        test_words = set(normalise(rows[test_id]["text"]).split())
        if not test_words:
            continue
        for train_id, train_words in train_sets:
            if not train_words:
                continue
            intersection = len(test_words & train_words)
            score = intersection / (len(test_words) + len(train_words) - intersection)
            if score > max_score:
                max_score = score
                max_pair = {"test_id": test_id, "train_id": train_id, "jaccard": round(score, 4)}
            if score >= NEAR_DUP_JACCARD:
                pairs.append({
                    "test_id": test_id,
                    "train_id": train_id,
                    "jaccard": round(score, 4),
                })
    pairs.sort(key=lambda item: (-item["jaccard"], item["test_id"], item["train_id"]))
    return {
        "method": "Jaccard on normalised word sets",
        "near_duplicate_at": NEAR_DUP_JACCARD,
        "max_jaccard": None if max_pair is None else max_pair,
        "pairs": pairs,
    }


def fit_all(train_texts, train_labels):
    models = {"multinomial_nb": make_nb(), "calibrated_linear_svc": make_svc()}
    for model in models.values():
        model.fit(train_texts, train_labels)
    return models


def main():
    rows = load_rows()
    split = json.loads((ROOT / "eval" / "clf_split.json").read_text(encoding="utf-8"))
    train_ids = split["train"]
    val_ids = split["val"]
    test_ids = split["test"]
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise SystemExit("train, validation, and test ids overlap")
    if len(train_ids) + len(val_ids) + len(test_ids) != len(rows):
        raise SystemExit("split does not cover every article")

    train_texts, train_labels = texts_labels(rows, train_ids)
    val_texts, val_labels = texts_labels(rows, val_ids)
    test_texts, test_labels = texts_labels(rows, test_ids)

    majority = Counter(train_labels).most_common(1)[0][0]
    majority_pred = [majority] * len(test_labels)
    majority_report = forced_report(test_labels, majority_pred)

    models = fit_all(train_texts, train_labels)
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(models["multinomial_nb"], model_dir / "multinomial_nb.joblib")
    joblib.dump(models["calibrated_linear_svc"], model_dir / "calibrated_linear_svc.joblib")

    keyword = json.loads((ROOT / "results" / "keyword_baseline.json").read_text(encoding="utf-8"))
    classification = {
        "target_weighted_f1": 0.85,
        "target_note": "The 0.85 figure is the objective from the proposal. The measured test weighted F1 is the result.",
        "primary_metric": "weighted F1 of the forced argmax prediction on the 100-article test split",
        "hyperparameters_fixed_before_test": {
            "tfidf": "unigrams, english stop words, min_df 2, max_df 0.95",
            "multinomial_nb": "sklearn defaults",
            "linear_svc": "C 1, random_state 42, sigmoid calibration, 5-fold on the training split only",
        },
        "split": {
            "train": len(train_ids),
            "validation": len(val_ids),
            "test": len(test_ids),
            "threshold_selected_on": "validation",
            "test_used_for_fitting": False,
            "test_used_for_threshold": False,
        },
        "majority_baseline": {"predicted_class": majority, **majority_report},
        "keyword_baseline": {
            "weighted_f1_abstention_counts_as_miss": keyword["weighted_f1_abstention_counts_as_miss"],
            "source": "results/keyword_baseline.json",
        },
        "models": {},
    }
    abstention = {
        "rule": (
            "Abstain when the highest class probability is strictly below the threshold. "
            "The threshold is chosen on the validation split only. Among thresholds from "
            "0.20 to 0.95 in steps of 0.01 whose abstention rate is between 10% and 25%, "
            "pick the highest accuracy on accepted predictions. Ties break toward an "
            "abstention rate closer to 15%, then the lower threshold. If the band is empty, "
            "pick the threshold whose abstention rate is closest to 15% and say so."
        ),
        "three_numbers": [
            "abstention_rate",
            "accuracy_accepted",
            "accuracy_abstained_if_forced",
        ],
        "models": {},
    }
    mistakes = {}

    for name, model in models.items():
        val_proba = model.predict_proba(val_texts)
        test_proba = model.predict_proba(test_texts)
        # Column order follows model.classes_, which may not match LABELS.
        class_order = list(model.named_steps["clf"].classes_)
        val_aligned = np.column_stack([val_proba[:, class_order.index(label)] for label in LABELS])
        test_aligned = np.column_stack([test_proba[:, class_order.index(label)] for label in LABELS])
        sweep, chosen, reason, band_empty = sweep_thresholds(val_labels, val_aligned)
        test_forced = [LABELS[index] for index in test_aligned.argmax(axis=1)]
        report = forced_report(test_labels, test_forced)
        test_abstain = apply_threshold(test_labels, test_aligned, chosen["threshold"])
        val_abstain = apply_threshold(val_labels, val_aligned, chosen["threshold"])
        classification["models"][name] = {
            **report,
            "weighted_f1_abstention_counts_as_miss": test_abstain["weighted_f1_abstention_counts_as_miss"],
            "target_met": report["weighted_f1"] > 0.85,
            "above_keyword_baseline_on_forced_f1": (
                report["weighted_f1"] > keyword["weighted_f1_abstention_counts_as_miss"]
            ),
        }
        abstention["models"][name] = {
            "threshold_selected_on": "validation",
            "band_empty": band_empty,
            "selection_reason": reason,
            "chosen_threshold": chosen["threshold"],
            "validation": {
                "abstention_rate": val_abstain["abstention_rate"],
                "accuracy_accepted": val_abstain["accuracy_accepted"],
                "accuracy_abstained_if_forced": val_abstain["accuracy_abstained_if_forced"],
                "n": val_abstain["n"],
                "abstained": val_abstain["abstained"],
            },
            "test": {
                "abstention_rate": test_abstain["abstention_rate"],
                "accuracy_accepted": test_abstain["accuracy_accepted"],
                "accuracy_abstained_if_forced": test_abstain["accuracy_abstained_if_forced"],
                "n": test_abstain["n"],
                "abstained": test_abstain["abstained"],
                "weighted_f1_abstention_counts_as_miss": test_abstain["weighted_f1_abstention_counts_as_miss"],
            },
            "validation_sweep": round_sweep(sweep),
        }
        mistakes[name] = [
            {"article_id": article_id, "truth": truth, "prediction": prediction}
            for article_id, truth, prediction in zip(test_ids, test_labels, test_forced)
            if truth != prediction
        ]
    classification["test_errors_forced"] = mistakes

    stripped_train = [LABEL_WORD.sub(" ", text) for text in train_texts]
    stripped_test = [LABEL_WORD.sub(" ", text) for text in test_texts]
    stripped_models = fit_all(stripped_train, train_labels)
    leakage_models = {}
    for name, model in stripped_models.items():
        predicted = list(model.predict(stripped_test))
        report = forced_report(test_labels, predicted)
        leakage_models[name] = {
            "weighted_f1_before": classification["models"][name]["weighted_f1"],
            "weighted_f1_after_stripping_label_words": report["weighted_f1"],
            "per_label_after": report["per_label"],
        }
    duplicates = near_duplicates(rows, train_ids, test_ids)
    leaked_test_ids = sorted({pair["test_id"] for pair in duplicates["pairs"]})
    held_out_scores = {}
    if leaked_test_ids:
        keep = [index for index, article_id in enumerate(test_ids) if article_id not in set(leaked_test_ids)]
        for name, model in models.items():
            predicted = list(model.predict([test_texts[index] for index in keep]))
            truth = [test_labels[index] for index in keep]
            report = forced_report(truth, predicted)
            held_out_scores[name] = {
                "dropped_test_ids": leaked_test_ids,
                "n": len(keep),
                "weighted_f1": report["weighted_f1"],
            }
    leakage = {
        "label_words": label_word_counts(rows, test_ids),
        "models": leakage_models,
        "near_duplicates": duplicates,
        "score_without_near_duplicate_test_articles": held_out_scores or None,
        "note": (
            "Before is the forced test weighted F1 of the model trained on the original text. "
            "After retrains on training text with the five category names removed and scores "
            "test text with the same words removed. The published test score is the before number."
        ),
    }

    (ROOT / "results" / "classification.json").write_text(
        json.dumps(classification, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "results" / "abstention.json").write_text(
        json.dumps(abstention, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "results" / "leakage.json").write_text(
        json.dumps(leakage, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        name: {
            "weighted_f1": classification["models"][name]["weighted_f1"],
            "target_met": classification["models"][name]["target_met"],
            "abstention_rate": abstention["models"][name]["test"]["abstention_rate"],
            "accuracy_accepted": abstention["models"][name]["test"]["accuracy_accepted"],
            "accuracy_abstained_if_forced": abstention["models"][name]["test"]["accuracy_abstained_if_forced"],
            "threshold": abstention["models"][name]["chosen_threshold"],
            "band_empty": abstention["models"][name]["band_empty"],
            "f1_after_stripping_label_words": leakage_models[name]["weighted_f1_after_stripping_label_words"],
        }
        for name in models
    }
    print(json.dumps({
        "majority_class": majority,
        "majority_weighted_f1": majority_report["weighted_f1"],
        "keyword_weighted_f1": keyword["weighted_f1_abstention_counts_as_miss"],
        "models": summary,
        "near_duplicate_pairs": len(duplicates["pairs"]),
        "max_jaccard": duplicates["max_jaccard"],
        "label_word_own": leakage["label_words"]["articles_containing_their_own_label_word"],
    }, indent=2))


if __name__ == "__main__":
    main()
