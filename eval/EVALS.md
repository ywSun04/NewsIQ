# Evaluations

Gold files in this directory were written before any model output. They were not edited to match a wrong prediction. Normalisation everywhere is the same: lowercase, delete every character that is not a letter, number, or space, and collapse whitespace.

## Split

`clf_split.json` fixes the classification split. Seed 42. Within each category, ids are shuffled with `random.Random(42)`, categories in alphabetical order. Test is 100 articles, validation is 200, train is the rest (1,925). Validation is only for the abstention threshold. The test ids are not used to fit a model, edit the keyword list, or choose a threshold.

## Keyword baseline

`keyword_lexicon.json` is a hand list per class. `src/keyword_baseline.py` scores the test split and writes `results/keyword_baseline.json`. The reported number is weighted F1 0.9125, with abstention counted as a miss. The list was not retuned after that score.

## Classification

`src/classify.py` trains on the train ids only. The headline number is forced weighted F1 in `results/classification.json`: Multinomial NB 0.9499, calibrated LinearSVC 0.9798. The objective was above 0.85. Both met it.

`results/abstention.json` holds the three abstention numbers. The LinearSVC threshold is 0.90 and the NB threshold is 0.75, each chosen on validation only. On the test set, LinearSVC abstains on 0.16 of articles, accepted accuracy is 0.9881, and accuracy on abstained articles if forced is 0.9375.

`results/leakage.json` repeats the forced F1 after the five category words are stripped, and again after ten near-duplicate test articles are dropped. The published score stays the full-test number.

## Archive questions

`rag_50.json` has 50 questions: 40 grounded and 10 adversarial. A grounded item is correct only when the system answers and the normalised answer equals `gold_answer`. An adversarial item is correct only when the system abstains.

`src/retrieve.py` writes Recall at 3 for the 40 grounded questions to `results/retrieval.json`. That score is 0.825. The retrieval cutoff is cosine 0.50, frozen before generation.

`src/rag_answer.py` writes `results/rag.json`. The official answer correctness is 0.74 (37 of 50). The objective was above 0.80. It was not met. Grounded items correct: 27 of 40. Adversarial items correct: 10 of 10.

## Extraction

`extraction_50.json` has 50 articles, 10 per class. The annotation rules are in that file. A list field scores 1 only when the whole normalised set matches. The topic must match one fixed phrase. If the JSON shape check fails, all five fields score 0.

`src/extract.py` writes `results/extraction.json`. The official field score is `field_correctness_exact` 0.484. The objective was above 0.75. It was not met. Schema pass rate is 1.0.

`entity_micro_f1` is 0.8013 on people, organisations, locations, and dates from the same outputs. It shows what exact set-match treats as a total miss. It is not a replacement score and it does not meet the 0.75 objective. Topic stays at exact match 0.06 and is outside that F1.
