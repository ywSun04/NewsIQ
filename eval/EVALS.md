# Evaluations

Gold files in this directory were written before any model output. They were not edited to match a wrong prediction. Normalisation everywhere is the same: lowercase, delete every character that is not a letter, number, or space, and collapse whitespace.

## Split

`clf_split.json` fixes the classification split. Seed 42. Within each category, ids are shuffled with `random.Random(42)`, categories in alphabetical order. Test is 100 articles, validation is 200, train is the rest (1,925). Validation is only for the abstention threshold. The test ids are not used to fit a model, edit the keyword list, or choose a threshold.

## Keyword baseline

`keyword_lexicon.json` is a hand list per class. `src/keyword_baseline.py` scores the test split and writes `results/keyword_baseline.json`. The reported number is weighted F1 0.9125, with abstention counted as a miss. The list was not retuned after that score.

## Classification

`src/classify.py` trains on the train ids only. The headline number is forced weighted F1 in `results/classification.json`: Multinomial NB 0.9499, calibrated LinearSVC 0.9798. The objective was above 0.85. Both met it.

`results/abstention.json` holds the three abstention numbers. The LinearSVC threshold is 0.90 and the NB threshold is 0.75, each chosen on validation only. On the page, LinearSVC withholds a desk below 0.90. NB withholds a desk below 0.75. Priya is told to take it. There is no review button. On the test set, LinearSVC abstains on 0.16 of articles (16 of 100), accepted accuracy is 0.9881, and accuracy on abstained articles if forced is 0.9375. Fifteen of those sixteen forced labels would have been right. The 10% to 25% abstention band is the selection rule, not a measured newsroom workload. If those abstentions count as misses, LinearSVC falls to 0.8996. The published classification result stays the forced F1, 0.9798.

`results/leakage.json` repeats the forced F1 after the five category words are stripped, and again after ten near-duplicate test articles are dropped. The published score stays the full-test number.

## Archive questions

`rag_50.json` has 50 questions: 40 grounded and 10 adversarial. A grounded item is correct only when the system answers and the normalised answer equals `gold_answer`. An adversarial item is correct only when the system abstains.

`src/retrieve.py` writes Recall at 3 for the 40 grounded questions to `results/retrieval.json`. That score is 0.825. The retrieval cutoff is cosine 0.50. It was chosen on this same set of 50 questions: at 0.35 none of the 10 adversarial questions would have been stopped, and at 0.50 the index stops 8 of those 10 and also 5 of the 40 grounded questions. It was frozen before any answer was generated and was not moved after the answers. This is a development check, not an unseen test. On the page, a best passage below 0.50 means the generator is not called.

Of the 40 grounded questions, the gold article is business for 17, entertainment for 11, politics for 9, sport for 2, and tech for 1. Those questions point at 29 articles.

`src/rag_answer.py` writes `results/rag.json`. The official answer correctness is 0.74 (37 of 50). The objective was above 0.80. It was not met. Grounded items correct: 27 of 40. Adversarial items correct: 10 of 10. The score compares the answer string, or checks that an adversarial item abstained. It does not check whether the citation supports the answer. If `results/rag.json` already exists, ids in that file are kept and are not sent to the generator again.

## Extraction

`extraction_50.json` has 50 articles, 10 per class. The annotation rules are in that file. A list field scores 1 only when the whole normalised set matches. The topic must match one fixed phrase. If the JSON shape check fails, all five fields score 0.

`src/extract.py` writes `results/extraction.json`. The official field score is `field_correctness_exact` 0.484. The objective was above 0.75. It was not met. The shape check passed on all 50 outputs. That check looks at keys and types. An empty topic, a topic outside three to eight words, or a relative date can still pass it. Exact match by field is people 0.84, organisations 0.52, locations 0.48, dates 0.52, and topic 0.06. The four list fields average 0.59, so the miss is not only topic paraphrase. If `results/extraction.json` already exists, ids in that file are kept and are not sent to the generator again.

The 50 articles are balanced, 10 per class, and short. Their mean length is 175 words. The corpus mean is 390 words. The full-corpus cost in the result file multiplies the short-sample mean by 2,225. It is a price-card estimate, not an invoice.

`entity_micro_f1` is 0.8013 on people, organisations, locations, and dates from the same outputs. It shows partial list credit. It is not a replacement score and it does not meet the 0.75 objective. Topic stays at exact match 0.06 and is outside that F1.
