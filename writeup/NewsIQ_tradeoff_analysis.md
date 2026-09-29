# NewsIQ: business and technical trade-offs

Sun Yawen, PE6201

I built NewsIQ for Priya, a content operator who sorts English news, looks up an old article, and pulls a few structured fields. The corpus is the public BBC News classification set: 2,225 articles from 2004-2005, with no personal data. Other languages, live ingest, and automatic publishing are out of scope. This is a proof of concept on a historical archive, not a product for today's news.

Priya's three jobs are fixed, so I did not use an agent. An agent would choose tools that the workflow already names. Classification uses a narrow TF-IDF model. Questions use retrieval over the archive, then a rented generator that may answer only from the retrieved passages. Extraction uses the same generator with a fixed JSON schema. I skipped a low-code or AutoML platform. The corpus is small, so that tooling would not remove much setup, and it would tie the pipeline to one vendor. I built the classifiers, the local embedding index, the evaluation, and the Streamlit page. I rent only the generator: GPT-4o-mini through OpenRouter, at USD 0.15 per million input tokens and USD 0.60 per million output tokens, temperature 0.

I did not measure how long an editor spends on an article. A planning assumption of USD 20 an hour and 50 articles an hour is USD 0.40 per article. That number is an assumption, not a measured saving. I do not use F1 to claim that time was saved. The costs below are token costs from the runs.

## Classification

The objective in the proposal was weighted F1 above 0.85 on a stratified holdout of 100 articles. The split is seed 42: 1,925 train, 200 validation, 100 test. The test set was not used to fit a model, to edit the keyword list, or to choose a threshold.

A majority baseline that always predicts sport, the largest training class, scores weighted F1 0.086. The hand keyword baseline, with abstention counted as a miss, scores 0.9125. TF-IDF plus Multinomial NB scores 0.9499 on forced predictions. TF-IDF plus LinearSVC, with sigmoid calibration, scores 0.9798. Both learned models meet the 0.85 objective, and both are above the keyword baseline on this forced F1. Tech was the weak keyword class, at F1 0.7333. It is 0.9189 for NB and 0.973 for LinearSVC.

The page uses the calibrated LinearSVC. It abstains when the top probability is below 0.90. I chose that threshold only on the validation split: among cutoffs with an abstention rate from 10% to 25%, I took the one with the highest accuracy on accepted articles. On the test set the three abstention numbers are an abstention rate of 0.16 (16 of 100), accuracy 0.9881 on the accepted articles, and accuracy 0.9375 on the abstained articles if the model is forced to predict. NB, with its own validation threshold of 0.75, abstains on 0.23 of the test set, with accepted accuracy 0.987 and forced accuracy on the abstained cases of 0.8261. If abstentions are scored as misses, LinearSVC falls to 0.8996 and NB to 0.8463, both below the keyword baseline, because the rule requires dropping between 10% and 25% of articles. I report 0.9798 as the classification result. The three abstention numbers describe the gate. They are not a second F1 target.

Removing the five category words and retraining left both forced F1 scores unchanged at four decimals, so the models are not relying on the label string. Ten test articles are near-duplicates of training articles. Dropping those ten leaves NB at 0.9446 and LinearSVC at 0.9776. I did not redraw the split after seeing the scores.

## Questions over the archive

The objective was answer correctness above 0.80 on 50 questions written before any model output. A grounded answer counts only if the normalised string matches the gold phrase. An out-of-archive question counts only if the system abstains. The result is 0.74 (37 of 50). The objective was not met. Of 40 grounded questions, 27 were correct. All 10 out-of-archive questions were abstained. Recall at 3 on the 40 grounded questions is 0.825 (33 of 40), using all-MiniLM-L6-v2, 200-word chunks with an overlap of 40, and the top 3 chunks.

I froze the retrieval cutoff at cosine 0.50 before any answer was generated. A lower cutoff of 0.35 would have abstained on none of the 10 out-of-archive questions. At 0.50 the index abstains on 8 of those 10 and on 5 of the 40 grounded questions. I did not move the cutoff after reading the answers. The run made 37 generator calls, used 27,381 prompt tokens and 1,032 completion tokens, and cost USD 0.004726, a mean of USD 0.000128 per call. Misses are mostly a phrase that is longer or shorter than the gold string, or a grounded question stopped by the 0.50 cutoff. I left those misses in place.

## Extraction

The objective was field correctness above 0.75. The pre-registered rule is exact: a list field scores 1 only when the whole normalised set matches, and the topic must match one fixed phrase of three to eight words. The score on 50 articles is 0.484. The objective was not met. Every output passed the schema check. Exact match by field is people 0.84, organisations 0.52, locations 0.48, dates 0.52, and topic 0.06. Topic is low because a paraphrase of the same event scores 0. On the same outputs, overlap of people, organisations, locations, and dates has micro F1 0.8013 (precision 0.8465, recall 0.7606). I report that overlap only to show what the exact rule treats as a total miss. It does not replace 0.484, and it does not meet the 0.75 objective. The 50 calls cost USD 0.006157, a mean of USD 0.000123 per article. At that mean, all 2,225 articles would cost about USD 0.274. These measured costs replace my earlier planning estimates.

## Controls

If the best passage is below cosine 0.50, or the generator finds no answer in the passages, the page says the corpus does not contain sufficient evidence. An article below the classification threshold is not labelled; it is handed to a person. Answers are limited to 400 tokens and extraction to 500. A session stops at 30 calls or USD 0.05, whichever comes first. The page states that the corpus is BBC News 2004-2005 and is not for automated publishing, fact-checking, or legal review. I report per-class F1 so a weak class cannot hide in the average. These controls follow the Singapore IMDA Model AI Governance Framework on a stated purpose and human oversight, and they limit prompt injection as described in the OWASP Top 10 for LLM Applications 2025: the generator sees only the retrieved passages and must abstain when those passages do not contain the answer.
