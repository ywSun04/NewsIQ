# NewsIQ: business and technical trade-offs

Sun Yawen, PE6201

I built NewsIQ for Priya, a content operator who sorts English news, looks up an old article, and pulls a few structured fields. The corpus is the public BBC News classification set: 2,225 articles from 2004-2005. I did not collect private user records. The articles do name public figures. Other languages, live ingest, and automatic publishing are out of scope. This is a proof of concept on a historical archive, not a product for today's news.

Priya's three jobs are fixed, so I did not use an agent. An agent would choose tools that the workflow already names. Classification uses a narrow TF-IDF model. Questions use retrieval over the archive, then a rented generator that may answer only from the retrieved passages. Extraction uses the same generator with a fixed JSON schema. I skipped a low-code or AutoML platform. The corpus is small, so that tooling would not remove much setup, and it would tie the pipeline to one vendor. I built the classifiers, the local embedding index, the evaluation, and the Streamlit page. I rent only the generator: GPT-4o-mini through OpenRouter, at USD 0.15 per million input tokens and USD 0.60 per million output tokens, temperature 0.

I did not measure how long an editor spends on an article. A planning assumption of USD 20 an hour and 50 articles an hour is USD 0.40 per article. That number is an assumption, not a measured saving. I do not use F1 to claim that time was saved. The costs below are token costs from the runs.

## What changes for Priya

Today Priya reads each article, chooses a desk, searches the archive by memory, and copies names into a sheet. NewsIQ keeps those three jobs and changes the first pass. The classifier proposes a desk, an archive question returns a cited span, and extraction returns JSON she can check. She still owns the decision when the system abstains. Because the archive stops in 2005, the page does not replace a current newsroom system.

## Classification

The objective in the proposal was weighted F1 above 0.85 on a stratified holdout of 100 articles. The split is seed 42: 1,925 train, 200 validation, 100 test. The test set was not used to fit a model, to edit the keyword list, or to choose a threshold.

A majority baseline that always predicts sport, the largest training class, scores weighted F1 0.086. The hand keyword baseline, with abstention counted as a miss, scores 0.9125. TF-IDF plus Multinomial NB scores 0.9499 on forced predictions. TF-IDF plus LinearSVC, with sigmoid calibration, scores 0.9798. Both learned models meet the 0.85 objective, and both are above the keyword baseline on this forced F1. Tech was the weak keyword class, at F1 0.7333. It is 0.9189 for NB and 0.973 for LinearSVC.

The page uses the calibrated LinearSVC. It abstains when the top probability is below 0.90. I chose that threshold only on the validation split: among cutoffs with an abstention rate from 10% to 25%, I took the one with the highest accuracy on accepted articles. On the test set the three abstention numbers are an abstention rate of 0.16 (16 of 100), accuracy 0.9881 on the accepted articles, and accuracy 0.9375 on the abstained articles if the model is forced to predict. Fifteen of those sixteen would have been right, so the gate sends sixteen articles to a person to catch one error. One accepted error remains. I report 0.9798 as the classification result. If abstentions count as misses, that score falls to 0.8996, below the keyword baseline. Ten test articles are near-duplicates of training articles. Dropping them leaves LinearSVC at 0.9776. I did not redraw the split. Removing the five category words and retraining left both forced F1 scores unchanged at four decimals.

## Questions over the archive

The objective was answer correctness above 0.80 on 50 questions written before any model output. A grounded answer counts only if the normalised string matches the gold phrase. An out-of-archive question counts only if the system abstains. The result is 0.74 (37 of 50). The objective was not met. Of 40 grounded questions, 27 were correct. All 10 out-of-archive questions were abstained. Recall at 3 on the 40 grounded questions is 0.825 (33 of 40), using all-MiniLM-L6-v2, 200-word chunks with an overlap of 40, and the top 3 chunks.

I set cosine 0.50 after seeing how these same 50 questions behaved, and before any answer was generated. At 0.35 none of the 10 out-of-archive questions would have been stopped. At 0.50 the index stops 8 of those 10 and also 5 of the 40 grounded questions. I did not move it after the answers. This is a development check on one set, not an unseen test. One miss is a wrong article, not a wording difference. Question rag-015 asks until which year the Treasury said its spending plans were funded. The gold article is 665 and the gold answer is 2008. The top passages came from articles 295 and 813, not 665, and the model answered 2009. The score does not check whether the citation supports the answer. The 37 calls used 27,381 prompt tokens and 1,032 completion tokens and cost USD 0.004726. I left the misses in place.

## Extraction

The objective was field correctness above 0.75. The pre-registered rule is exact: a list field scores 1 only when the whole normalised set matches, and the topic must match one fixed phrase of three to eight words. The score on 50 articles is 0.484. The objective was not met. Exact match by field is people 0.84, organisations 0.52, locations 0.48, dates 0.52, and topic 0.06. The four list fields average 0.59 even with topic left out, so the miss is not only paraphrase. On article ext-001 the gold dates are only 2003. The model also wrote "last year", and the whole dates field scores 0. The gold topic is "burren energy wins egyptian oil contracts"; the model wrote "Burren Energy awarded contracts in Egypt", which also scores 0. Overlap of the four list fields on the same outputs has micro F1 0.8013. That describes partial list credit. The reported extraction result remains 0.484. The 50 calls cost USD 0.006157, a mean of USD 0.000123. Those articles average 175 words; the corpus averages 390. Mean cost times 2,225 is about USD 0.274 at the price card. That is not an invoice, and it can understate a longer article.

## Controls

If the best passage is below cosine 0.50, or the generator finds no answer in the passages, the page says the corpus does not contain sufficient evidence. An article below the classification threshold is not labelled. The page tells Priya to take it. There is no review queue. Answers are limited to 400 tokens and extraction to 500. Before another request, the page stops at 30 recorded attempts or USD 0.05. A bad-JSON retry is checked first. Inside one request, HTTP 429, 500, 502, or 503 can still be tried up to three times. The page states that the corpus is BBC News 2004-2005 and is not for automated publishing, fact-checking, or legal review. I report per-class F1 so a weak class cannot hide in the average. The generator is told to use only the passages. I did not test a malicious instruction planted inside a passage. The stated purpose and the handoff follow the Singapore IMDA Model AI Governance Framework. The passage-only instruction is aimed at the prompt-injection risk in the OWASP Top 10 for LLM Applications 2025.

## A later version

I would not edit the gold to lift the 0.484 extraction score. A later version could ask a person to judge whether a topic paraphrase is faithful, and it could separate duplicate articles before the split. I would not add an agent or a live feed until the handoff is a check Priya actually uses.
