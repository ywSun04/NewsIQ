# Data

This file explains the corpus. It is not a model output.

## Corpus

Source: https://storage.googleapis.com/download.tensorflow.org/data/bbc-text.csv

The file is the public BBC News classification set distributed with TensorFlow. It has 2,225 rows and two columns, `category` and `text`. Counts are business 510, entertainment 386, politics 417, sport 511, tech 401. Articles are from 2004-2005. This project did not collect private user records. The articles do name public figures.

The full CSV is not in git. From the repository root:

```bash
python scripts/download_bbc.py
```

The script writes `data/bbc-text.csv`. That path is gitignored. SHA-256 of the file used for every id in this repository:

`fdaee0f7451cd8db2709d00e992886fe1c387ee332b8c7b7c1554ed3d3e3382e`

If that file is already present, the script checks this digest and does not download again. If the digest differs, the script exits and leaves the existing file in place.

`results/data_profile.json` is a saved count record. It is not written by the download script. `scripts/make_split.py` writes it, and that same script also rewrites `eval/clf_split.json`. Do not run it. The split already in git is the one the scores use.

An article id is the 0-based row index in that CSV. The same id is used in the split, the gold files, and the result files.

## What is checked in

`data/sample/` has one article text per class, for a quick look. Those five are not the evaluation set. `below_threshold.txt` is article 1731 from the classification test split. The demo loads it only to show abstention. It is not a new evaluation.

The evaluation rows are named by id in `eval/`. See `eval/EVALS.md`.
