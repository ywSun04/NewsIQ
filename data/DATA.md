# Data

This file explains the corpus. It is not a model output.

## Corpus

Source: https://storage.googleapis.com/download.tensorflow.org/data/bbc-text.csv

The file is the public BBC News classification set distributed with TensorFlow. It has 2,225 rows and two columns, `category` and `text`. Counts are business 510, entertainment 386, politics 417, sport 511, tech 401. Articles are from 2004-2005. There is no personal data.

The full CSV is not in git. From the repository root:

```bash
python scripts/download_bbc.py
```

The script writes `data/bbc-text.csv`. That path is gitignored. `results/data_profile.json` records the row count, columns, and class counts after download.

An article id is the 0-based row index in that CSV. The same id is used in the split, the gold files, and the result files.

## What is checked in

`data/sample/` has one article text per class, for a quick look. Those samples are not the evaluation set.

The evaluation rows are named by id in `eval/`. See `eval/EVALS.md`.
