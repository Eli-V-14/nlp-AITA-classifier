# nlp-AITA-classifier

Takes a story about some interpersonal conflict and guesses whether the person
telling it is in the wrong. It's trained on r/AmItheAsshole posts and whatever
verdict the community landed on, using a fine-tuned `roberta-base` that predicts
**NTA** (not the a-hole) or **YTA** (you're the a-hole).

This is the prototype milestone, so it's the whole pipeline working end to end
(data, training, inference, evaluation) with a baseline evaluation, not a
finished model.

## What's in here

```
preprocess/   cleaning + the pipeline that builds the labeled dataset (stdlib only)
eda/          read-only scripts for poking at the raw/processed data
model/        training, inference, evaluation, baselines
out/          outputs: dataset samples, the saved model, metrics (big files git-ignored)
```

The dataset itself (`posts.jsonl`, ~757 MB) lives on external storage, not in
git. See `preprocess/REPORT.md`.

## Requirements

- **Python 3.12** for the model code. PyTorch doesn't have 3.14 wheels yet, and
  3.14 is what was on the machine, so the model code runs in a 3.12 venv. The
  `preprocess/` and `eda/` scripts are stdlib-only and run on anything.
- A CUDA GPU helps a lot (built on an RTX 4070 SUPER, 12 GB). CPU works but it's
  slow.
- Packages: `torch` (CUDA build), `transformers`, `datasets`, `scikit-learn`,
  `accelerate`.

## Setup

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install transformers datasets scikit-learn accelerate
```

## Data

The model trains from `posts.jsonl` (one labeled post per line, already split
into train/val/test). Build it from the raw dumps:

```bash
python preprocess/run_all.py --data-dir "<folder with the raw .jsonl dumps>"
```

Or just point the scripts at a copy you already have with `--data <path>`. The
default is `D:/MSOE/CSC/NLP Data/processed/posts.jsonl`.

## Running it

Train (saves the model to `out/model/`, so you only do this once):

```bash
.venv/Scripts/python.exe model/train.py --max-train 60000 --max-eval 8000 --epochs 3
```

Predict on any post (loads the saved model, nothing retrains):

```bash
.venv/Scripts/python.exe model/infer.py --text "AITA for skipping my friend's wedding because I can't afford the trip?"
.venv/Scripts/python.exe model/infer.py --file my_post.txt
echo "WIBTA if I reported my coworker for taking credit for my work?" | .venv/Scripts/python.exe model/infer.py
```

Baselines (majority + TF-IDF, with cross-validation):

```bash
.venv/Scripts/python.exe model/baselines.py
```

Evaluate (test split, natural + rebalanced, plus the hand-written cases):

```bash
.venv/Scripts/python.exe model/evaluate.py --max-test 8000
```

There's also `test_model.ipynb` if you'd rather poke at it in a notebook.

### What the output looks like

Input: `AITA for refusing to give up my paid-for window seat so a family could sit together?`

```
Verdict: NTA   (you are NOT the asshole)
Probabilities: {'NTA': 0.9x, 'YTA': 0.0x}
```

Input: `AITA for eating the lunch my roommate labeled with her name in the fridge?`

```
Verdict: YTA   (YOU are the asshole)
Probabilities: {'NTA': 0.1x, 'YTA': 0.8x}
```

The verdict is the prediction, and the probabilities are how confident it is.
Something close to 50/50 means it's genuinely unsure, which happens on the
ambiguous ones.

## Limitations

- **Binary only.** We squash the five AITA verdicts down to NTA vs YTA (NAH→NTA,
  ESH→YTA, and INFO/meta dropped), so there's no "everyone sucks" or "no a-holes
  here" yet.
- **Imbalanced.** About 76% of posts are NTA. We weight the loss to help, but the
  model still leans NTA when it's unsure.
- **The labels are subjective.** "Ground truth" is a Reddit vote, which is noisy
  and reflects that crowd's norms, not an objective answer.
- **Small scale on purpose.** Trains on a 60k slice of ~300k to keep it fast, and
  it isn't tuned for best accuracy.
- **Gap in the data.** The comments stop in March 2019 but posts run to mid-2022,
  so the val/test posts have labels but no comment signal. More in
  `preprocess/REPORT.md`.
- **No explanations yet.** This is the verdict classifier only; the explanation
  generator is still to come.

## Reproducing the numbers

`out/eval_report.json` and `out/baseline_report.json` have the metrics;
`preprocess/REPORT.md` has the dataset stats.
