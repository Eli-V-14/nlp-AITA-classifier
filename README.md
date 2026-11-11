# nlp-AITA-classifier

A text classifier that predicts whether the author of an interpersonal-conflict
story is in the wrong. It is trained on Reddit r/AmItheAsshole posts and the
community's final verdict, and fine-tuned with **LoRA** on top of `roberta-base`
to predict **NTA** (Not the A-hole) vs **YTA** (You're the A-hole).

This is the prototype milestone: an end-to-end, runnable system (data →
training → inference → evaluation) with a baseline evaluation, not a polished
final model.

## Repository layout

```
preprocess/   data cleaning + the pipeline that builds the labeled dataset (stdlib only)
eda/          read-only scripts to explore the raw/processed data
model/         LoRA training, inference, and evaluation
out/           outputs: dataset samples, saved LoRA adapter, metrics (large files git-ignored)
```

The labeled dataset itself (`posts.jsonl`, ~757 MB) is kept on external storage,
not in git — see `preprocess/REPORT.md` and "Data" below.

## Requirements

- **Python 3.12** for the model code (PyTorch has no 3.14 wheels yet). The
  `preprocess/` and `eda/` scripts are stdlib-only and run on any 3.x.
- A CUDA GPU is recommended (developed on an RTX 4070 SUPER, 12 GB). CPU works
  but is slow.
- Python packages: `torch` (CUDA build), `transformers`, `peft`, `datasets`,
  `scikit-learn`, `accelerate`.

## Setup

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install transformers peft datasets scikit-learn accelerate
```

## Data

The model trains from the processed dataset `posts.jsonl` (one labeled post per
line, with a train/val/test split). Build it from the raw Reddit dumps with:

```bash
python preprocess/run_all.py --data-dir "<folder with the raw .jsonl dumps>"
```

Or point the model scripts at an existing copy with `--data <path to posts.jsonl>`.
The default path is `D:/MSOE/CSC/NLP Data/processed/posts.jsonl`.

## Execution

**Train** (saves a LoRA adapter to `out/lora_adapter/` — done once, reused after):

```bash
.venv/Scripts/python.exe model/train.py --max-train 60000 --max-eval 8000 --epochs 3
```

**Predict on any new post** (adapter is loaded, nothing is retrained):

```bash
.venv/Scripts/python.exe model/infer.py --text "AITA for skipping my friend's wedding because I can't afford the trip?"
.venv/Scripts/python.exe model/infer.py --file my_post.txt
echo "WIBTA if I reported my coworker for taking credit for my work?" | .venv/Scripts/python.exe model/infer.py
```

**Evaluate** (test-split metrics + hand-built cases):

```bash
.venv/Scripts/python.exe model/evaluate.py --max-test 5000
```

### Example inputs and outputs

Input: `AITA for refusing to give up my paid-for window seat so a family could sit together?`
Output:
```
Verdict: NTA   (you are NOT the asshole)
Probabilities: {'NTA': 0.9x, 'YTA': 0.0x}
```

Input: `AITA for eating the lunch my roommate labeled with her name in the fridge?`
Output:
```
Verdict: YTA   (YOU are the asshole)
Probabilities: {'NTA': 0.1x, 'YTA': 0.8x}
```

The verdict is the model's prediction; the probabilities show its confidence.
A value near 0.5/0.5 means the model is unsure (common on genuinely ambiguous
cases).

## Known limitations

- **Binary only.** Collapses the subreddit's 5 verdicts into NTA vs YTA (NAH→NTA,
  ESH→YTA; INFO/meta dropped). No "everyone sucks" / "no a-holes" distinction yet.
- **Class imbalance.** ~76% of posts are NTA; training uses class weighting, but
  the model still favors NTA on ambiguous input.
- **Subjective labels.** The "ground truth" is a Reddit vote, which is noisy and
  reflects that community's norms, not an objective judgment.
- **Baseline scale.** Trains on a 60k subset of 300k for speed; not tuned for
  best accuracy.
- **Temporal gap in the data.** Comments (used for vote counts/explanations) stop
  in 2019-03 while posts run to 2022-06; the val/test splits have labels but no
  comment-derived signal. See `preprocess/REPORT.md`.
- The explanation-generation component (`explanations.jsonl` targets) is future
  work; this prototype is the verdict classifier only.

## Reproducing the numbers

See `out/eval_report.json` (written by `model/evaluate.py`) and
`preprocess/REPORT.md` for the dataset statistics.
