# Prototype Results

Quick reference so you can see where things stand without digging through the
code: the roadmap stages, the stats the assignment wants, and the actual numbers.
The formal writeup is in [PART5_SUMMARY.md](PART5_SUMMARY.md), and
[README.md](README.md) covers how to run it.

The system fine-tunes `roberta-base` to guess an AITA post's verdict, NTA (not
the a-hole) or YTA (you're the a-hole).

## Where we are on the roadmap

Stages 1 through 5 are done, stage 6 hasn't been started, and stage 7 is half
finished.

| Stage | Status | Where |
|---|---|---|
| 1. Collection | done | raw Reddit dumps |
| 2. Cleaning (drop removed/deleted/meta/short, strip edits + verdict tags) | done | `preprocess/clean_aita.py` |
| 3. Labeling (flair to binary NTA/YTA, keep top-level verdict counts) | done | `preprocess/clean_aita.py` |
| 4. Baseline (majority + TF-IDF LogReg/NB) | done | `model/baselines.py` |
| 5. Classify (fine-tune RoBERTa-base, fixed hyperparams) | done | `model/train.py` |
| 6. Explain (BART generator from agreeing top comments) | not started | future work |
| 7. Integrate (user text to verdict + community-norms note) | partial, classifier side only | `model/infer.py` |

We used a fixed train/val/test split for the transformer. Cross-validation is on
the TF-IDF baseline (5-fold).

## Assignment deliverables

Everything the prof asked for in Parts 1 through 5 is covered.

- Part 1 (working prototype): runs end to end, uses our data, fine-tuned model,
  takes new input, prints output.
- Part 2 (evaluation): metrics, baseline comparison, 13 hand-written cases, with
  success and failure examples.
- Part 3 (artifacts): datasets, saved model, preprocessing scripts (see below).
- Part 4 (running it): `README.md`.
- Part 5 (summary): `PART5_SUMMARY.md`.

## Dataset

From `preprocess/` over the full download (more in `preprocess/REPORT.md`):

- Raw posts: 1,385,631. After cleaning and labeling: 373,340 (about 27% survive).
- Labels: NTA 284,151 (76.1%), YTA 89,189 (23.9%).
- Split, oldest to newest: train 298,672, val 37,334, test 37,334.
- Verdict comments on kept posts: 482,208. Explanation targets: 50,355.

Training setup: roberta-base, full fine-tune, 60,000-post train subset, 3 epochs,
batch 32, lr 2e-5, max length 256, class weights NTA/YTA = 0.670 / 1.969.

## The numbers

Natural test set (n = 8,000):

| Model | Accuracy | Macro-F1 | NTA-F1 | YTA-F1 |
|---|---:|---:|---:|---:|
| Majority (all-NTA) | 0.815 | 0.449 | 0.898 | 0.000 |
| TF-IDF + Naive Bayes | 0.815 | 0.452 | 0.898 | 0.005 |
| TF-IDF + LogReg | 0.703 | 0.584 | 0.806 | 0.362 |
| roberta-base (fine-tuned) | 0.735 | **0.622** | 0.828 | **0.416** |

Per-class for the fine-tuned model: NTA precision/recall/F1 = 0.876 / 0.786 /
0.828, YTA = 0.351 / 0.509 / 0.416. Confusion matrix (rows = true, cols = pred,
order NTA/YTA): `[[5122, 1395], [728, 755]]`. Cross-validated macro-F1 for
TF-IDF + LogReg: 0.592 ± 0.003. Validation during training: accuracy 0.735,
macro-F1 0.626.

Rebalanced 50/50 test set (n = 2,966), fine-tuned: accuracy 0.642, macro-F1
0.635, NTA 0.612 / 0.774 / 0.684, YTA 0.693 / 0.509 / 0.587.

### How to read it
The fine-tuned model scores lower on raw accuracy than the majority baseline
(0.735 vs 0.815), which looks wrong until you notice the baselines only get there
by almost never saying YTA (their YTA-F1 is basically zero). Ours actually catches
YTA cases, so it wins on macro-F1. That's why we treat macro-F1 as the real number
here instead of accuracy. Efstathiadis et al. ran into the same thing.

### Hand-built cases: 12 of 13 right
13 short scenarios we typed ourselves, the way someone would actually phrase a
question.

- It got the clear ones: "paid-for airplane seat" (NTA), "labeled lunch" (YTA),
  "reported coworker for stealing credit" (NTA).
- The one it missed: "WIBTA if I skipped my best friend's destination wedding
  because I can't afford the $3000 trip?" It said YTA, we'd say NTA. It's a
  hypothetical built around wedding guilt, and the model grabs onto "skipping the
  event" over the can't-afford-it part.

## Artifacts and how to run

| Artifact | Location | In git? |
|---|---|---|
| Preprocessing scripts | `preprocess/` | yes |
| Model scripts (train/infer/evaluate/baselines) | `model/` | yes |
| Hand-built eval cases | `model/eval_cases.jsonl` | yes |
| Dataset samples | `out/sample_posts.jsonl`, `out/sample_comments.jsonl` | yes |
| Saved model (~499 MB) | `out/model/` | no, too big, hosted separately |
| Full dataset (`posts.jsonl`, 757 MB) | `D:/MSOE/CSC/NLP Data/processed/` | no, on external storage |
| Eval + baseline reports | `out/eval_report.json`, `out/baseline_report.json` | yes |

Run (after the `README.md` setup):
```bash
.venv/Scripts/python.exe model/train.py --max-train 60000 --epochs 3     # train (saves out/model/)
.venv/Scripts/python.exe model/baselines.py                               # baselines + CV
.venv/Scripts/python.exe model/evaluate.py                                # test + rebalanced + cases
.venv/Scripts/python.exe model/infer.py --text "AITA for ...?"            # predict any new post
```

## What's left
1. Build the explanation generator (stage 6, BART) and connect it to the classifier (stage 7).
2. Get YTA detection up: more training data, threshold tuning, handling long posts.
3. Add the Lourie-style soft-label / vote-proportion evaluation using the counts we stored.
4. For submission, upload the saved model and dataset somewhere accessible and link them
   (Part 3), since both are too big for git.
