# Prototype Implementation & Initial Evaluation — Summary

**Project:** AITA verdict classifier (predict NTA vs YTA from an interpersonal-conflict story)
**Milestone:** Prototype Implementation and Initial Evaluation

## 1. Current implementation

**What works (end-to-end).** The prototype runs the full roadmap pipeline:
data → preprocessing → training → inference → evaluation.

- **Data (roadmap Stages 1–3).** Raw Reddit dumps are cleaned and labeled by
  `preprocess/clean_aita.py`: removed/deleted/meta/short posts dropped, EDIT/UPDATE
  sections and verdict tokens stripped (leakage control), verdict flair mapped to a
  binary **NTA/YTA** label (NAH→NTA, ESH→YTA), top-level comment verdict counts kept,
  and an oldest→newest train/val/test split. Result: **373,340 labeled posts**
  (76% NTA / 24% YTA).
- **Classifier (Stage 5).** `model/train.py` fine-tunes **roberta-base** (full
  fine-tune, fixed hyperparameters, no search) with inverse-frequency class weighting
  for the imbalance. The trained model is saved to `out/model/` and reused by
  inference/evaluation (no retraining).
- **Inference (Stage 7, partial).** `model/infer.py` accepts **any new post** (arg,
  file, or stdin) and returns the verdict + class probabilities, with a note that the
  result reflects how the Reddit community would likely vote — not an objective moral
  judgement.
- **Baselines & evaluation (Stage 4 + eval plan).** `model/baselines.py` (majority,
  TF-IDF+LogReg, TF-IDF+NaiveBayes, 5-fold CV) and `model/evaluate.py` (natural +
  rebalanced test sets, hand-built cases).

**What is incomplete.** The **explanation generator** (Stage 6, BART fine-tuned on
agreeing top comments) is not built — only the verdict classifier. Full Stage-7
integration (classifier → generator → combined output, web UI) is therefore partial:
the classifier is wired end-to-end, the generator is future work. We also used a
fixed train/val/test split for the transformer rather than k-fold cross-validation
(CV is applied to the TF-IDF baseline).

## 2. Initial evaluation

**Task & metrics.** Binary verdict classification. Because the data is imbalanced
(~81% NTA in the test split), we report **macro-F1 and per-class precision/recall**,
not just accuracy, and compare against a **majority-class baseline** and **TF-IDF
baselines** — exactly the failure mode the roadmap's literature (Efstathiadis et al.)
warns about.

**Evaluation data.** Held-out **test split** (n=8,000, never seen in training) plus a
hand-written probe set of **13 typed cases** (easy + ambiguous), in the style a user
would actually type — which differs from real Reddit posts.

**Baseline results (natural test set, n=8,000):**

| Model | Accuracy | Macro-F1 | YTA-F1 |
|---|---:|---:|---:|
| Majority (all-NTA) | 0.815 | 0.449 | 0.000 |
| TF-IDF + Naive Bayes | 0.815 | 0.452 | 0.005 |
| TF-IDF + LogReg | 0.703 | 0.584 | 0.362 |
| **roberta-base (fine-tuned)** | 0.735 | **0.622** | **0.416** |

5-fold CV macro-F1 (TF-IDF+LogReg): **0.592 ± 0.003**.
Rebalanced 50/50 test set (n=2,966), fine-tuned model: accuracy **0.642**, macro-F1 **0.635**.
Confusion matrix [true][pred], order [NTA, YTA]: `[[5122, 1395], [728, 755]]`.

**Reading the results.** The fine-tuned model has **lower raw accuracy than the
majority baseline (0.735 vs 0.815)** but the **best macro-F1 (0.622)** by a clear
margin, because the baselines win accuracy only by predicting NTA for (almost)
everything — they get YTA-F1 ≈ 0. The transformer actually identifies the YTA
minority (recall 0.51). This is the same pattern reported in the cited papers, and is
why macro metrics are the headline. On the rebalanced set the model is well above the
0.5 chance level (0.642 accuracy, 0.635 macro-F1).

**Hand-built cases: 12 / 13 correct (0.92).**

- *Successes:* "telling my sister her cooking was terrible in front of her guests" → **YTA**;
  "refusing to give up my paid-for airplane seat" → **NTA**;
  "eating the lunch my roommate labeled with her name" → **YTA**. Clear-cut cases are
  handled confidently and correctly.
- *Failure:* "WIBTA if I skipped my best friend's destination wedding because I can't
  afford the $3000 trip?" → predicted **YTA**, expected **NTA**. A hypothetical
  ("WIBTA") framed around wedding guilt; the model leans YTA, likely over-weighting the
  "skipping a friend's big event" pattern over the financial-hardship justification.

**What the metrics do NOT measure.** Agreement with the Reddit verdict measures how an
online community would judge, not moral correctness; raw accuracy hides minority-class
failure (hence macro metrics); and none of this measures how users react to the output.

## 3. Challenges and next steps

**Challenges encountered.**
- *Class imbalance* is the central difficulty — accuracy is misleading, so the whole
  evaluation is built around macro metrics, class weighting, and a rebalanced set.
- *Noisy, subjective labels* (a Reddit vote) cap achievable accuracy, consistent with
  the literature (posts are much noisier than comments).
- *Data coverage gap:* comments stop at 2019-03 while posts run to 2022-06, so the
  val/test posts have labels but no comment-derived signal (see `preprocess/REPORT.md`).
- *Tooling:* running on a brand-new Python (3.14) with no PyTorch wheels required a
  separate 3.12 environment; preprocessing streams multi-GB files in binary mode to
  avoid a Windows text-mode error.

**Biggest current weaknesses.** YTA recall (0.51) and precision (0.35) are modest — the
model still misses or over-calls roughly half of YTA cases; and the system is
classifier-only (no explanations yet).

**Priorities before final submission.**
1. Build the **explanation generator** (Stage 6) and integrate it (Stage 7).
2. Improve YTA detection — train on more of the 300k set, tune the decision threshold,
   and handle posts longer than 256 tokens deliberately.
3. Add the Lourie-style **soft-label / vote-proportion** evaluation using the stored
   vote counts.
4. Package a simple interface for the integrated verdict + explanation.

**Artifacts:** `out/model/` (saved model), `out/eval_report.json`,
`out/baseline_report.json`, `out/train.log`; dataset built by `preprocess/`; see the
top-level `README.md` to run everything.
