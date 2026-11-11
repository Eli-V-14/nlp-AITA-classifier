# Prototype Results

A teammate-facing summary of where the prototype stands: our **roadmap stages**,
the **stats the assignment asks for**, and all the **actual numbers**. For the
formal 1–2 page writeup see [`PART5_SUMMARY.md`](PART5_SUMMARY.md); to run things
see [`README.md`](README.md).

Core system: fine-tune **roberta-base** to predict an AITA post's verdict,
**NTA** (not the a-hole) vs **YTA** (you're the a-hole).

---

## 1. Roadmap stage status

| Stage (from our proposal) | Status | Where |
|---|---|---|
| 1. Collection | ✅ done | raw Reddit dumps |
| 2. Cleaning (drop removed/deleted/meta/short, strip edits + verdict tags) | ✅ done | `preprocess/clean_aita.py` |
| 3. Labeling (flair → binary NTA/YTA, keep top-level verdict counts) | ✅ done | `preprocess/clean_aita.py` |
| 4. Baseline (majority + TF-IDF LogReg/NB, scikit-learn) | ✅ done | `model/baselines.py` |
| 5. Classify (fine-tune RoBERTa-base, fixed hyperparams) | ✅ done | `model/train.py` |
| 6. Explain (BART generator from agreeing top comments) | ⬜ **future work** | — |
| 7. Integrate (accept user text → verdict + community-norms note) | 🟡 partial | `model/infer.py` (classifier side done; generator pending) |

We used a fixed train/val/test split for the transformer; **cross-validation**
is applied to the TF-IDF baseline (5-fold).

---

## 2. Assignment deliverables (what the prof wants)

| Part | Requirement | Status |
|---|---|---|
| 1. Working prototype | End-to-end, uses our data, core NLP approach, processes new input, observable output | ✅ |
| 2. Initial evaluation | Metric(s), ≥10 eval cases, baseline comparison, success/failure examples | ✅ (13 cases) |
| 3. Data & model artifacts | Datasets, saved model, preprocessing scripts, configs | ✅ (see §5) |
| 4. Running the system | README with setup/run/examples/limitations | ✅ `README.md` |
| 5. Summary report | 1–2 page implementation + evaluation writeup | ✅ `PART5_SUMMARY.md` |

---

## 3. Dataset stats

From `preprocess/` over the full download (details in `preprocess/REPORT.md`):

- Raw posts: **1,385,631** → cleaned/labeled: **373,340** (~27% usable)
- Labels: **NTA 284,151 (76.1%)** / **YTA 89,189 (23.9%)**
- Split (oldest→newest): **train 298,672 · val 37,334 · test 37,334**
- Verdict comments on kept posts: 482,208 · explanation targets: 50,355

**Training config:** roberta-base, full fine-tune, **60,000**-post train subset,
3 epochs, batch 32, lr 2e-5, max_len 256, class weights NTA/YTA = 0.670 / 1.969.

---

## 4. Results (the metrics)

### Baselines vs fine-tuned — natural test set (n = 8,000)

| Model | Accuracy | Macro-F1 | NTA-F1 | YTA-F1 |
|---|---:|---:|---:|---:|
| Majority (all-NTA) | 0.815 | 0.449 | 0.898 | 0.000 |
| TF-IDF + Naive Bayes | 0.815 | 0.452 | 0.898 | 0.005 |
| TF-IDF + LogReg | 0.703 | 0.584 | 0.806 | 0.362 |
| **roberta-base (fine-tuned)** | **0.735** | **0.622** | 0.828 | **0.416** |

- Fine-tuned per-class: NTA P/R/F1 = 0.876 / 0.786 / 0.828 · YTA P/R/F1 = 0.351 / 0.509 / 0.416
- Confusion matrix [true][pred], order [NTA, YTA]: **[[5122, 1395], [728, 755]]**
- 5-fold CV macro-F1 (TF-IDF+LogReg): **0.592 ± 0.003**
- Validation (during training): accuracy 0.735, macro-F1 0.626

### Rebalanced 50/50 test set (n = 2,966), fine-tuned

| Metric | Value |
|---|---:|
| Accuracy | 0.642 |
| Macro-F1 | 0.635 |
| NTA P/R/F1 | 0.612 / 0.774 / 0.684 |
| YTA P/R/F1 | 0.693 / 0.509 / 0.587 |

### Hand-built cases — 12 / 13 correct (0.92)

13 typed scenarios (easy + ambiguous), the way a user would phrase them.
- **Successes:** "paid-for airplane seat" → NTA; "trashed my parents' donuts" style /
  "labeled lunch" → YTA; "reported coworker for stealing credit" → NTA.
- **Only miss:** *"WIBTA if I skipped my best friend's destination wedding because I
  can't afford the $3000 trip?"* → predicted **YTA**, expected **NTA** (hypothetical
  framed around wedding guilt; model over-weighted "skipping the event").

### How to read it
The fine-tuned model has **lower raw accuracy than the majority baseline**
(0.735 vs 0.815) but the **best macro-F1** — the baselines only win accuracy by
predicting NTA for almost everything (YTA-F1 ≈ 0). The transformer actually catches
the YTA minority. This matches the papers in our research (Efstathiadis et al.), which
is why **macro-F1 is the headline metric, not accuracy**.

---

## 5. Artifacts & how to run

| Artifact | Location | In git? |
|---|---|---|
| Preprocessing scripts | `preprocess/` | ✅ |
| Model scripts (train/infer/evaluate/baselines) | `model/` | ✅ |
| Hand-built eval cases | `model/eval_cases.jsonl` | ✅ |
| Dataset samples | `out/sample_posts.jsonl`, `out/sample_comments.jsonl` | ✅ |
| **Saved model (~499 MB)** | `out/model/` | ❌ too large — provide accessible location |
| **Full dataset (`posts.jsonl`, 757 MB)** | `D:/MSOE/CSC/NLP Data/processed/` | ❌ too large — on external storage |
| Eval + baseline reports | `out/eval_report.json`, `out/baseline_report.json` | ✅ |

Run (after `README.md` setup):
```bash
.venv/Scripts/python.exe model/train.py --max-train 60000 --epochs 3     # train (saves out/model/)
.venv/Scripts/python.exe model/baselines.py                               # baselines + CV
.venv/Scripts/python.exe model/evaluate.py                                # test + rebalanced + cases
.venv/Scripts/python.exe model/infer.py --text "AITA for ...?"            # predict any new post
```

---

## 6. Known gaps / next steps
1. Build the **explanation generator** (Stage 6, BART) and integrate it (Stage 7).
2. Improve YTA detection (more training data, threshold tuning, long-post handling).
3. Add the Lourie-style **soft-label / vote-proportion** evaluation using stored counts.
4. For submission: upload the **saved model** and **dataset** somewhere accessible and
   link them (Part 3), since both are too large for git.
