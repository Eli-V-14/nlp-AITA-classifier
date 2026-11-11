# nlp-AITA-classifier

A text classifier that predicts whether the author of an interpersonal conflict story is in the wrong. It is trained on Reddit r/AmItheAsshole posts and the community's final verdict. We fine-tune `roberta-base` (all weights, no LoRA) to predict **NTA** (Not the A-hole) or **YTA** (You're the A-hole).

This is the prototype milestone. It is a working end-to-end system (data, training, inference, evaluation) with baseline results, not a finished model. It only predicts the verdict. The explanation generator (BART) is not built yet.

## Repository layout

```
preprocess/   cleaning script and the pipeline that builds the labeled dataset (standard library only)
eda/          read-only scripts that explore the raw and processed data
model/        training, baselines, inference, evaluation, and the hand-built eval cases
out/          dataset samples and metric reports (the saved model and large files are not in git)
requirements.txt   Python packages for the model code
```

## Requirements

- **Python 3.12** for the model code. PyTorch has no wheels for Python 3.14 yet, so 3.14 will not work.
- The `preprocess/` and `eda/` scripts use only the standard library and run on any Python 3.
- A CUDA GPU is recommended (we used a consumer NVIDIA GPU with 12 GB of VRAM). CPU works but is slow, and training on CPU is not practical.
- Packages are in `requirements.txt`: torch, transformers, datasets, accelerate, scikit-learn, numpy.
- No API keys or outside services are needed. `roberta-base` is downloaded from Hugging Face the first time you train.

## Setup

Windows:

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

macOS or Linux: use `python3.12 -m venv .venv` and replace `.venv/Scripts/python.exe` with `.venv/bin/python` in every command in this README. Install the default `torch` build from pytorch.org instead of the cu124 one if you have no NVIDIA GPU.

## Getting the model and the data

Two files are too large for git. Download them and put them where the commands can find them.

- **Saved model** (about 499 MB): download from `<model link>` and unzip it so the files sit in `out/model/`. With this you can run inference and evaluation without training anything.
- **Dataset** `posts.jsonl` (about 757 MB, one cleaned labeled post per line with a train, val, or test split): download from `<dataset link>` and pass its location to the scripts with `--data path/to/posts.jsonl`.

The dataset is only needed for training, baselines, and evaluation on the test split. Inference on new text needs only the saved model.

To rebuild the dataset from scratch, download the r/AmItheAsshole posts and comments with the Arctic Shift download tool (https://arctic-shift.photon-reddit.com/download-tool), then run:

```bash
python preprocess/run_all.py --data-dir "<folder with the raw .jsonl files>"
```

or just the cleaning step:

```bash
python preprocess/clean_aita.py --posts <raw posts .jsonl> --comments <raw comments .jsonl> --out <output folder>
```

See `preprocess/README.md` and `preprocess/REPORT.md` for the dataset statistics.

## Execution

**Predict on any new post** (loads the saved model, nothing is retrained):

```bash
.venv/Scripts/python.exe model/infer.py --text "AITA for skipping my friend's wedding because I can't afford the trip?"
.venv/Scripts/python.exe model/infer.py --file my_post.txt
echo "WIBTA if I reported my coworker for taking credit for my work?" | .venv/Scripts/python.exe model/infer.py
```

You can also open `test_model.ipynb` with the `.venv` kernel if you would rather click through predictions in a notebook. It has example posts, a cell to type your own, and a batch run over the hand-built cases.

**Evaluate** (test split metrics, rebalanced test set, and the 13 hand-built cases in `model/eval_cases.jsonl`):

```bash
.venv/Scripts/python.exe model/evaluate.py --data path/to/posts.jsonl --max-test 8000
```

**Baselines** (majority class, TF-IDF with logistic regression, TF-IDF with naive Bayes, and 5-fold cross-validation):

```bash
.venv/Scripts/python.exe model/baselines.py --data path/to/posts.jsonl --max-train 60000 --max-test 8000
```

**Train** (only needed if you do not download the saved model; it saves to `out/model/`):

```bash
.venv/Scripts/python.exe model/train.py --data path/to/posts.jsonl --max-train 60000 --max-eval 8000 --epochs 3
```

Training settings are `roberta-base`, full fine-tune, a random 60,000 post subset of the train split, 3 epochs, batch size 32, learning rate 2e-5, max length 256 tokens, and class weights of 0.670 for NTA and 1.969 for YTA. Add `--max-train 500 --max-eval 200 --epochs 1` for a quick smoke test.

## Example inputs and outputs

Input:

```bash
python model/infer.py --text "AITA for refusing to give up my paid-for airplane seat so a family could sit together? I booked and paid extra for a window seat months ago. At boarding a mom asked me to move to a middle seat in the back so her and her kids could sit together. I said no and kept my seat."
```

Output:

```
Verdict: NTA   (you are NOT the asshole)
Probabilities: {'NTA': ..., 'YTA': ...}

Note: this reflects how the r/AmItheAsshole community would likely vote, not an objective or moral judgement of the situation.
```

Input:

```bash
python model/infer.py --text "AITA for eating the lunch my roommate labeled with her name in the fridge because I was hungry and didn't feel like going to the store?"
```

Output:

```
Verdict: YTA   (YOU are the asshole)
Probabilities: {'NTA': ..., 'YTA': ...}

Note: this reflects how the r/AmItheAsshole community would likely vote, not an objective or moral judgement of the situation.
```

Both of these were predicted correctly in our evaluation. The exact probabilities will vary slightly between machines.

**How to read the output.** The verdict is the class with the higher probability. The two probabilities add up to 1 and show how confident the model is. A split close to 0.5 and 0.5 means the model is unsure, which is common for ambiguous stories. The note at the bottom is a reminder that the label is a prediction of the Reddit vote.

## Results

Natural test set (8,000 random posts from the test split):

| Model | Accuracy | Macro-F1 | YTA-F1 |
|---|---:|---:|---:|
| Majority (all NTA) | 0.815 | 0.449 | 0.000 |
| TF-IDF + Naive Bayes | 0.815 | 0.452 | 0.005 |
| TF-IDF + Logistic Regression | 0.703 | 0.584 | 0.362 |
| roberta-base (fine-tuned) | 0.735 | 0.622 | 0.416 |

On a rebalanced 50/50 test set (2,966 posts) the fine-tuned model gets 0.642 accuracy and 0.635 macro-F1. On the 13 hand-built cases it gets 12 correct. The full numbers are in `out/eval_report.json` and `out/baseline_report.json`. Macro-F1 is our main metric because the majority baseline wins on accuracy only by predicting NTA for almost everything.

## Known limitations

- **Verdict only.** The explanation generator is not built yet. The system returns a verdict and probabilities, not a written explanation.
- **Two labels.** The subreddit's five verdicts are folded into NTA and YTA (NAH goes to NTA, ESH goes to YTA, and INFO and Too Close to Call posts are dropped).
- **Class imbalance.** About 76% of the posts are NTA. Training uses class weights, but the model still leans NTA on unclear input and its YTA precision is low (0.35).
- **Noisy labels.** The label is a Reddit vote. It reflects that community's norms and not an objective judgement.
- **Long posts are cut.** Only the first 256 tokens of a post are used, so the end of a long story is ignored.
- **Subset training.** The model is trained on 60,000 of the 298,672 training posts, with fixed settings and no tuning.
- **Test sample.** Evaluation uses a random 8,000 post sample of the 37,334 post test split, and the hand-built cases are only 13 examples.
- **Comment gap.** The comments we downloaded stop in March 2019 while the posts run to June 2022. The validation and test posts have labels but no comment-based vote counts or explanation targets.
- **Explicit language.** The data comes from Reddit and contains explicit language.
- **System paths.** The setup commands assume Windows; on other systems use the path changes described in Setup. Pass the dataset location with `--data`.

## Reproducing the numbers

`model/evaluate.py` writes `out/eval_report.json` and `model/baselines.py` writes `out/baseline_report.json`. Data sampling uses a fixed random seed (13), so the same posts are picked each run. The dataset statistics are in `preprocess/REPORT.md`.