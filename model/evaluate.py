"""
evaluate.py — Part 2 initial evaluation of the fine-tuned AITA classifier.

Reports on the held-out TEST split (never seen in training):
  * NATURAL test set: accuracy, per-class P/R/F1, macro-F1, confusion matrix
  * REBALANCED test set (50/50 by downsampling NTA) — the roadmap's eval plan
  * majority-class baseline for comparison
Then runs the hand-built eval_cases.jsonl and prints each prediction with
success/failure, so the report can quote concrete examples.

Usage:
    python model/evaluate.py --max-test 8000
"""
from __future__ import annotations

import argparse, json, os, random
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

from common import DEFAULT_DATA, MODEL_DIR, OUT_DIR, ID2LABEL, load_split
from infer import load_model

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def predict_batch(tok, model, texts, max_len=256, bs=32):
    preds = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        preds.extend(model(**enc).logits.argmax(-1).tolist())
    return preds


def scores(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                  labels=[0, 1], zero_division=0)
    _, _, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                   zero_division=0)
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1m, 4),
            "NTA": {"precision": round(p[0], 4), "recall": round(r[0], 4), "f1": round(f1[0], 4)},
            "YTA": {"precision": round(p[1], 4), "recall": round(r[1], 4), "f1": round(f1[1], 4)}}


def rebalance(texts, labels, seed=13):
    """Downsample the majority class to a 50/50 set."""
    rng = random.Random(seed)
    idx0 = [i for i, y in enumerate(labels) if y == 0]
    idx1 = [i for i, y in enumerate(labels) if y == 1]
    n = min(len(idx0), len(idx1))
    keep = rng.sample(idx0, n) + rng.sample(idx1, n)
    rng.shuffle(keep)
    return [texts[i] for i in keep], [labels[i] for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--max-test", type=int, default=8000)
    ap.add_argument("--cases", default=os.path.join(HERE, "eval_cases.jsonl"))
    args = ap.parse_args()

    tok, model = load_model(args.model)

    print(f"loading test split (max {args.max_test})...")
    tx, ty = load_split(args.data, "test", args.max_test)
    print(f"  test n={len(tx)}")
    pred = predict_batch(tok, model, tx)

    natural = scores(ty, pred)
    cm = confusion_matrix(ty, pred, labels=[0, 1]).tolist()
    majority = scores(ty, [0] * len(ty))

    # rebalanced test set
    rtx, rty = rebalance(tx, ty)
    rpred = predict_batch(tok, model, rtx)
    rebalanced = scores(rty, rpred)

    print("\n=== TEST SET (natural) ===")
    print("fine-tuned:", json.dumps(natural))
    print("majority (all-NTA):", json.dumps(majority))
    print(f"confusion [true][pred] {['NTA','YTA']}: {cm}")
    print(f"\n=== TEST SET (rebalanced 50/50, n={len(rty)}) ===")
    print("fine-tuned:", json.dumps(rebalanced))

    # hand-built cases
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]
    cpred = predict_batch(tok, model, [c["text"] for c in cases])
    correct = 0
    rows = []
    print("\n=== HAND-BUILT CASES ===")
    for c, pi in zip(cases, cpred):
        pl = ID2LABEL[pi]
        ok = (pl == c["expected"])
        correct += ok
        print(f"  [{'OK ' if ok else 'XX '}] pred={pl} exp={c['expected']}  {c['text'][:65]}...")
        rows.append({"text": c["text"], "expected": c["expected"], "predicted": pl,
                     "correct": ok, "note": c.get("note")})
    case_acc = round(correct / len(cases), 4)
    print(f"\ncases correct: {correct}/{len(cases)}  ({case_acc})")

    report = {"test_n": len(tx),
              "test_natural_finetuned": natural,
              "test_natural_majority": majority,
              "confusion_matrix_true_pred": cm, "labels": ["NTA", "YTA"],
              "test_rebalanced_finetuned": rebalanced, "rebalanced_n": len(rty),
              "hand_cases_accuracy": case_acc, "hand_cases": rows}
    with open(os.path.join(OUT_DIR, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nsaved -> out/eval_report.json")


if __name__ == "__main__":
    main()
