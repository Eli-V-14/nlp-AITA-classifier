"""
Simple baselines, so the fine-tuned model has something to be measured against.

This is roadmap stage 4: a majority-class predictor plus TF-IDF with Logistic
Regression and Naive Bayes (scikit-learn). They're scored with the same metrics as
the transformer so the comparison is fair, and there's a 5-fold cross-validated
macro-F1 on the train split since the roadmap calls for cross-validation.

    python model/baselines.py --max-train 60000 --max-test 8000
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

from common import DEFAULT_DATA, OUT_DIR, load_split


def scores(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1],
                                                  average=None, zero_division=0)
    _, _, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                   zero_division=0)
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1m, 4),
            "NTA": {"precision": round(p[0], 4), "recall": round(r[0], 4), "f1": round(f1[0], 4)},
            "YTA": {"precision": round(p[1], 4), "recall": round(r[1], 4), "f1": round(f1[1], 4)}}


def tfidf_pipe(clf):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                               max_features=50000, sublinear_tf=True)),
                     ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--max-test", type=int, default=8000)
    args = ap.parse_args()

    print("loading data...")
    xtr, ytr = load_split(args.data, "train", args.max_train)
    xte, yte = load_split(args.data, "test", args.max_test)
    print(f"  train={len(xtr)}  test={len(yte)}")

    report = {"train_n": len(xtr), "test_n": len(xte), "models": {}}

    # majority-class baseline (predict NTA=0)
    report["models"]["majority_NTA"] = scores(yte, [0] * len(yte))

    # TF-IDF + Logistic Regression (class-weight balanced for the imbalance)
    lr = tfidf_pipe(LogisticRegression(max_iter=1000, class_weight="balanced"))
    lr.fit(xtr, ytr)
    report["models"]["tfidf_logreg"] = scores(yte, lr.predict(xte))

    # TF-IDF + Multinomial Naive Bayes
    nb = tfidf_pipe(MultinomialNB())
    nb.fit(xtr, ytr)
    report["models"]["tfidf_nb"] = scores(yte, nb.predict(xte))

    # 5-fold cross-validated macro-F1 on train (roadmap names cross-validation)
    print("5-fold CV (tfidf+logreg, macro-F1) on a subset...")
    sub = min(len(xtr), 20000)
    cv = cross_val_score(tfidf_pipe(LogisticRegression(max_iter=1000, class_weight="balanced")),
                         xtr[:sub], ytr[:sub], cv=5, scoring="f1_macro")
    report["cv_tfidf_logreg_macro_f1"] = {"folds": [round(float(x), 4) for x in cv],
                                          "mean": round(float(cv.mean()), 4),
                                          "std": round(float(cv.std()), 4), "n": sub}

    print("\n=== BASELINES (test set) ===")
    for name, s in report["models"].items():
        print(f"  {name:16} acc={s['accuracy']:.4f}  macroF1={s['macro_f1']:.4f}  "
              f"YTA-F1={s['YTA']['f1']:.4f}")
    print(f"  CV macro-F1 (logreg): {report['cv_tfidf_logreg_macro_f1']['mean']:.4f} "
          f"+/- {report['cv_tfidf_logreg_macro_f1']['std']:.4f}")

    with open(os.path.join(OUT_DIR, "baseline_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("saved -> out/baseline_report.json")


if __name__ == "__main__":
    main()
