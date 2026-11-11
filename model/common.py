"""Shared helpers for the classifier: loading the data, label maps, and paths."""
from __future__ import annotations

import json
import os
import random

# Default dataset location (the big file lives on the Passport, not in git).
DEFAULT_DATA = r"D:/MSOE/CSC/NLP Data/processed/posts.jsonl"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "out")
MODEL_DIR = os.path.join(OUT_DIR, "model")               # saved fine-tuned model
BASE_MODEL = "roberta-base"

LABEL2ID = {"NTA": 0, "YTA": 1}
ID2LABEL = {0: "NTA", 1: "YTA"}
LABEL_NAMES = ["NTA", "YTA"]


def format_text(rec: dict) -> str:
    """Model input = title + body (title carries real signal on this subreddit)."""
    title = (rec.get("title") or "").strip()
    body = (rec.get("text") or "").strip()
    return (title + "\n\n" + body).strip() if title else body


def load_split(path: str, split: str, max_n: int | None = None, seed: int = 13):
    """Stream posts.jsonl and return (texts, labels) for one split.

    If max_n is set, keep a random (stratified-ish) sample of that size. We
    reservoir-sample while streaming so we never hold the whole file in memory.
    """
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[int] = []
    seen = 0
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("split") != split:
                continue
            lab = LABEL2ID.get(rec.get("label"))
            if lab is None:
                continue
            txt = format_text(rec)
            if not txt:
                continue
            if max_n is None:
                texts.append(txt); labels.append(lab)
            else:
                # reservoir sampling
                seen += 1
                if len(texts) < max_n:
                    texts.append(txt); labels.append(lab)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < max_n:
                        texts[j] = txt; labels[j] = lab
    return texts, labels


def class_weights(labels):
    """Inverse-frequency weights for the 2 classes (handles 76/24 imbalance)."""
    import collections
    c = collections.Counter(labels)
    n = len(labels)
    return [n / (2 * c.get(i, 1)) for i in range(2)]
