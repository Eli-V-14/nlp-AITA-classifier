"""
Predict the AITA verdict (NTA or YTA) for a new post.

Loads the saved model (no retraining) and prints the verdict, the probability for
each class, and a reminder that this is only how the Reddit community would likely
vote, not an objective call. That caveat is roadmap stage 7, and it echoes the
ethics notes in Lourie et al. and Jiang et al.

    python model/infer.py --text "AITA for eating my roommate's leftovers?"
    python model/infer.py --file some_post.txt
    echo "WIBTA if I skipped the wedding?" | python model/infer.py
"""
from __future__ import annotations

import argparse, sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common import MODEL_DIR, ID2LABEL

DISCLAIMER = ("Note: this reflects how the r/AmItheAsshole community would likely "
              "vote, not an objective or moral judgement of the situation.")


def load_model(model_dir=MODEL_DIR):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return tok, model


@torch.no_grad()
def predict(tok, model, text, max_len=256):
    enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
    pred = int(max(range(len(probs)), key=lambda i: probs[i]))
    return ID2LABEL[pred], {ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--file")
    ap.add_argument("--model", default=MODEL_DIR)
    args = ap.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        text = open(args.file, encoding="utf-8").read()
    else:
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        sys.exit("No input text provided.")

    tok, model = load_model(args.model)
    label, probs = predict(tok, model, text)
    meaning = "you are NOT the asshole" if label == "NTA" else "YOU are the asshole"
    print(f"\nVerdict: {label}   ({meaning})")
    print(f"Probabilities: {probs}")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
