"""
Fine-tune roberta-base to predict an AITA verdict (NTA or YTA).

This is a full fine-tune (all the weights, not LoRA), which is what the roadmap's
"fine-tune a RoBERTa-base encoder with fixed hyperparameters" stage asks for. The
loss is weighted by class frequency so the 76/24 imbalance doesn't just push every
prediction to NTA. The model, tokenizer, and metrics get saved to out/model/, so
you never have to retrain just to run inference.

Quick smoke test:
    python model/train.py --max-train 500 --max-eval 200 --epochs 1

Real run:
    python model/train.py --max-train 60000 --max-eval 8000 --epochs 3
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

from common import (DEFAULT_DATA, MODEL_DIR, OUT_DIR, BASE_MODEL,
                    ID2LABEL, LABEL2ID, load_split, class_weights)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    _, _, f1m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_yta": f1, "precision_yta": p, "recall_yta": r, "f1_macro": f1m}


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy for the imbalance."""
    def __init__(self, *args, class_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cw = torch.tensor(class_weight, dtype=torch.float) if class_weight else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self._cw.to(outputs.logits.device) if self._cw is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--max-eval", type=int, default=8000)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)          # full fine-tune LR
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", default=MODEL_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"cuda: {torch.cuda.is_available()}  device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    print("loading data...")
    tr_txt, tr_lab = load_split(args.data, "train", args.max_train)
    va_txt, va_lab = load_split(args.data, "val", args.max_eval)
    print(f"  train={len(tr_txt)}  val={len(va_txt)}")
    cw = class_weights(tr_lab)
    print(f"  class weights NTA/YTA: {cw[0]:.3f} / {cw[1]:.3f}")

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    dtr = Dataset.from_dict({"text": tr_txt, "labels": tr_lab}).map(
        tokenize, batched=True, remove_columns=["text"])
    dva = Dataset.from_dict({"text": va_txt, "labels": va_lab}).map(
        tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

    use_bf16 = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "trainer_tmp"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=200,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        bf16=use_bf16,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=dtr, eval_dataset=dva,
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics, class_weight=cw,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("val metrics:", json.dumps({k: round(float(v), 4) for k, v in metrics.items()
                                      if isinstance(v, (int, float))}, indent=2))

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_val_metrics.json"), "w") as f:
        json.dump({"args": vars(args), "val_metrics": {k: float(v) for k, v in metrics.items()
                   if isinstance(v, (int, float))}, "class_weights": cw}, f, indent=2)
    print("saved model ->", args.out)


if __name__ == "__main__":
    main()
