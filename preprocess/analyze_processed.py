"""Item 3: cleaned word count + earliest date in each split, from posts.jsonl."""
import json, sys, datetime as dt

path = sys.argv[1]
total_words = 0
n = 0
first = {}  # split -> min created_utc
per_split_words = {}
with open(path, "rb") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        w = len((r.get("text") or "").split())
        total_words += w
        s = r.get("split")
        per_split_words[s] = per_split_words.get(s, 0) + w
        cu = r.get("created_utc")
        if cu is not None and (s not in first or cu < first[s]):
            first[s] = cu


def iso(x):
    return dt.datetime.fromtimestamp(int(x), dt.timezone.utc).strftime("%Y-%m-%d") if x is not None else None


print(json.dumps({
    "posts": n,
    "cleaned_text_words_total": total_words,
    "cleaned_words_per_split": per_split_words,
    "first_date_train": iso(first.get("train")),
    "first_date_val": iso(first.get("val")),
    "first_date_test": iso(first.get("test")),
}, indent=2))
