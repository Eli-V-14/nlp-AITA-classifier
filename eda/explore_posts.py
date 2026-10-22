"""Explore the r/AmItheAsshole POSTS dump.

Streams the posts JSONL and reports the things that matter for a verdict
classifier: how many posts there are, the verdict (label) distribution, how
much text each post has, how many are deleted/removed, and how the volume and
label mix change over time. Writes a plain-text summary to stdout and a few
CSVs you can plot (or feed to eda/plot.py).

Usage:
    python eda/explore_posts.py "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl"
    python eda/explore_posts.py <path> --limit 100000      # quick sample
    python eda/explore_posts.py <path> --outdir eda/out     # where CSVs go
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

from common import (
    VERDICT_LABELS,
    is_removed,
    iter_records,
    normalize_verdict,
    ym,
)

# selftext length buckets (characters)
LEN_BUCKETS = [0, 1, 250, 500, 1000, 2000, 4000, 8000, float("inf")]
LEN_LABELS = ["empty", "1-249", "250-499", "500-999", "1000-1999",
              "2000-3999", "4000-7999", "8000+"]


def bucket_len(n: int) -> str:
    for i in range(len(LEN_BUCKETS) - 1):
        if LEN_BUCKETS[i] <= n < LEN_BUCKETS[i + 1]:
            return LEN_LABELS[i]
    return LEN_LABELS[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="path to r_AmItheAsshole_posts.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="only read N records")
    ap.add_argument("--outdir", default="eda/out", help="directory for CSV output")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    total = 0
    verdicts = Counter()
    len_buckets = Counter()
    removed = 0
    over_18 = 0
    by_month = Counter()                       # YYYY-MM -> count
    verdict_by_month = {}                       # YYYY-MM -> Counter(verdict)
    score_sum = 0
    comments_sum = 0
    len_sum = 0
    len_judged_sum = 0                          # text length of judged posts only
    judged = 0                                  # posts with a real verdict

    for rec in iter_records(args.path, args.limit):
        total += 1
        v = normalize_verdict(rec.get("link_flair_text"))
        verdicts[v] += 1

        selftext = rec.get("selftext")
        if is_removed(selftext):
            removed += 1
            tlen = 0
        else:
            tlen = len(selftext)
        len_buckets[bucket_len(tlen)] += 1
        len_sum += tlen

        if rec.get("over_18"):
            over_18 += 1

        score_sum += rec.get("score") or 0
        comments_sum += rec.get("num_comments") or 0

        m = ym(rec.get("created_utc"))
        by_month[m] += 1
        verdict_by_month.setdefault(m, Counter())[v] += 1

        if v in ("NTA", "YTA", "ESH", "NAH", "INFO"):
            judged += 1
            len_judged_sum += tlen

        if total % 100000 == 0:
            print(f"  ...{total:,} posts read", flush=True)

    # ---- text report ----
    print("\n" + "=" * 60)
    print("r/AmItheAsshole POSTS - summary")
    print("=" * 60)
    print(f"Total posts read      : {total:,}")
    print(f"Removed/deleted/empty : {removed:,} ({pct(removed, total)})")
    print(f"NSFW (over_18)        : {over_18:,} ({pct(over_18, total)})")
    print(f"Avg score             : {safe_div(score_sum, total):.1f}")
    print(f"Avg # comments        : {safe_div(comments_sum, total):.1f}")
    print(f"Avg selftext length   : {safe_div(len_sum, total):.0f} chars")
    print(f"Posts with a verdict  : {judged:,} ({pct(judged, total)})")
    print(f"Avg length (judged)   : {safe_div(len_judged_sum, judged):.0f} chars")

    print("\nVerdict distribution")
    print("-" * 60)
    for code, n in verdicts.most_common():
        label = VERDICT_LABELS.get(code, code)
        bar = "#" * int(40 * n / max(verdicts.values()))
        print(f"  {code:10} {n:>10,}  {pct(n, total):>6}  {bar}")

    print("\nVerdict distribution AMONG JUDGED posts (model label balance)")
    print("-" * 60)
    for code in ("NTA", "YTA", "ESH", "NAH", "INFO"):
        n = verdicts.get(code, 0)
        print(f"  {code:10} {n:>10,}  {pct(n, judged):>6}")

    print("\nSelftext length buckets")
    print("-" * 60)
    for lab in LEN_LABELS:
        n = len_buckets.get(lab, 0)
        print(f"  {lab:12} {n:>10,}  {pct(n, total):>6}")

    # ---- CSVs ----
    write_counter_csv(os.path.join(args.outdir, "verdict_distribution.csv"),
                      ["verdict", "label", "count"],
                      [(c, VERDICT_LABELS.get(c, c), n) for c, n in verdicts.most_common()])

    write_counter_csv(os.path.join(args.outdir, "length_buckets.csv"),
                      ["bucket", "count"],
                      [(lab, len_buckets.get(lab, 0)) for lab in LEN_LABELS])

    write_counter_csv(os.path.join(args.outdir, "posts_by_month.csv"),
                      ["month", "count"],
                      sorted(by_month.items()))

    # verdict share per month (wide table)
    codes = ["NTA", "YTA", "ESH", "NAH", "INFO", "TOO_CLOSE", "META", "UNFLAIRED"]
    with open(os.path.join(args.outdir, "verdict_by_month.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["month"] + codes)
        for m in sorted(verdict_by_month):
            row = verdict_by_month[m]
            w.writerow([m] + [row.get(c, 0) for c in codes])

    print(f"\nCSVs written to {os.path.abspath(args.outdir)}")
    print("Next: python eda/plot.py --outdir", args.outdir)


def pct(n, d):
    return f"{(100.0 * n / d):.1f}%" if d else "0.0%"


def safe_div(a, b):
    return a / b if b else 0.0


def write_counter_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    main()
