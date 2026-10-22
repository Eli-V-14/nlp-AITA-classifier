"""Explore the r/AmItheAsshole COMMENTS dump.

Streams the comments JSONL and reports volume, score stats, and how often a
canonical verdict code (YTA/NTA/ESH/NAH/INFO) appears in comment bodies. Also
tallies the verdict codes found in *top-level* comments (parent_id starts with
't3_'), since those are the ones the community uses to decide a post's verdict.

Usage:
    python eda/explore_comments.py "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl"
    python eda/explore_comments.py <path> --limit 200000
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

from common import iter_records, verdict_in_comment, ym

from explore_posts import pct, safe_div, write_counter_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="path to r_AmItheAsshole_comments.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--outdir", default="eda/out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    total = 0
    removed = 0
    top_level = 0
    score_sum = 0
    body_len_sum = 0
    verdict_any = Counter()        # verdict code found anywhere in body
    verdict_toplevel = Counter()   # verdict code in top-level comments
    by_month = Counter()

    for rec in iter_records(args.path, args.limit):
        total += 1
        body = rec.get("body")
        if body in ("[deleted]", "[removed]", None, ""):
            removed += 1
            body = ""
        body_len_sum += len(body)
        score_sum += rec.get("score") or 0
        by_month[ym(rec.get("created_utc"))] += 1

        is_top = str(rec.get("parent_id", "")).startswith("t3_")
        if is_top:
            top_level += 1

        v = verdict_in_comment(body)
        if v:
            verdict_any[v] += 1
            if is_top:
                verdict_toplevel[v] += 1

        if total % 200000 == 0:
            print(f"  ...{total:,} comments read", flush=True)

    print("\n" + "=" * 60)
    print("r/AmItheAsshole COMMENTS - summary")
    print("=" * 60)
    print(f"Total comments read     : {total:,}")
    print(f"Removed/deleted/empty   : {removed:,} ({pct(removed, total)})")
    print(f"Top-level comments      : {top_level:,} ({pct(top_level, total)})")
    print(f"Avg score               : {safe_div(score_sum, total):.1f}")
    print(f"Avg body length         : {safe_div(body_len_sum, total):.0f} chars")
    print(f"Comments with a verdict : {sum(verdict_any.values()):,} "
          f"({pct(sum(verdict_any.values()), total)})")

    print("\nVerdict code mentions (anywhere in body)")
    print("-" * 60)
    tot_any = sum(verdict_any.values()) or 1
    for code, n in verdict_any.most_common():
        print(f"  {code:6} {n:>10,}  {pct(n, tot_any):>6}")

    print("\nVerdict code mentions in TOP-LEVEL comments")
    print("-" * 60)
    tot_top = sum(verdict_toplevel.values()) or 1
    for code, n in verdict_toplevel.most_common():
        print(f"  {code:6} {n:>10,}  {pct(n, tot_top):>6}")

    write_counter_csv(os.path.join(args.outdir, "comment_verdicts_any.csv"),
                      ["verdict", "count"], verdict_any.most_common())
    write_counter_csv(os.path.join(args.outdir, "comment_verdicts_toplevel.csv"),
                      ["verdict", "count"], verdict_toplevel.most_common())
    write_counter_csv(os.path.join(args.outdir, "comments_by_month.csv"),
                      ["month", "count"], sorted(by_month.items()))

    print(f"\nCSVs written to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
