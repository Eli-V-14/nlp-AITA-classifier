"""Turn the CSVs from explore_posts.py / explore_comments.py into PNG charts.

This is the ONLY script that needs a third-party library. Install it with:
    pip install matplotlib

Usage:
    python eda/plot.py --outdir eda/out
PNGs are written next to the CSVs.
"""
from __future__ import annotations

import argparse
import csv
import os


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="eda/out")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")  # no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib not installed — run: pip install matplotlib")

    out = args.outdir

    # 1) verdict distribution bar chart
    p = os.path.join(out, "verdict_distribution.csv")
    if os.path.exists(p):
        rows = read_csv(p)[1:]
        labels = [r[1] for r in rows]
        counts = [int(r[2]) for r in rows]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels[::-1], counts[::-1], color="#4C78A8")
        ax.set_title("Verdict distribution (all posts)")
        ax.set_xlabel("posts")
        fig.tight_layout()
        save(fig, out, "verdict_distribution.png", plt)

    # 2) posts per month
    p = os.path.join(out, "posts_by_month.csv")
    if os.path.exists(p):
        rows = [r for r in read_csv(p)[1:] if r[0] != "unknown"]
        rows.sort()
        months = [r[0] for r in rows]
        counts = [int(r[1]) for r in rows]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(range(len(months)), counts, color="#E45756")
        step = max(1, len(months) // 12)
        ax.set_xticks(range(0, len(months), step))
        ax.set_xticklabels(months[::step], rotation=45, ha="right")
        ax.set_title("Posts per month")
        ax.set_ylabel("posts")
        fig.tight_layout()
        save(fig, out, "posts_by_month.png", plt)

    # 3) length buckets
    p = os.path.join(out, "length_buckets.csv")
    if os.path.exists(p):
        rows = read_csv(p)[1:]
        labels = [r[0] for r in rows]
        counts = [int(r[1]) for r in rows]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(labels, counts, color="#54A24B")
        ax.set_title("Post selftext length (characters)")
        ax.set_ylabel("posts")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        save(fig, out, "length_buckets.png", plt)

    # 4) verdict share over time (stacked, judged verdicts only)
    p = os.path.join(out, "verdict_by_month.csv")
    if os.path.exists(p):
        rows = read_csv(p)
        header = rows[0]
        data = [r for r in rows[1:] if r[0] != "unknown"]
        data.sort()
        codes = ["NTA", "YTA", "ESH", "NAH", "INFO"]
        idx = {c: header.index(c) for c in codes}
        months = [r[0] for r in data]
        # normalize to shares of judged verdicts
        shares = {c: [] for c in codes}
        for r in data:
            vals = {c: int(r[idx[c]]) for c in codes}
            tot = sum(vals.values()) or 1
            for c in codes:
                shares[c].append(100.0 * vals[c] / tot)
        fig, ax = plt.subplots(figsize=(11, 5))
        bottom = [0.0] * len(months)
        palette = ["#4C78A8", "#E45756", "#F58518", "#54A24B", "#B279A2"]
        for c, col in zip(codes, palette):
            ax.bar(range(len(months)), shares[c], bottom=bottom, label=c, color=col, width=1.0)
            bottom = [b + s for b, s in zip(bottom, shares[c])]
        step = max(1, len(months) // 12)
        ax.set_xticks(range(0, len(months), step))
        ax.set_xticklabels(months[::step], rotation=45, ha="right")
        ax.set_ylabel("% of judged posts")
        ax.set_title("Verdict mix over time")
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
        fig.tight_layout()
        save(fig, out, "verdict_by_month.png", plt)

    print("Done.")


def save(fig, out, name, plt):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
