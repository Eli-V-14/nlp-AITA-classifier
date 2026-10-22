"""
run_all.py - one command to produce every required result.

Runs, in order:
  1. clean_aita.py        -> processed/posts.jsonl, explanations.jsonl, stats.json
  2. raw_posts_scan.py    -> posts_scan.json + sample_post.jsonl (raw 2019+ EDIT post)
  3. raw_comments_scan.py -> comments_scan.json + sample_comments.jsonl
                             (link-id taken automatically from the sample post)
  4. analyze_processed.py -> cleaned word count + first date per split

Everything is written under --out, and a combined summary is printed at the end.

Usage (defaults assume the raw dumps sit in "D:/MSOE/CSC/NLP Data"):
    python preprocess/run_all.py
    python preprocess/run_all.py --data-dir "D:/MSOE/CSC/NLP Data" --out "D:/MSOE/CSC/NLP Data/processed"
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *script_args, capture=False):
    cmd = [sys.executable, "-u", os.path.join(HERE, script), *map(str, script_args)]
    print("\n$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    if capture:
        out = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(out.stdout)
        return out.stdout
    subprocess.run(cmd, check=True)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="D:/MSOE/CSC/NLP Data",
                    help="folder containing the two raw *.jsonl dumps")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <data-dir>/processed)")
    args = ap.parse_args()

    posts = os.path.join(args.data_dir, "r_AmItheAsshole_posts.jsonl")
    comments = os.path.join(args.data_dir, "r_AmItheAsshole_comments.jsonl")
    out = args.out or os.path.join(args.data_dir, "processed")
    os.makedirs(out, exist_ok=True)

    for p in (posts, comments):
        if not os.path.exists(p):
            sys.exit(f"Missing input file: {p}")

    posts_scan = os.path.join(out, "posts_scan.json")
    comments_scan = os.path.join(out, "comments_scan.json")
    sample_post = os.path.join(out, "sample_post.jsonl")
    sample_comments = os.path.join(out, "sample_comments.jsonl")

    # 1. preprocessing (writes stats.json / posts.jsonl / explanations.jsonl)
    print("\n" + "#" * 70 + "\n# 1/4  clean_aita.py (preprocessing)\n" + "#" * 70)
    run("clean_aita.py", "--posts", posts, "--comments", comments, "--out", out)

    # 2. raw post counts + the sample post
    print("\n" + "#" * 70 + "\n# 2/4  raw_posts_scan.py\n" + "#" * 70)
    run("raw_posts_scan.py", "--posts", posts,
        "--out-sample", sample_post, "--out-stats", posts_scan)

    with open(posts_scan, encoding="utf-8") as f:
        pscan = json.load(f)
    sample_id = pscan.get("sample_id")
    link_id = f"t3_{sample_id}" if sample_id else "t3_none"

    # 3. raw comment counts + that post's comments
    print("\n" + "#" * 70 + f"\n# 3/4  raw_comments_scan.py (link-id {link_id})\n" + "#" * 70)
    run("raw_comments_scan.py", "--comments", comments, "--link-id", link_id,
        "--out-sample", sample_comments, "--out-stats", comments_scan)

    # 4. cleaned word count + split dates
    print("\n" + "#" * 70 + "\n# 4/4  analyze_processed.py\n" + "#" * 70)
    analysis_out = run("analyze_processed.py", os.path.join(out, "posts.jsonl"), capture=True)

    # ---- combined summary ----
    with open(os.path.join(out, "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    with open(comments_scan, encoding="utf-8") as f:
        cscan = json.load(f)
    analysis = json.loads(analysis_out)

    print("\n" + "=" * 70)
    print("ALL RESULTS")
    print("=" * 70)
    print("\n[1] Raw counts")
    print(f"    posts lines      : {pscan['posts_lines']:,}")
    print(f"    comments lines   : {cscan['comments_lines']:,}")
    print(f"    raw post words   : {pscan['posts_selftext_words']:,}")
    print(f"    posts dates      : {pscan['posts_created_min']} -> {pscan['posts_created_max']}")
    print(f"    comments dates   : {cscan['comments_created_min']} -> {cscan['comments_created_max']}")
    print("\n[2] stats.json")
    for k, v in stats.items():
        print(f"    {k:32} {v:,}" if isinstance(v, int) else f"    {k:32} {v}")
    print("\n[3] Cleaned word count + split dates")
    print(f"    cleaned text words : {analysis['cleaned_text_words_total']:,}")
    print(f"    first train date   : {analysis['first_date_train']}")
    print(f"    first val date     : {analysis['first_date_val']}")
    print(f"    first test date    : {analysis['first_date_test']}")
    print("\n[4] Sample (item 4)")
    print(f"    post id            : {sample_id}  ({pscan['sample_created_date']})")
    print(f"    sample_post.jsonl  : {sample_post}")
    print(f"    sample_comments    : {sample_comments}  ({cscan['sample_comment_lines']} lines)")
    print("\n[5] Processed outputs in:", out)
    print("    posts.jsonl, explanations.jsonl, stats.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
