"""
Build a small, real sample we can use to rebuild processed examples by hand.

Picks N raw posts (5 by default) from 2019 or later that have an EDIT the cleaner
would cut, a flair that maps to a label, and at least one comment. For each of
those posts it pulls the top K top-level comments by score (2 by default), found
by post id, and writes out their raw comment lines.

It writes the lines exactly as they appear in the dump, so they can be fed back
through the cleaner:
    --out-posts     one raw post line per sample post
    --out-comments  the top-K raw comment lines per sample post

Both default to the repo's out/ folder (the files are small and meant to be
committed); pass --out-* to send them somewhere else.

    python preprocess/build_samples.py \
        --posts "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl" \
        --comments "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl"
"""
import argparse, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "out")  # <repo>/out

EDIT_IGNORE_FRACTION = 0.30
EDIT_LINE_RE = re.compile(
    r"^[ \t*_>~(\[]*(?:edit(?:ed)?|updated?|eta)\b[ \t]*\d*[ \t]*"
    r"(?:[:\-/\u2013\u2014]|\(|$|\bto\b|\bafter\b|\bfor\b)", re.I | re.M)
EDIT_INLINE_RE = re.compile(r"\b(?:EDIT|UPDATE|ETA)\s*\d*\s*:")
FLAIR_MAP = [
    (re.compile(r"^not the a[- ]?(hole|sshole)"), "NTA"),
    (re.compile(r"^no a[- ]?holes? here"), "NTA"),
    (re.compile(r"^(you'?re the a[- ]?hole|asshole)"), "YTA"),
    (re.compile(r"^everyone sucks"), "YTA"),
]
REMOVED = {"", "[removed]", "[deleted]"}


def map_flair(flair):
    if not flair:
        return None
    f = flair.strip().lower().replace("\u2019", "'")
    for pat, lab in FLAIR_MAP:
        if pat.search(f):
            return lab
    return None


def edit_is_cut(text):
    starts = [m.start() for m in EDIT_LINE_RE.finditer(text)]
    starts += [m.start() for m in EDIT_INLINE_RE.finditer(text)]
    return any(s >= EDIT_IGNORE_FRACTION * len(text) for s in starts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--comments", required=True)
    ap.add_argument("--out-posts", default=os.path.join(OUT_DIR, "sample_posts.jsonl"))
    ap.add_argument("--out-comments", default=os.path.join(OUT_DIR, "sample_comments.jsonl"))
    ap.add_argument("--n-posts", type=int, default=5)
    ap.add_argument("--top", type=int, default=2, help="top comments per post, by score")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_posts), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_comments), exist_ok=True)

    # ---- pass 1: pick N qualifying posts (file is chronological, so stop early) ----
    chosen = []  # list of (id, raw_line, flair_label)
    with open(args.posts, "rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            line = raw.decode("utf-8", "replace")
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cu = rec.get("created_utc")
            try:
                cu = int(cu)
            except (TypeError, ValueError):
                continue
            if cu < 1546300800:  # before 2019-01-01 UTC
                continue
            st = (rec.get("selftext") or "").strip()
            if (st not in REMOVED and not rec.get("removed_by_category")
                    and map_flair(rec.get("link_flair_text"))
                    and (rec.get("num_comments") or 0) > 0
                    and edit_is_cut(st)):
                chosen.append((rec.get("id"), line.rstrip("\n"),
                               map_flair(rec.get("link_flair_text"))))
                if len(chosen) >= args.n_posts:
                    break

    ids = [c[0] for c in chosen]
    link_ids = {f"t3_{i}" for i in ids}
    print(f"chosen posts: {ids}")

    # ---- pass 2: top-level comments per chosen post, keep top-K by score ----
    buckets = {i: [] for i in ids}  # post id -> list of (score, raw_line)
    with open(args.comments, "rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            line = raw.decode("utf-8", "replace")
            try:
                rec = json.loads(line)
            except Exception:
                continue
            lid = rec.get("link_id")
            if lid not in link_ids:
                continue
            if not str(rec.get("parent_id", "")).startswith("t3_"):
                continue  # top-level only
            body = (rec.get("body") or "").strip()
            if body in REMOVED:
                continue
            buckets[lid[3:]].append((rec.get("score") or 0, line.rstrip("\n")))

    # ---- write ----
    with open(args.out_posts, "w", encoding="utf-8") as f:
        for _, raw_line, _ in chosen:
            f.write(raw_line + "\n")

    total_comments = 0
    with open(args.out_comments, "w", encoding="utf-8") as f:
        for pid in ids:
            top = sorted(buckets[pid], key=lambda t: t[0], reverse=True)[: args.top]
            for score, raw_line in top:
                f.write(raw_line + "\n")
                total_comments += 1
            print(f"  {pid}: {len(buckets[pid])} top-level comments -> kept {len(top)}")

    print(f"\nWrote {len(chosen)} posts and {total_comments} comments.")


if __name__ == "__main__":
    main()
