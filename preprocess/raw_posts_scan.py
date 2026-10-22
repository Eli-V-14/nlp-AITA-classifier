"""Single streaming pass over the raw POSTS file.

Produces (item 1) line count + total raw selftext word count + date range, and
(item 4) picks one real sample post: created 2019 or later, with an EDIT the
cleaner would actually cut, a mappable verdict flair, real text, and comments.
The sample's ORIGINAL raw line is written out verbatim.
"""
import argparse, datetime as dt, json, re

EDIT_IGNORE_FRACTION = 0.30
EDIT_LINE_RE = re.compile(
    r"^[ \t*_>~(\[]*(?:edit(?:ed)?|updated?|eta)\b[ \t]*\d*[ \t]*"
    r"(?:[:\-/–—]|\(|$|\bto\b|\bafter\b|\bfor\b)", re.I | re.M)
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
    f = flair.strip().lower().replace("’", "'")
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
    ap.add_argument("--out-sample", required=True)
    ap.add_argument("--out-stats", required=True)
    args = ap.parse_args()

    lines = 0
    words = 0
    cmin = None
    cmax = None
    sample = None  # (raw_line_str, id, created_utc)

    with open(args.posts, "rb") as f:
        for raw_bytes in f:
            if not raw_bytes.strip():
                continue
            lines += 1
            line = raw_bytes.decode("utf-8", "replace")
            try:
                rec = json.loads(line)
            except Exception:
                continue
            st = rec.get("selftext") or ""
            words += len(st.split())
            cu = rec.get("created_utc")
            try:
                cu = int(cu)
                cmin = cu if cmin is None else min(cmin, cu)
                cmax = cu if cmax is None else max(cmax, cu)
            except (TypeError, ValueError):
                cu = None

            if sample is None and cu is not None and cu >= 1546300800:  # 2019-01-01 UTC
                s = st.strip()
                if (s not in REMOVED and not rec.get("removed_by_category")
                        and map_flair(rec.get("link_flair_text"))
                        and (rec.get("num_comments") or 0) > 0
                        and edit_is_cut(s)):
                    sample = (line.rstrip("\n"), rec.get("id"), cu)

            if lines % 200000 == 0:
                print(f"  ...{lines:,} posts", flush=True)

    if sample:
        with open(args.out_sample, "w", encoding="utf-8") as f:
            f.write(sample[0] + "\n")

    def iso(x):
        return dt.datetime.fromtimestamp(x, dt.timezone.utc).strftime("%Y-%m-%d") if x else None

    out = {
        "posts_lines": lines,
        "posts_selftext_words": words,
        "posts_created_min": iso(cmin),
        "posts_created_max": iso(cmax),
        "sample_id": sample[1] if sample else None,
        "sample_created_utc": sample[2] if sample else None,
        "sample_created_date": iso(sample[2]) if sample else None,
    }
    with open(args.out_stats, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
