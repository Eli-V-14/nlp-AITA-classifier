"""Read the actual TEXT of the collected data (not just aggregate stats).

Browse post bodies with their community verdict, filter them, look one up by
id, or pull the comments attached to a post. Streams the files, so it works on
the full multi-GB dumps. Stdlib only.

Examples
--------
# Show 5 posts (title + verdict + body, truncated)
python eda/read_text.py posts "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl"

# Only YTA verdicts, print the FULL body, 3 of them
python eda/read_text.py posts <posts.jsonl> --verdict YTA --full --n 3

# Posts whose text mentions a word (case-insensitive)
python eda/read_text.py posts <posts.jsonl> --contains "wedding" --n 10

# One specific post by id (the base36 id, e.g. 1fy0bx)
python eda/read_text.py posts <posts.jsonl> --id 1fy0bx --full

# The comments under one post (use its t3_ name or bare id)
python eda/read_text.py comments <comments.jsonl> --link-id t3_1fy0bx --n 20
"""
from __future__ import annotations

import argparse
import datetime as dt
import textwrap

from common import VERDICT_LABELS, iter_records, normalize_verdict, verdict_in_comment

RULE = "=" * 78
THIN = "-" * 78


def when(created_utc) -> str:
    try:
        return dt.datetime.fromtimestamp(int(created_utc), dt.timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return "????-??-??"


def show_body(text: str, full: bool, width: int = 76) -> None:
    text = text or ""
    if not full and len(text) > 1200:
        text = text[:1200] + "\n   [... truncated; use --full to see all ...]"
    for para in text.split("\n"):
        if para.strip():
            print(textwrap.fill(para, width=width))
        else:
            print()


def cmd_posts(args) -> None:
    want = args.verdict.upper() if args.verdict else None
    needle = args.contains.lower() if args.contains else None
    shown = 0

    for rec in iter_records(args.path):
        if args.id and rec.get("id") != args.id:
            continue

        v = normalize_verdict(rec.get("link_flair_text"))
        if want and v != want:
            continue

        selftext = rec.get("selftext") or ""
        if selftext in ("[deleted]", "[removed]"):
            if args.id:
                print(f"(post {args.id} body was {selftext})")
            continue
        if args.min_len and len(selftext) < args.min_len:
            continue
        if needle and needle not in selftext.lower() and needle not in (rec.get("title") or "").lower():
            continue

        print("\n" + RULE)
        print(f"[{v}] {VERDICT_LABELS.get(v, v)}   "
              f"score={rec.get('score')}  comments={rec.get('num_comments')}  "
              f"{when(rec.get('created_utc'))}  id={rec.get('id')}")
        print(f"TITLE: {rec.get('title')}")
        print(THIN)
        show_body(selftext, args.full)
        shown += 1

        if args.id:
            break
        if shown >= args.n:
            break

    if shown == 0:
        print("No matching posts found.")
    else:
        print("\n" + RULE)
        print(f"Showed {shown} post(s).")


def _passes_comment_filters(rec, args):
    body = rec.get("body") or ""
    if body in ("[deleted]", "[removed]", ""):
        return None
    if args.top_level and not str(rec.get("parent_id", "")).startswith("t3_"):
        return None
    if args.verdicts_only and not verdict_in_comment(body):
        return None
    return body


def _print_comment(rec, body, full):
    v = verdict_in_comment(body)
    tag = f"[{v}] " if v else ""
    top = "TOP" if str(rec.get("parent_id", "")).startswith("t3_") else "reply"
    print("\n" + THIN)
    print(f"{tag}score={rec.get('score')}  {top}  {when(rec.get('created_utc'))}  "
          f"by {rec.get('author')}")
    print(textwrap.fill(body if full else body[:600], width=76))


def cmd_comments(args) -> None:
    link = args.link_id
    if link and not link.startswith("t3_"):
        link = "t3_" + link

    # --top N: rank a post's comments by score and keep the N highest.
    # This needs all of that post's comments in hand, so --link-id is required.
    if args.top:
        if not link:
            raise SystemExit("--top needs --link-id (it ranks one post's comments by score)")
        bucket = []
        for rec in iter_records(args.path):
            if rec.get("link_id") != link:
                continue
            body = _passes_comment_filters(rec, args)
            if body is None:
                continue
            bucket.append((rec.get("score") or 0, rec, body))
        bucket.sort(key=lambda t: t[0], reverse=True)
        kept = bucket[: args.top]
        for _, rec, body in kept:
            _print_comment(rec, body, args.full)
        print("\n" + RULE)
        print(f"Top {len(kept)} of {len(bucket)} comment(s) on {link} by score."
              if kept else f"No matching comments found on {link}.")
        return

    # default: stream in file order, stop after --n
    shown = 0
    for rec in iter_records(args.path):
        if link and rec.get("link_id") != link:
            continue
        body = _passes_comment_filters(rec, args)
        if body is None:
            continue
        _print_comment(rec, body, args.full)
        shown += 1
        if shown >= args.n:
            break

    print("\n" + RULE)
    print(f"Showed {shown} comment(s)." if shown else "No matching comments found.")


def _clip(s, width):
    s = " ".join(str(s or "").split())  # collapse newlines/tabs to one line
    return s if len(s) <= width else s[: width - 3] + "..."


def cmd_rows(args) -> None:
    """Print one compact row per record (a scannable table), posts or comments."""
    want = args.verdict.upper() if args.verdict else None
    needle = args.contains.lower() if args.contains else None

    import csv as _csv
    import sys as _sys
    writer = _csv.writer(_sys.stdout) if args.csv else None

    kind = None            # decided from the first record
    header_printed = False
    shown = 0

    for rec in iter_records(args.path):
        if kind is None:
            kind = "comments" if "body" in rec and "link_flair_text" not in rec else "posts"

        if kind == "posts":
            v = normalize_verdict(rec.get("link_flair_text"))
            if want and v != want:
                continue
            selftext = rec.get("selftext") or ""
            if args.exclude_removed and selftext in ("[deleted]", "[removed]", ""):
                continue
            title = rec.get("title") or ""
            if needle and needle not in selftext.lower() and needle not in title.lower():
                continue
            row = [when(rec.get("created_utc")), v, rec.get("score"),
                   rec.get("num_comments"), len(selftext), rec.get("id"), title]
            cols = ["date", "verdict", "score", "ncmt", "len", "id", "title"]
            widths = [10, 9, 6, 5, 6, 8, 60]
        else:  # comments
            v = verdict_in_comment(rec.get("body")) or ""
            if want and v != want:
                continue
            body = rec.get("body") or ""
            if args.exclude_removed and body in ("[deleted]", "[removed]", ""):
                continue
            if needle and needle not in body.lower():
                continue
            top = "TOP" if str(rec.get("parent_id", "")).startswith("t3_") else "reply"
            row = [when(rec.get("created_utc")), v, rec.get("score"), top,
                   rec.get("author"), rec.get("link_id"), body]
            cols = ["date", "verdict", "score", "level", "author", "link_id", "body"]
            widths = [10, 7, 6, 5, 16, 11, 60]

        if writer is not None:
            if not header_printed:
                writer.writerow(cols)
                header_printed = True
            writer.writerow(row)
        else:
            if not header_printed:
                print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
                print("  ".join("-" * w for w in widths))
                header_printed = True
            print("  ".join(_clip(val, w).ljust(w) for val, w in zip(row, widths)))

        shown += 1
        if shown >= args.n:
            break

    if not args.csv:
        if shown:
            print(f"\n({shown} rows)")
        else:
            print("No matching rows found.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("rows", help="print a compact one-line-per-record table")
    r.add_argument("path")
    r.add_argument("--verdict", help="filter to one verdict code")
    r.add_argument("--contains", help="only rows whose text contains this")
    r.add_argument("--exclude-removed", action="store_true",
                   help="skip deleted/removed/empty bodies")
    r.add_argument("--n", type=int, default=50, help="how many rows")
    r.add_argument("--csv", action="store_true", help="emit CSV to stdout instead of a table")
    r.set_defaults(func=cmd_rows)

    p = sub.add_parser("posts", help="read post bodies")
    p.add_argument("path")
    p.add_argument("--verdict", help="filter to one verdict code (NTA/YTA/ESH/NAH/INFO/...)")
    p.add_argument("--contains", help="only posts whose title/body contains this text")
    p.add_argument("--min-len", type=int, default=0, help="minimum selftext length")
    p.add_argument("--id", help="show one specific post by its base36 id")
    p.add_argument("--n", type=int, default=5, help="how many to show")
    p.add_argument("--full", action="store_true", help="print full body, no truncation")
    p.set_defaults(func=cmd_posts)

    c = sub.add_parser("comments", help="read comment bodies")
    c.add_argument("path")
    c.add_argument("--link-id", help="only comments on this post (t3_xxx or bare id)")
    c.add_argument("--top", type=int, metavar="N",
                   help="show the N highest-SCORED comments for --link-id (not file order)")
    c.add_argument("--top-level", action="store_true",
                   help="only top-level comments (replies to the post itself)")
    c.add_argument("--verdicts-only", action="store_true", help="only comments containing a verdict code")
    c.add_argument("--n", type=int, default=10)
    c.add_argument("--full", action="store_true")
    c.set_defaults(func=cmd_comments)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
