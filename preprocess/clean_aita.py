"""
clean_aita.py

Cleans the AITA data downloaded with the Arctic Shift download tool.
Uses only the Python standard library.

Usage:
    python preprocess/clean_aita.py \
        --posts "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl" \
        --comments "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl" \
        --out "D:/MSOE/CSC/NLP Data/processed"

Outputs (in --out):
    posts.jsonl         one cleaned, labeled post per line, with vote counts and a time split
    explanations.jsonl  agreeing top comments used as targets for the explanation model
    stats.json          how many records were dropped at each step, and why

Note: read_jsonl opens the files in binary mode ("rb"). Windows text-mode line
iteration raises OSError 22 on the 2.9 GB comments file; reading bytes and
letting json.loads parse them is byte-for-byte identical in every result.
"""
import argparse
import hashlib
import heapq
import html
import json
import os
import re
from collections import Counter

# ---------------------------------------------------------------- settings
MIN_POST_WORDS = 50        # drops very short posts
MIN_POST_COMMENTS = 1      # a post with no comments has no votes to learn from
EDIT_IGNORE_FRACTION = 0.30  # an EDIT marker in the first 30 percent of a post is not cut
MIN_EXPL_SCORE = 5         # an explanation comment needs at least this many upvotes
MIN_EXPL_WORDS = 15
MAX_EXPL_WORDS = 200
EXPL_PER_POST = 3          # keep at most this many explanations per post
DEDUP_PREFIX_CHARS = 300
SPLIT = (0.8, 0.1, 0.1)    # train, validation, test (oldest to newest by created_utc)

# ------------------------------------------------------- step 1: flair mapping
FLAIR_MAP = [
    (re.compile(r"^not the a[- ]?(hole|sshole)"), "NTA"),
    (re.compile(r"^no a[- ]?holes? here"), "NTA"),
    (re.compile(r"^(you'?re the a[- ]?hole|asshole)"), "YTA"),
    (re.compile(r"^everyone sucks"), "YTA"),
]

def map_flair(flair):
    if not flair:
        return None
    f = flair.strip().lower().replace("’", "'")
    for pattern, label in FLAIR_MAP:
        if pattern.search(f):
            return label
    return None

# ---------------------------------------------- step 2: removing edits and tags
EDIT_LINE_RE = re.compile(
    r"^[ \t*_>~(\[]*(?:edit(?:ed)?|updated?|eta)\b[ \t]*\d*[ \t]*"
    r"(?:[:\-/–—]|\(|$|\bto\b|\bafter\b|\bfor\b)",
    re.I | re.M,
)
EDIT_INLINE_RE = re.compile(r"\b(?:EDIT|UPDATE|ETA)\s*\d*\s*:")   # uppercase, mid-line
VERDICT_TOKEN_RE = re.compile(r"\b(?:YTA|NTA|ESH|NAH|YWBTA|YWNBTA)\b")

def cut_edits(text):
    """Cut everything from the first EDIT or UPDATE marker to the end of the text."""
    starts = [m.start() for m in EDIT_LINE_RE.finditer(text)]
    starts += [m.start() for m in EDIT_INLINE_RE.finditer(text)]
    starts = [s for s in starts if s >= EDIT_IGNORE_FRACTION * len(text)]
    return text[:min(starts)] if starts else text

# ------------------------------------------------- step 3: text and title cleanup
TITLE_PREFIX_RE = re.compile(r"^\s*(\[\s*aita\s*\]|aita\s*:)\s*", re.I)

def clean_text(text):
    text = html.unescape(text).replace("​", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)   # markdown link -> link text
    text = re.sub(r"https?://\S+", "", text)                # bare urls
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)            # bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)                # italics
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paragraphs if p)

def clean_title(title):
    title = TITLE_PREFIX_RE.sub("", html.unescape(title or ""))
    return re.sub(r"\s+", " ", VERDICT_TOKEN_RE.sub("", title)).strip()

def word_count(text):
    return len(text.split())

# ------------------------------------------------------------ post processing
REMOVED_TEXT = {"", "[removed]", "[deleted]"}

def process_post(raw):
    """Returns (record, None) if kept, or (None, reason) if dropped."""
    selftext = (raw.get("selftext") or "").strip()
    if selftext in REMOVED_TEXT:
        return None, "removed_or_deleted_text"
    if raw.get("removed_by_category"):
        return None, "removed_by_moderator"
    label = map_flair(raw.get("link_flair_text"))
    if label is None:
        return None, "no_verdict_flair"
    if (raw.get("num_comments") or 0) < MIN_POST_COMMENTS:
        return None, "no_comments"
    text = clean_text(VERDICT_TOKEN_RE.sub("", cut_edits(selftext)))
    if word_count(text) < MIN_POST_WORDS:
        return None, "too_short"
    return {
        "id": raw["id"],
        "title": clean_title(raw.get("title")),
        "text": text,
        "label": label,
        "score": raw.get("score"),
        "num_comments": raw.get("num_comments"),
        "created_utc": raw.get("created_utc"),
    }, None

def dedup_key(text):
    normalized = re.sub(r"[^a-z0-9]", "", text.lower())[:DEDUP_PREFIX_CHARS]
    return hashlib.sha1(normalized.encode()).hexdigest()

# --------------------------------------------------------- comment processing
TAG_RE = re.compile(r"^[\s*_>\[(\"']*(YTA|NTA|ESH|NAH|INFO|YWBTA|YWNBTA)\b", re.I)
TAG_TO_LABEL = {"YTA": "YTA", "YWBTA": "YTA", "ESH": "YTA",
                "NTA": "NTA", "YWNBTA": "NTA", "NAH": "NTA"}   # INFO has no side
PHRASE_YTA = re.compile(
    r"^\W*(?:(?:yep|yes|yeah)\W+)?you(?:'|’)?re\s+(?:the\s+)?(?:an?\s+)?(?:asshole|a-hole)\b", re.I)
PHRASE_NTA = re.compile(
    r"^\W*(?:you(?:'|’)?re\s+not\s+(?:the\s+|an?\s+)?|not\s+the\s+)(?:asshole|a-hole)\b", re.I)

def is_bot(author, body):
    a = (author or "").lower()
    head = body[:120].lower()
    return (a in {"automoderator", "[deleted]"} or a.endswith("bot")
            or "i am a bot" in head or "i'm a bot" in head)

def parse_verdict(body):
    """Returns (label or None, body with a leading acronym tag removed)."""
    m = TAG_RE.match(body)
    if m:
        rest = body[m.end():].lstrip(" \t*_:,;.-–—)]!\"'")
        return TAG_TO_LABEL.get(m.group(1).upper()), rest
    if PHRASE_YTA.match(body):
        return "YTA", body
    if PHRASE_NTA.match(body):
        return "NTA", body
    return None, body

def process_comment(raw):
    """Keeps top-level, non-bot comments that state a verdict. Returns a record or None."""
    if not (raw.get("parent_id") or "").startswith("t3_"):
        return None
    body = (raw.get("body") or "").strip()
    if body in {"[removed]", "[deleted]", ""} or is_bot(raw.get("author"), body):
        return None
    label, rest = parse_verdict(body)
    if label is None:
        return None
    return {"post_id": raw["link_id"][3:], "label": label, "score": raw.get("score") or 0,
            "text": clean_text(cut_edits(rest))}

# ------------------------------------------------------------------- main run
def read_jsonl(path):
    with open(path, "rb") as f:            # CHANGED: binary mode (Windows OSError 22 fix)
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--comments", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stats = Counter()

    # pass 1: posts
    posts, seen = {}, set()
    for raw in read_jsonl(args.posts):
        stats["raw_posts"] += 1
        rec, reason = process_post(raw)
        if rec is None:
            stats["dropped_" + reason] += 1
            continue
        key = dedup_key(rec["text"])
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        rec.update(n_nta=0, n_yta=0, w_nta=0, w_yta=0)
        posts[rec["id"]] = rec
    stats["posts_after_cleaning"] = len(posts)

    # pass 2: comments (votes and explanation candidates)
    best = {}   # post id -> min-heap of (score, text) for agreeing comments
    for raw in read_jsonl(args.comments):
        stats["raw_comments"] += 1
        c = process_comment(raw)
        if c is None or c["post_id"] not in posts:
            continue
        stats["verdict_comments_on_kept_posts"] += 1
        p = posts[c["post_id"]]
        side = c["label"].lower()
        p["n_" + side] += 1
        p["w_" + side] += max(c["score"], 1)
        words = word_count(c["text"])
        if (c["label"] == p["label"] and c["score"] >= MIN_EXPL_SCORE
                and MIN_EXPL_WORDS <= words <= MAX_EXPL_WORDS):
            heap = best.setdefault(p["id"], [])
            heapq.heappush(heap, (c["score"], c["text"]))
            if len(heap) > EXPL_PER_POST:
                heapq.heappop(heap)

    # time split: oldest to newest by created_utc
    ordered = sorted(posts.values(), key=lambda r: r["created_utc"] or 0)
    n = len(ordered)
    cut1, cut2 = int(SPLIT[0] * n), int((SPLIT[0] + SPLIT[1]) * n)
    for i, rec in enumerate(ordered):
        rec["split"] = "train" if i < cut1 else "val" if i < cut2 else "test"
        stats["split_" + rec["split"]] += 1
        stats["label_" + rec["label"]] += 1

    # write outputs, with a leakage check on the way out
    with open(os.path.join(args.out, "posts.jsonl"), "w", encoding="utf-8") as f:
        for rec in ordered:
            assert not VERDICT_TOKEN_RE.search(rec["text"]), rec["id"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "explanations.jsonl"), "w", encoding="utf-8") as f:
        for pid, heap in best.items():
            for score, text in sorted(heap, reverse=True):
                f.write(json.dumps({"post_id": pid, "label": posts[pid]["label"],
                                    "score": score, "explanation": text}, ensure_ascii=False) + "\n")
                stats["explanations"] += 1
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(dict(stats), f, indent=2)
    print(json.dumps(dict(stats), indent=2))

if __name__ == "__main__":
    main()
