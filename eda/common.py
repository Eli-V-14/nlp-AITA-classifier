"""Shared helpers for streaming the raw r/AmItheAsshole JSONL dumps.

The dumps are multi-GB, so everything here reads one JSON object per line and
never holds the whole file in memory. json from the stdlib is used so there are
no third-party dependencies; if `orjson` happens to be installed it's used
automatically because it's a few times faster.
"""
from __future__ import annotations

import datetime as _dt
from typing import Iterator

try:  # optional speedup, not required
    import orjson as _orjson

    def loads(line: bytes | str):
        return _orjson.loads(line)
except ImportError:  # pragma: no cover - fallback path
    import json as _json

    def loads(line: bytes | str):
        return _json.loads(line)


def iter_records(path: str, limit: int | None = None) -> Iterator[dict]:
    """Yield one parsed record per line. `limit` caps how many are read.

    The file is opened in BINARY mode and each line decoded individually. Text
    mode chokes on these multi-GB dumps on Windows (an odd byte triggers
    OSError 22), so we decode with errors='replace' and skip unparseable lines.
    """
    n = 0
    with open(path, "rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield loads(raw)  # json/orjson both accept bytes
            except Exception:
                # A handful of records in big dumps are truncated/corrupt.
                continue
            n += 1
            if limit is not None and n >= limit:
                return


# --- Verdict normalization -------------------------------------------------
# The subreddit's flair wording and capitalization changed over the years, so
# every variant is folded down to one of the six canonical AITA verdict codes.
CANONICAL = ["NTA", "YTA", "ESH", "NAH", "INFO", "TOO_CLOSE", "META", "UNFLAIRED"]

VERDICT_LABELS = {
    "NTA": "Not the A-hole",
    "YTA": "You're the A-hole",
    "ESH": "Everyone Sucks Here",
    "NAH": "No A-holes Here",
    "INFO": "Not Enough Info",
    "TOO_CLOSE": "Too Close to Call",
    "META": "Meta / Shitpost / Troll",
    "UNFLAIRED": "Unflaired (no verdict)",
}


def normalize_verdict(flair: str | None) -> str:
    """Map a raw link_flair_text value to a canonical verdict code."""
    if not flair:
        return "UNFLAIRED"
    f = flair.strip().lower()
    if not f:
        return "UNFLAIRED"
    # Order matters: check the "not ..." phrases before bare "asshole".
    if "not the a" in f or f == "nta":
        return "NTA"
    if ("no a" in f and "hole" in f) or f == "nah":  # "no a--holes here"
        return "NAH"
    if "everyone sucks" in f or f == "esh":
        return "ESH"
    if "enough info" in f or f == "info":
        return "INFO"
    if "too close" in f:
        return "TOO_CLOSE"
    if any(k in f for k in ("shitpost", "shit post", "meta", "troll")):
        return "META"
    if "asshole" in f or "a-hole" in f or f == "yta":
        return "YTA"
    return "META"  # anything else unexpected


# Verdict codes as they appear in comment bodies (top-level judgements).
# Match is CASE-SENSITIVE on purpose: the subreddit convention is to type the
# verdict in uppercase (YTA, NTA, ...). Allowing lowercase would count ordinary
# English words like "nah" and "info" as votes and badly inflate those two.
import re as _re

_COMMENT_VERDICT_RE = _re.compile(r"\b(YTA|NTA|ESH|NAH|INFO)\b")


def verdict_in_comment(body: str | None) -> str | None:
    """Return the first (uppercase) AITA verdict code in a comment body, or None."""
    if not body:
        return None
    m = _COMMENT_VERDICT_RE.search(body)
    return m.group(1) if m else None


def ym(created_utc) -> str:
    """created_utc -> 'YYYY-MM' bucket (UTC)."""
    try:
        return _dt.datetime.fromtimestamp(int(created_utc), _dt.timezone.utc).strftime("%Y-%m")
    except (TypeError, ValueError, OSError):
        return "unknown"


def is_removed(selftext: str | None) -> bool:
    return selftext in ("[deleted]", "[removed]", None, "")
