# EDA — exploring the collected r/AmItheAsshole data

Dependency-free (stdlib-only) scripts to look at the raw Reddit dumps before
training the classifier. They **stream** the JSONL line by line, so they work
on the multi-GB files without loading them into memory.

Data files (not in this repo):
- `D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl`  (~3.8 GB)
- `D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl` (~2.9 GB)

## Run

```bash
# posts: verdict/label distribution, text length, volume & verdict mix over time
python eda/explore_posts.py "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl"

# comments: volume, scores, and uppercase verdict codes (YTA/NTA/...) in bodies
python eda/explore_comments.py "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl"
```

Use `--limit N` for a fast sample (e.g. `--limit 100000`) while iterating; drop
it to process the whole file (a few minutes each). Both write CSVs to
`eda/out/` (override with `--outdir`).

## Look at the actual text / rows (`read_text.py`)

`explore_*.py` give you numbers; `read_text.py` shows the data itself.

```bash
# compact table: one row per record (auto-detects posts vs comments)
python eda/read_text.py rows <posts.jsonl> --exclude-removed --n 50
python eda/read_text.py rows <posts.jsonl> --verdict YTA --n 30
python eda/read_text.py rows <posts.jsonl> --csv --n 1000 > sample.csv   # export

# full post bodies with the verdict
python eda/read_text.py posts <posts.jsonl> --verdict YTA --full --n 3
python eda/read_text.py posts <posts.jsonl> --contains "wedding" --n 10
python eda/read_text.py posts <posts.jsonl> --id 1fy0bx --full

# the comments under one post
python eda/read_text.py comments <comments.jsonl> --link-id 1fy0bx --n 20

# the TOP 2 comments for a post, ranked by score (top-level only)
python eda/read_text.py comments <comments.jsonl> --link-id 1fy0bx --top 2 --top-level --full
```

`--top N` collects that post's comments and returns the N highest-scored (it
requires `--link-id`). Add `--top-level` to ignore replies, and `--verdicts-only`
to keep just comments that contain a verdict code. Without `--top`, comments come
out in file order, capped by `--n`.

Filters (`--verdict`, `--contains`, `--exclude-removed`, `--min-len`) work the
same across `rows` and `posts`. `rows --csv` streams a CSV to stdout so you can
redirect it to a file or open it in a spreadsheet.

## Charts (optional)

The only script needing a third-party library:

```bash
pip install matplotlib
python eda/plot.py --outdir eda/out
```

Produces `verdict_distribution.png`, `posts_by_month.png`, `length_buckets.png`,
and `verdict_by_month.png` (verdict mix over time).

## What the label looks like

The classifier's label is the community verdict on each **post**, stored in
`link_flair_text`. The wording/capitalization drifted over the years, so
`common.normalize_verdict()` folds every variant into one canonical code:

| code | meaning | example raw flairs |
|------|---------|--------------------|
| `NTA` | Not the A-hole | `not the a-hole`, `Not the A-hole`, `not the asshole` |
| `YTA` | You're the A-hole | `asshole`, `Asshole` |
| `ESH` | Everyone Sucks Here | `everyone sucks`, `Everyone Sucks` |
| `NAH` | No A-holes Here | `no a--holes here`, `No A-holes here` |
| `INFO` | Not Enough Info | `not enough info` |
| `TOO_CLOSE` | Too Close to Call | `too close to call` |
| `META` | meta/shitpost/troll | `META`, `Shitpost`, `Troll` |
| `UNFLAIRED` | no verdict yet | `null` / empty |

## Caveats worth knowing before training

- **Many posts are unusable as training rows.** On an early sample ~44% have
  `selftext` of `[deleted]`/`[removed]`/empty, and ~33% are `UNFLAIRED` (no
  verdict). `explore_posts.py` reports both — the usable set is posts that have
  BOTH real text AND a verdict in {NTA, YTA, ESH, NAH, INFO}.
- **The label is heavily imbalanced** (~60%+ NTA). Plan for class weighting or
  resampling.
- **Comment verdict detection is case-sensitive on purpose.** Verdicts are typed
  in uppercase; matching lowercase would count the English words "nah"/"info" as
  votes. Even so, treat comment-derived verdicts as noisy — the post flair is the
  authoritative label.
- The very earliest data (2013+) predates the modern YTA/NTA convention, so old
  flairs/comments are sparser and messier.
