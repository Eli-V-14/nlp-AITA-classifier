# EDA: looking at the r/AmItheAsshole data

A few stdlib-only scripts for poking at the raw Reddit dumps before training.
They stream the JSONL line by line, so they handle the multi-GB files without
loading everything into memory.

The data files aren't in the repo:
- `D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl` (~3.8 GB)
- `D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl` (~2.9 GB)

## Run

```bash
# posts: verdict/label distribution, text length, volume and verdict mix over time
python eda/explore_posts.py "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl"

# comments: volume, scores, and uppercase verdict codes (YTA/NTA/...) in bodies
python eda/explore_comments.py "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl"
```

Add `--limit N` for a quick sample while you're iterating (say `--limit 100000`),
or drop it to run the whole file, which takes a few minutes each. Both write CSVs
to `eda/out/` (change that with `--outdir`).

## Reading the actual text or rows (read_text.py)

`explore_*.py` give you numbers; `read_text.py` shows the data itself.

```bash
# compact table, one row per record (figures out posts vs comments on its own)
python eda/read_text.py rows <posts.jsonl> --exclude-removed --n 50
python eda/read_text.py rows <posts.jsonl> --verdict YTA --n 30
python eda/read_text.py rows <posts.jsonl> --csv --n 1000 > sample.csv   # export

# full post bodies with the verdict
python eda/read_text.py posts <posts.jsonl> --verdict YTA --full --n 3
python eda/read_text.py posts <posts.jsonl> --contains "wedding" --n 10
python eda/read_text.py posts <posts.jsonl> --id 1fy0bx --full

# the comments under one post
python eda/read_text.py comments <comments.jsonl> --link-id 1fy0bx --n 20

# the top 2 comments for a post, ranked by score (top-level only)
python eda/read_text.py comments <comments.jsonl> --link-id 1fy0bx --top 2 --top-level --full
```

`--top N` grabs that post's comments and keeps the N highest-scored, so it needs
`--link-id`. Add `--top-level` to skip replies, or `--verdicts-only` to keep just
the comments with a verdict code in them. Without `--top`, comments come out in
file order, capped by `--n`.

The filters (`--verdict`, `--contains`, `--exclude-removed`, `--min-len`) work the
same way on `rows` and `posts`. `rows --csv` prints a CSV to stdout, so you can
redirect it to a file or open it in a spreadsheet.

## Charts (optional)

This is the one script that needs a third-party library:

```bash
pip install matplotlib
python eda/plot.py --outdir eda/out
```

It makes `verdict_distribution.png`, `posts_by_month.png`, `length_buckets.png`,
and `verdict_by_month.png` (verdict mix over time).

## What the label looks like

The classifier's label is the community verdict on each post, stored in
`link_flair_text`. The wording and capitalization drifted over the years, so
`common.normalize_verdict()` folds all the variants into one code:

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

## Things to know before training

- A lot of posts can't be used as training rows. On an early sample about 44% have
  `selftext` of `[deleted]`, `[removed]`, or empty, and about 33% are `UNFLAIRED`
  (no verdict). `explore_posts.py` reports both. The usable set is the posts that
  have both real text and a verdict in {NTA, YTA, ESH, NAH, INFO}.
- The labels are heavily imbalanced (60%+ NTA), so plan on class weighting or
  resampling.
- Comment verdict detection is case-sensitive on purpose. People type the verdicts
  in uppercase, and matching lowercase would count the ordinary words "nah" and
  "info" as votes. Even then, treat comment-derived verdicts as noisy; the post
  flair is the label we actually trust.
- The earliest data (2013 on) predates the modern YTA/NTA convention, so those old
  flairs and comments are sparser and messier.
