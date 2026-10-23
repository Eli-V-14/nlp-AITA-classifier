# preprocess/

Turns the raw Arctic Shift dumps into the model-ready dataset, plus the scripts
that produced the grading report. Stdlib only. Point the paths at wherever the
raw `r_AmItheAsshole_posts.jsonl` / `_comments.jsonl` live (here: `D:/MSOE/CSC/NLP Data`).

**All outputs go to the repo's `out/` folder by default.** The large generated
files there (`posts.jsonl`, `explanations.jsonl`, scan JSON, logs) are
git-ignored; only the small `out/sample_posts.jsonl`, `out/sample_comments.jsonl`
and `out/stats.json` are committed. The raw dumps live elsewhere (here:
`D:/MSOE/CSC/NLP Data`) and are never committed.

## One command (everything)

`run_all.py` runs all four steps in order and prints a combined summary. It
auto-feeds the sample post id from the posts scan into the comments scan.

```bash
python preprocess/run_all.py --data-dir "D:/MSOE/CSC/NLP Data"
```

Outputs land in `<repo>/out/` (override with `--out`). The individual steps
below are the same scripts, if you want to run one on its own.

## The pipeline

`clean_aita.py` — the preprocessing. Drops removed/deleted/short posts, maps
flair to NTA/YTA labels, cuts EDIT/UPDATE sections, strips markdown/links/verdict
tokens, dedups, counts votes from comments, selects explanation comments, makes
the oldest→newest train/val/test split, and writes `posts.jsonl`,
`explanations.jsonl`, `stats.json`.

```bash
python preprocess/clean_aita.py \
  --posts "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl" \
  --comments "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl" \
  --out out
```

`build_samples.py` — the 5-post sample (each with its top 2 top-level comments
by score). Defaults its output to `<repo>/out/`:

```bash
python preprocess/build_samples.py \
  --posts "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl" \
  --comments "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl"
```

## Reporting helpers (used for REPORT.md)

- `raw_posts_scan.py` — raw post line count, total raw selftext word count, date
  range, and one real sample post (2019+, with an EDIT the cleaner cuts).
- `raw_comments_scan.py` — raw comment line count, date range, and every raw
  comment line for one post (`--link-id t3_<id>`).
- `analyze_processed.py` — cleaned word count and the first date in each split,
  from `out/posts.jsonl`.

```bash
python preprocess/raw_posts_scan.py    --posts "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl"    --out-sample out/sample_post.jsonl     --out-stats out/posts_scan.json
python preprocess/raw_comments_scan.py --comments "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl" --link-id t3_abdo6m --out-sample out/sample_comments.jsonl --out-stats out/comments_scan.json
python preprocess/analyze_processed.py out/posts.jsonl
```

## Results

See `REPORT.md` for the full numbers from the latest full run.

**Heads-up for anyone using this data:** the comments dump stops at **2019-03-28**
but posts run to **2022-06-13**, so posts after Mar-2019 have no comments — the
val (first date 2021-09-10) and test (2022-02-19) splits have labels from flair
but no vote counts or explanations.

## Windows note

`clean_aita.py` reads files in binary mode on purpose; plain
`open(path, encoding="utf-8")` iteration raises `OSError 22` on the 2.9 GB
comments file on Windows. Parsing is identical either way.
