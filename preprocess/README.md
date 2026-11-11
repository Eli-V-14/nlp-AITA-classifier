# preprocess/

These scripts turn the raw Arctic Shift dumps into the labeled dataset the model
trains on, plus a few helpers that produced the numbers in the report. Everything
here is stdlib only, so it runs on any Python. Point the paths at wherever the raw
`r_AmItheAsshole_posts.jsonl` and `_comments.jsonl` live (on this machine that's
`D:/MSOE/CSC/NLP Data`).

By default everything writes to the repo's `out/` folder. The big generated files
there (`posts.jsonl`, `explanations.jsonl`, the scan JSON, the logs) are
git-ignored; only the small `out/sample_posts.jsonl`, `out/sample_comments.jsonl`,
and `out/stats.json` get committed. The raw dumps stay on external storage and are
never committed.

## Run the whole thing at once

`run_all.py` does all four steps in order and prints a combined summary. It even
feeds the sample post id from the posts scan into the comments scan for you.

```bash
python preprocess/run_all.py --data-dir "D:/MSOE/CSC/NLP Data"
```

Output lands in `<repo>/out/` (use `--out` to change that). If you'd rather run a
single step, they're below.

## The actual pipeline

`clean_aita.py` is the preprocessing. It drops removed/deleted/short posts, maps
the flair to NTA/YTA, cuts out EDIT/UPDATE sections, strips markdown, links, and
verdict tokens, de-dupes, counts the verdict votes from the comments, picks the
explanation comments, makes the oldest-to-newest train/val/test split, and writes
`posts.jsonl`, `explanations.jsonl`, and `stats.json`.

```bash
python preprocess/clean_aita.py \
  --posts "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl" \
  --comments "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl" \
  --out out
```

`build_samples.py` makes the small 5-post sample, each post with its top 2
top-level comments by score. It writes to `<repo>/out/` by default:

```bash
python preprocess/build_samples.py \
  --posts "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl" \
  --comments "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl"
```

## Helpers that built the report

- `raw_posts_scan.py`: raw post line count, total raw word count, the date range,
  and one real sample post (2019 or later, with an EDIT the cleaner would cut).
- `raw_comments_scan.py`: raw comment line count, date range, and every raw
  comment line for one post (`--link-id t3_<id>`).
- `analyze_processed.py`: cleaned word count and the first date in each split,
  read from `out/posts.jsonl`.

```bash
python preprocess/raw_posts_scan.py    --posts "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_posts.jsonl"    --out-sample out/sample_post.jsonl     --out-stats out/posts_scan.json
python preprocess/raw_comments_scan.py --comments "D:/MSOE/CSC/NLP Data/r_AmItheAsshole_comments.jsonl" --link-id t3_abdo6m --out-sample out/sample_comments.jsonl --out-stats out/comments_scan.json
python preprocess/analyze_processed.py out/posts.jsonl
```

## Results

`REPORT.md` has the full numbers from the last run.

One thing to watch out for if you use this data: the comment dump stops at
2019-03-28 but the posts run to 2022-06-13, so anything after March 2019 has no
comments. That means the val split (starts 2021-09-10) and test split (starts
2022-02-19) have labels from the flair but no vote counts or explanations.

## Windows note

`clean_aita.py` reads the files in binary mode on purpose. The plain
`open(path, encoding="utf-8")` loop throws `OSError 22` on the 2.9 GB comments
file on Windows. Parsing the bytes gives identical results.
