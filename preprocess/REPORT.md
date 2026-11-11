# AITA download report

Numbers from the full download in `D:\MSOE\CSC\NLP Data` (run 2026-10-22).

## Download facts
- Tool: Arctic Shift download tool (from the `clean_aita.py` header).
- Content: both posts and comments (two JSONL files).
- Download date: 2026-10-21 (file timestamps around 20:22-20:23).
- Posts date range: 2013-06-08 to 2022-06-13.
- Comments date range: 2013-06-11 to 2019-03-28.
- Stored at `D:\MSOE\CSC\NLP Data` (local). No shared/cloud link yet.

Worth flagging: the comments stop at 2019-03 but the posts go to 2022-06, so
every post after March 2019 has no comments in this dump, which means no vote
counts and no explanation targets. Both the val split (starts 2021-09-10) and the
test split (starts 2022-02-19) fall entirely past the comment window. The labels
still exist (they come from the flair), but `n_nta/n_yta` and the explanations are
train-era only.

## 1. Raw counts
| file | lines | size (bytes) | size |
|------|------:|-------------:|------|
| posts.jsonl | 1,385,631 | 3,848,523,354 | 3.58 GiB |
| comments.jsonl | 2,247,059 | 2,957,599,315 | 2.75 GiB |

Total raw post word count (sum of `selftext` words): 141,135,463.

## 2. stats.json (full clean_aita.py run)
This ran with a copy of the script whose only change is that `read_jsonl` opens
the file in binary mode. Windows text-mode iteration throws `OSError 22` on the
2.9 GB comments file; the bytes parse identically and the leakage assert still
passed.

```json
{
  "raw_posts": 1385631,
  "dropped_removed_or_deleted_text": 993013,
  "dropped_no_verdict_flair": 17987,
  "dropped_too_short": 1061,
  "dropped_duplicate": 229,
  "dropped_removed_by_moderator": 1,
  "posts_after_cleaning": 373340,
  "label_NTA": 284151,
  "label_YTA": 89189,
  "raw_comments": 2247059,
  "verdict_comments_on_kept_posts": 482208,
  "split_train": 298672,
  "split_val": 37334,
  "split_test": 37334,
  "explanations": 50355
}
```
Clean labeled posts: 373,340. NTA 284,151 (76.1%), YTA 89,189 (23.9%).

## 3. Cleaned word count and split dates
- Cleaned post-text word count (total): 128,110,937.
  - train 100,984,903, val 13,520,670, test 13,605,364.
- First date in each split (oldest to newest by `created_utc`):
  - train 2013-06-08
  - val 2021-09-10
  - test 2022-02-19

## 4. Sample
`sample_posts.jsonl` holds 5 raw post lines, all from 2019-01-01, each with an
EDIT the cleaner cuts and a flair that maps to a label:

| id | flair | label | #comments |
|----|-------|-------|----------:|
| abdo6m | Asshole | YTA | 38 |
| abe0ca | Asshole | YTA | 25 |
| abee17 | Asshole | YTA | 79 |
| abei8b | Not the A-hole | NTA | 116 |
| abfi6o | Asshole | YTA | 34 |

`sample_comments.jsonl` holds the top 2 top-level comments by score for each of
those posts (10 lines), looked up by `t3_<id>`. Both come from
`preprocess/build_samples.py`.

## 5. Processed output location
`D:\MSOE\CSC\NLP Data\processed\`
- `posts.jsonl` (757,478,304 bytes)
- `explanations.jsonl` (18,311,407 bytes)
- `stats.json` (463 bytes)
