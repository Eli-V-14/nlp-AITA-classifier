# AITA download — grading report

_Generated 2026-10-22 from the full download in `D:\MSOE\CSC\NLP Data`._

## Download facts
- **Tool:** Arctic Shift download tool (per `clean_aita.py` header).
- **Content:** posts **and** comments (two JSONL files).
- **Download date:** 2026-10-21 (file mtimes ~20:22–20:23).
- **Posts date range:** 2013-06-08 → 2022-06-13.
- **Comments date range:** 2013-06-11 → **2019-03-28**.
- **Stored at:** `D:\MSOE\CSC\NLP Data` (local). Shared/cloud link: _not set — tell me the destination and I'll put the processed files there._

> ⚠️ **Coverage mismatch:** comments stop at **2019-03**, posts run to **2022-06**. Every post after Mar-2019 has **no comments** in this dump, so it has zero vote counts and no explanation targets. The val split (first date 2021-09-10) and test split (first date 2022-02-19) are **entirely past the comment window** — labels (from flair) exist, but `n_nta/n_yta` and explanations are train-era only.

## 1. Raw counts
| file | lines | size (bytes) | size |
|------|------:|-------------:|------|
| posts.jsonl | 1,385,631 | 3,848,523,354 | 3.58 GiB |
| comments.jsonl | 2,247,059 | 2,957,599,315 | 2.75 GiB |

- **Total raw post word count** (sum of `selftext` words): **141,135,463**

## 2. stats.json (full `clean_aita.py` run)
_Run with an identical copy; only change: `read_jsonl` opens the file in binary mode, because Windows text-mode iteration throws `OSError 22` on the 2.9 GB comments file. Results are unchanged (the leakage assert passed)._

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
Clean labeled posts: **373,340** — NTA **284,151** (76.1%), YTA **89,189** (23.9%).

## 3. Cleaned word count + split dates
- **Cleaned post-text word count (total):** **128,110,937**
  - train 100,984,903 · val 13,520,670 · test 13,605,364
- **First date in each split** (oldest→newest by `created_utc`):
  - train **2013-06-08**
  - val **2021-09-10**
  - test **2022-02-19**

## 4. Real sample (sent separately)
- `sample_post.jsonl` — one raw post line: id **abdo6m**, flair **"Asshole"** (→ YTA), 38 comments, created **2019-01-01**, contains a late "Edit:" the cleaner cuts.
- `sample_comments.jsonl` — the **39** raw comment lines for `t3_abdo6m`.

## 5. Processed output location
`D:\MSOE\CSC\NLP Data\processed\`
- `posts.jsonl` (757,478,304 bytes)
- `explanations.jsonl` (18,311,407 bytes)
- `stats.json` (463 bytes)

_Upload to the shared location is pending a destination._
