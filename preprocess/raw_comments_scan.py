"""Single streaming pass over the raw COMMENTS file.

Produces (item 1) line count + date range, and (item 4) writes out verbatim
every raw comment line belonging to the sample post (--link-id t3_<id>).
"""
import argparse, datetime as dt, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comments", required=True)
    ap.add_argument("--link-id", required=True, help="t3_<postid>")
    ap.add_argument("--out-sample", required=True)
    ap.add_argument("--out-stats", required=True)
    args = ap.parse_args()

    link = args.link_id if args.link_id.startswith("t3_") else "t3_" + args.link_id
    lines = 0
    matched = 0
    cmin = None
    cmax = None

    with open(args.comments, "rb") as f, \
            open(args.out_sample, "w", encoding="utf-8") as outf:
        for raw_bytes in f:
            if not raw_bytes.strip():
                continue
            lines += 1
            line = raw_bytes.decode("utf-8", "replace")
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cu = rec.get("created_utc")
            try:
                cu = int(cu)
                cmin = cu if cmin is None else min(cmin, cu)
                cmax = cu if cmax is None else max(cmax, cu)
            except (TypeError, ValueError):
                pass
            if rec.get("link_id") == link:
                outf.write(line.rstrip("\n") + "\n")
                matched += 1
            if lines % 500000 == 0:
                print(f"  ...{lines:,} comments ({matched} matched)", flush=True)

    def iso(x):
        return dt.datetime.fromtimestamp(x, dt.timezone.utc).strftime("%Y-%m-%d") if x else None

    out = {
        "comments_lines": lines,
        "comments_created_min": iso(cmin),
        "comments_created_max": iso(cmax),
        "sample_link_id": link,
        "sample_comment_lines": matched,
    }
    with open(args.out_stats, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
