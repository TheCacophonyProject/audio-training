#!/usr/bin/env python3
"""Compare the windows best_windows.py picks with the current set up, where
each track's 3s window is its best rms segment (Track with tighten and
filter_rms, see audiodataset.Track.tighten_track) and secondary tags get no
window at all.

Per tag of every track in the best_windows csv, one of:
  same          both pick a window and they overlap by at least --same-overlap
  moved         both pick a window but they overlap less
  now_dropped   rms has a window but the probe's best is under --min-score
  rms_filtered  dropped now by filter_rms (flat or missing rms), the probe
                keeps it
  both_dropped  rms_filtered and the probe's best is under --min-score
  no_rms        not tightened (e.g. not a bird track), the probe keeps it
  no_rms_dropped  not tightened and the probe's best is under --min-score
  new_secondary   a secondary tag the probe places, gets no window now
  secondary_dropped  a secondary tag the probe doesn't place

Needs the tier1 metadata the csv points at (the file column), so run it where
that is:
    python3 compare_best_windows.py tier1-best-windows-spectv12AutoNoiseClean.csv \
        --out tier1-best-windows-vs-rms.csv
"""

import argparse
import json
import logging
import multiprocessing
from collections import defaultdict

import pandas as pd

GENERATED_BY = "best_windows"
CATEGORIES = [
    "same",
    "moved",
    "now_dropped",
    "rms_filtered",
    "both_dropped",
    "no_rms",
    "no_rms_dropped",
    "new_secondary",
    "secondary_dropped",
]


def window_scores(value):
    """ "0-3:0.1;2-5:0.9" -> [(0, 3, 0.1), (2, 5, 0.9)]"""
    windows = []
    if isinstance(value, str):
        for item in value.split(";"):
            span, _, score = item.rpartition(":")
            start, _, end = span.partition("-")
            windows.append((float(start), float(end), float(score)))
    return windows


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def compare_file(job):
    txt, rows, segment_length = job
    from audiodataset import Track

    try:
        with open(txt) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning("Can't read %s %s", txt, e)
        return []
    tracks = {
        str(t.get("id")): t
        for t in meta.get("Tracks") or []
        if t.get("generated_by") != GENERATED_BY
    }
    results = []
    rms_by_track = {}
    for r in rows:
        track_id = str(r["track_id"])
        if track_id not in rms_by_track:
            track_meta = tracks.get(track_id)
            if track_meta is None:
                rms_by_track[track_id] = None
            else:
                track = Track(
                    track_meta,
                    r["audio"],
                    meta.get("id"),
                    None,
                    segment_length=segment_length,
                    tighten=True,
                    filter_rms=True,
                )
                rms_by_track[track_id] = (
                    track.best_start,
                    track.best_end,
                    track.rms_filtered,
                )
        rms = rms_by_track[track_id]
        out = dict(r)
        out.update(rms_start="", rms_end="", rms_filtered="", rms_probe_score="")
        if rms is None:
            out["category"] = "missing_track"
            results.append(out)
            continue
        rms_start, rms_end, rms_filtered = rms
        out["rms_filtered"] = rms_filtered
        keep = bool(r["keep"])
        if r["tag_type"] == "secondary":
            out["category"] = "new_secondary" if keep else "secondary_dropped"
        elif rms_filtered:
            out["category"] = "rms_filtered" if keep else "both_dropped"
        elif rms_start is None:
            out["category"] = "no_rms" if keep else "no_rms_dropped"
        else:
            out["rms_start"] = round(rms_start, 2)
            out["rms_end"] = round(rms_end, 2)
            # the probe's score for the window most like the rms one
            windows = window_scores(r.get("window_scores"))
            if windows:
                nearest = max(
                    windows, key=lambda w: overlap(w[0], w[1], rms_start, rms_end)
                )
                out["rms_probe_score"] = nearest[2]
            if not keep:
                out["category"] = "now_dropped"
            else:
                shared = overlap(
                    rms_start, rms_end, float(r["best_start"]), float(r["best_end"])
                )
                out["shared"] = round(shared, 2)
                out["category"] = "same" if shared >= r["same_overlap"] else "moved"
        results.append(out)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("best_windows_csv", help="output of best_windows.py")
    parser.add_argument("--out", required=True, help="per tag comparison csv")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Override the keep column with this score, default use keep, or "
        "0.4 if the csv has no keep column",
    )
    parser.add_argument(
        "--same-overlap",
        type=float,
        default=2.0,
        help="Seconds the two windows must share to count as the same",
    )
    parser.add_argument("--segment-length", type=float, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    bw = pd.read_csv(args.best_windows_csv, low_memory=False)
    missing = {"file", "audio", "track_id", "tag", "tag_type", "best_score"} - set(
        bw.columns
    )
    if missing:
        parser.error(
            f"{args.best_windows_csv} has no {sorted(missing)} columns, remake it "
            "with the current best_windows.py"
        )
    if args.min_score is None and "keep" not in bw.columns:
        args.min_score = 0.4
        logging.info("No keep column, keeping best scores >= %s", args.min_score)
    if args.min_score is not None:
        bw["keep"] = bw.best_score.notna() & (bw.best_score >= args.min_score)
        if "source" in bw.columns:
            # rescued by neighbour agreement, not by score
            bw["keep"] |= bw.source == "knn"
    bw["same_overlap"] = args.same_overlap
    by_file = defaultdict(list)
    for r in bw.to_dict("records"):
        by_file[r["file"]].append(r)
    jobs = [(txt, rows, args.segment_length) for txt, rows in by_file.items()]
    logging.info("Comparing %s tags in %s files", len(bw), len(jobs))

    results = []
    with multiprocessing.Pool(args.workers) as pool:
        for i, file_results in enumerate(
            pool.imap_unordered(compare_file, jobs, chunksize=64)
        ):
            results.extend(file_results)
            if i % 20000 == 0:
                logging.info("%s / %s files", i, len(jobs))
    df = pd.DataFrame(results)
    for c in ("rms_start", "rms_end", "rms_probe_score", "best_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.drop(columns=["same_overlap"]).to_csv(args.out, index=False)
    logging.info("Wrote %s to %s", len(df), args.out)

    counts = df.pivot_table(
        index="tag", columns="category", values="audio", aggfunc="size", fill_value=0
    )
    counts = counts.reindex(
        columns=[c for c in CATEGORIES + ["missing_track"] if c in counts.columns]
    )
    counts.insert(0, "tags", counts.sum(axis=1))
    both = counts.get("same", 0) + counts.get("moved", 0)
    counts["moved_%"] = (100 * counts.get("moved", 0) / both.where(both > 0)).round(1)
    primary = df[df.tag_type == "primary"]
    changed = primary.category.isin(
        ["moved", "now_dropped", "rms_filtered", "no_rms", "no_rms_dropped"]
    )
    counts["primary_changed_%"] = (
        100 * changed.groupby(primary.tag).mean()
    ).round(1)
    moved = df[df.category == "moved"]
    counts["moved_score_rms"] = moved.groupby("tag").rms_probe_score.median().round(2)
    counts["moved_score_best"] = moved.groupby("tag").best_score.median().round(2)
    counts = counts.sort_values("tags", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 50)
    print(counts.to_string())
    total = df.category.value_counts()
    print("\nAll tags:", total.to_dict())
    print(
        "Primary tags changed:",
        f"{changed.mean():.1%}",
        "of",
        len(primary),
    )
    summary = args.out.rsplit(".", 1)[0] + "-per-label.csv"
    counts.to_csv(summary)
    logging.info("Per label summary in %s", summary)


if __name__ == "__main__":
    main()
