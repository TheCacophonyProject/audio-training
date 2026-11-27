#!/usr/bin/env python3
"""Writes tighter tracks of the cacophony data into its metadata as a
"confirmed-tracks" list, by the rules of scripts/tighten_tracks.py, from a
perchembed --analyse flags csv (e.g. cacophony-flags-<model>.csv). "Tracks" is
left as is.

Per recording and label:
  - a track with fewer than --min-windows windows, a --keep-labels label or a
    label the probe has no class for is kept whole
  - a longer track keeps only its windows whose probe probability for its own
    label is at least --min-prob, clipped to the track
  - the kept parts of the label's tracks are merged where they overlap, so
    overlapping tracks of the same label become one
Each merged segment is a confirmed track tagged with the track's one human
tag (no secondaries), with
  id            the original track's id, or <id>_1, <id>_2 ... when it
                becomes several tracks. A segment merging several tracks takes
                the id of the first of them (by start)
  source_track  that original track's id, merged_tracks all of them if more
  source        original (kept whole), tightened or both
  score         the best probe probability of the windows kept, if any
  minFreq / maxFreq of the tracks merged
  the original's rms data (upper_rms, bird_rms ...) over the segment, when it
                comes from one track, so the build still tightens it by rms
A label with no window over --min-prob has no confirmed tracks, it's dropped.
Tracks not in the flags csv (no human tag) are copied as they are.

    python3 write_tightened_tracks.py cacophony-flags-<model>.csv --dry-run
"""

import argparse
import copy
import json
import logging
import multiprocessing
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CONFIRMED_KEY = "confirmed-tracks"
COLUMNS = [
    "file",
    "recording_id",
    "track_id",
    "label",
    "track_start",
    "track_end",
    "window_start",
    "window_end",
    "probe_own_prob",
]
# per frame rms over the track, sliced to a segment
RMS_LISTS = ["upper_rms", "noise_rms", "bird_rms", "track_rms"]
# kept as they are, they say how the rms was made
RMS_KEYS = [
    "upper_noise_bin",
    "lower_nose_bin",
    "bird_rms_bin",
    "track_rms_bin",
    "rms_hop_length",
    "rms_sr",
]


def merge(intervals):
    """Merges overlapping (start, end) intervals, as tighten_tracks.py"""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def segments(rows, args):
    """[(start, end, sources, score, track ids)] of one recording label's
    windows by the tighten_tracks.py rules, rows are (track id, start, end,
    window start, window end, probe prob, keep whole)"""
    by_track = defaultdict(list)
    for r in rows:
        by_track[r[0]].append(r)
    intervals = []
    for track_id, windows in by_track.items():
        _, start, end, *_rest = windows[0]
        if len(windows) < args.min_windows or windows[0][6]:
            intervals.append((start, end, "original", None, track_id))
            continue
        for _, start, end, w_start, w_end, prob, _ in windows:
            if prob >= args.min_prob:
                intervals.append(
                    (max(w_start, start), min(w_end, end), "tightened", prob, track_id)
                )
    out = []
    for s, e in merge((i[0], i[1]) for i in intervals):
        # zero length tracks are kept too, so <= at the edges
        inside = [i for i in intervals if i[0] <= e and i[1] >= s]
        probs = [i[3] for i in inside if i[3] is not None]
        out.append(
            (
                s,
                e,
                ";".join(sorted({i[2] for i in inside})),
                max(probs) if probs else None,
                list(dict.fromkeys(i[4] for i in sorted(inside))),
            )
        )
    return out


def slice_rms(original, start, end):
    """The original track's rms data over start - end, None if it has none.
    Rms frames start at the track start"""
    if "upper_rms" not in original:
        return None
    hop = original.get("rms_hop_length", 281)
    sr = original.get("rms_sr", 48000)
    frame = hop / sr
    first = max(0, int(round((start - float(original["start"])) / frame)))
    last = first + int(round((end - start) / frame)) + 1
    rms = {k: original[k] for k in RMS_KEYS if k in original}
    for k in RMS_LISTS:
        if k in original:
            rms[k] = original[k][first:last]
    return rms


def write_file(job):
    txt, label_rows, args = job
    counts = Counter()
    try:
        with open(txt) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning("Can't read %s %s", txt, e)
        counts["unreadable"] += 1
        return counts
    originals = {str(t.get("id")): t for t in meta.get("Tracks") or []}
    # owner track id -> its segments, in start order
    owned = defaultdict(list)
    scored = set()
    for label, rows in label_rows.items():
        scored.update(str(r[0]) for r in rows)
        found = segments(rows, args)
        if not found:
            counts["labels dropped, no window over min prob"] += 1
        for s, e, source, score, track_ids in found:
            owner = str(track_ids[0])
            owned[owner].append((s, e, source, score, [str(t) for t in track_ids]))
            counts[f"segments {source}"] += 1

    confirmed = []
    for original in meta.get("Tracks") or []:
        track_id = str(original.get("id"))
        if track_id not in scored:
            # no human tag, not judged
            confirmed.append(copy.deepcopy(original))
            counts["tracks not scored, copied"] += 1
            continue
        mine = sorted(owned.get(track_id, []))
        if not mine:
            counts["tracks dropped or merged into another"] += 1
            continue
        if len(mine) > 1:
            counts["tracks split"] += 1
        tags = [
            t
            for t in original.get("tags", [])
            if t.get("automatic") is False and not t.get("secondary")
        ]
        for i, (s, e, source, score, track_ids) in enumerate(mine):
            track = {
                "id": original.get("id") if len(mine) == 1 else f"{track_id}_{i + 1}",
                "start": round(float(s), 4),
                "end": round(float(e), 4),
                "tags": copy.deepcopy(tags),
                "source_track": original.get("id"),
                "source": source,
            }
            if len(track_ids) > 1:
                track["merged_tracks"] = [originals[t].get("id") for t in track_ids]
                counts["segments merging tracks"] += 1
            if score is not None:
                track["score"] = round(float(score), 3)
            merged = [originals[t] for t in track_ids if t in originals]
            for k, pick in (("minFreq", min), ("maxFreq", max)):
                values = [m.get(k) for m in merged]
                if values and all(v is not None for v in values):
                    track[k] = pick(float(v) for v in values)
            if len(merged) == 1:
                rms = slice_rms(original, track["start"], track["end"])
                if rms is not None:
                    track.update(rms)
            confirmed.append(track)
    meta.pop("filtered-tracks", None)
    meta[CONFIRMED_KEY] = confirmed
    counts["confirmed tracks"] += len(confirmed)
    if not args.dry_run:
        from addtracks import write_meta

        write_meta(Path(txt), meta)
    counts["files"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("flags", help="perchembed --analyse flags csv")
    parser.add_argument("--labels", nargs="+", help="Only these labels, default all")
    parser.add_argument(
        "--keep-labels",
        nargs="+",
        default=["comred", "nezpig2"],
        help="Tracks with any of these labels are kept whole",
    )
    parser.add_argument("--min-prob", type=float, default=0.4)
    parser.add_argument(
        "--min-windows",
        type=int,
        default=2,
        help="Only tighten tracks with at least this many windows, shorter are "
        "kept whole",
    )
    parser.add_argument("--dry-run", action="store_true", help="Count, don't write")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    d = pd.read_csv(args.flags, usecols=COLUMNS, low_memory=False)
    d = d[d.label.notna()]
    tags = d.label.str.split(";").apply(set)
    if args.labels:
        keep = tags.apply(lambda t: bool(t & set(args.labels)))
        d, tags = d[keep], tags[keep]
    keep_whole = tags.apply(lambda t: bool(t & set(args.keep_labels)))
    d = d.assign(keep_whole=keep_whole | d.probe_own_prob.isna())

    jobs = []
    for txt, g in d.groupby("file"):
        label_rows = {
            label: list(
                zip(
                    lg.track_id,
                    lg.track_start,
                    lg.track_end,
                    lg.window_start,
                    lg.window_end,
                    lg.probe_own_prob,
                    lg.keep_whole,
                )
            )
            for label, lg in g.groupby("label")
        }
        jobs.append((txt, label_rows, args))
    logging.info(
        "%s %s windows of %s files%s",
        "Checking" if args.dry_run else "Writing",
        len(d),
        len(jobs),
        ", dry run" if args.dry_run else "",
    )
    total = Counter()
    with multiprocessing.Pool(args.workers) as pool:
        for i, counts in enumerate(pool.imap_unordered(write_file, jobs, chunksize=32)):
            total.update(counts)
            if i % 2000 == 0:
                logging.info("%s / %s files", i, len(jobs))
    for k, v in sorted(total.items()):
        logging.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
