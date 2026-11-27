#!/usr/bin/env python3
"""Writes the refined windows of each tier1 track into its metadata as a
"confirmed-tracks" list, the 3s tracks to keep, from a compare_best_windows.py csv. "Tracks" is left
as is, apart from adding to each scored track
  best_start / best_end  the best rms window, if it doesn't have them
  primary_starts         start of every window where the probe scores the
                         primary at least --min-score
  secondary_starts       {secondary tag what: [starts]} the same for each
                         secondary
  probe_starts           {label: [starts]} for every probe label scoring at
                         least --min-score, not only the track's tags
  knn_starts             {label: [starts]} where label is the windows' most
                         common neighbour tag (best_windows --knn-fraction)
                         (the build checks labels from overlaps with these)
so where each bird is can be found later, e.g. to add secondaries from
overlapping tracks of the same clip when recordings are merged. Times are in
the clip's own time, as start / end. An empty list is scored but under
--min-score everywhere, a label the probe has no class for isn't there.

Per track in "Tracks", into "confirmed-tracks":
  - a track with the same id over the primary's window (rules below), tagged
    with the primary and every secondary placed in that window (sharing at
    least --same-overlap seconds with it)
  - a track per other window secondaries are placed in, id <id>_secondary
    (then <id>_secondary_2 ...), tagged with those secondaries as its own
    tags. Secondaries placed in the same window share the track
Secondaries the probe and knn don't place aren't confirmed so aren't in any
confirmed track, they're only in "Tracks" (with their secondary_starts).
A confirmed track has source_track, the id of the "Tracks" track it came from,
the original's offset_start / offset_end and frequency
box, and source (rms, probe, knn or original) and score (the probe score of
its window).

Primary window:
  - the best rms window, unless
  - the probe (or knn) window is a clear fix: the rms window's probe score is
    under --min-score, the new window's is at least it (or knn rescued it) and
    they share under --max-shared seconds, then that window
  - neither the probe nor knn place it (now_dropped) and the label isn't in
    --exempt (default none): no primary track, it's dropped. Exempt labels
    keep the rms window, as do labels the probe has no class for (it can't
    score them) and primaries that aren't --labels (the model's)
  - an rms_filtered track the probe keeps gets the probe window
  - a track with no rms window (not tightened) is copied as is
Secondaries that aren't --labels are never placed.

    python3 write_best_windows.py tier1-best-windows-vs-rms.csv \
        --labels labels.txt --dry-run
"""

import argparse
import json
import logging
import multiprocessing
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CONFIRMED_KEY = "confirmed-tracks"
# written by earlier versions, removed on a re-run
OLD_KEYS = ["filtered-tracks"]
# copied from the original track to its confirmed tracks
COPY_KEYS = ["offset_start", "offset_end", "minFreq", "maxFreq"]


def parse_labels(value):
    if value is None:
        return None
    path = Path(value)
    if "," not in value and path.is_file():
        text = path.read_text().strip()
        if text.startswith("["):
            return set(json.loads(text))
        return {t.strip() for t in text.replace(",", "\n").splitlines() if t.strip()}
    return {t.strip() for t in value.split(",") if t.strip()}


def num(value):
    return None if value is None or pd.isna(value) or value == "" else float(value)


def shared(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def window_scores(value):
    """ "0-3:0.1;2-5:0.9" -> [(0, 3, 0.1), (2, 5, 0.9)]"""
    windows = []
    if isinstance(value, str):
        for item in value.split(";"):
            span, _, score = item.rpartition(":")
            start, _, end = span.partition("-")
            windows.append((float(start), float(end), float(score)))
    return windows


def starts_over(r, min_score):
    """Window starts the probe scores r's tag at least min_score, None if it
    has no probe scores"""
    if r is None or r.get("source") not in ("probe", "knn"):
        return None
    windows = window_scores(r.get("window_scores"))
    if not windows:
        return None
    return [round(w[0], 3) for w in windows if w[2] >= min_score]


def primary_window(r, args):
    """((start, end), source, score) or None, and the reason"""
    category = r["category"]
    source = r.get("source")
    rms = None
    if num(r.get("rms_start")) is not None:
        rms = (
            (num(r["rms_start"]), num(r["rms_end"])),
            "rms",
            num(r.get("rms_probe_score")),
        )
    best = None
    if r.get("keep") and num(r.get("best_start")) is not None:
        best = ((num(r["best_start"]), num(r["best_end"])), source, num(r["best_score"]))
    if args.labels is not None and r["tag"] not in args.labels:
        return rms, "not a model label"
    if category == "same":
        return rms, "same"
    if category == "moved":
        rms_score = num(r.get("rms_probe_score"))
        clear = (
            rms_score is not None
            and rms_score < args.min_score
            and (num(r.get("shared")) or 0) < args.max_shared
        )
        return (best, "clear_fix") if clear else (rms, "moved_kept_rms")
    if category == "now_dropped":
        if source in ("model_top", "none") or r["tag"] in args.exempt:
            return rms, "exempt_kept_rms"
        return None, "dropped"
    if category == "rms_filtered" and best is not None:
        return best, "rms_filtered_probe"
    if rms is None and best is not None:
        # not tightened, the probe's window keeps it to one 3s track
        return best, f"{category}_probe"
    return rms, category


def confirmed_track(original, track_id, window, tags, source, score):
    track = {"id": track_id}
    if window is None:
        track["start"], track["end"] = original["start"], original["end"]
    else:
        track["start"], track["end"] = round(window[0], 4), round(window[1], 4)
    track["tags"] = tags
    # the "Tracks" track it came from, for its primary_starts / secondary_starts
    track["source_track"] = original.get("id")
    for k in COPY_KEYS:
        if k in original:
            track[k] = original[k]
    track["source"] = source
    if score is not None:
        track["score"] = round(score, 3)
    return track


def write_file(job):
    txt, rows, args = job
    from birdsconfig import RELABEL_MAP
    from utils import get_ebird_id, get_label_to_ebird_map

    ebird_map = get_label_to_ebird_map()

    def to_label(what):
        ebird_id = get_ebird_id(what, ebird_map)
        return RELABEL_MAP.get(ebird_id, ebird_id)

    counts = Counter()
    try:
        with open(txt) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning("Can't read %s %s", txt, e)
        counts["unreadable"] += 1
        return counts
    by_track = defaultdict(list)
    for r in rows:
        by_track[str(r["track_id"])].append(r)

    confirmed = []
    for original in meta.get("Tracks") or []:
        track_id = original.get("id")
        track_rows = by_track.get(str(track_id))
        if not track_rows:
            # not in the probe csv, e.g. no human tags, keep it as it is
            confirmed.append(dict(original))
            counts["track not scored, copied"] += 1
            continue
        primary_row = next((r for r in track_rows if r["tag_type"] == "primary"), None)
        secondary_rows = {r["tag"]: r for r in track_rows if r["tag_type"] == "secondary"}

        original.pop("primary_starts", None)
        original.pop("secondary_starts", None)
        original.pop("probe_starts", None)
        original.pop("knn_starts", None)
        # every label's windows, so the build can check labels from overlaps
        if primary_row is not None:
            for k in ("probe_starts", "knn_starts"):
                value = primary_row.get(k)
                if isinstance(value, str) and value:
                    original[k] = json.loads(value)
        starts = starts_over(primary_row, args.min_score)
        if starts is not None:
            original["primary_starts"] = starts
        secondary_starts = {}
        for tag in original.get("tags", []):
            if tag.get("secondary") and tag.get("what"):
                starts = starts_over(
                    secondary_rows.get(to_label(tag["what"])), args.min_score
                )
                if starts is not None:
                    secondary_starts[tag["what"]] = starts
        if secondary_starts:
            original["secondary_starts"] = secondary_starts

        window, source, score = None, "original", None
        if primary_row is not None:
            if num(primary_row.get("rms_start")) is not None and "best_start" not in original:
                original["best_start"] = num(primary_row["rms_start"])
                original["best_end"] = num(primary_row["rms_end"])
                counts["best_start added to Tracks"] += 1
            chosen, reason = primary_window(primary_row, args)
            counts[f"primary {reason}"] += 1
            if chosen is not None:
                window, source, score = chosen
            elif reason == "dropped":
                source = None

        primary_tags = []
        # secondary windows -> their tags, in the order first seen
        elsewhere = {}
        for tag in original.get("tags", []):
            if not tag.get("secondary"):
                primary_tags.append(tag)
                continue
            r = secondary_rows.get(to_label(tag["what"])) if tag.get("what") else None
            placed = (
                r is not None
                and r["category"] == "new_secondary"
                and num(r.get("best_start")) is not None
                and (args.labels is None or r["tag"] in args.labels)
            )
            if not placed:
                # where it is is unknown, so it's not confirmed
                counts["secondary not placed, not confirmed"] += 1
                continue
            sec_window = (num(r["best_start"]), num(r["best_end"]))
            if (
                source is not None
                and window is not None
                and shared(window, sec_window) >= args.same_overlap
            ):
                primary_tags.append(tag)
                counts["secondary in primary window"] += 1
                continue
            own = {k: v for k, v in tag.items() if k != "secondary"}
            # secondaries overlapping an earlier one's window share its track,
            # so every label is in one 3s track
            group_window = next(
                (w for w in elsewhere if shared(w, sec_window) >= args.same_overlap),
                sec_window,
            )
            group = elsewhere.setdefault(group_window, {"tags": [], "rows": []})
            # one tag per track (the build keeps only those), birds sharing
            # the window after the first are its secondaries
            if group["tags"]:
                own["secondary"] = True
            group["tags"].append(own)
            group["rows"].append(r)
            counts["secondary own track"] += 1

        if source is None:
            counts["primary track dropped"] += 1
        else:
            if window is None:
                counts["primary kept at full track length, no window"] += 1
            confirmed.append(
                confirmed_track(original, track_id, window, primary_tags, source, score)
            )
        for i, (sec_window, group) in enumerate(elsewhere.items()):
            suffix = "_secondary" if i == 0 else f"_secondary_{i + 1}"
            sources = {r.get("source") for r in group["rows"]}
            confirmed.append(
                confirmed_track(
                    original,
                    f"{track_id}{suffix}",
                    sec_window,
                    group["tags"],
                    sources.pop() if len(sources) == 1 else ";".join(sorted(sources)),
                    max(num(r["best_score"]) or 0 for r in group["rows"]),
                )
            )
    for k in OLD_KEYS:
        meta.pop(k, None)
    meta[CONFIRMED_KEY] = confirmed
    counts["confirmed tracks"] += len(confirmed)
    if not args.dry_run:
        from addtracks import write_meta

        write_meta(Path(txt), meta)
    counts["files"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("compare_csv", help="per tag csv from compare_best_windows.py")
    parser.add_argument(
        "--labels",
        help="The model's labels, comma separated or a file of them (one per "
        "line, comma separated or a json list). Default all labels",
    )
    parser.add_argument(
        "--exempt",
        default="",
        help="Labels whose primary keeps the rms window when the probe and knn "
        "don't place it, comma separated or a file. Default none, every label "
        "goes through the same rules and is worked out again with the next "
        "model",
    )
    parser.add_argument("--min-score", type=float, default=0.4)
    parser.add_argument(
        "--max-shared",
        type=float,
        default=1.5,
        help="A probe window sharing at least this many seconds with the rms "
        "window isn't a clear fix, the rms window is kept",
    )
    parser.add_argument(
        "--same-overlap",
        type=float,
        default=2.0,
        help="A secondary sharing at least this many seconds with the primary's "
        "window goes on the primary's track",
    )
    parser.add_argument("--dry-run", action="store_true", help="Count, don't write")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.labels = parse_labels(args.labels)
    args.exempt = parse_labels(args.exempt) or set()

    df = pd.read_csv(args.compare_csv, low_memory=False)
    if "source" not in df.columns or "keep" not in df.columns:
        parser.error(
            f"{args.compare_csv} has no source or keep column, remake it from a "
            "current best_windows.py csv"
        )
    by_file = defaultdict(list)
    for r in df.to_dict("records"):
        by_file[r["file"]].append(r)
    jobs = [(txt, rows, args) for txt, rows in by_file.items()]
    logging.info(
        "%s %s tags in %s files%s",
        "Checking" if args.dry_run else "Writing",
        len(df),
        len(jobs),
        ", dry run" if args.dry_run else "",
    )
    total = Counter()
    with multiprocessing.Pool(args.workers) as pool:
        for i, counts in enumerate(pool.imap_unordered(write_file, jobs, chunksize=64)):
            total.update(counts)
            if i % 20000 == 0:
                logging.info("%s / %s files", i, len(jobs))
    for k, v in sorted(total.items()):
        logging.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
