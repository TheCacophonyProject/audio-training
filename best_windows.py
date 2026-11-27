#!/usr/bin/env python3
"""For every track in a perchembed --load-probe output (tier1-probe-<model>.csv)
pick, for each of its tags, primary and secondary, the window the probe scores
highest for that tag, the most likely place in the track that bird is.

Tier1 tags are per track and the birds usually aren't calling the whole
length, so this narrows each tag to one window.

Tags the probe has no class for fall back to the window where the model's own
top label is that tag, if there is one, otherwise they are written without a
window so they can be found.

A tag's window is kept if its score is at least --min-score. A tag under it
is still kept if --knn-fraction of a window's nearest neighbours (from other
recordings) have that tag as their most common one, then the window with the
most neighbour agreement is used, the probe score breaking ties, and its
source is knn. Neighbours share track level tags so they say a lot less about
where in the track the bird is than whether it's there. With --write-meta
every kept tag is added to the recording's metadata as a track of its own over
that window, tagged only with that tag (copied from the original track, a
secondary becomes the track's tag). The original tracks aren't changed, tracks
this added before are replaced.

    python3 best_windows.py tier1-probe-spectv12AutoNoiseClean.csv \
        --out tier1-best-windows-spectv12AutoNoiseClean.csv --min-score 0.4
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

GENERATED_BY = "best_windows"

COLUMNS = [
    "file",
    "audio",
    "recording_id",
    "track_id",
    "track_start",
    "track_end",
    "window_start",
    "window_end",
    "label",
    "secondary_label",
    "top_label",
    "top_prob",
    "probe_scores",
    "knn_label",
    "knn_label_fraction",
]


def split(value):
    return [t for t in str(value).split(";") if t] if isinstance(value, str) else []


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("probe_csv", help="tier1-probe-<model>.csv from perchembed")
    parser.add_argument("--out", required=True, help="CSV to write")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.4,
        help="Keep a tag's best window only if it scores at least this",
    )
    parser.add_argument(
        "--knn-fraction",
        type=float,
        default=0.7,
        help="Keep a tag under --min-score if at least this fraction of a "
        "window's neighbours have it as their most common tag, 0 never",
    )
    parser.add_argument(
        "--write-meta",
        action="store_true",
        help="Add a track per kept tag to each recording's metadata txt",
    )
    parser.add_argument("--chunksize", type=int, default=50000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tracks = {}
    # (track key, tag) -> [(window_start, window_end, probe score, model prob,
    # fraction of neighbours with the tag as their most common)]
    windows = defaultdict(list)
    # track key -> {label: [window starts]} for every label, not only the
    # track's tags, the probe scores at least --min-score / knn agrees with at
    # least --knn-fraction, so the build can check labels from overlaps
    probe_starts = defaultdict(lambda: defaultdict(list))
    knn_starts = defaultdict(lambda: defaultdict(list))
    probe_classes = None
    for chunk in pd.read_csv(
        args.probe_csv, usecols=COLUMNS, chunksize=args.chunksize, low_memory=False
    ):
        for row in chunk.itertuples(index=False):
            key = (row.audio, row.track_id)
            primary = split(row.label)
            secondary = [t for t in split(row.secondary_label) if t not in primary]
            if key not in tracks:
                tracks[key] = dict(
                    file=row.file,
                    audio=row.audio,
                    recording_id=row.recording_id,
                    track_id=row.track_id,
                    track_start=row.track_start,
                    track_end=row.track_end,
                    tags=[(t, "primary") for t in primary]
                    + [(t, "secondary") for t in secondary],
                )
            scores = {}
            if isinstance(row.probe_scores, str):
                for item in row.probe_scores.split(";"):
                    label, _, score = item.rpartition(":")
                    scores[label] = float(score)
                if probe_classes is None:
                    probe_classes = set(scores)
                for label, score in scores.items():
                    if score >= args.min_score:
                        probe_starts[key][label].append(round(row.window_start, 3))
            if (
                args.knn_fraction > 0
                and isinstance(row.knn_label, str)
                and pd.notna(row.knn_label_fraction)
                and float(row.knn_label_fraction) >= args.knn_fraction
            ):
                knn_starts[key][row.knn_label].append(round(row.window_start, 3))
            for tag, _ in tracks[key]["tags"]:
                model_prob = row.top_prob if row.top_label == tag else None
                knn = (
                    float(row.knn_label_fraction)
                    if row.knn_label == tag and pd.notna(row.knn_label_fraction)
                    else 0.0
                )
                windows[(key, tag)].append(
                    (
                        row.window_start,
                        row.window_end,
                        scores.get(tag),
                        model_prob,
                        knn,
                    )
                )
        logging.info("Read %s tracks", len(tracks))
    probe_classes = probe_classes or set()

    out_rows = []
    for key, track in tracks.items():
        for tag, tag_type in track["tags"]:
            tag_windows = sorted(windows[(key, tag)])
            result = dict(
                file=track["file"],
                audio=track["audio"],
                recording_id=track["recording_id"],
                track_id=track["track_id"],
                track_start=track["track_start"],
                track_end=track["track_end"],
                tag=tag,
                tag_type=tag_type,
                windows=len(tag_windows),
                source="",
                best_start="",
                best_end="",
                best_score="",
                second_score="",
                margin="",
                window_scores="",
                knn_score=max(w[4] for w in tag_windows),
                # on the primary row, the track's whole {label: [starts]}
                probe_starts="",
                knn_starts="",
            )
            if tag_type == "primary":
                result["probe_starts"] = json.dumps(
                    {k: sorted(v) for k, v in sorted(probe_starts[key].items())}
                )
                result["knn_starts"] = json.dumps(
                    {k: sorted(v) for k, v in sorted(knn_starts[key].items())}
                )
            if tag in probe_classes:
                result["source"] = "probe"
                ranked = sorted(tag_windows, key=lambda w: -w[2])
                best = ranked[0]
                second = ranked[1][2] if len(ranked) > 1 else None
                result.update(
                    best_start=best[0],
                    best_end=best[1],
                    best_score=round(best[2], 3),
                    second_score="" if second is None else round(second, 3),
                    margin="" if second is None else round(best[2] - second, 3),
                    window_scores=";".join(
                        f"{w[0]:g}-{w[1]:g}:{w[2]:.3f}" for w in tag_windows
                    ),
                )
            else:
                model_hits = [w for w in tag_windows if w[3] is not None]
                if model_hits:
                    best = max(model_hits, key=lambda w: w[3])
                    result.update(
                        source="model_top",
                        best_start=best[0],
                        best_end=best[1],
                        best_score=round(best[3], 3),
                    )
                else:
                    result["source"] = "none"
            result["keep"] = (
                result["best_score"] != "" and result["best_score"] >= args.min_score
            )
            if (
                not result["keep"]
                and args.knn_fraction > 0
                and result["knn_score"] >= args.knn_fraction
            ):
                best = max(tag_windows, key=lambda w: (w[4], w[2] or w[3] or 0))
                score = best[2] if best[2] is not None else best[3]
                result.update(
                    source="knn",
                    best_start=best[0],
                    best_end=best[1],
                    best_score="" if score is None else round(score, 3),
                    keep=True,
                )
            out_rows.append(result)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    summary = pd.DataFrame(out_rows)
    logging.info(
        "Wrote %s tags of %s tracks to %s, by source %s",
        len(out_rows),
        len(tracks),
        args.out,
        summary.source.value_counts().to_dict(),
    )
    kept = summary[summary.keep]
    logging.info(
        "%s tags kept with a score >= %s or knn >= %s, by tag type %s, by source %s",
        len(kept),
        args.min_score,
        args.knn_fraction,
        kept.tag_type.value_counts().to_dict(),
        kept.source.value_counts().to_dict(),
    )
    if args.write_meta:
        write_tracks(tracks, out_rows)


def write_tracks(tracks, out_rows):
    from addtracks import write_meta
    from birdsconfig import RELABEL_MAP
    from utils import get_ebird_id, get_label_to_ebird_map

    # same mapping perchembed uses for label, so a code can be matched back to
    # the tag on the track it came from
    ebird_map = get_label_to_ebird_map()

    def to_label(what):
        ebird_id = get_ebird_id(what, ebird_map)
        return RELABEL_MAP.get(ebird_id, ebird_id)

    by_file = defaultdict(list)
    for r in out_rows:
        if r["keep"]:
            by_file[tracks[(r["audio"], r["track_id"])]["file"]].append(r)

    added = unmatched = 0
    for txt, rows in by_file.items():
        txt = Path(txt)
        with open(txt) as f:
            meta = json.load(f)
        meta_tracks = [
            t for t in meta.get("Tracks") or [] if t.get("generated_by") != GENERATED_BY
        ]
        originals = {str(t.get("id")): t for t in meta_tracks}
        new_tracks = []
        for r in rows:
            original = originals.get(str(r["track_id"]))
            tag = None
            if original is not None:
                tag = next(
                    (
                        g
                        for g in original.get("tags", [])
                        if g.get("automatic") is False
                        and g.get("what")
                        and to_label(g["what"]) == r["tag"]
                    ),
                    None,
                )
            if tag is None:
                unmatched += 1
                logging.warning(
                    "No tag %s on track %s of %s", r["tag"], r["track_id"], txt
                )
                continue
            tag = {k: v for k, v in tag.items() if k != "secondary"}
            track = {
                "id": f"{original.get('id')}_{r['tag']}",
                "start": float(r["best_start"]),
                "end": float(r["best_end"]),
                "tags": [tag],
                "generated_by": GENERATED_BY,
                "source_track": original.get("id"),
                "tag_type": r["tag_type"],
                "score": r["best_score"],
                "score_source": r["source"],
                "knn_score": r["knn_score"],
            }
            for k in ("minFreq", "maxFreq"):
                if k in original:
                    track[k] = original[k]
            new_tracks.append(track)
        meta["Tracks"] = meta_tracks + new_tracks
        write_meta(txt, meta)
        added += len(new_tracks)
    logging.info(
        "Added %s tracks to %s metadata files, %s tags not found on their track",
        added,
        len(by_file),
        unmatched,
    )


if __name__ == "__main__":
    main()
