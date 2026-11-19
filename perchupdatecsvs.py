"""Adds secondary tags and rec_datetime to existing perchembed shard csvs and
updates their recording_id, and track_id for Train_001 and Train_002 txts, by
reloading the txt file each row references.
Only the csvs change, rows keep their order so row i still matches embedding i
of the shard's npy."""

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path

from perchembed import (
    FIELDNAMES,
    ID_PREFIX_DIRS,
    rec_datetime,
    recording_id,
    secondary_tags,
    shard_paths,
)


def load_meta(txt, cache):
    if txt not in cache:
        try:
            with open(txt) as f:
                meta = json.load(f)
            cache[txt] = meta if isinstance(meta, dict) else None
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            cache[txt] = None
    return cache[txt]


def match_track(row, tracks):
    """The txt's track for a csv row and None, or None and why it wasn't found.
    Track ids in Train_001 and Train_002 txts have been changed but they only
    have one track, elsewhere the ids are unchanged so are matched on"""
    if any(prefix in row["file"] for prefix in ID_PREFIX_DIRS):
        if len(tracks) != 1:
            return None, "Train txt without exactly one track"
        return tracks[0], None
    for t in tracks:
        if str(t.get("id")) == str(row["track_id"]):
            return t, None
    return None, "track not found"


def update_row(row, meta, changes):
    new_id = recording_id(row["file"], meta)
    if new_id is not None and new_id != row.get("recording_id"):
        row["recording_id"] = new_id
        changes["recording_id"] += 1

    when = rec_datetime(meta)
    if when and when != row.get("rec_datetime"):
        row["rec_datetime"] = when
        changes["rec_datetime"] += 1

    track, problem = match_track(row, meta.get("Tracks", []))
    if track is None:
        changes[problem] += 1
        return
    track_id = str(track.get("id"))
    if track_id != str(row["track_id"]):
        row["track_id"] = track_id
        changes["track_id"] += 1
    secondary = ";".join(t["what"] for t in secondary_tags(track))
    if secondary != row.get("secondary_tags", ""):
        row["secondary_tags"] = secondary
        changes["secondary_tags"] += 1


def update_shard(csv_path, cache, dry_run):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        existing = list(reader.fieldnames or [])
        rows = list(reader)
    # the current columns in order, then any extra ones the csv had
    fieldnames = FIELDNAMES + [c for c in existing if c not in FIELDNAMES]
    changes = Counter()
    for row in rows:
        meta = load_meta(row["file"], cache)
        if meta is None:
            changes["txt missing"] += 1
            continue
        update_row(row, meta, changes)
    for row in rows:
        for column in fieldnames:
            row.setdefault(column, "")
    logging.info("%s: %s rows %s", csv_path, len(rows), dict(changes) or "unchanged")
    updated = any(
        changes[c]
        for c in ("recording_id", "track_id", "rec_datetime", "secondary_tags")
    )
    if dry_run or (not updated and existing == fieldnames):
        return changes
    # write to a temp name then rename so a crash never leaves a partial csv
    tmp = csv_path.with_name(csv_path.stem + ".tmp.csv")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out",
        nargs="?",
        default="perch-embeddings",
        help="Shard directory made by perchembed --out",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Also update shards in sub directories"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change only"
    )
    args = parser.parse_args()

    shards = shard_paths(args.out, args.recursive)
    if not shards:
        logging.info("No shards in %s", args.out)
        return
    cache = {}
    total = Counter()
    for npy in shards:
        total += update_shard(npy.with_suffix(".csv"), cache, args.dry_run)
        # rows are grouped by file so a shard's txts are rarely needed again
        cache.clear()
    logging.info(
        "%s %s shards: %s",
        "Would update" if args.dry_run else "Updated",
        len(shards),
        dict(total),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
