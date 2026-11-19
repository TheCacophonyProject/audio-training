"""Add recording info from tier1 clip filenames to each clip's metadata .txt.

Train_001 clips, e.g. AE142_BIRX_120321_102857_016.flac (DOC Tier 1 plot,
recorder at the plot, date, start time, clip number):
    id: "AE142_BIRX_120321_102857", site: "AE142", device: "BIRX",
    rec_datetime: "2012-03-21T10:28:57", track id: 16
Train_002 clips, e.g. 00024E7C-..._27.flac (recording UUID, clip start second):
    id: "00024E7C-...", track id: 27
Both also get the csv start/end (seconds into the original recording) on the
track as offset_start, offset_end. The clip's original id (the csv row) is kept
as backup_id.

The metadata is <tier1 dir>/<Train_00N>/train_audio/<label>/<clip>.txt as written
by otherdata.tier1_data. Clips without a .txt are counted and skipped.

python3 scripts/tier1_add_meta.py /data/.../tier1 [--dry-run]
"""
import argparse, csv, json, re
from collections import Counter
from datetime import datetime
from pathlib import Path

# some plots have the recorder letter (AR126X) or BIR (L169BIR) appended, the
# site is the leading letters and number
PLOT_RE = re.compile(
    r"^([A-Z]+\d+)[A-Z]*_(BIR[A-Z])_(\d{6}|\d{8})_(\d{6})_(\d+)\.flac$"
)
LATEST_YEAR = 2025
UUID_RE = re.compile(r"^([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})_(\d+)\.flac$")


def parse_datetime(date, time):
    # YYYYMMDD or YYMMDD, a few (e.g. CV52_BIRP_311015) are DDMMYY, which can
    # parse as a YYMMDD date in the future
    formats = ["%Y%m%d"] if len(date) == 8 else ["%y%m%d", "%d%m%y"]
    for f in formats:
        try:
            dt = datetime.strptime(date + time, f + "%H%M%S")
        except ValueError:
            continue
        if dt.year <= LATEST_YEAR:
            return dt
    return None


def clip_meta(name):
    """(recording fields, track id) for a clip filename, or None"""
    m = PLOT_RE.match(name)
    if m:
        site, recorder, date, time, clip = m.groups()
        dt = parse_datetime(date, time)
        if dt is None:
            return None
        # BIRA..BIRX repeat at every plot, a recorder is site + device
        return {
            # the original recording, the clip name without the clip number
            "id": name.rsplit("_", 1)[0],
            "site": site,
            "device": recorder,
            "rec_datetime": dt.isoformat(),
        }, int(clip)
    m = UUID_RE.match(name)
    if m:
        uuid, clip = m.groups()
        return {"id": uuid}, int(clip)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path, help="tier1 dir containing Train_001, Train_002")
    p.add_argument("--folders", nargs="+", default=["Train_001", "Train_002"])
    p.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = p.parse_args()

    stats = Counter()
    for folder in a.folders:
        csvs = list((a.dir / folder).glob("*.csv"))
        if len(csvs) != 1:
            print(f"{folder}: expected one csv, found {len(csvs)}")
            continue
        with open(csvs[0], newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            filename = row["filename"]
            parsed = clip_meta(Path(filename).name)
            if parsed is None:
                stats[f"{folder} unparsed name"] += 1
                print("Can't parse", filename)
                continue
            fields, track_id = parsed
            meta_file = (a.dir / folder / "train_audio" / filename).with_suffix(".txt")
            if not meta_file.exists():
                stats[f"{folder} no metadata file"] += 1
                continue
            meta = json.loads(meta_file.read_text())
            # Recording reads Tracks before tracks
            tracks = meta.get("Tracks") or meta.get("tracks") or []
            if len(tracks) != 1:
                stats[f"{folder} not 1 track, track id not set"] += 1
            else:
                tracks[0]["id"] = track_id
                # where the clip is in the original recording, seconds
                tracks[0]["offset_start"] = float(row["start"])
                tracks[0]["offset_end"] = float(row["end"])
            # keep the original id (the csv row) once, so re-running doesn't
            # overwrite it with the recording id
            if "backup_id" not in meta:
                meta["backup_id"] = meta.get("id")
            meta.update(fields)
            if not a.dry_run:
                meta_file.write_text(json.dumps(meta, indent=4))
            else:
                dry_test = Path(meta_file.name)
                dry_test.write_text(json.dumps(meta, indent=4))
                print("Saving too ",dry_test)
                break
            stats[f"{folder} updated"] += 1

    for k, v in sorted(stats.items()):
        print(f"{k}: {v}")
    if a.dry_run:
        print("dry run, nothing written")


if __name__ == "__main__":
    main()
