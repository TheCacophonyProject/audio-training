"""How much evidence there is of a morepork in each tier1 morepork clip's best
3 second window (track best_start - best_end).

For each track's best window:
  birdnet: match (Morepork), related (another owl, same genus / family),
      different (another bird only) or no_bird, from the birdnet detections
      overlapping the window
  rms: bird_rms (the track's bird_rms_bin frequencies) once rms noise is
      removed, as build does. Active if it stays RMS_ACTIVE_DB above the
      track's background (20th percentile) for RMS_MIN_ACTIVE seconds in
      the window, the same rule build uses to trim tracks
  signal: a signal from the tracking algorithm overlapping the window, any
      frequency and in the morepork band (300Hz - 1.2kHz)

--drop-a / --drop-b write the tracks to leave out of training, one
"rec_id,track_id" per line for tfdataset --drop-tracks:
  A: birdnet doesn't find a morepork or other owl in the best window
  B: A, plus owls with neither rms active nor a morepork band signal

python3 scripts/tier1_morepork_check.py tier1-moreporks --csv tier1-morepork-check.csv \\
    --drop-a tier1-morepork-drop-a.txt --drop-b tier1-morepork-drop-b.txt
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import birdnetcompare as bc  # noqa: E402
from audiodataset import (  # noqa: E402
    RMS_ACTIVE_DB,
    RMS_MIN_ACTIVE,
    rms_best_segment,
)

MOREPORK_MIN_FREQ = bc.SPECIES_SIGNAL_MIN_FREQ["morepo2"]
MOREPORK_MAX_FREQ = 1200


def window_rms(track, start, end):
    """(active seconds, peak dB above background) of the track's noise removed
    bird rms in start - end"""
    _, _, rms, noise_removed, frame_length, _ = rms_best_segment(
        track, track["start"], track["end"]
    )
    rms_db = 20 * np.log10(np.maximum(np.asarray(rms), 1e-10))
    keep = ~noise_removed
    if not keep.any():
        return 0.0, None
    floor = np.percentile(rms_db[keep], 20)
    first = max(0, int((start - track["start"]) / frame_length))
    last = min(len(rms_db), int(np.ceil((end - track["start"]) / frame_length)))
    window_db = rms_db[first:last]
    window_keep = keep[first:last]
    if len(window_db) == 0:
        return 0.0, None
    above = (window_db > floor + RMS_ACTIVE_DB) & window_keep
    min_frames = max(1, int(round(RMS_MIN_ACTIVE / frame_length)))
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(np.int8), 0]))
    active = sum(
        e - s for s, e in zip(edges[::2], edges[1::2]) if e - s >= min_frames
    )
    peak = float(np.max(window_db[window_keep]) - floor) if window_keep.any() else None
    return active * frame_length, peak


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dir")
    p.add_argument("--csv", help="Write a row per track")
    p.add_argument("--drop-a", help="Write version A's tracks to drop")
    p.add_argument("--drop-b", help="Write version B's tracks to drop")
    p.add_argument(
        "--taxonomy", default=str(Path(bc.__file__).parent / "eBird_taxonomy_v2024.csv")
    )
    a = p.parse_args()
    taxonomy = bc.load_taxonomy(a.taxonomy)

    rows = []
    for f in sorted(Path(a.dir).glob("*.txt")):
        meta = json.load(open(f))
        detections = meta.get("birdnet") or []
        for t in meta.get("Tracks", []):
            start = t.get("best_start", t["start"])
            end = t.get("best_end", min(t["start"] + 3, t["end"]))
            tag_taxa = [
                x for x in (bc.tag_taxon(g, taxonomy) for g in t["tags"]) if x is not None
            ]
            found = bc.overlapping(detections, start, end, 0)
            status, best, birds = bc.evaluate(found, tag_taxa, False, [], taxonomy)
            match = [d for d, x in birds if x["code"] == "morepo2"]
            signals = [s for s in meta.get("signal") or [] if s[0] < end and s[1] > start]
            active, peak = window_rms(t, start, end)
            rows.append(
                {
                    "file": f.name,
                    "rec_id": meta["id"],
                    "track_id": t["id"],
                    "best_start": round(start, 2),
                    "best_end": round(end, 2),
                    "birdnet": status,
                    "birdnet_best": "" if best is None else best["common_name"],
                    "birdnet_confidence": "" if best is None else best["confidence"],
                    "morepork_confidence": max((d["confidence"] for d in match), default=""),
                    "birdnet_any": bool(detections),
                    "rms_active_s": round(active, 2),
                    "rms_peak_db": "" if peak is None else round(peak, 1),
                    "signals": len(signals),
                    "signals_band": sum(
                        s[3] > MOREPORK_MIN_FREQ and s[2] < MOREPORK_MAX_FREQ
                        for s in signals
                    ),
                }
            )
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    n = len(rows)

    def pct(k):
        return f"{k:6} ({100 * k / n:5.1f}%)"

    print(f"{n} tracks in {len({r['file'] for r in rows})} files, best 3s window of each\n")
    status = Counter(r["birdnet"] for r in rows)
    print("BirdNET in the window")
    for s in bc.STATUS_ORDER:
        print(f"  {s:10} {pct(status[s])}")
    print(f"  (files with no birdnet detections at all {pct(sum(not r['birdnet_any'] for r in rows))})")
    for s in (bc.RELATED, bc.DIFFERENT):
        names = Counter(r["birdnet_best"] for r in rows if r["birdnet"] == s)
        print(f"  top {s}: " + ", ".join(f"{k} {v}" for k, v in names.most_common(8)))

    rms = [r["rms_active_s"] > 0 for r in rows]
    sig = [r["signals"] > 0 for r in rows]
    sig300 = [r["signals_band"] > 0 for r in rows]
    print(f"\nRMS active (>{RMS_ACTIVE_DB}dB for {RMS_MIN_ACTIVE}s) {pct(sum(rms))}")
    print(f"Tracker signal, any freq     {pct(sum(sig))}")
    print(
        f"Tracker signal, {MOREPORK_MIN_FREQ}-{MOREPORK_MAX_FREQ}Hz   {pct(sum(sig300))}"
    )

    owl = [r["birdnet"] in (bc.MATCH, bc.RELATED) for r in rows]
    print("\nCombined (birdnet morepork/owl, rms active, morepork band signal)")
    combos = Counter(zip(owl, rms, sig300))
    for (o, r_, s), k in sorted(combos.items(), key=lambda kv: -kv[1]):
        print(
            f"  birdnet owl {'Y' if o else 'n'}  rms {'Y' if r_ else 'n'}"
            f"  signal {'Y' if s else 'n'}  {pct(k)}"
        )
    none = sum(not (o or r_ or s) for o, r_, s in zip(owl, rms, sig300))
    print(f"\nNo evidence at all {pct(none)}")

    drop_a = [r for r, o in zip(rows, owl) if not o]
    drop_b = [
        r for r, o, r_, s in zip(rows, owl, rms, sig300) if not o or not (r_ or s)
    ]
    for path, drop, name in ((a.drop_a, drop_a, "A"), (a.drop_b, drop_b, "B")):
        print(f"{name} drops {pct(len(drop))}")
        if path:
            with open(path, "w") as fh:
                fh.writelines(f"{r['rec_id']},{r['track_id']}\n" for r in drop)
            print(f"  wrote {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
