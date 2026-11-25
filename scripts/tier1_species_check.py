"""BirdNET against Perch for one species' tier1 clips, in each track's best 3s
window (best_start - best_end), to see what would be lost by keeping only the
tracks one or both agree with.

For each track:
  birdnet: species (birdnet hears the species), genus (only a species of the
      same genus e.g. tomtit -> NZ robin), other (only other birds), none.
      Detections at least --birdnet-conf overlapping the window
  perch: from a perch flags csv (e.g. flags-best.csv), agrees if the probe's
      top label is the species or its own prob >= 0.5, or the knn label is the
      species with fraction >= 0.5. confident_other if the probe's top label is
      another label with prob >= --perch-other-prob and knn doesn't agree
  signal: a tracker signal in the window overlapping --band

--drop-neither writes the tracks neither birdnet nor perch agree with,
--drop-confident those perch confidently calls something else and birdnet
doesn't hear the species in. "rec_id,track_id" lines for --drop-tracks.

python3 scripts/tier1_species_check.py /data2/audio-training/tier1 \\
    --species tomtit1 --perch flags-best.csv --csv tier1-tomtit-check.csv \\
    --drop-neither tier1-tomtit-drop-neither.txt \\
    --drop-confident tier1-tomtit-drop-confident.txt
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import birdnetcompare as bc  # noqa: E402

BANDS = {"tomtit1": (3000, 8000), "morepo2": (300, 1200)}


def load_perch(path, species):
    """{file name: [rows]} of the species' perch rows"""
    rows = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["label"] == species:
                rows[Path(row["file"]).name].append(row)
    return rows


def perch_row(rows, start):
    # a clip usually has one track, otherwise the row whose rms window starts
    # nearest the track's best window
    if not rows:
        return None
    return min(rows, key=lambda r: abs(float(r["rms_start"] or 0) - start))


def perch_result(row, species, other_prob):
    if row is None:
        return "missing", ""
    own = float(row["probe_own_prob"] or 0)
    probe_ok = row["probe_label"] == species or own >= 0.5
    knn_ok = (
        row["knn_label"] == species and float(row["knn_label_fraction"] or 0) >= 0.5
    )
    if probe_ok or knn_ok:
        return "agrees", row["probe_label"]
    if float(row["probe_prob"] or 0) >= other_prob:
        return "confident_other", row["probe_label"]
    return "unsure", row["probe_label"]


def birdnet_result(detections, start, end, species, species_genus, min_conf, taxonomy):
    found = [
        d
        for d in bc.overlapping(detections, start, end, 0)
        if d["confidence"] >= min_conf
    ]
    birds = [
        (d, t) for d in found if (t := bc.detection_taxon(d, taxonomy)) is not None
    ]
    heard = [d["confidence"] for d, t in birds if t["code"] == species]
    if heard:
        return "species", max(heard), ""
    best = max(birds, key=lambda dt: dt[0]["confidence"], default=None)
    if best is None:
        return "none", "", ""
    if any(bc.genus(t) == species_genus for _, t in birds):
        return "genus", "", best[1]["common"]
    return "other", "", best[1]["common"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dir", help="tier1 dir, clips are in **/train_audio/<species>/*.txt")
    p.add_argument("--species", default="tomtit1")
    p.add_argument("--perch", required=True, help="perch flags csv e.g. flags-best.csv")
    p.add_argument("--birdnet-conf", type=float, default=0.1, help="build's default")
    p.add_argument("--perch-other-prob", type=float, default=0.8)
    p.add_argument("--band", type=float, nargs=2, help="Signal Hz, default per species")
    p.add_argument("--csv")
    p.add_argument("--drop-neither")
    p.add_argument("--drop-confident")
    p.add_argument(
        "--taxonomy", default=str(Path(bc.__file__).parent / "eBird_taxonomy_v2024.csv")
    )
    a = p.parse_args()
    taxonomy = bc.load_taxonomy(a.taxonomy)
    species_genus = bc.genus(taxonomy[0][a.species])
    band = a.band or BANDS.get(a.species)
    perch = load_perch(a.perch, a.species)

    files = sorted(Path(a.dir).glob(f"**/train_audio/{a.species}/*.txt"))
    print(f"{len(files)} {a.species} clips, {len(perch)} with perch rows")
    rows = []
    no_birdnet = 0
    for f in files:
        meta = json.load(open(f))
        if "birdnet" not in meta:
            no_birdnet += 1
        detections = meta.get("birdnet") or []
        for t in meta.get("Tracks", []):
            start = t.get("best_start", t["start"])
            end = t.get("best_end", min(t["start"] + 3, t["end"]))
            bn, bn_conf, bn_other = birdnet_result(
                detections,
                start,
                end,
                a.species,
                species_genus,
                a.birdnet_conf,
                taxonomy,
            )
            pr, pr_label = perch_result(
                perch_row(perch.get(f.name), start), a.species, a.perch_other_prob
            )
            signals = [
                s for s in meta.get("signal") or [] if s[0] < end and s[1] > start
            ]
            in_band = band is not None and any(
                s[3] > band[0] and s[2] < band[1] for s in signals
            )
            rows.append(
                {
                    "file": str(f),
                    "set": next((x for x in f.parts if x.startswith("Train_")), ""),
                    "rec_id": meta["id"],
                    "track_id": t["id"],
                    "best_start": round(start, 2),
                    "best_end": round(end, 2),
                    "secondary": ";".join(
                        x["what"] for x in t.get("tags", []) if x.get("secondary")
                    ),
                    "birdnet": bn,
                    "birdnet_conf": bn_conf,
                    "birdnet_other": bn_other,
                    "perch": pr,
                    "perch_label": pr_label,
                    "band_signal": in_band,
                }
            )
    if not rows:
        print("No tracks found")
        return
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    n = len(rows)

    def pct(k, total=n):
        return f"{k:6} ({100 * k / max(1, total):5.1f}%)"

    print(f"{n} tracks, {no_birdnet} clips have no birdnet results\n")
    for s in sorted({r["set"] for r in rows}):
        sub = [r for r in rows if r["set"] == s]
        bn = sum(r["birdnet"] == "species" for r in sub)
        pr = sum(r["perch"] == "agrees" for r in sub)
        print(
            f"{s or 'other'}: {len(sub)} tracks, birdnet hears {pct(bn, len(sub))}, perch agrees {pct(pr, len(sub))}"
        )

    print("\nBirdNET in the best window")
    for k, v in Counter(r["birdnet"] for r in rows).most_common():
        print(f"  {k:16} {pct(v)}")
    print("Perch")
    for k, v in Counter(r["perch"] for r in rows).most_common():
        print(f"  {k:16} {pct(v)}")
    if band:
        print(
            f"Tracker signal {band[0]:.0f}-{band[1]:.0f}Hz {pct(sum(r['band_signal'] for r in rows))}"
        )

    print("\nBirdNET x Perch (rows birdnet, columns perch)")
    perch_keys = ["agrees", "unsure", "confident_other", "missing"]
    print(f"  {'':10}" + "".join(f"{k:>17}" for k in perch_keys))
    for b in ["species", "genus", "other", "none"]:
        cells = [
            sum(r["birdnet"] == b and r["perch"] == pk for r in rows)
            for pk in perch_keys
        ]
        print(f"  {b:10}" + "".join(f"{c:>9} {100 * c / n:5.1f}%" for c in cells))

    for status in ("genus", "other"):
        names = Counter(r["birdnet_other"] for r in rows if r["birdnet"] == status)
        print(
            f"\nBirdNET {status} top: "
            + ", ".join(f"{k} {v}" for k, v in names.most_common(8))
        )
    names = Counter(r["perch_label"] for r in rows if r["perch"] == "confident_other")
    print(
        "Perch confident other top: "
        + ", ".join(f"{k} {v}" for k, v in names.most_common(8))
    )

    keep_rules = {
        "birdnet hears it": lambda r: r["birdnet"] == "species",
        "perch agrees": lambda r: r["perch"] == "agrees",
        "birdnet or perch": lambda r: r["birdnet"] == "species"
        or r["perch"] == "agrees",
        "birdnet and perch": lambda r: r["birdnet"] == "species"
        and r["perch"] == "agrees",
        "not (perch confident other, birdnet doesn't hear it)": lambda r: not (
            r["perch"] == "confident_other" and r["birdnet"] != "species"
        ),
    }
    print("\nKeeping only tracks where")
    for name, rule in keep_rules.items():
        lost = sum(not rule(r) for r in rows)
        print(f"  {name:55} loses {pct(lost)}")

    drops = (
        (a.drop_neither, lambda r: not keep_rules["birdnet or perch"](r)),
        (a.drop_confident, lambda r: not keep_rules[list(keep_rules)[-1]](r)),
    )
    for path, drop in drops:
        if path:
            dropped = [r for r in rows if drop(r)]
            with open(path, "w") as fh:
                fh.writelines(f"{r['rec_id']},{r['track_id']}\n" for r in dropped)
            print(f"Wrote {len(dropped)} to {path}")


if __name__ == "__main__":
    main()
