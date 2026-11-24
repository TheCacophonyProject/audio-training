"""Summarise summit-road tracks and model predictions, then print the bird
calls made at night.

Writes a CSV of every track sorted by NZ time, with each model's mean window
score per label above --threshold, and marker rows at each sunrise/sunset.
Sunrise/sunset are calculated per day for the recording's location (NOAA
solar equations).

Then prints every track a model calls a bird (mean score of a bird label above
--threshold) between --hours after sunset and --hours before the next sunrise.

python3 scripts/summit_night_calls.py --hours 2
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

NZ = ZoneInfo("Pacific/Auckland")
MODELS = ["spectv10-fine-val_pr_auc", "spectv11NoiseLbael", "spectv11NoiseShare"]
TAG_MODELS = ["Master", "efficientnetv2bm-kokako", "efficientnetv2b3-rms", "human"]
KOKAKO = "efficientnetv2bm-kokako"
# kokako model common names to the spect model labels, so they line up
KOKAKO_LABELS = {
    "morepork": "morepo2", "tomtit": "tomtit1", "kaka": "nezkak1", "kiwi": "kiwi",
    "blackbird": "eurbla", "chaffinch": "comcha", "redpoll": "comred",
    "silvereye": "silver3", "yellowhammer": "yellow2", "skylark": "skylar",
    "warbler": "gryger1", "fantail": "nezfan1", "dunnock": "dunnoc1",
    "bellbird": "nezbel1", "pukeko": "purswa6", "song thrush": "sonthr1",
    "tui": "tui1", "rifleman": "riflem1", "canada goose": "cangoo",
    "new zealand dotterel": "rebdot1", "robin": "nezrob3", "greenfinch": "eurgre1",
    "weka": "weka1", "kea": "kea1", "kokako": "kokako3", "whitehead": "whiteh1",
    "falcon": "nezfal1", "pigeon": "nezpig2", "sacred kingfisher": "sackin3",
}
NOT_BIRDS = {"animal", "human", "insect", "noise", "frog", "mammal", "cat", "dog"}


def sun_times(day, lat, lng):
    """(sunrise, sunset) as UTC datetimes for the local date day, NOAA
    equations with the standard 90.833 degree zenith"""
    n = day.timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(g)
        - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g)
        - 0.040849 * math.sin(2 * g)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(g)
        + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g)
        + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g)
        + 0.00148 * math.sin(3 * g)
    )
    lat_r = math.radians(lat)
    ha = math.degrees(
        math.acos(
            math.cos(math.radians(90.833)) / (math.cos(lat_r) * math.cos(decl))
            - math.tan(lat_r) * math.tan(decl)
        )
    )
    midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    rise = midnight + timedelta(minutes=720 - 4 * (lng + ha) - eqtime)
    sset = midnight + timedelta(minutes=720 - 4 * (lng - ha) - eqtime)
    return rise, sset


class Sun:
    """Cached sun times per local date"""

    def __init__(self):
        self.cache = {}

    def times(self, day, lat, lng):
        key = (day, round(lat, 2), round(lng, 2))
        if key not in self.cache:
            rise, sset = sun_times(day, lat, lng)
            self.cache[key] = (rise.astimezone(NZ), sset.astimezone(NZ))
        return self.cache[key]

    def night(self, ts, lat, lng):
        """(sunset, next sunrise) of the night ts is in or nearest to"""
        day = ts.date() if ts.hour >= 12 else ts.date() - timedelta(days=1)
        return self.times(day, lat, lng)[1], self.times(day + timedelta(days=1), lat, lng)[0]


def kokako_scores(track):
    """[(label, mean confidence 0-1, is bird)] of the kokako model's tags on
    track, highest first. Labels are mapped to the spect model codes and a
    label tagged more than once has the mean of its confidences"""
    confs = defaultdict(list)
    bird = {}
    for g in track.get("tags", []):
        if g.get("model") != KOKAKO:
            continue
        label = KOKAKO_LABELS.get(g["what"], g["what"])
        confs[label].append((g.get("confidence") or 0) / 100)
        bird[label] = (g.get("path") or "").startswith("all.bird")
    return sorted(
        ((k, sum(v) / len(v), bird[k]) for k, v in confs.items()), key=lambda x: -x[1]
    )


def mean_scores(pred):
    if not pred or not pred.get("windows"):
        return []
    sums = defaultdict(float)
    for w in pred["windows"]:
        for k, v in w["scores"].items():
            sums[k] += v
    n = len(pred["windows"])
    return sorted(((k, v / n) for k, v in sums.items()), key=lambda kv: -kv[1])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", default="summit-road/metadata")
    p.add_argument("--out", default="summit-road/summit-road-tracks-v11.csv")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--hours",
        type=float,
        default=2,
        help="Night calls are from this many hours after sunset to this many"
        " hours before sunrise",
    )
    p.add_argument(
        "--all-day",
        action="store_true",
        help="Print bird calls at any time of day, not just at night",
    )
    p.add_argument(
        "--models",
        nargs="+",
        help="Prediction keys in the metadata, default every model with window"
        " predictions on a track",
    )
    a = p.parse_args()
    recs = [(f, json.load(open(f))) for f in sorted(glob.glob(os.path.join(a.dir, "*.txt")))]
    if not a.models:
        found = {
            k
            for _, rec in recs
            for t in rec.get("Tracks") or rec.get("tracks") or []
            for k, v in t.items()
            if isinstance(v, dict) and "windows" in v
        }
        # the old ones first so columns stay where they were
        a.models = [m for m in MODELS if m in found] + sorted(found - set(MODELS))
        print("Models", ", ".join(a.models))

    sun = Sun()
    tracks = []
    locations = []
    for f, rec in recs:
        rec_dt = datetime.fromisoformat(
            rec["recordingDateTime"].replace("Z", "+00:00")
        ).astimezone(NZ)
        loc = rec.get("location") or {}
        if "lat" not in loc:
            print(f"{f} has no location, skipping")
            continue
        locations.append((loc["lat"], loc["lng"]))
        for t in rec.get("Tracks") or rec.get("tracks") or []:
            if "start" not in t:
                continue
            ts = rec_dt + timedelta(seconds=t["start"])
            tags = defaultdict(list)
            birds = {}
            kokako = kokako_scores(t)
            for g in t.get("tags", []):
                m = g.get("model") or ("human" if not g.get("automatic") else "unknown")
                if m == KOKAKO:
                    continue
                conf = g.get("confidence")
                tags[m].append(g["what"] + (f" ({conf})" if conf is not None else ""))
            found = [(k, v, bird) for k, v, bird in kokako if bird]
            if len(found) > 1:
                # generic bird tag alongside the species
                found = [f for f in found if f[0] != "bird"] or found
            if found:
                birds[KOKAKO] = [(k, v) for k, v, _ in found]
            row = {
                "rec_id": rec["id"],
                "station": rec.get("stationName", ""),
                "datetime_nz": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "track_id": t["id"],
                "start": round(t["start"], 2),
                "end": round(t["end"], 2),
            }
            for m in TAG_MODELS + sorted(set(tags) - set(TAG_MODELS)):
                if m != KOKAKO:
                    row[m] = ";".join(tags.get(m, []))
            row[f"{KOKAKO}_labels"] = ";".join(f"{k} ({v:.2f})" for k, v, _ in kokako)
            row[f"{KOKAKO}_top"] = f"{kokako[0][0]} ({kokako[0][1]:.2f})" if kokako else ""
            for m in a.models:
                ranked = mean_scores(t.get(m))
                above = [(k, v) for k, v in ranked if v > a.threshold]
                row[f"{m}>{a.threshold}"] = ";".join(f"{k} ({v:.2f})" for k, v in above)
                row[f"{m}_top"] = f"{ranked[0][0]} ({ranked[0][1]:.2f})" if ranked else ""
                found = [(k, v) for k, v in above if k not in NOT_BIRDS]
                if found:
                    birds[m] = found
            tracks.append((ts, loc["lat"], loc["lng"], row, birds))
    tracks.sort(key=lambda x: (x[0], x[3]["rec_id"], x[3]["track_id"]))
    if not tracks:
        print("No tracks found")
        return

    # sunrise/sunset marker rows between tracks
    lat, lng = locations[0]
    events = []
    day = tracks[0][0].date() - timedelta(days=1)
    while day <= tracks[-1][0].date() + timedelta(days=1):
        rise, sset = sun.times(day, lat, lng)
        events += [(rise, "SUNRISE"), (sset, "SUNSET")]
        day += timedelta(days=1)
    events.sort()
    rows, ei = [], 0
    for ts, _, _, row, _ in tracks:
        passed = None
        while ei < len(events) and events[ei][0] <= ts:
            passed = events[ei]
            ei += 1
        if passed and rows:
            rows.append(
                {
                    "rec_id": f"---- {passed[1]} {passed[0]:%Y-%m-%d %H:%M} ----",
                    "datetime_nz": passed[0].strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        rows.append(row)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(tracks)} tracks to {a.out}")

    # night time bird calls
    gap = timedelta(hours=a.hours)
    night = []
    for ts, lat, lng, row, birds in tracks:
        sset, rise = sun.night(ts, lat, lng)
        if birds and (a.all_day or sset + gap <= ts <= rise - gap):
            night.append((ts, sset, rise, row, birds))
    print(
        f"\n{len(night)} bird calls "
        + ("at any time" if a.all_day else f"from {a.hours:g}h after sunset to {a.hours:g}h before sunrise")
        + f" (mean score > {a.threshold})"
    )
    current = None
    for ts, sset, rise, row, birds in night:
        if current != sset:
            current = sset
            print(
                f"\nNight of {sset:%a %Y-%m-%d}: sunset {sset:%H:%M}, sunrise"
                f" {rise:%H:%M}, window {sset + gap:%H:%M} - {rise - gap:%H:%M}"
            )
        calls = "  ".join(
            f"{m}: " + ",".join(f"{k} {v:.2f}" for k, v in found)
            for m, found in birds.items()
        )
        print(
            f"  {ts:%H:%M:%S} rec {row['rec_id']} track {row['track_id']}"
            f" ({row['start']}-{row['end']}s) Master: {row.get('Master') or '-'}  {calls}"
        )
    per_label = defaultdict(lambda: defaultdict(int))
    columns = [KOKAKO] + a.models
    for *_, birds in night:
        for m, found in birds.items():
            for k, _ in found:
                per_label[k][m] += 1
    if per_label:
        print("\nNight calls per label (tracks)")
        print(f"  {'label':10}" + "".join(f"{m:>26}" for m in columns))
        for k in sorted(per_label, key=lambda k: -sum(per_label[k].values())):
            print(f"  {k:10}" + "".join(f"{per_label[k][m]:>26}" for m in columns))
        # agreement of each spect model with the kokako model, per night track
        print(f"\nAgreement with {KOKAKO} on night tracks")
        for m in a.models:
            both = only_k = only_m = 0
            for *_, birds in night:
                k_labels = {l for l, _ in birds.get(KOKAKO, [])}
                m_labels = {l for l, _ in birds.get(m, [])}
                both += bool(k_labels & m_labels)
                only_k += bool(k_labels) and not m_labels
                only_m += bool(m_labels) and not k_labels
            print(
                f"  {m:26} same label {both:3}  kokako only {only_k:3}"
                f"  {m} only {only_m:3}"
            )


if __name__ == "__main__":
    main()
