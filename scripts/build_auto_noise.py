"""Make extra noise tfrecords from tracks that only have automatic tags, all
saying noise, for recordings already in an existing build's split.

A track is used when:
  - it has no human tags
  - Master says noise and no automatic tag (e.g. efficientnetv2bm-kokako)
    says morepork, other automatic tags are ignored
  - Master's noise confidence is at least --min-confidence
  - it doesn't overlap a track that has any non noise tag (human or
    automatic) or no tags
Then each of its samples is dropped if a birdnet bird detection of at least
--birdnet-conf overlaps it. Recordings without birdnet results are skipped
unless --allow-no-birdnet.

Then --per-recording (default 1) random samples are kept per recording so a
few recordings with lots of noise tracks don't dominate.

Recordings stay in the split (train / validation / test) they are in the
split file. Recordings not in it (e.g. a new download of automatically tagged
recordings) go to the split of the split file recordings with the same split
key (site / device, utils.split_key), the most held out if several, so no
noise comes from another split's sites. Recordings with a new key go to
--new-split (default train, skip to leave them out). The records
go to <training dir>/noise/auto-noise/<split> with the build's labels and
config, so add <training dir>/noise/auto-noise to train.sh's --noise-dirs.

python3 scripts/build_auto_noise.py -d /data/audio-data/cacophony /data/new-noise-meta \\
    /data2/audio-training/audio-training/training-data
"""

import argparse
import copy
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audiodataset import AudioDataset, Config, Recording, load_metadata  # noqa: E402
from audiowriter import create_tf_records  # noqa: E402
from birdnetcompare import load_taxonomy  # noqa: E402
from build import birdnet_bird  # noqa: E402
from utils import prefixed_id, split_key  # noqa: E402

AUTO_DIR = "auto-noise"
MOREPORK = {"morepork", "morepo2"}
MASTER = "Master"
AUDIO_EXT = (".m4a", ".wav", ".mp3", ".flac")


def is_noise(what):
    # exactly noise, NOISE_LABELS has e.g. chirping_birds
    return what == "noise"


def track_kind(track, min_confidence):
    """'auto' an automatic only track we can use, 'noise' a human noise only
    track, 'other' anything else (human birds, untagged, not Master noise,
    an automatic morepork)"""
    tags = track.get("tags", [])
    human = [t for t in tags if not t.get("automatic") and not t.get("secondary")]
    if human:
        return "noise" if all(is_noise(t.get("what")) for t in human) else "other"
    auto = [t for t in tags if t.get("automatic")]
    master = [t for t in auto if t.get("model") == MASTER]
    if not master or not all(is_noise(t.get("what")) for t in master):
        return "other"
    if any(t.get("what") in MOREPORK for t in auto):
        return "morepork"
    if min(t.get("confidence") or 0 for t in master) < min_confidence:
        return "low"
    return "auto"


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def auto_tracks(meta, min_confidence, counts):
    tracks = meta.get("Tracks") or meta.get("tracks") or []
    kinds = [track_kind(t, min_confidence) for t in tracks]
    keep = []
    for t, kind in zip(tracks, kinds):
        counts[f"{kind} tracks"] += 1
        if kind != "auto":
            continue
        if any(
            k not in ("auto", "noise") and o is not t and overlaps(t, o)
            for o, k in zip(tracks, kinds)
        ):
            counts["overlaps a non noise track"] += 1
            continue
        t = copy.deepcopy(t)
        # the automatic noise tags as the track's label so Recording keeps it
        t["tags"] = [{"what": "noise", "automatic": False}]
        keep.append(t)
    return keep


def load_config(meta):
    config = Config()
    for k, v in meta.items():
        if k in config.__dict__:
            setattr(config, k, v)
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("training_dir", help="The build's training-data dir")
    p.add_argument(
        "-d",
        "--dir",
        nargs="+",
        required=True,
        help="Metadata / audio dirs, the one build used and any new downloads",
    )
    p.add_argument(
        "--new-split",
        default="train",
        help="Split for recordings whose site / device key isn't in the split file "
        "(a split name or skip)",
    )
    p.add_argument("--split-file", help="Default <training dir>/training-split.json")
    p.add_argument("--min-confidence", type=float, default=0, help="Master noise confidence")
    p.add_argument("--birdnet-conf", type=float, default=0.3)
    p.add_argument("--allow-no-birdnet", action="store_true")
    p.add_argument(
        "--per-recording",
        type=int,
        default=1,
        help="Random samples kept per recording after the birdnet check, 0 for all",
    )
    p.add_argument("--max-samples", type=int, help="Most samples per split, random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Only print counts")
    a = p.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)

    training_dir = Path(a.training_dir)
    train_meta = json.load(open(training_dir / "training-meta.json"))
    labels = train_meta["primary_labels"]
    config = load_config(train_meta)
    split = json.load(open(a.split_file or training_dir / "training-split.json"))["recs"]
    rec_split = {str(r): name for name, recs in split.items() for r in recs}

    taxonomy = load_taxonomy(Path(__file__).resolve().parent.parent / "eBird_taxonomy_v2024.csv")
    datasets = {name: AudioDataset(name, config) for name in split}
    counts = Counter()
    # a split key (site / device) in more than one split goes to the most
    # held out one so new recordings never add to train from a test device
    held_out = {"test": 0, "validation": 1}
    key_split = {}
    candidates = []
    for f in sorted(f for d in a.dir for f in Path(d).glob("**/*.txt")):
        try:
            meta = load_metadata(f)
        except Exception:
            continue
        if not isinstance(meta, dict) or "id" not in meta:
            continue
        rec_id = str(prefixed_id(f, meta["id"]))
        key = split_key(f, meta)
        name = rec_split.get(rec_id)
        if name is not None:
            counts["split recordings"] += 1
            old = key_split.get(key)
            if old is None or held_out.get(name, 2) < held_out.get(old, 2):
                key_split[key] = name
        else:
            counts["new recordings"] += 1
        tracks = auto_tracks(meta, a.min_confidence, counts)
        if not tracks:
            continue
        if "birdnet" not in meta and not a.allow_no_birdnet:
            counts["no birdnet, tracks skipped"] += len(tracks)
            continue
        meta = dict(meta)
        meta["Tracks"] = tracks
        meta.pop("tracks", None)
        candidates.append((f, meta, name, key))

    for f, meta, name, key in candidates:
        if name is None:
            name = key_split.get(key)
            if name is None:
                name = a.new_split
                counts["new key recordings"] += 1
            if name not in datasets:
                counts["new key recordings skipped"] += 1
                continue
        audio = next((f.with_suffix(e) for e in AUDIO_EXT if f.with_suffix(e).exists()), None)
        if audio is None:
            logging.error("No audio for %s", f)
            continue
        tracks = meta["Tracks"]
        datasets[name].add_recording(Recording(meta, audio, config, load_samples=False))
        counts[f"{name} tracks"] += len(tracks)

    for name, ds in datasets.items():
        ds.load_samples()
        keep = []
        with_bird = 0
        for rec in ds.recs.values():
            ok = [s for s in rec.samples if not birdnet_bird(rec, s, a.birdnet_conf, taxonomy)]
            with_bird += len(rec.samples) - len(ok)
            if a.per_recording and len(ok) > a.per_recording:
                ok = random.sample(ok, a.per_recording)
            keep.extend(ok)
        if a.max_samples and len(keep) > a.max_samples:
            keep = random.sample(keep, a.max_samples)
        keep_ids = {id(s) for s in keep}
        for rec_id, rec in list(ds.recs.items()):
            rec.samples = [s for s in rec.samples if id(s) in keep_ids]
            if not rec.samples:
                del ds.recs[rec_id]
        ds.samples = [s for s in ds.samples if id(s) in keep_ids]
        counts[f"{name} birdnet bird dropped"] = with_bird
        counts[f"{name} samples"] = len(ds.samples)
        counts[f"{name} recordings"] = len(ds.recs)

    for k, v in counts.items():
        print(f"{k:35} {v}")
    if a.dry_run:
        return
    out = training_dir / "noise" / AUTO_DIR
    for name, ds in datasets.items():
        if ds.samples:
            create_tf_records(ds, out / name, labels, num_shards=100)
    with open(out / "auto-noise-meta.json", "w") as fh:
        json.dump({"args": vars(a), "counts": counts}, fh, indent=4)
    print("Wrote", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
