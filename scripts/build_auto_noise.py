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
noise comes from another split's sites. Recordings with a new key are split
by build.py's split_randomly: random new keys go to validation and test until
each has --val-share and --test-share of all the samples, counting what the
split file and --test-devices already give it, the rest to train. The records
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

import build  # noqa: E402
from audiodataset import AudioDataset, Config, Recording, load_metadata  # noqa: E402
from audiowriter import create_tf_records  # noqa: E402
from birdnetcompare import detection_taxon, load_taxonomy  # noqa: E402
from birdsconfig import RELABEL_MAP  # noqa: E402
from utils import prefixed_id, split_key  # noqa: E402

AUTO_DIR = "auto-noise"
MOREPORK = {"morepork", "morepo2"}
MASTER = "Master"
KOKAKO = "efficientnetv2bm-kokako"
# the kokako model's tag for no bird
KOKAKO_NO_BIRD = "unidentified"
# Summit Road A213, the site we judge night false positives on, never trained on
SUMMIT_ROAD_DEVICES = [3267]
AUDIO_EXT = (".m4a", ".wav", ".mp3", ".flac")
# NOISE_LABELS the build relabels to noise that usually have birds in them, not
# used as automatic noise
NOISY_BIRD_LABELS = {"chirping_birds", "forest", "rainforest"}


KIND_NAMES = {
    "auto": "automatic noise, candidates",
    "noise": "human noise (already in the build)",
    "other": "birds, untagged or Master not noise",
    "morepork": "Master noise but an automatic morepork",
    "kokako": "Master noise but kokako a bird (--kokako-noise)",
    "low": "Master noise below --min-confidence",
}


def is_noise(what):
    """A tag the build trains as noise (RELABEL_MAP, e.g. wind, rain, static),
    except the NOISE_LABELS that are likely to have birds in them"""
    return what not in NOISY_BIRD_LABELS and RELABEL_MAP.get(what, what) == "noise"


def track_kind(track, min_confidence, kokako_noise=False):
    """'auto' an automatic only track we can use, 'noise' a human noise only
    track, 'other' anything else (human birds, untagged, not Master noise),
    'morepork' an automatic morepork, with kokako_noise 'kokako' if a kokako
    tag isn't noise or unidentified"""
    tags = track.get("tags", [])
    human = [t for t in tags if not t.get("automatic") and not t.get("secondary")]
    if human:
        return "noise" if all(is_noise(t.get("what")) for t in human) else "other"
    auto = [t for t in tags if t.get("automatic")]
    # a track has at most one Master tag
    master = next((t for t in auto if t.get("model") == MASTER), None)
    if master is None or not is_noise(master.get("what")):
        return "other"
    if any(t.get("what") in MOREPORK for t in auto):
        return "morepork"
    if kokako_noise and any(
        t.get("model") == KOKAKO
        and t.get("what") != KOKAKO_NO_BIRD
        and not is_noise(t.get("what"))
        for t in auto
    ):
        return "kokako"
    if (master.get("confidence") or 0) < min_confidence:
        return "low"
    return "auto"


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def auto_tracks(meta, min_confidence, counts, kokako_noise=False):
    tracks = meta.get("Tracks") or meta.get("tracks") or []
    kinds = [track_kind(t, min_confidence, kokako_noise) for t in tracks]
    keep = []
    for t, kind in zip(tracks, kinds):
        counts[f"tracks: {KIND_NAMES[kind]}"] += 1
        if kind != "auto":
            continue
        if any(
            k not in ("auto", "noise") and o is not t and overlaps(t, o)
            for o, k in zip(tracks, kinds)
        ):
            counts["automatic noise tracks: overlap a non noise track, dropped"] += 1
            continue
        t = copy.deepcopy(t)
        # the automatic noise tags as the track's label so Recording keeps it
        t["tags"] = [{"what": "noise", "automatic": False}]
        keep.append(t)
    return keep


def print_counts(counts, splits):
    """counts grouped by the section before the colon, in pipeline order"""
    order = ["recordings", "tracks", "automatic noise tracks", *splits]
    width = max(len(k) for k in counts)
    for section in order:
        rows = [(k, v) for k, v in counts.items() if k.split(":")[0] == section]
        if not rows:
            continue
        print()
        for k, v in sorted(rows):
            print(f"{k:{width}} {v:>8}")


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
        "--val-share",
        type=float,
        default=0.15,
        help="Validation's share of all the samples, new site / device keys are "
        "added to it until it has this (build.py VAL_PERCENT)",
    )
    p.add_argument(
        "--test-share",
        type=float,
        default=0.05,
        help="Test's share of all the samples, new site / device keys are added "
        "to it until it has this (build.py TEST_PERCENT)",
    )
    p.add_argument(
        "--test-devices",
        type=int,
        nargs="*",
        default=SUMMIT_ROAD_DEVICES,
        help="deviceIds whose recordings always go to test whatever the split "
        "file says, default Summit Road (A213)",
    )
    p.add_argument("--split-file", help="Default <training dir>/training-split.json")
    p.add_argument(
        "--kokako-noise",
        action="store_true",
        help="Also drop tracks the kokako model tags as anything but noise or "
        "unidentified (e.g. bird or a species), not only morepork",
    )
    p.add_argument(
        "--min-confidence", type=float, default=0, help="Master noise confidence"
    )
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
    split = json.load(open(a.split_file or training_dir / "training-split.json"))[
        "recs"
    ]
    rec_split = {str(r): name for name, recs in split.items() for r in recs}

    taxonomy = load_taxonomy(
        Path(__file__).resolve().parent.parent / "eBird_taxonomy_v2024.csv"
    )
    datasets = {name: AudioDataset(name, config) for name in split}
    counts = Counter()
    # a split key (site / device) in more than one split goes to the preferred
    # (lowest, most held out) one so new recordings never add to train from a
    # test device
    dataset_preference = {"test": 0, "validation": 1, "train": 2}
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
            counts["recordings: in the split file"] += 1
            old = key_split.get(key)
            if old is None or dataset_preference[name] < dataset_preference[old]:
                key_split[key] = name
        else:
            counts["recordings: not in the split file"] += 1
        tracks = auto_tracks(meta, a.min_confidence, counts, a.kokako_noise)
        if not tracks:
            continue
        if "birdnet" not in meta and not a.allow_no_birdnet:
            counts["automatic noise tracks: no birdnet results, dropped"] += len(tracks)
            continue
        meta = dict(meta)
        meta["Tracks"] = tracks
        meta.pop("tracks", None)
        candidates.append((f, meta, name, key))

    if not key_split:
        # every recording would be "new", keys from validation / test sites
        # included, so the split would put noise from held out sites in train
        sys.exit(
            "No recordings in the split file were found in --dir so every site / "
            "device would be new, validation and test sites included. Add the "
            "metadata dir the build used to -d (its recordings give each site's "
            "split)"
        )

    def filter_samples(ds):
        """Makes the samples, drops those birdnet hears a bird in and keeps
        --per-recording random ones per recording. Returns the number dropped
        for a bird"""
        ds.load_samples()
        keep = []
        with_bird = 0
        for rec in ds.recs.values():
            # the recording's bird detections once, not per sample
            birds = [
                (d["start"], d["end"])
                for d in rec.metadata.get("birdnet") or []
                if d["confidence"] >= a.birdnet_conf
                and detection_taxon(d, taxonomy) is not None
            ]
            ok = [
                s
                for s in rec.samples
                if not any(start < s.end and end > s.start for start, end in birds)
            ]
            with_bird += len(rec.samples) - len(ok)
            if a.per_recording and len(ok) > a.per_recording:
                ok = random.sample(ok, a.per_recording)
            keep.extend(ok)
        keep_only(ds, keep)
        return with_bird

    def keep_only(ds, keep):
        keep_ids = {id(s) for s in keep}
        for rec_id, rec in list(ds.recs.items()):
            rec.samples = [s for s in rec.samples if id(s) in keep_ids]
            if not rec.samples:
                del ds.recs[rec_id]
        ds.samples = [s for s in ds.samples if id(s) in keep_ids]

    test_devices = set(a.test_devices or [])
    # recordings whose site / device isn't in the split file, split below
    new_ds = AudioDataset("new", config)
    for f, meta, name, key in candidates:
        if meta.get("deviceId") in test_devices:
            if name != "test":
                counts["recordings: --test-devices, moved to test"] += 1
            name = "test"
        elif name is None:
            name = key_split.get(key)
            if name is not None:
                counts[f"recordings: site / device in the split file, to {name}"] += 1
        if name is None:
            target = new_ds
            counts["recordings: new site / device, split randomly"] += 1
        else:
            target = datasets[name]
        audio = next(
            (f.with_suffix(e) for e in AUDIO_EXT if f.with_suffix(e).exists()), None
        )
        if audio is None:
            logging.error("No audio for %s", f)
            continue
        target.add_recording(Recording(meta, audio, config, load_samples=False))
        counts[f"{target.name}: automatic noise tracks"] += len(meta["Tracks"])

    for name, ds in [*datasets.items(), ("new", new_ds)]:
        counts[f"{name}: samples birdnet hears a bird in, dropped"] = filter_samples(ds)
    new_samples = {id(s) for s in new_ds.samples}
    if new_ds.samples:
        # build.py's split, counting what the split file and --test-devices
        # already give each split, every sample is the noise label
        build.VAL_PERCENT = a.val_share
        build.TEST_PERCENT = a.test_share
        build.split_randomly(new_ds, datasets=list(datasets.values()))
    counts["new: samples"] = len(new_samples)

    for name, ds in datasets.items():
        if a.max_samples and len(ds.samples) > a.max_samples:
            keep_only(ds, random.sample(ds.samples, a.max_samples))
        from_new = [s for s in ds.samples if id(s) in new_samples]
        counts[f"{name}: samples written"] = len(ds.samples)
        counts[f"{name}: samples written, from new sites"] = len(from_new)
        counts[f"{name}: recordings"] = len(ds.recs)
        counts[f"{name}: new site bins"] = len({s.bin_id for s in from_new})

    print_counts(counts, [*datasets, "new"])
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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
