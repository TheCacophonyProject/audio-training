import os

# Replace '1' with the actual index of the GPU you want to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import logging
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import birdnet
import librosa

# pip install "birdnet[pt]"

AUDIO_EXTS = {".m4a", ".flac", ".wav", ".mp3", ".ogg"}


def birdnet_week(dt):
    # birdnet geo model uses 48 "weeks", 4 per month
    return (dt.month - 1) * 4 + min(dt.day - 1, 27) // 7 + 1


def find_files(root):
    files = []
    for txt in Path(root).rglob("*.txt"):
        audio = None
        for ext in AUDIO_EXTS:
            candidate = txt.with_suffix(ext)
            if candidate.exists():
                audio = candidate
                break
        if audio is not None:
            files.append((audio, txt))
    return files


def location_key(meta, use_geo):
    if not use_geo:
        return None
    loc = meta.get("location")
    if not loc or loc.get("lat") is None or loc.get("lng") is None:
        return None
    week = None
    rec_time = meta.get("recordingDateTime")
    if rec_time:
        week = birdnet_week(datetime.fromisoformat(rec_time.replace("Z", "+00:00")))
    # group nearby recordings so they share one species list and one predict call
    return (round(loc["lat"], 1), round(loc["lng"], 1), week)


def split_name(name):
    # birdnet labels are "Scientific name_Common name"
    if "_" in name:
        scientific, common = name.split("_", 1)
        return scientific, common
    return name, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir", help="Directory to search for audio files with a matching .txt"
    )
    parser.add_argument("--version", default="3.0", choices=["2.4", "3.0"])
    parser.add_argument("--backend", default="pt")
    parser.add_argument("--device", default="CPU", help="CPU or GPU")
    parser.add_argument(
        "--workers", type=int, default=None, help="Inference processes (use 1 for GPU)"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--decoders", type=int, default=8, help="Threads decoding audio"
    )
    parser.add_argument(
        "--chunk", type=int, default=64, help="Files decoded and predicted per call"
    )
    parser.add_argument("--min-conf", type=float, default=0.1)
    parser.add_argument("--geo-min-conf", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-geo", action="store_true", help="Don't filter species by location"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run files already analysed by this model",
    )
    args = parser.parse_args()

    if args.version == "2.4" and args.backend not in ("tf", "pb"):
        args.backend = "tf"
    model_id = f"birdnet-{args.version}-{args.backend}"
    if args.no_geo:
        model_id += "-nogeo"

    groups = defaultdict(list)
    for audio, txt in find_files(args.dir):
        with open(txt) as f:
            try:
                meta = json.load(f)
            except:
                continue
        if not args.overwrite and meta.get("birdnet_model") == model_id:
            continue
        groups[location_key(meta, not args.no_geo)].append((audio, txt))

    total = sum(len(v) for v in groups.values())
    logging.info(
        "Analysing %s files in %s location groups with %s", total, len(groups), model_id
    )
    if total == 0:
        return

    model = birdnet.load("acoustic", args.version, args.backend)
    geo_model = None
    if not args.no_geo and any(k is not None for k in groups):
        geo_model = birdnet.load("geo", args.version, args.backend)

    for key, files in groups.items():
        try:
            species_list = None
            if key is not None:
                lat, lng, week = key
                geo = geo_model.predict(
                    lat, lng, week=week, min_confidence=args.geo_min_conf
                )
                # geo model knows some species the acoustic model doesn't
                species_list = geo.to_set() & set(model.species_list)
                logging.info(
                    "lat %s lng %s week %s: %s species",
                    lat,
                    lng,
                    week,
                    len(species_list),
                )

            for i in range(0, len(files), args.chunk):
                analyse_chunk(
                    model, files[i : i + args.chunk], species_list, key, model_id, args
                )
        except Exception:
            logging.exception(
                "Failed on group %s (%s files), skipping rest of group", key, len(files)
            )


def analyse_chunk(model, files, species_list, key, model_id, args):
    # birdnet can't read m4a so decode with librosa and pass arrays
    def load(audio):
        try:
            return librosa.load(audio, sr=None, mono=True)
        except Exception:
            logging.exception("Could not load %s", audio)
            return None

    with ThreadPoolExecutor(args.decoders) as pool:
        results = list(pool.map(load, [audio for audio, _ in files]))
    loaded = [f for f, r in zip(files, results) if r is not None]
    arrays = [r for r in results if r is not None]
    if not arrays:
        return

    # run unfiltered (no top_k) and apply the geo list ourselves, so segments with
    # no geo species above threshold can fall back to the model's full species list
    predictions = model.predict_arrays(
        arrays,
        top_k=None,
        default_confidence_threshold=args.min_conf,
        device=args.device,
        n_workers=args.workers,
        batch_size=args.batch_size,
    )

    segments = defaultdict(list)
    for row in predictions.to_structured_array():
        scientific, common = split_name(row["species_name"])
        start = round(float(row["start_time"]), 2)
        segments[(int(row["input"]), start)].append(
            {
                "start": start,
                "end": round(float(row["end_time"]), 2),
                "scientific_name": scientific,
                "common_name": common,
                "confidence": round(float(row["confidence"]), 4),
                "in_geo": species_list is None or row["species_name"] in species_list,
            }
        )

    by_file = defaultdict(list)
    for (index, _), segment in segments.items():
        in_geo = [d for d in segment if d["in_geo"]]
        segment = in_geo if in_geo else segment
        segment.sort(key=lambda d: -d["confidence"])
        by_file[index].extend(segment[: args.top_k])

    for index, (audio, txt) in enumerate(loaded):
        with open(txt) as f:
            meta = json.load(f)
        detections = sorted(
            by_file.get(index, []), key=lambda d: (d["start"], -d["confidence"])
        )
        meta["birdnet"] = detections
        meta["birdnet_model"] = model_id
        meta["birdnet_geo"] = (
            None if key is None else {"lat": key[0], "lng": key[1], "week": key[2]}
        )
        with open(txt, "w") as f:
            json.dump(meta, f, indent=4)
        logging.info("%s: %s detections", audio.name, len(detections))


if __name__ == "__main__":
    # m4a falls back to audioread which warns on every file
    warnings.filterwarnings("ignore", message="PySoundFile failed")
    warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
