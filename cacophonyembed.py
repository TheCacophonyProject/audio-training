"""Cacophony model embeddings and predictions of every manually tagged track.

Each track is covered by model length windows (3s for spectv10), spread evenly
from the track start to its end so they overlap by at least --overlap seconds.
Tracks shorter than a window get one window centred on them.

Per window:
  - the pooled embedding before the classifier and the logits are saved to
    shards in --out, 00000.npy embeddings, 00000.logits.npy logits and
    00000.csv with a row per window, the same layout as perchembed.py so its
    shard loading works on them
  - the probability of every label is saved in the metadata, on the track
    under the model key e.g. track["spectv10-val_pr_auc"]

Re-running skips windows already in --out and tracks that already have
predictions under the model key.
"""

import argparse
import csv
import json
import logging
import warnings
from pathlib import Path

import numpy as np

from addtracks import (
    CacophonyModel,
    load_recording,
    setup_gpu,
    windows,
    write_meta,
)
from perchembed import find_audio, load_shard_rows, shard_paths

FIELDNAMES = [
    "index",
    "file",
    "audio",
    "recording_id",
    "device_id",
    "station_id",
    "recording_date",
    "rec_tags",
    "track_id",
    "manual_tags",
    "secondary_tags",
    "automatic_tags",
    "track_start",
    "track_end",
    "min_freq",
    "max_freq",
    "window_start",
    "window_end",
    "top_label",
    "top_prob",
    "model",
]


def window_key(recording_id, track_id, start):
    return (str(recording_id), str(track_id), f"{float(start):.2f}")


def tag_whats(tags, automatic=False, secondary=False):
    return ";".join(
        t["what"]
        for t in tags
        if bool(t.get("automatic")) == automatic
        and bool(t.get("secondary")) == secondary
        and t.get("what")
    )


def model_key(model):
    # e.g. spectv10-val_pr_auc, so other models or weights get their own key
    weights = Path(model.weights).name.split(".")[0]
    return f"{model.model_name}-{weights}"


class Shards:
    def __init__(self, out_dir, shard_size):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.done = set()
        shards = shard_paths(self.out_dir)
        for npy in shards:
            for r in load_shard_rows(npy):
                self.done.add(
                    window_key(r["recording_id"], r["track_id"], r["window_start"])
                )
        self.next_shard = int(shards[-1].stem) + 1 if shards else 0
        self.total = len(self.done)
        if self.done:
            logging.info(
                "Resuming, %s windows already in %s shards", self.total, len(shards)
            )
        self.embeddings, self.logits, self.rows = [], [], []

    def add(self, embedding, logits, row):
        row["index"] = self.total
        self.embeddings.append(embedding)
        self.logits.append(logits)
        self.rows.append(row)
        self.done.add(window_key(row["recording_id"], row["track_id"], row["window_start"]))
        self.total += 1

    def maybe_save(self):
        # called per recording so a recording is never split across shards
        if len(self.rows) >= self.shard_size:
            self.save()

    def save(self):
        if not self.rows:
            return
        # temp names then rename so a crash never leaves a partial shard, the
        # csv is written last as that is what marks a shard complete
        npy = self.out_dir / f"{self.next_shard:05d}.npy"
        for path, data in (
            (npy, self.embeddings),
            (npy.with_name(f"{npy.stem}.logits.npy"), self.logits),
        ):
            tmp = path.with_name(path.name[: -len(".npy")] + ".tmp.npy")
            np.save(tmp, np.stack(data).astype(np.float32))
            tmp.replace(path)
        tmp_csv = npy.with_name(npy.stem + ".tmp.csv")
        with open(tmp_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)
        tmp_csv.replace(npy.with_suffix(".csv"))
        logging.info("Saved shard %s with %s windows", npy, len(self.rows))
        self.next_shard += 1
        self.embeddings, self.logits, self.rows = [], [], []


def get_jobs(meta, args, model, key, shards, stride, duration):
    """Return the tracks in this recording that still need processing.

    Only human-tagged tracks are considered unless --all-tracks is set. A track
    is a job if any of its windows is missing from the shards, or (unless
    --no-meta) its predictions under the model key are missing or --overwrite
    is set.

    Returns a list of (track, windows) tuples, where windows are the
    (start, end) times the track is split into for the model.
    """
    tracks = meta.get("Tracks") or []
    if not args.all_tracks:
        # only tracks a person has tagged
        tracks = [
            t
            for t in tracks
            if any(g.get("automatic") is False for g in t.get("tags", []))
        ]
    recording_id = meta.get("id")
    todo = []
    for track in tracks:
        if "start" not in track or "end" not in track:
            continue
        track_windows = windows(
            track["start"], track["end"], model.length, stride, duration
        )
        missing = any(
            window_key(recording_id, track.get("id"), round(w[0], 2))
            not in shards.done
            for w in track_windows
        )
        meta_missing = not args.no_meta and (args.overwrite or key not in track)
        if missing or meta_missing:
            todo.append((track, track_windows))
    return todo


def process(txt, meta, y, sr, todo, args, model, embed_model, key, shards):
    recording_id = meta.get("id")
    samples = []
    owners = []
    for track, track_windows in todo:
        for start, end in track_windows:
            samples.append(model.sample(y, sr, start))
            owners.append((track, start, end))
    embeddings, logits, probs = embed_model.predict(
        np.stack(samples), batch_size=args.batch_size, verbose=0
    )

    rec_tags = tag_whats(meta.get("tags") or [])
    results = {}
    added = 0
    for (track, start, end), emb, logit, prob in zip(owners, embeddings, logits, probs):
        result = results.get(id(track))
        if result is None:
            result = {
                "model_name": model.model_name,
                "weights": model.weights,
                "overlap": args.overlap,
                "windows": [],
            }
            results[id(track)] = result
            track[key] = result
            print("Sasving result for ", key)
        top = int(np.argmax(prob))
        result["windows"].append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "scores": {
                    label: round(float(p), 4) for label, p in zip(model.labels, prob)
                },
            }
        )
        if window_key(recording_id, track.get("id"), round(start, 2)) in shards.done:
            continue
        tags = track.get("tags", [])
        shards.add(
            emb,
            logit,
            {
                "file": str(txt),
                "audio": meta.get("file", ""),
                "recording_id": recording_id,
                "device_id": meta.get("deviceId", ""),
                "station_id": meta.get("stationId", ""),
                "recording_date": meta.get("recordingDateTime", ""),
                "rec_tags": rec_tags,
                "track_id": track.get("id"),
                "manual_tags": tag_whats(tags),
                "secondary_tags": tag_whats(tags, secondary=True),
                "automatic_tags": tag_whats(tags, automatic=True),
                "track_start": track.get("start"),
                "track_end": track.get("end"),
                "min_freq": track.get("minFreq", ""),
                "max_freq": track.get("maxFreq", ""),
                "window_start": round(start, 2),
                "window_end": round(end, 2),
                "top_label": model.labels[top],
                "top_prob": round(float(prob[top]), 4),
                "model": key,
            },
        )
        added += 1
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir", help="Directory to search for metadata .txt files")
    parser.add_argument(
        "--out",
        default="cacophony-embeddings",
        help="Directory for shards, re-running skips windows already in it",
    )
    parser.add_argument(
        "--model",
        default="~/cacophony/audio-model/spectv10/spectv10NoMixKokakoFine/spectv10NoMixKokakoFine.keras",
        help="Cacophony .keras model file, metadata.txt and --weights are read from its directory",
    )
    parser.add_argument("--weights", default="val_pr_auc.weights.h5")
    parser.add_argument(
        "--overlap", type=float, default=1.0, help="Seconds of overlap between windows"
    )
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--all-tracks",
        action="store_true",
        help="Also do tracks with no manual tags",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redo the metadata predictions of tracks that already have them",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Only save the shards, don't change the metadata",
    )
    args = parser.parse_args()

    setup_gpu()
    model = CacophonyModel(args)
    embed_model = model.embedding_model()
    key = model_key(model)
    stride = model.length - args.overlap
    if stride <= 0:
        parser.error(f"--overlap must be less than the {model.length}s window")
    logging.info(
        "Saving under %s, %ss windows %ss stride, embedding size %s",
        key,
        model.length,
        stride,
        embed_model.output_shape[0][-1],
    )
    shards = Shards(args.out, args.shard_size)

    files = 0
    try:
        for txt in sorted(Path(args.dir).rglob("*.txt")):
            try:
                with open(txt) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(meta, dict) or not meta.get("Tracks"):
                continue
            # get jobs with the metadata duration first so done recordings
            # aren't loaded, then again with the real duration
            tracks = meta.get("Tracks")
            duration = meta.get("duration") or max(t.get("end", 0) for t in tracks)
            if not get_jobs(meta, args, model, key, shards, stride, duration):
                continue
            audio = find_audio(txt, meta)
            if audio is None:
                logging.info("No audio for %s", txt)
                continue
            try:
                y, sr = load_recording(audio)
            except Exception:
                logging.exception("Could not load %s", audio)
                continue
            todo = get_jobs(meta, args, model, key, shards, stride, len(y) / sr)
            if not todo:
                continue
            try:
                process(txt, meta, y, sr, todo, args, model, embed_model, key, shards)
            except Exception:
                logging.exception("Failed on %s", txt)
                continue
            if not args.no_meta:
                write_meta(txt, meta)
            shards.maybe_save()
            files += 1
            if files % 100 == 0:
                logging.info("%s windows from %s files", shards.total, files)
    finally:
        # also save on ctrl-c or an error so the run can be resumed
        shards.save()
    logging.info("%s windows in %s", shards.total, args.out)


if __name__ == "__main__":
    # m4a falls back to audioread which warns on every file
    warnings.filterwarnings("ignore", message="PySoundFile failed")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
