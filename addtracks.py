"""Fills in missing metadata for a directory of recordings, see addtracks.md.

Per recording, each step only runs when its version key is missing or old, so
the script can be re-run and resumes where it left off:
  signal      identifytracks.signal_noise, same settings as audio-analysis
  Tracks      get_tracks_from_signals, only when a recording has no Tracks
  rms         rmsdata.add_rms_data_to_tracks
  birdnet     per track, saved in track["birdnet"]
  perch       per track window, saved as shards in --out like perchembed.py
  cacophony   per track window, saved in track["cacophony-model"]
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import csv
import json
import logging
import math
import multiprocessing
import warnings
from datetime import datetime
from pathlib import Path

import audioread.ffdec  # Use ffmpeg decoder
import librosa
import numpy as np

from identifytracks import Signal, get_end, get_tracks_from_signals, signal_noise

AUDIO_EXTS = [".flac", ".wav", ".m4a", ".mp3", ".ogg"]
SR = 48000
TOP_FREQ = SR / 2

# 2.0 is identifytracks.signal_noise with audio-analysis settings, 1.x was
# otherdata.signal_noise (n_fft 4096 and no minimum size)
SIGNAL_VERSION = 2.0
# marks Tracks made by this script, so they can be remade when signals change
GENERATED_BY = "addtracks"

PERCH_MODEL = "perch_v2"
PERCH_FIELDNAMES = [
    "index",
    "file",
    "recording_id",
    "track_id",
    "manual_tags",
    "track_start",
    "track_end",
    "perch_start",
    "perch_end",
    "perch_model",
]


def load_recording(file, resample=SR):
    # librosa.load(file) gives strange results on some m4a, same as audiowriter
    aro = audioread.ffdec.FFmpegAudioFile(file)
    frames, sr = librosa.load(aro, sr=None)
    aro.close()
    if resample is not None and resample != sr:
        frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
        sr = resample
    return frames, sr


def find_audio(root):
    files = []
    for ext in AUDIO_EXTS:
        files.extend(Path(root).rglob(f"*{ext}"))
    return sorted(files)


def write_meta(txt, meta):
    # temp file then rename so a crash never leaves half a metadata file
    tmp = txt.with_name(txt.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=4)
    tmp.replace(txt)


def windows(start, end, length, stride, duration):
    """Model context length windows over a track. A track shorter than the
    context gets one window centred on it, kept inside the recording. Longer
    tracks get the fewest windows at most stride apart that cover the track,
    spread evenly from the track start to the track end, so windows overlap by
    at least length - stride and the last one isn't a near copy of the one
    before"""
    end = min(end, duration)
    if end - start <= length:
        centre = (start + end) / 2
        w_start = max(0, min(centre - length / 2, duration - length))
        return [(w_start, w_start + length)]
    # rounded so float error doesn't add a window e.g. 10.0000001s
    gaps = math.ceil(round((end - start - length) / stride, 3))
    step = (end - start - length) / gaps
    return [(start + i * step, start + i * step + length) for i in range(gaps + 1)]


def clip(y, sr, start, length):
    """length seconds from start, recordings shorter than this are padded
    either side so the audio stays centred"""
    size = int(round(length * sr))
    first = max(0, int(round(start * sr)))
    chunk = y[first : first + size]
    missing = size - len(chunk)
    if missing > 0:
        chunk = np.pad(chunk, (missing // 2, missing - missing // 2))
    return chunk


def model_sample(y, sr, start, length):
    """A model input window, separate from CacophonyModel so it can be cut in
    processes without tensorflow"""
    chunk = clip(y, sr, start, length)
    # peak normalise to [-1, 1] as in training
    chunk = chunk / max(float(np.max(np.abs(chunk))), 1e-6)
    return chunk.astype(np.float32)


def manual_tags(track):
    return [
        t["what"]
        for t in track.get("tags", [])
        if t.get("automatic") is False and not t.get("secondary")
    ]


def signals_from_meta(meta):
    # signal arrays are [start, end, freq_start, freq_end, strength], mass
    # isn't saved but is only summed when merging
    signals = []
    for s in meta.get("signal") or []:
        strength = s[4] if len(s) > 4 else None
        signals.append(Signal(s[0], s[1], s[2], s[3], 0, strength=strength))
    return signals


def track_meta(signal, track_id, duration):
    return {
        "id": track_id,
        "start": round(float(signal.start), 3),
        "end": round(float(signal.end), 3),
        "minFreq": round(float(signal.freq_start), 1),
        "maxFreq": round(float(signal.freq_end), 1),
        "automatic": True,
        "generated_by": GENERATED_BY,
        "tags": [],
        "positions": [
            {
                "x": signal.start / duration,
                "y": signal.freq_start / TOP_FREQ,
                "width": signal.length / duration,
                "height": (signal.freq_end - signal.freq_start) / TOP_FREQ,
            }
        ],
    }


def generated_tracks(meta):
    tracks = meta.get("Tracks") or []
    return bool(tracks) and all(t.get("generated_by") == GENERATED_BY for t in tracks)


def add_signals_and_tracks(meta, y, sr, duration, overwrite):
    changed = False
    signals = None
    if overwrite or meta.get("signal_version", 0) < SIGNAL_VERSION:
        end = get_end(y, sr)
        signals, _ = signal_noise(y[: int(sr * end)], sr)
        meta["signal"] = [[float(v) for v in s.to_array(decimals=2)] for s in signals]
        meta["noise"] = []
        meta["rec_end"] = float(end)
        meta["signal_version"] = SIGNAL_VERSION
        changed = True

    # manual tracks are never touched, ours are remade when the signals change
    remake = generated_tracks(meta) and (
        signals is not None or meta.get("tracks_version") != SIGNAL_VERSION
    )
    if not meta.get("Tracks") or remake:
        if signals is None:
            signals = signals_from_meta(meta)
        rec_end = meta.get("rec_end", duration)
        tracks = get_tracks_from_signals([s.copy() for s in signals], rec_end)
        tracks.sort(key=lambda s: (s.start, s.freq_start))
        meta["Tracks"] = [track_meta(s, i, duration) for i, s in enumerate(tracks)]
        meta["tracks_version"] = SIGNAL_VERSION
        # new tracks need their rms
        meta.pop("rms_version", None)
        changed = True
        logging.info("Made %s tracks from %s signals", len(tracks), len(signals))
    return changed


def add_rms(meta, y, sr, overwrite):
    from rmsdata import RMS_VERSION, add_rms_data_to_tracks

    if not overwrite and meta.get("rms_version", 0) >= RMS_VERSION:
        return False
    add_rms_data_to_tracks(y, sr, meta.get("Tracks", []))
    meta["rms_version"] = RMS_VERSION
    return True


def birdnet_week(dt):
    # birdnet geo model uses 48 "weeks", 4 per month
    return (dt.month - 1) * 4 + min(dt.day - 1, 27) // 7 + 1


def location_key(meta):
    loc = meta.get("location")
    if not loc or loc.get("lat") is None or loc.get("lng") is None:
        return None
    week = None
    rec_time = meta.get("recordingDateTime")
    if rec_time:
        week = birdnet_week(datetime.fromisoformat(rec_time.replace("Z", "+00:00")))
    return (round(loc["lat"], 1), round(loc["lng"], 1), week)


def split_name(name):
    # birdnet labels are "Scientific name_Common name"
    if "_" in name:
        return tuple(name.split("_", 1))
    return name, name


class BirdNet:
    def __init__(self, args):
        import birdnet

        self.args = args
        self.model_id = f"birdnet-{args.birdnet_version}-{args.backend}"
        if args.no_geo:
            self.model_id += "-nogeo"
        logging.info("Loading %s", self.model_id)
        self.model = birdnet.load("acoustic", args.birdnet_version, args.backend)
        self.geo_model = None
        if not args.no_geo:
            self.geo_model = birdnet.load("geo", args.birdnet_version, args.backend)
        self.species_lists = {}
        self.session = None
        self.length = args.birdnet_length or self.window_length()
        self.stride = self.length - args.overlap
        logging.info("BirdNET window %ss stride %ss", self.length, self.stride)

    def predict(self, arrays, min_conf, top_k):
        args = self.args
        return self.model.predict_arrays(
            arrays,
            top_k=top_k,
            default_confidence_threshold=min_conf,
            device=args.device,
            n_workers=args.workers,
            batch_size=args.batch_size,
        ).to_structured_array()

    def session_predict(self, arrays):
        """predict_arrays starts and stops birdnet's worker processes on every
        call, so keep one session open for the whole run"""
        if self.session is None:
            args = self.args
            self.session = self.model.predict_session(
                top_k=None,
                default_confidence_threshold=args.min_conf,
                device=args.device,
                n_workers=args.workers,
                batch_size=args.batch_size,
            )
            self.session.__enter__()
        return self.session.run_arrays(arrays).to_structured_array()

    def close(self):
        if self.session is not None:
            self.session.__exit__(None, None, None)
            self.session = None

    def window_length(self):
        # read the context length off the model's own output segments
        try:
            rows = self.predict([(np.zeros(SR * 10, dtype=np.float32), SR)], 0.0, 1)
            if len(rows):
                return round(float(rows[0]["end_time"] - rows[0]["start_time"]), 3)
        except Exception:
            logging.exception("Could not get BirdNET window length")
        logging.warning("Using 3s BirdNET windows, set --birdnet-length to change")
        return 3.0

    def species_list(self, key):
        if key is None or self.geo_model is None:
            return None
        if key not in self.species_lists:
            lat, lng, week = key
            geo = self.geo_model.predict(
                lat, lng, week=week, min_confidence=self.args.geo_min_conf
            )
            # geo model knows some species the acoustic model doesn't
            self.species_lists[key] = geo.to_set() & set(self.model.species_list)
            logging.info(
                "lat %s lng %s week %s: %s species",
                lat,
                lng,
                week,
                len(self.species_lists[key]),
            )
        return self.species_lists[key]

    def add(self, meta, y, sr, duration, overwrite):
        tracks = [
            t
            for t in meta.get("Tracks", [])
            if overwrite or t.get("birdnet_model") != self.model_id
        ]
        if not tracks:
            return False
        key = None if self.args.no_geo else location_key(meta)
        species = self.species_list(key)

        arrays = []
        owners = []
        for track in tracks:
            for start, _ in windows(
                track["start"], track["end"], self.length, self.stride, duration
            ):
                arrays.append((clip(y, sr, start, self.length), sr))
                owners.append((track, start))

        # run unfiltered and apply the geo list here, so windows with no geo
        # species above threshold fall back to the full species list
        rows = self.session_predict(arrays) if arrays else []
        # only marked done once predict has worked
        for track in tracks:
            track["birdnet"] = []
            track["birdnet_model"] = self.model_id
        meta["birdnet_geo"] = (
            None if key is None else {"lat": key[0], "lng": key[1], "week": key[2]}
        )
        segments = {}
        for row in rows:
            track, w_start = owners[int(row["input"])]
            start = round(w_start + float(row["start_time"]), 2)
            scientific, common = split_name(row["species_name"])
            segments.setdefault((id(track), start), (track, []))[1].append(
                {
                    "start": start,
                    "end": round(w_start + float(row["end_time"]), 2),
                    "scientific_name": scientific,
                    "common_name": common,
                    "confidence": round(float(row["confidence"]), 4),
                    "in_geo": species is None or row["species_name"] in species,
                }
            )
        for track, detections in segments.values():
            in_geo = [d for d in detections if d["in_geo"]]
            detections = in_geo if in_geo else detections
            detections.sort(key=lambda d: -d["confidence"])
            track["birdnet"].extend(detections[: self.args.top_k])
        for track in tracks:
            track["birdnet"].sort(key=lambda d: (d["start"], -d["confidence"]))
        return True


def perch_key(recording_id, track_id, start):
    return (str(recording_id), str(track_id), f"{float(start):.2f}")


class Perch:
    def __init__(self, args):
        from perch_hoplite.zoo import model_configs

        from perchembed import load_shard_rows, shard_paths

        self.out_dir = Path(args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = args.shard_size
        # resume, skip windows already embedded and carry on numbering
        self.done = set()
        shards = shard_paths(self.out_dir)
        for npy in shards:
            for r in load_shard_rows(npy):
                self.done.add(
                    perch_key(r["recording_id"], r["track_id"], r["perch_start"])
                )
        self.next_shard = int(shards[-1].stem) + 1 if shards else 0
        self.total = len(self.done)
        logging.info("%s perch windows already in %s shards", self.total, len(shards))

        logging.info("Loading %s", PERCH_MODEL)
        self.model = model_configs.load_model_by_name(PERCH_MODEL)
        self.sr = int(getattr(self.model, "sample_rate", 32000))
        self.length = float(getattr(self.model, "window_size_s", 5.0))
        self.stride = self.length - args.overlap
        logging.info(
            "Perch window %ss at %sHz stride %ss", self.length, self.sr, self.stride
        )
        self.embeddings = []
        self.rows = []

    def embed(self, y, start):
        chunk = clip(y, self.sr, start, self.length)
        embeddings = np.asarray(self.model.embed(chunk).embeddings, dtype=np.float32)
        # one window gives a single frame, average anything else down
        return embeddings.reshape(-1, embeddings.shape[-1]).mean(axis=0)

    def add(self, txt, meta, y, sr, duration):
        y_perch = None
        recording_id = meta.get("id")
        for track in meta.get("Tracks", []):
            for start, end in windows(
                track["start"], track["end"], self.length, self.stride, duration
            ):
                key = perch_key(recording_id, track.get("id"), round(start, 2))
                if key in self.done:
                    continue
                if y_perch is None:
                    y_perch = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
                self.embeddings.append(self.embed(y_perch, start))
                self.rows.append(
                    {
                        "index": self.total,
                        "file": str(txt),
                        "recording_id": recording_id,
                        "track_id": track.get("id"),
                        "manual_tags": ";".join(manual_tags(track)),
                        "track_start": track.get("start"),
                        "track_end": track.get("end"),
                        "perch_start": round(start, 2),
                        "perch_end": round(end, 2),
                        "perch_model": PERCH_MODEL,
                    }
                )
                self.done.add(key)
                self.total += 1
        # checked per recording so a recording is never split across shards
        if len(self.rows) >= self.shard_size:
            self.save()

    def save(self):
        if not self.rows:
            return
        # temp names then rename so a crash never leaves a partial shard, the
        # csv is written last as that is what marks a shard complete
        npy = self.out_dir / f"{self.next_shard:05d}.npy"
        tmp_npy = npy.with_name(npy.stem + ".tmp.npy")
        np.save(tmp_npy, np.stack(self.embeddings))
        tmp_npy.replace(npy)
        tmp_csv = npy.with_name(npy.stem + ".tmp.csv")
        with open(tmp_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PERCH_FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)
        tmp_csv.replace(npy.with_suffix(".csv"))
        logging.info("Saved shard %s with %s embeddings", npy, len(self.rows))
        self.next_shard += 1
        self.embeddings, self.rows = [], []


class CacophonyModel:
    def __init__(self, args):
        import tensorflow as tf

        # custom layers in the saved model, MagTransform is the one audiomodel.py
        # trains with. Passed explicitly as maglayers registers another class
        # under the same name, which doesn't load
        import badwinner2
        import mellayer  # noqa: F401

        model_file = Path(args.model).expanduser()
        model_dir = model_file.parent
        with open(model_dir / "metadata.txt") as f:
            # model_name is set at the top of the file, a later model_name from
            # training is the architecture, so the first of a duplicate key wins
            model_meta = json.load(
                f,
                object_pairs_hook=lambda pairs: {k: v for k, v in reversed(pairs)},
            )
        self.model_meta = model_meta
        self.labels = model_meta["ebird_labels"]
        self.model_name = model_meta.get("model_name") or model_file.stem
        self.weights = args.weights
        logging.info("Loading %s with %s", model_file, self.weights)
        mag_transform = badwinner2.MagTransform
        self.model = tf.keras.models.load_model(
            str(model_file),
            compile=False,
            custom_objects={
                "MyLayers>MagTransform": mag_transform,
                "MagTransform": mag_transform,
            },
        )
        self.model.load_weights(model_dir / self.weights)
        self.model.trainable = False
        self.model.summary()
        # the model makes its own mel from raw audio, so its input size is the
        # context length
        input_shape = self.model.input_shape
        if len(input_shape) != 2:
            raise ValueError(
                f"Expected a raw audio input [batch, samples], model has {input_shape}"
            )
        self.length = input_shape[1] / SR
        self.stride = model_meta.get("segment_stride", 1)
        outputs = self.model.output_shape[-1]
        if outputs != len(self.labels):
            raise ValueError(
                f"Model has {outputs} outputs but {len(self.labels)} ebird_labels"
            )
        self.batch_size = args.batch_size
        logging.info(
            "%s window %ss stride %ss, %s labels",
            self.model_name,
            self.length,
            self.stride,
            len(self.labels),
        )

    def sample(self, y, sr, start):
        return model_sample(y, sr, start, self.length)

    def embedding_model(self):
        """The model with three outputs, the pooled embedding before the
        classifier, the logits and the probabilities, so one pass gives all"""
        import tensorflow as tf

        pools = [
            l
            for l in self.model.layers
            if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)
        ]
        if not pools:
            raise ValueError("No GlobalAveragePooling2D layer to embed from")
        names = [l.name for l in self.model.layers]
        # model.output is a list for models built with outputs=[birds], which
        # would nest the probabilities and give them a leading dim, so take the
        # tensor. "predictions" is the linear layer before the sigmoid
        probs = self.model.outputs[0]
        logits = (
            self.model.get_layer("predictions").output
            if "predictions" in names
            else probs
        )
        return tf.keras.Model(self.model.input, [pools[-1].output, logits, probs])

    def done(self, track):
        return any(
            r.get("model_name") == self.model_name and r.get("weights") == self.weights
            for r in track.get("cacophony-model", [])
        )

    def add(self, meta, y, sr, duration, overwrite):
        tracks = [t for t in meta.get("Tracks", []) if overwrite or not self.done(t)]
        if not tracks:
            return False
        samples = []
        owners = []
        for track in tracks:
            for start, end in windows(
                track["start"], track["end"], self.length, self.stride, duration
            ):
                samples.append(self.sample(y, sr, start))
                owners.append((track, start, end))
        predictions = self.model.predict(
            np.stack(samples), batch_size=self.batch_size, verbose=0
        )
        results = {}
        for (track, start, end), p in zip(owners, predictions):
            result = results.get(id(track))
            if result is None:
                result = {
                    "model_name": self.model_name,
                    "weights": self.weights,
                    "windows": [],
                }
                results[id(track)] = result
                # replace an older result of this model, keep other models
                track["cacophony-model"] = [
                    r
                    for r in track.get("cacophony-model", [])
                    if r.get("model_name") != self.model_name
                ] + [result]
            result["windows"].append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "scores": {
                        label: round(float(s), 4) for label, s in zip(self.labels, p)
                    },
                }
            )
        return True


def setup_gpu():
    try:
        import tensorflow as tf

        # tf takes all gpu memory by default, leave some for birdnet and perch
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        logging.exception("Could not set gpu memory growth")


def process(audio, args, birdnet, perch, cacophony):
    txt = audio.with_suffix(".txt")
    if txt.exists():
        try:
            with open(txt) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logging.error("Could not read %s, skipping", txt)
            return
        if not isinstance(meta, dict):
            logging.error("%s isn't a metadata object, skipping", txt)
            return
        changed = False
    else:
        meta = {"id": audio.stem}
        changed = True

    try:
        y, sr = load_recording(audio)
    except Exception:
        logging.exception("Could not load %s", audio)
        return
    duration = len(y) / sr
    if meta.get("duration") is None:
        meta["duration"] = round(duration, 3)
        changed = True
    if meta.get("file") != str(audio):
        meta["file"] = str(audio)
        changed = True

    try:
        changed |= add_signals_and_tracks(meta, y, sr, duration, args.overwrite)
        changed |= add_rms(meta, y, sr, args.overwrite)
        # a model failing on a file doesn't stop the others, it isn't marked
        # done so is retried on the next run
        for name, model in (("BirdNET", birdnet), ("cacophony model", cacophony)):
            if model is None:
                continue
            try:
                changed |= model.add(meta, y, sr, duration, args.overwrite)
            except Exception:
                logging.exception("%s failed on %s", name, audio)
    finally:
        # save whatever was done, even if a later step failed
        if changed:
            write_meta(txt, meta)
            logging.info("Updated %s", txt)
    if perch is not None:
        perch.add(txt, meta, y, sr, duration)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir", help="Directory to search for recordings")
    parser.add_argument(
        "--out", default="perch-embeddings", help="Directory for perch shards"
    )
    parser.add_argument(
        "--model",
        default="~/cacophony/audio-model/spectv10/spectv10NoMixKokakoFine/spectv10NoMixKokakoFine.keras",
        help="Cacophony .keras model file, metadata.txt and --weights are read from its directory",
    )
    parser.add_argument("--weights", default="val_pr_auc.weights.h5")
    parser.add_argument(
        "--overlap",
        type=float,
        default=1.0,
        help="Seconds of overlap between birdnet and perch windows on long tracks",
    )
    parser.add_argument("--overwrite", action="store_true", help="Redo every step")
    parser.add_argument("--no-birdnet", action="store_true")
    parser.add_argument("--no-perch", action="store_true")
    parser.add_argument("--no-cacophony", action="store_true")
    parser.add_argument("--birdnet-version", default="3.0", choices=["2.4", "3.0"])
    parser.add_argument("--backend", default="pt")
    parser.add_argument("--device", default="GPU", help="BirdNET device, CPU or GPU")
    parser.add_argument(
        "--birdnet-length",
        type=float,
        help="BirdNET window seconds, read from the model by default",
    )
    parser.add_argument("--no-geo", action="store_true")
    parser.add_argument("--min-conf", type=float, default=0.1)
    parser.add_argument("--geo-min-conf", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4, help="BirdNET workers")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=1000)
    args = parser.parse_args()
    if args.birdnet_version == "2.4" and args.backend not in ("tf", "pb"):
        args.backend = "tf"

    setup_gpu()
    # each model is loaded once for the whole directory
    cacophony = None if args.no_cacophony else CacophonyModel(args)
    birdnet = None if args.no_birdnet else BirdNet(args)
    perch = None if args.no_perch else Perch(args)

    files = find_audio(args.dir)
    logging.info("%s recordings in %s", len(files), args.dir)
    try:
        for i, audio in enumerate(files):
            try:
                process(audio, args, birdnet, perch, cacophony)
            except Exception:
                logging.exception("Failed on %s", audio)
            if (i + 1) % 100 == 0:
                logging.info("Done %s / %s", i + 1, len(files))
    finally:
        # also save on ctrl-c or an error so the run can be resumed
        if perch is not None:
            perch.save()
        if birdnet is not None:
            birdnet.close()


if __name__ == "__main__":
    # birdnet runs in worker processes, forked ones can't use the gpu once
    # tensorflow has initialised cuda in this process ("CUDA error:
    # initialization error"). Newer birdnet defaults to spawn, older forks
    multiprocessing.set_start_method("spawn", force=True)
    # m4a falls back to audioread which warns on every file
    warnings.filterwarnings("ignore", message="PySoundFile failed")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
