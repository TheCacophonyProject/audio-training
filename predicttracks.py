"""Cacophony model predictions of every track in a folder of recordings.

Each track, manually tagged or not, is covered by model length windows, spread
evenly from the track start to its end so they overlap by at least --overlap
seconds. Tracks shorter than a window get one window centred on them.

The probability of every label per window is saved in the metadata on the
track under the model name (metadata.txt model_name, or --name), replacing any
predictions already saved under that name e.g.
    track["spectv11NoiseLabel"] = {"model_name": ..., "weights": ...,
        "overlap": 1.0, "windows": [{"start", "end", "scores": {label: p}}]}

Several models can be given, each recording's audio is loaded once for all of
them.

python3 predicttracks.py summit-road/metadata \\
    --model ~/cacophony/audio-model/spectv11/noiselabel/spectv11NoiseLabel.keras \\
    --model ~/cacophony/audio-model/spectv11/noisemix/spectv11NoiseShare.keras
"""

import argparse
import json
import logging
import multiprocessing
import os
import signal
import traceback
import warnings
from argparse import Namespace
from collections import deque
from pathlib import Path

import numpy as np

from addtracks import (
    CacophonyModel,
    load_recording,
    model_sample,
    setup_gpu,
    windows,
    write_meta,
)
from perchembed import find_audio


def default_name(model, model_file):
    """The model_name in its metadata.txt, or the model file's name if that is
    only the architecture e.g. efficientnetv2b3 (metadata.txt without a
    model_name of its own at the top)"""
    if model.model_name == model.model_meta.get("name"):
        return Path(model_file).stem
    return model.model_name


def recording_tracks(meta):
    # Recording reads Tracks before tracks, API metadata may only have tracks
    return meta.get("Tracks") or meta.get("tracks") or []


def init_worker():
    # ctrl-c is handled by the main process, which terminates the pool
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    warnings.filterwarnings("ignore", message="PySoundFile failed")
    # m4a falls back to librosa's deprecated audioread loader
    warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")


def load_samples(txt, audio_file, spans, configs):
    """Runs in a worker process so the GPU isn't waiting on audio decoding.
    Loads the recording and cuts each model's windows of each (start, end)
    track span, configs is a (length, stride) per model.

    Returns ([windows per span per model], [samples per model]) or (None, error
    message)"""
    try:
        audio = find_audio(txt, {"file": audio_file})
        if audio is None:
            return None, "No audio"
        y, sr = load_recording(audio)
        duration = len(y) / sr
        all_windows, all_samples = [], []
        for length, stride in configs:
            track_windows = [
                windows(start, end, length, stride, duration) for start, end in spans
            ]
            all_windows.append(track_windows)
            all_samples.append(
                np.stack(
                    [
                        model_sample(y, sr, start, length)
                        for span_windows in track_windows
                        for start, _ in span_windows
                    ]
                )
            )
        return all_windows, all_samples
    except Exception:
        return None, traceback.format_exc()


def find_jobs(dir):
    """Yields (txt, meta, tracks) for each recording with tracks"""
    for txt in sorted(Path(dir).rglob("*.txt")):
        try:
            with open(txt) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        tracks = [t for t in recording_tracks(meta) if "start" in t and "end" in t]
        if tracks:
            yield txt, meta, tracks


def predict(model, name, overlap, tracks, track_windows, samples, batch_size):
    """Saves model's predictions of samples on each track under name,
    replacing any already there"""
    probs = model.model.predict(samples, batch_size=batch_size, verbose=0)
    i = 0
    for track, span_windows in zip(tracks, track_windows):
        result = {
            "model_name": model.model_name,
            "weights": model.weights,
            "overlap": overlap,
            "windows": [],
        }
        for start, end in span_windows:
            result["windows"].append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "scores": {
                        label: round(float(p), 4)
                        for label, p in zip(model.labels, probs[i])
                    },
                }
            )
            i += 1
        track[name] = result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir", help="Directory to search for metadata .txt files")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Cacophony .keras model file, metadata.txt and --weights are read"
        " from its directory, can be given more than once",
    )
    parser.add_argument("--weights", default="val_pr_auc.weights.h5")
    parser.add_argument(
        "--name",
        action="append",
        help="Metadata key for each --model's predictions, in order, default"
        " the model_name in its metadata.txt",
    )
    parser.add_argument(
        "--overlap", type=float, default=1.0, help="Seconds of overlap between windows"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="Processes loading audio while the main process runs the models",
    )
    args = parser.parse_args()
    if args.name and len(args.name) != len(args.model):
        parser.error("give a --name for every --model")

    setup_gpu()
    models = []
    for i, model_file in enumerate(args.model):
        model = CacophonyModel(
            Namespace(
                model=model_file, weights=args.weights, batch_size=args.batch_size
            )
        )
        stride = model.length - args.overlap
        if stride <= 0:
            parser.error(f"--overlap must be less than the {model.length}s window")
        name = args.name[i] if args.name else default_name(model, model_file)
        logging.info(
            "Saving %s under %s, %ss windows %ss stride",
            model_file,
            name,
            model.length,
            stride,
        )
        models.append((model, name, stride))
    names = [name for _, name, _ in models]
    if len(set(names)) != len(names):
        parser.error(f"models would save under the same name {names}, give --name")
    configs = [(model.length, stride) for model, _, stride in models]

    files = 0
    jobs = find_jobs(args.dir)
    # spawn so workers don't inherit tensorflow and the GPU from this process
    pool = multiprocessing.get_context("spawn").Pool(
        args.workers, initializer=init_worker
    )
    # loaded recordings waiting for the GPU, bounded so decoded audio can't
    # pile up in memory when the models are slower than loading
    pending = deque()
    try:
        while True:
            while len(pending) < args.workers * 2:
                job = next(jobs, None)
                if job is None:
                    break
                txt, meta, tracks = job
                spans = [(t["start"], t["end"]) for t in tracks]
                result = pool.apply_async(
                    load_samples, (txt, meta.get("file"), spans, configs)
                )
                pending.append((txt, meta, tracks, result))
            if not pending:
                break
            txt, meta, tracks, result = pending.popleft()
            all_windows, all_samples = result.get()
            if all_windows is None:
                logging.info("Could not load %s: %s", txt, all_samples)
                continue
            try:
                for (model, name, _), track_windows, samples in zip(
                    models, all_windows, all_samples
                ):
                    predict(
                        model,
                        name,
                        args.overlap,
                        tracks,
                        track_windows,
                        samples,
                        args.batch_size,
                    )
            except Exception:
                logging.exception("Failed on %s", txt)
                continue
            write_meta(txt, meta)
            files += 1
            if files % 100 == 0:
                logging.info("%s files done", files)
    finally:
        pool.terminate()
    logging.info("Saved predictions of %s files under %s", files, names)


if __name__ == "__main__":
    # m4a falls back to audioread which warns on every file
    warnings.filterwarnings("ignore", message="PySoundFile failed")
    # m4a falls back to librosa's deprecated audioread loader
    warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
