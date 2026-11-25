"""Audio loading shared by training (audiowriter.py) and prediction
(addtracks.py, predict.py) so both see exactly the same samples.

Decodes with ffmpeg to 16 bit pcm, as audioread's ffmpeg backend did before
(librosa.load(audioread.ffdec.FFmpegAudioFile(file)), deprecated in librosa
0.10 and removed in 1.0): the same decoder, the same handling of AAC
encoder delay and the same 16 bit rounding, then scaled to float32 by 1/32768
and averaged to mono like librosa. Resampling is librosa.resample with its
default (soxr_hq), as before.

scripts/compare_audio_loaders.py checks it matches the old way.
"""

import json
import subprocess

import numpy as np


class AudioLoadError(Exception):
    pass


def probe(file):
    """(sample rate, channels) of the file's first audio stream"""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            str(file),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AudioLoadError(f"ffprobe failed on {file}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise AudioLoadError(f"No audio stream in {file}")
    return int(streams[0]["sample_rate"]), int(streams[0]["channels"])


def load_audio(file, sr=None, mono=True):
    """(samples float32, sample rate). sr resamples to that rate, None keeps
    the file's. Mono averages the channels, otherwise (channels, samples)"""
    native_sr, channels = probe(file)
    result = subprocess.run(
        # the first audio stream, as probed
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(file), "-map", "0:a:0", "-f", "s16le", "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise AudioLoadError(
            f"ffmpeg failed on {file}: {result.stderr.decode(errors='replace').strip()}"
        )
    y = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / np.float32(32768)
    if channels > 1:
        y = y[: len(y) - len(y) % channels].reshape((-1, channels)).T
        if mono:
            y = np.mean(y, axis=0)
    if sr is not None and sr != native_sr:
        import librosa

        y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
        native_sr = sr
    return y, native_sr
