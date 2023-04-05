# Build a segment dataset for training.
# Segment headers will be extracted from a track database and balanced
# according to class. Some filtering occurs at this stage as well, for example
# tracks with low confidence are excluded.
import json
import argparse
import os
import random
import datetime
import logging
import pickle
import json
import audioread.ffdec  # Use ffmpeg decoder
import math

# from dateutil.parser import parse as parse_date
import sys
import itertools

# import tensorflow_addons as tfa

# from config.config import Config
import numpy as np

# from audiodataset import AudioDataset

# from audiowriter import create_tf_records

# import tensorflow as tf
# from tfdataset import get_dataset
import time
from pathlib import Path

# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import librosa

# from audiomodel import get_preprocess_fn

# from tfdataset import get_dataset

import soundfile as sf
import matplotlib
import librosa.display

matplotlib.use("TkAgg")

import cv2
from audiodataset import Recording

#
#
def load_recording(file, resample=48000):
    aro = audioread.ffdec.FFmpegAudioFile(file)
    frames, sr = librosa.load(aro)
    aro.close()
    if resample is not None and resample != sr:
        frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
        sr = resample
    return frames, sr


def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def signal_noise(file, hop_length=281):
    frames, sr = load_recording(file)
    # frames = frames[:sr]
    n_fft = sr // 10
    # frames = frames[: sr * 3]
    spectogram = np.abs(librosa.stft(frames, n_fft=n_fft, hop_length=hop_length))
    # plot_spec(spectogram)
    a_max = np.amax(spectogram)
    spectogram = spectogram / a_max
    row_medians = np.median(spectogram, axis=1)
    column_medians = np.median(spectogram, axis=0)
    rows, columns = spectogram.shape

    column_medians = column_medians[np.newaxis, :]
    row_medians = row_medians[:, np.newaxis]
    row_medians = np.repeat(row_medians, columns, axis=1)
    column_medians = np.repeat(column_medians, rows, axis=0)

    signal = (spectogram > 3 * column_medians) & (spectogram > 3 * row_medians)
    noise = (spectogram > 2.5 * column_medians) & (spectogram > 2.5 * row_medians)
    noise[signal == noise] = 0
    noise = noise.astype(np.uint8)
    signal = signal.astype(np.uint8)
    kernel = np.ones((4, 4), np.uint8)
    signal = cv2.morphologyEx(signal, cv2.MORPH_OPEN, kernel)
    noise = cv2.morphologyEx(noise, cv2.MORPH_OPEN, kernel)
    # plot_spec(spectogram)

    signal_indicator_vector = np.amax(signal, axis=0)
    noise_indicator_vector = np.amax(noise, axis=0)

    signal_indicator_vector = signal_indicator_vector[np.newaxis, :]
    signal_indicator_vector = cv2.dilate(
        signal_indicator_vector, np.ones((4, 1), np.uint8)
    )
    signal_indicator_vector = np.where(signal_indicator_vector > 0, 1, 0)
    signal_indicator_vector = signal_indicator_vector * 255

    noise_indicator_vector = noise_indicator_vector[np.newaxis, :]
    noise_indicator_vector = cv2.dilate(
        noise_indicator_vector, np.ones((4, 1), np.uint8)
    )
    noise_indicator_vector = np.where(noise_indicator_vector > 0, 1, 0)

    noise_indicator_vector = noise_indicator_vector * 128

    indicator_vector = np.concatenate(
        (signal_indicator_vector, noise_indicator_vector), axis=0
    )
    i = 0
    indicator_vector = np.uint8(indicator_vector)
    s_start = -1
    noise_start = -1
    signals = []
    noise = []
    for c in indicator_vector.T:
        # print("indicator", c)
        if c[0] == 255:
            if s_start == -1:
                s_start = i
        elif s_start != -1:
            signals.append((s_start * 281 / sr, (i - 1) * 281 / sr))
            s_start = -1
        if c[1] == 128:
            if noise_start == -1:
                noise_start = i
        elif noise_start != -1:
            noise.append((noise_start * 281 / sr, (i - 1) * 281 / sr))
            noise_start = -1

        i += 1
    if s_start != -1:
        signals.append((s_start * 281 / sr, (i - 1) * 281 / sr))
    if noise_start != -1:
        noise.append((noise_start * 281 / sr, (i - 1) * 281 / sr))
    # cv2.imshow("b", indicator_vector)
    # cv2.waitKey()
    signal_frames = []
    for s in signals:
        s_f = int((s[0]) * sr)
        s_e = int((s[1]) * sr)
        s_f = max(0, s_f)
        signal_frames.extend(frames[s_f:s_e])

    signal_frames = np.array(signal_frames)
    name = file.parent / f"{file.stem}-signal.wav"
    sf.write(str(name), signal_frames, sr)

    return signals, noise


def load_metadata(filename):
    """
    Loads a metadata file for a clip.
    :param filename: full path and filename to meta file
    :return: returns the stats file
    """
    with open(str(filename), "r") as t:
        # add in some metadata stats
        meta = json.load(t)
    return meta


from multiprocessing import Pool


def process(base_path):
    meta_files = Path(base_path).glob("**/*.txt")
    files = list(meta_files)
    with Pool(processes=8) as pool:
        [0 for x in pool.imap_unordered(process_signal, files, chunksize=8)]
    # pool.wait()
    print("Finished pool")


def process_signal(f):
    try:
        meta = load_metadata(f)
        file = f.with_suffix(".m4a")
        if not file.exists():
            logging.info("Not recording for %s", f)
            return
        r = Recording(meta, file)

        logging.info("Calcing %s", file)
        signals, noise = signal_noise(file)
        meta["signal"] = signals
        meta["noise"] = noise
        json.dump(
            meta,
            open(
                f,
                "w",
            ),
            indent=4,
        )
        logging.info("Updated %s", f)
    except:
        logging.error("Error processing %s", f, exc_info=True)
    return


def main():
    init_logging()
    args = parse_args()
    process(args.file)
    # data = np.array(data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confusion", help="Save confusion matrix for model")
    parser.add_argument("--dataset", help="Dataset to predict")

    parser.add_argument("file", help="Run name")

    args = parser.parse_args()
    return args


def init_logging():
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """

    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


class Track:
    def __init__(self, label, start, end, confidence):
        self.start = start
        self.label = label
        self.end = end
        self.confidences = [confidence]

    def get_meta(self):
        meta = {}
        meta["begin_s"] = self.start
        meta["end_s"] = self.end
        meta["species"] = self.label
        likelihood = float(round((100 * np.mean(np.array(self.confidences))), 2))
        meta["likelihood"] = likelihood
        return meta


def plot_spec(spec):
    plt.figure(figsize=(10, 10))

    ax = plt.subplot(1, 1, 1)

    img = librosa.display.specshow(spec, sr=48000, y_axis="log", x_axis="time", ax=ax)
    ax.set_title("Power spectrogram")
    plt.show()
    # plt.savefig(f"mel-power-{i}.png", format="png")
    plt.clf()
    plt.close()


def plot_mfcc(mfcc):
    plt.figure(figsize=(10, 10))

    img = librosa.display.specshow(mfcc, x_axis="time", ax=ax)
    ax = plt.subplot(1, 1, 1)
    plt.show()
    # plt.savefig(f"mel-power-{i}.png", format="png")
    plt.clf()
    plt.close()


def plot_mel(mel, i=0):
    plt.figure(figsize=(10, 10))

    ax = plt.subplot(1, 1, 1)
    img = librosa.display.specshow(
        mel, x_axis="time", y_axis="mel", sr=48000, fmax=11000, ax=ax
    )
    plt.show()
    # plt.savefig(f"mel-power-{i}.png", format="png")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    main()
