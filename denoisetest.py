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
from plot_utils import plot_spec, plot_mel_signals
from custommels import mel_spec

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
    frames, sr = librosa.load(aro, sr=None)
    aro.close()
    if resample is not None and resample != sr:
        frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
        sr = resample
    return frames, sr


def signal_noise(file, hop_length=281):
    frames, sr = load_recording(file)
    end = get_end(frames, sr)
    print("got end of ", end, " for file of ", len(frames) / sr)
    1 / 0
    frames = frames[: int(sr * 37.2)]
    n_fft = sr // 10
    # frames = frames[: sr * 3]
    spectogram = np.abs(librosa.stft(frames, n_fft=n_fft, hop_length=hop_length))
    return signal_noise_data(spectogram, sr, hop_length=hop_length, n_fft=n_fft)


def signal_noise_data(spectogram, sr, min_bin=None, hop_length=281, n_fft=None):
    a_max = np.amax(spectogram)
    # spectogram = spectogram / a_max
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
    min_width = 0.1
    min_width = min_width * sr / 281
    min_width = int(min_width)

    width = 0.25  # seconds
    width = width * sr / 281
    width = int(width)
    freq_range = 1000
    height = 0
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    for i, f in enumerate(freqs):
        if f > freq_range:
            height = i + 1
            break

    kernel = np.ones((4, 4), np.uint8)
    signal = cv2.morphologyEx(signal, cv2.MORPH_OPEN, kernel)
    noise = cv2.morphologyEx(noise, cv2.MORPH_OPEN, kernel)
    #
    signal = cv2.dilate(signal, np.ones((height, width), np.uint8))
    signal = cv2.erode(signal, np.ones((height // 10, width), np.uint8))

    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(signal)
    for i, s in enumerate(stats):
        print(i, s)
        if i == 1:
            small_mask[small_mask == i] = 200

        if i == 0:
            continue
        if s[2] <= min_width:
            small_mask[small_mask == i] = 200
    stats = sorted(stats, key=lambda stat: stat[0])
    stats = stats[1:]
    # # for x in small_mask:
    # # print(x[-10:])
    stats = [s for s in stats if s[2] > min_width]
    # for s in stats:
    #     print(s)
    # # stats = np.uint8(stats)
    # # small_mask = np.uint8(small_mask)
    # # small_mask[small_mask > 0] = 255
    print("Signals", len(stats))
    # plot_spec(small_mask)
    # #
    # signal_indicator_vector = np.amax(signal, axis=0)
    #
    # noise_indicator_vector = np.amax(noise, axis=0)
    #
    # signal_indicator_vector = signal_indicator_vector[np.newaxis, :]
    # signal_indicator_vector = cv2.dilate(
    #     signal_indicator_vector, np.ones((4, 1), np.uint8)
    # )
    # print(signal_indicator_vector.shape)
    # signal_indicator_vector = np.where(signal_indicator_vector > 0, 1, 0)
    # signal_indicator_vector = signal_indicator_vector * 255
    #
    # noise_indicator_vector = noise_indicator_vector[np.newaxis, :]
    # noise_indicator_vector = cv2.dilate(
    #     noise_indicator_vector, np.ones((4, 1), np.uint8)
    # )
    # noise_indicator_vector = np.where(noise_indicator_vector > 0, 1, 0)
    #
    # noise_indicator_vector = noise_indicator_vector * 128
    #
    # indicator_vector = np.concatenate(
    #     (signal_indicator_vector, noise_indicator_vector), axis=0
    # )
    i = 0
    # indicator_vector = np.uint8(indicator_vector)
    s_start = -1
    noise_start = -1
    signals = []

    bins = len(freqs)
    for s in stats:
        max_freq = min(len(freqs) - 1, s[1] + s[3])
        freq_range = (freqs[s[1]], freqs[max_freq])
        start = s[0] * 281 / sr
        end = (s[0] + s[2]) * 281 / sr
        signals.append(Signal(start, end, freq_range[0], freq_range[1]))
        # break
    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(noise)
    stats = stats[1:]
    stats = [s for s in stats if s[2] > 4]
    noise = []

    for s in stats:
        max_freq = min(len(freqs) - 1, s[1] + s[3])
        freq_range = (freqs[s[1]], freqs[max_freq])
        start = s[0] * 281 / sr
        end = (s[0] + s[2]) * 281 / sr
        noise.append((start, end, freq_range[0], freq_range[1]))

    # for c in indicator_vector.T:
    #     # print("indicator", c)
    #     if c[0] == 255:
    #         if s_start == -1:
    #             s_start = i
    #     elif s_start != -1:
    #         signals.append((s_start * 281 / sr, (i - 1) * 281 / sr))
    #         s_start = -1
    #     if c[1] == 128:
    #         if noise_start == -1:
    #             noise_start = i
    #     elif noise_start != -1:
    #         noise.append((noise_start * 281 / sr, (i - 1) * 281 / sr))
    #         noise_start = -1
    #
    #     i += 1
    # if s_start != -1:
    #     signals.append((s_start * 281 / sr, (i - 1) * 281 / sr))
    # if noise_start != -1:
    #     noise.append((noise_start * 281 / sr, (i - 1) * 281 / sr))

    # signal_frames = []
    # for s in signals:
    #     s_f = int((s[0]) * sr)
    #     s_e = int((s[1]) * sr)
    #     s_f = max(0, s_f)
    #     signal_frames.extend(frames[s_f:s_e])
    #
    # signal_frames = np.array(signal_frames)
    # name = file.parent / f"{file.stem}-signal.wav"
    # sf.write(str(name), signal_frames, sr)
    # print("signals are", signals)
    # signals = space_signals(signals, spacing=0.1)
    # print("spaced", signals)
    # spectogram = librosa.amplitude_to_db(spectogram, ref=np.max)
    # plot_spec(spectogram, signals, len(frames) / sr)
    # print(signals, noise)
    for s in signals:
        print("Signal", s)
    return signals, noise, spectogram


def space_signals(signals, spacing=0.1):
    # print("prev have", len(self.signals))
    # for s in self.signals:
    #     print(s)
    new_signals = []
    prev_s = None
    for s in signals:
        if prev_s is None:
            prev_s = s
        else:
            if s[0] < prev_s[1] + spacing:
                # combine them
                prev_s = (prev_s[0], s[1])
            else:
                new_signals.append(prev_s)
                prev_s = s
    if prev_s is not None:
        new_signals.append(prev_s)

    return new_signals


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
        if meta.get("signal", None) is not None:
            print("Zeroing existing signal")
            meta["signal"] = None
            # print("Already have signal data")
            # return
        file = f.with_suffix(".m4a")
        if not file.exists():
            file = f.with_suffix(".wav")
        if not file.exists():
            file = f.with_suffix(".mp3")
        if not file.exists():
            logging.info("Not recording for %s", f)
            return
        # r = Recording(meta, file, None)

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


def add_noise(file):
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),


def add_white_noise(file):
    frames, sr = load_recording(file)
    transform = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)
    augmented_sound = transform(frames, sample_rate=sr)


def mix_file(file, mix):
    from audiomentations import AddBackgroundNoise, PolarityInversion

    print("mixxing", mix, " with ", file)
    transform = AddBackgroundNoise(
        sounds_path=[mix],
        min_snr_in_db=3.0,
        max_snr_in_db=30.0,
        noise_transform=PolarityInversion(),
        p=1.0,
    )
    frames, sr = load_recording(file)
    augmented_sound = transform(frames, sample_rate=sr)
    name = Path(".") / f"mixed.wav"
    sf.write(str(name), augmented_sound, 48000)


def get_end(frames, sr):
    hop_length = 281
    spectogram = np.abs(librosa.stft(frames, n_fft=sr // 10, hop_length=hop_length))
    mel = mel_spec(
        spectogram,
        sr,
        sr // 10,
        hop_length,
        120,
        50,
        11000,
        1750,
        power=1,
    )
    start = 0
    chunk_length = sr // hop_length
    # this is roughtly a third of our spectogram used for classification
    end = start + chunk_length
    file_length = len(frames) / sr
    print(mel.shape)
    while end < mel.shape[1]:
        data = mel[:, start:end]
        if np.amax(data) == np.amin(data):
            # end of data
            return start * hop_length // sr
        start = end
        end = start + chunk_length
    return file_length


def main():
    init_logging()
    args = parse_args()
    # mix_file(args.file, args.mix)
    signal, noise, spectogram = signal_noise(args.file)
    unique_signals = []
    f_overlap = 500
    for s in signal:
        merged = False
        for u_i, u in enumerate(unique_signals):
            overlap = s.time_overlap(u)
            freq_overlap = (
                abs(s.mel_freq_start - u.mel_freq_start) < f_overlap
                and abs(s.mel_freq_end - u.mel_freq_end) < f_overlap
            )
            if not freq_overlap:
                continue
            print("start", s.start, "u end", u.end)
            if overlap > 0 or (s.start - u.end) < 3:
                print("Merging", u, " and ", s)
                u.merge(s)
                merged = True
                break
        if not merged:
            unique_signals.append(s)
    to_delete = []
    min_length = 0.5
    for s in unique_signals:
        if s in to_delete:
            continue
        if s.length < min_length:
            to_delete.append(s)
            continue
        for s2 in unique_signals:
            if s2 in to_delete:
                continue
            if s == s2:
                continue
            overlap = s.time_overlap(s2)
            engulfed = overlap >= 0.9 * s2.length
            f_overlap = s.mel_freq_overlap(s2)
            range = s2.mel_freq_range
            range *= 0.9
            if f_overlap > range and engulfed:
                to_delete.append(s2)
    for s in to_delete:
        unique_signals.remove(s)

    for s in unique_signals:
        s.enlarge(1.2)
        print("Final s", s)
    plot_mel_signals(spectogram, unique_signals)
    return
    process(args.file)
    # process_signal(args.file)
    # data = np.array(data)


def mel_freq(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


#
# def freq_overlap_amount(s, s2):
#     return ((s[3] - s[2]) + (s2[3] - s2[2])) - (max(s[3], s2[3]) - min(s[2], s2[2]))
#
#
# def signal_overlap(s, s2):
#     return (s[1] - s[0]) + (s2[1] - s2[0]) - (max(s[1], s2[1]) - min(s[0], s2[0]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confusion", help="Save confusion matrix for model")
    parser.add_argument("--dataset", help="Dataset to predict")
    parser.add_argument("--mix", help="File to mix name")

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


def plot_signals(spec, signals, length):
    plt.figure(figsize=(10, 10))

    ax = plt.subplot(1, 1, 1)

    img = librosa.display.specshow(
        spec, sr=48000, y_axis="log", x_axis="time", ax=ax, hop_length=281
    )
    plt.savefig("temp.png")
    import cv2

    img = cv2.imread("temp.png")

    height, width, _ = img.shape
    width = 900 - 130
    t_p = width / length

    for s in signals:
        start = int(s[0] * t_p) + 130
        end = int(s[1] * t_p) + 130
        print("Drawing signal", s, " at ", start, "-", end, " for shape", img.shape)
        cv2.rectangle(img, (start, 10), (end, height - 10), (0, 255, 0), 3)
    # ax.set_title("Power spectrogram")
    # plt.show()
    # plt.clf()
    # plt.close()
    cv2.imshow("a", img)
    cv2.moveWindow("a", 0, 0)
    cv2.waitKey()


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


def segment_overlap(first, second):
    return (
        (first[1] - first[0])
        + (second[1] - second[0])
        - (max(first[1], second[1]) - min(first[0], second[0]))
    )


class Signal:
    def __init__(self, start, end, freq_start, freq_end):
        self.start = start
        self.end = end
        self.freq_start = freq_start
        self.freq_end = freq_end

        self.mel_freq_start = mel_freq(freq_start)
        self.mel_freq_end = mel_freq(freq_end)

    def time_overlap(self, other):
        return segment_overlap(
            (self.start, self.end),
            (other.start, other.end),
        )

    def mel_freq_overlap(self, other):
        return segment_overlap(
            (self.mel_freq_start, self.mel_freq_end),
            (other.mel_freq_start, other.mel_freq_end),
        )

    def freq_overlap(s, s2):
        return segment_overlap(
            (self.mel_freq_start, self.mel_freq_end),
            (other.mel_freq_start, other.mel_freq_end),
        )

    @property
    def mel_freq_range(self):
        return self.mel_freq_end - self.mel_freq_start

    @property
    def freq_range(self):
        return self.freq_end - self.freq_start

    @property
    def length(self):
        return self.end - self.start

    def enlarge(self, scale):
        new_length = self.length * scale
        extension = new_length / 2
        self.start = self.start - extension
        self.end = self.end + extension
        self.start = max(self.start, 0)

    def merge(self, other):
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)
        self.freq_start = min(self.freq_start, other.freq_start)
        self.freq_end = max(self.freq_end, other.freq_end)
        self.mel_freq_start = mel_freq(self.freq_start)
        self.mel_freq_end = mel_freq(self.freq_end)

    def __str__(self):
        return f"Signal: {self.start}-{self.end} f: {self.freq_start}-{self.freq_end}"


if __name__ == "__main__":
    main()
