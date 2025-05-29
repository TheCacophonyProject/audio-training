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
from plot_utils import plot_spec, plot_mel_signals, plot_mel, plot_mel_weights
from custommel import mel_spec, mel_f
import tensorflow as tf
from identifytracks import (
    get_tracks_from_signals,
    Signal,
    signal_noise as track_signals,
)

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

# matplotlib.use("TkAgg")

import cv2
from audiodataset import Recording

SIGNAL_WIDTH = 0.25


#
#
# def load_recording(file, resample=48000):
#     aro = audioread.ffdec.FFmpegAudioFile(file)
#     frames, sr = librosa.load(aro, sr=None)
#     aro.close()
#     if resample is not None and resample != sr:
#         frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
#         sr = resample
#     return frames, sr
def load_recording(file, resample=48000):
    # librosa.load(file) giving strange results
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
    # end = 5
    # frames = frames[int(20*sr): int(sr * 40)]

    frames = frames[: int(sr * end)]
    # frames = frames[: sr * 120]
    # n_fft = sr // 10
    n_fft = 4096
    # spectogram = librosa.stft(frames, n_fft=n_fft, hop_length=hop_length)
    # plot_spec(spectogram)
    signals, spectogram = track_signals(frames, sr, hop_length=hop_length, n_fft=n_fft)
    # , min_width=0, min_height=0
    # )
    noise = []
    return signals, noise, spectogram, frames, end


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def signal_noise_data(spectogram, sr, min_bin=None, hop_length=281, n_fft=None):
    # normed = librosa.display.specshow(
    #     # S,
    #     librosa.amplitude_to_db(spectogram, ref=np.max),
    #     x_axis="time",
    #     y_axis="linear",
    #     sr=48000,
    #     fmax=30000,
    #     fmin=0,
    #     hop_length=281,
    # )
    # normed = scale_minmax(spectogram,0,255).astype(np.uint8)
    # # normed = 255 - normed
    # print(normed)
    # plt.imshow(normed,origin="lower")
    # plt.show()
    # # plot_spec(spectogram)
    # # a_max = np.amax(spectogram)
    # # a_min = np.amin(spectogram)
    # # normed = 255*(spectogram- a_min)/ (a_max-a_min)
    # # normed = np.uint8(normed)
    # # print(a_max,a_min)
    # edges = cv2.Canny(normed,100,200)
    # plt.imshow(edges,origin="lower")
    # plt.show()

    a_max = np.amax(spectogram)
    # spectogram = spectogram / a_max
    row_medians = np.median(spectogram, axis=1)
    column_medians = np.median(spectogram, axis=0)
    rows, columns = spectogram.shape

    column_medians = column_medians[np.newaxis, :]
    row_medians = row_medians[:, np.newaxis]
    row_medians = np.repeat(row_medians, columns, axis=1)
    column_medians = np.repeat(column_medians, rows, axis=0)
    kernel = np.ones((4, 4), np.uint8)
    spectogram = cv2.morphologyEx(spectogram, cv2.MORPH_OPEN, kernel)

    # kernel = np.ones((4, 4*3), np.uint8)
    # signal = (spectogram > 3 * column_medians) & (spectogram > 3 * row_medians)
    # noise = (spectogram > 2.5 * column_medians) & (spectogram > 2.5 * row_medians)

    # plt.imshow(np.uint8(signal)*255,origin="lower")
    # plt.show()
    # plt.imshow(np.uint8(noise) * 100 + np.uint8(signal)*255,origin="lower")
    # plt.show()
    # normed =(spectogram > 1 * column_medians) & (spectogram > 1 * row_medians)
    # normed = np.uint8(normed)

    # normed = np.float32(normed*125)+ np.uint8(signal)*255
    # np.clip(normed,0,255,out = normed)
    # # print(normed)
    # plt.imshow(np.uint8(normed),origin="lower")
    # plt.show()

    # 1/0
    signal_thresh = 2
    signal = (spectogram > 3 * column_medians) & (spectogram > 3 * row_medians)
    noise = (spectogram > 2.5 * column_medians) & (spectogram > 2.5 * row_medians)
    noise[signal == noise] = 0
    noise = noise.astype(np.uint8)
    signal = signal.astype(np.uint8)
    normed = scale_minmax(spectogram, 0, 100).astype(np.uint8)
    normed = (spectogram > 1 * column_medians) & (spectogram > 1 * row_medians)
    normed = normed + signal * 255
    normed = normed.astype(np.uint8)
    normed = normed * 255
    # plt.imshow(normed,origin="lower")
    # plt.show()
    otsus = True
    kernel2 = (45, 205)
    image = cv2.GaussianBlur(normed, kernel2, 0)
    flags = cv2.THRESH_BINARY
    if otsus:
        flags += cv2.THRESH_OTSU
    _, image = cv2.threshold(image, None, 255, flags)
    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel2)
    image = cv2.dilate(image, kernel2, iterations=4)

    # plt.imshow(image,origin="lower")
    # plt.show()
    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(image)
    # fig = plt.figure(figsize=(10, 10))
    # ax = plt.subplot(1, 1, 1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    import matplotlib.patches as patches

    # for s in stats[1:]:
    #     # max_freq = min(len(freqs) - 1, s[1] + s[3])
    #     # freq_range = (freqs[s[1]], freqs[max_freq])
    #     # start = s[0] * 281 / sr
    #     # end = (s[0] + s[2]) * 281 / sr
    #     # print(s,start,end, freq_range)
    #     rect = patches.Rectangle(
    #         (s[0], s[1]),
    #         s[2],
    #         s[3],
    #         linewidth=1,
    #         edgecolor="r",
    #         facecolor="none",
    #     )
    #     ax.add_patch(rect)
    # edges = cv2.Canny(normed,100,200)
    # ax.imshow(image,origin="lower")
    # plt.show()

    # print(signal)
    # print("SIgnal shape is ",signal.shape)
    # noise[noise>0]= 128
    # print(noise)
    signal[signal > 0] = 255
    normed = (spectogram > 1 * column_medians) & (spectogram > 1 * row_medians)
    normed = np.uint8(normed)
    freq_range = 300
    height = 0
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    for i, f in enumerate(freqs):
        if f > freq_range:
            height = i + 1
            break
    # min_width = 0.3
    # min_width = min_width * sr / 281
    # min_width = int(min_width)

    width = 0.5  # seconds
    width = width * sr / 281
    width = int(width)

    signal = cv2.morphologyEx(signal, cv2.MORPH_OPEN, kernel)
    # noise = cv2.morphologyEx(noise, cv2.MORPH_OPEN, kernel)
    #
    signal = cv2.dilate(signal, np.ones((height, width), np.uint8))
    signal = cv2.erode(signal, np.ones((height // 10, width), np.uint8))

    normed = np.float32(normed * 10) + np.uint8(signal)
    np.clip(normed, 0, 255, out=normed)
    signal = np.uint8(normed)
    # signal = signal + noise
    # plt.imshow(signal,origin="lower")
    # plt.show()
    # plt.imshow(noise,origin="lower")
    # plt.show()

    # signal = cv2.morphologyEx(signal, cv2.MORPH_OPEN, kernel)
    noise = cv2.morphologyEx(noise, cv2.MORPH_OPEN, kernel)
    #
    # signal = cv2.dilate(signal, np.ones((height, width), np.uint8))
    # signal = cv2.erode(signal, np.ones((height // 10, width), np.uint8))
    # plt.imshow(signal,origin="lower")
    # plt.show()
    # 1/0
    otsus = True
    # kernel2=(45, 205)
    # image = cv2.GaussianBlur(signal, kernel, 0)
    flags = cv2.THRESH_BINARY
    if otsus:
        flags += cv2.THRESH_OTSU
    _, image = cv2.threshold(signal, None, 255, flags)
    # plt.imshow(image,origin="lower")
    # plt.show()
    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(image)

    stats = stats[1:]
    print(stats)
    stats = sorted(stats, key=lambda stat: stat[0])
    # # for x in small_mask:
    # # print(x[-10:])
    min_width = 0.65 * width
    min_height = height - height // 10
    print("Min width is ", min_width, min_height, " height is ", height)
    stats = [s for s in stats if s[2] > min_width and s[3] > min_height]
    s_start = -1
    noise_start = -1
    signals = []

    bins = len(freqs)
    # print("Freqs are", freqs)
    for s in stats:
        max_freq = min(len(freqs) - 1, s[1] + s[3])
        freq_range = (freqs[s[1]], freqs[max_freq])
        start = s[0] * 281 / sr
        end = (s[0] + s[2]) * 281 / sr
        print(s)
        signals.append(Signal(start, end, freq_range[0], freq_range[1], s[4]))
        print(signals[-1])
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

    return signals, noise


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
        signals, noise, spectogram, frames = signal_noise(file)
        signals = [s.to_array() for s in signals]
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
    while end < mel.shape[1]:
        data = mel[:, start:end]
        if np.amax(data) == np.amin(data):
            # end of data
            return start * hop_length // sr
        start = end
        end = start + chunk_length
    return file_length


# def merge_signals(signals, time_overlap_percent = 0.75, freq_overlap_percent = 0.75):
#     unique_signals = []
#     to_delete = []
#     something_merged = False
#     i = 0

#     signals = sorted(signals, key=lambda s: s.mel_freq_end, reverse=True)
#     signals = sorted(signals, key=lambda s: s.start)

#     for s in signals:
#         if s in to_delete:
#             continue
#         merged = False
#         for u_i, u in enumerate(signals):
#             if u in to_delete:
#                 continue
#             if u == s:
#                 continue
#             in_freq = u.mel_freq_end < 1500 and s.mel_freq_end < 1500
#             in_freq = in_freq or u.mel_freq_start > 1500 and s.mel_freq_start > 1500
#             if not in_freq:
#                 # print("Skipping", s, " with ", u, " as freqs differ")
#                 continue
#             overlap = s.time_overlap(u)
#             if s.mel_freq_start > 1000 and u.mel_freq_start > 1000:

#                 freq_overlap_time = 0.5
#             else:
#                 freq_overlap_time = freq_overlap_percent
#             if s.start > u.end:
#                 time_diff = s.start - u.end
#             else:
#                 time_diff = u.start - s.end
#             mel_overlap = s.mel_freq_overlap(u)
#             # print("Checking over lap for", s, " with ", u, overlap, mel_overlap)
#             # ensure both are either below 1500 or abov
#             if overlap > u.length * time_overlap_percent and mel_overlap > -20:
#                 #  (
#                 #     mel_overlap > u.Freq_range * freq_overlap
#                 # ):
#                 # times overlap a lot be more leninant on freq
#                 # s.merge(u)
#                 s.merge(u)

#                 merged = True

#                 break
#             elif (time_diff <=2 or overlap > 0) and mel_overlap > u.mel_freq_range * freq_overlap_time:
#                 # time overlaps at all with more freq overlap
#                 s.merge(u)

#                 merged = True

#                 break

#             # elif mel_overlap > u.mel_freq_range * freq_overlap_time and time_diff <= 2:
#             #     if u.mel_freq_end > s.mel_freq_range:
#             #         range_overlap = s.mel_freq_range / u.mel_freq_range
#             #     else:
#             #         range_overlap = u.mel_freq_range / s.mel_freq_range

#             #         # atleast 75% overlap
#             #     if range_overlap < 0.75:
#             #         continue
#             #     # freq range similar

#             #     s.merge(u)
#             #     merged = True

#             #     break

#         if merged:
#             something_merged = True
#             to_delete.append(u)

#     for s in to_delete:
#         signals.remove(s)

#     return signals, something_merged


# def signals_to_tracks(unique_signals):
#     count = 0

#     # return
#     merged = True
#     while merged:
#         count += 1
#         unique_signals, merged = merge_signals(unique_signals)

#     min_length = 0.35
#     to_delete = []
#     # print("Enlarged are", s)
#     for s in unique_signals:
#         # continue
#         if s in to_delete:
#             continue
#         if s.length < min_length:
#             to_delete.append(s)
#             continue
#         s.enlarge(1.4)
#         for s2 in unique_signals:
#             if s2 in to_delete:
#                 continue
#             if s == s2:
#                 continue
#             overlap = s.time_overlap(s2)
#             # print("time over lap for", s, s2, overlap, s2.length)
#             engulfed = overlap >= 0.9 * s2.length
#             f_overlap = s.mel_freq_overlap(s2)
#             range = s2.mel_freq_range
#             range *= 0.7
#             # print(
#             #     "Comparing",
#             #     s,
#             #     " and ",
#             #     s2,
#             #     " f overlap",
#             #     f_overlap,
#             #     range,
#             #     " engulfed",
#             #     engulfed,
#             # )
#             if f_overlap > range and engulfed:
#                 to_delete.append(s2)
#             # elif engulfed and s2.freq_start > s.freq_start and s2.freq_end < s.freq_end:
#             # print("s2", s2, " is inside ", s)
#             # to_delete.append(s2)
#     for s in to_delete:
#         unique_signals.remove(s)
#     return unique_signals


def tracks_to_audio(tracks, spectogram, frames, sr=48000, hop_length=281):
    import soundfile as sf

    n_fft = sr // 10
    for t in tracks:
        print(
            "writing out tracks",
            t.start,
            t.end,
            t.freq_start,
            t.freq_end,
        )
        start = t.start * sr
        end = t.end * sr
        data = frames[int(start) : int(end)]
        start = start / hop_length
        end = end / hop_length
        end = min(spectogram.shape[1], end)
        start = int(start)
        end = int(end)
        spect_data = spectogram[:, start:end].copy()
        low_pass = t.freq_start
        high_pass = t.freq_end
        bins = 1 + n_fft / 2
        max_f = sr / 2
        gap = max_f / bins
        bandpassed = butter_bandpass_filter(data, low_pass, high_pass, sr, order=5)

        # if low_pass is not None:
        #     min_bin = low_pass // gap
        #     spect_data[: int(min_bin)] = 0
        #
        # if high_pass is not None:
        #     max_bin = high_pass // gap
        #     spect_data[int(max_bin) :] = 0

        # y_data = librosa.istft(spect_data, hop_length=281, n_fft=n_fft)
        # y_inv = librosa.griffinlim(S)
        sf.write(f"{t.start}-{t.end}-{t.freq_start}-{t.freq_end}.wav", bandpassed, sr)
        # 1 / 0


def means_merge(spectogram, signals):
    features = np.float32([s.to_features() for s in signals])
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS

    n_clusters = 2

    # model = KMeans(n_clusters).fit(features)
    model = AgglomerativeClustering(None, distance_threshold=500).fit(features)
    # model = DBSCAN(min_samples = 2).fit(features)
    # n_clusters = model.n_clusters_
    labels = model.labels_
    import matplotlib.colors as mcolors

    possible_colours = list(mcolors.CSS4_COLORS.keys())
    rect_colours = []
    print("Clusters are ", labels)
    print(possible_colours)
    merged = {}
    index = 0
    merged[-1] = []
    for l in labels:
        if l == -1:

            merged[-1].append(signals[index])
            print("Ignored", signals[index])
        elif l in merged:
            print("Merging l", l)
            signal = merged[l]
            signal.merge(signals[index])
        else:
            merged[l] = signals[index]
        # print("Possible colours at ",l,possible_colours[l*10])
        # rect_colours.append(possible_colours[l*4])
        index += 1

    signals = []
    colour_i = 0
    for signal in merged[-1]:
        signals.append(signal)
        rect_colours.append(possible_colours[0])
    del merged[-1]
    for l in labels:
        if l == -1:
            continue
        signals.append(merged[l])
        rect_colours.append(possible_colours[l + 1])

    print("Signals are ", len(signals))
    plot_mel_signals(np.abs(spectogram), signals, colours=rect_colours)


def test_plot(file):
    frames, sr = load_recording(file)
    frames = frames[18 * sr : 22 * sr]
    # frames,sr = librosa.load(file)
    # print("Oriuginal sr ",sr, np.amax(frames),np.amin(frames))
    # frames = librosa.resample(frames, orig_sr=sr, target_sr=96000)
    # frames = librosa.util.normalize(frames)
    # print("New sr ",24000, np.amax(frames),np.amin(frames))
    # return
    # end = get_end(frames, sr)
    # frames = frames[: int(sr * end)]
    # frames = frames[: sr * 120]
    # n_fft = sr // 10
    n_fft = 4096
    hop_length = 281

    # librosa.power_to_db(S, ref=np.max)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # S_dB = librosa.power_to_db(mel, ref=np.max)
    # img = librosa.display.specshow(S_dB, x_axis='time',
    #                     y_axis='mel', sr=sr,
    #                     fmax=8000, ax=ax)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectrogram')
    # # plt.show()
    # print("MELLL")
    # plot_mel(mel,"test.png")
    # frames_2 = butter_bandpass_filter(frames,150,15000,sr)
    # frames_2 = frames
    # frames_2 = normalize(frames_2)
    # # print("Max min ",np.amax(frames_2),np.amin(frames_2))
    # mel_2 = librosa.feature.melspectrogram(y=frames_2, sr=sr,power=1,fmin=50,fmax=11000,n_mels=160,hop_length=281)
    # plot_mel(mel_2,"default160.png")

    # spectogram = librosa.stft(frames_2, n_fft=4096, hop_length=281)
    # plot_spec(spectogram)
    n_fft = 4096
    spectogram = librosa.stft(normalize(frames), n_fft=n_fft, hop_length=281)
    # plot_spec(spectogram)
    fmin = 500
    fmax = 11000
    # hop_length=281
    mel = mel_spec(
        spectogram,
        sr,
        n_fft,
        hop_length,
        160,
        fmin,
        fmax,
        1000,
        power=1,
    )
    plot_mel(mel, filename="mel4096")

    # spectogram = librosa.stft(frames_2, n_fft=1024, hop_length=281)

    # mel = mel_spec(
    #     spectogram,
    #     sr,
    #     1024,
    #     hop_length,
    #     160,
    #     fmin,
    #     fmax,
    #     1750,
    #     power=1,
    # )
    # plot_mel(mel,filename="mel1024")


def normalize(input):
    min_v = tf.math.reduce_min(input, -1, keepdims=True)
    input = tf.math.subtract(input, min_v)
    max_v = tf.math.reduce_max(input, -1, keepdims=True)
    input = tf.math.divide(input, max_v) + 0.000001
    input = tf.math.subtract(input, 0.5)
    input = tf.math.multiply(input, 2)
    return np.array(input)


def merge_again(tracks):
    post_filter = []
    tracks_sorted = sorted(tracks, key=lambda track: track.start)
    current_track = None
    for t in tracks_sorted:
        if current_track is None:
            current_track = t
            post_filter.append(current_track)

            continue
        overlap = current_track.time_overlap(t)
        percent_overlap = overlap / t.length
        percent_overlap_2 = overlap / current_track.length

        f_overlap = current_track.mel_freq_overlap(t)
        f_percent_overlap = f_overlap / t.mel_freq_range

        if percent_overlap_2 > 0.5:
            post_filter = post_filter[: len(post_filter) - 1]
            post_filter.append(t)
            current_track = t
        elif percent_overlap > 0.5 or (percent_overlap > 0 and f_percent_overlap > 0.5):
            if f_percent_overlap > 0.5:
                current_track.end = max(current_track.end, t.end)
        else:
            # encountered another big track shall we just make this current track and allow some small overlap???
            current_track = t
            post_filter.append(current_track)

        # if  percent_overlap>0.5 or percent_overlap_2> 0.5:
        #     print("Current track ", current_track, " over laps ", t, " with ", overlap)
        #     print("Track 1 over lap is ",percent_overlap, " track 2 overlap is ",percent_overlap_2)
        # el
        if overlap <= 0:
            current_track = t
            post_filter.append(current_track)
    return post_filter


def birdnet_mel(sr, data, frame_length, hop_length=281, mel_weights=None):
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=160,
        num_spectrogram_bins=frame_length // 2 + 1,
        sample_rate=sr,
        lower_edge_hertz=50,
        upper_edge_hertz=11000,
        dtype=tf.float32,
    )
    # Perform STFT
    spec = tf.signal.stft(
        data,
        frame_length,
        hop_length,
        fft_length=frame_length,
        window_fn=tf.signal.hann_window,
        pad_end=True,
        name="stft",
    )
    # Cast from complex to float
    spec = tf.dtypes.cast(spec, tf.float32)

    if mel_weights is not None:

        # S = librosa.feature.melspectrogram(S=D, sr=sr)
        spec = tf.transpose(spec, [1, 0])
        # print(mel_weights.shape, spec.shape)
        spec = tf.tensordot(mel_weights, spec, 1)

        spec = tf.math.pow(spec, 2.0)
        return spec.numpy()

        # or

        mel_weights = tf.transpose(mel_weights, [1, 0])
        spec = tf.tensordot(spec, mel_weights, 1)

        spec = tf.math.pow(spec, 2.0)
        spec = tf.transpose(spec, [1, 0])

        return spec.numpy()

    spec = tf.tensordot(spec, mel_filterbank, 1)

    # Apply mel scale

    # Convert to power spectrogram
    spec = tf.math.pow(spec, 2.0)

    # Flip spec horizontally
    # spec = tf.reverse(spec, axis=[1])
    spec = tf.transpose(spec, [1, 0])

    # Swap axes to fit input shape
    return spec.numpy()


def main():
    init_logging()

    args = parse_args()
    n_fft = 4096
    hop_length = 281
    N_MELS = 160
    BREAK_FREQ = 1000
    MEL_WEIGHTS = mel_f(48000, N_MELS, 50, 11000, n_fft, BREAK_FREQ)
    # plot_mel_weights(MEL_WEIGHTS)
    MEL_WEIGHTS = librosa.filters.mel(
        sr=48000, n_fft=n_fft, n_mels=N_MELS, fmin=50, fmax=11000
    )
    # plot_mel_weights(MEL_WEIGHTS)

    frames, sr = load_recording(args.file)
    # mel = birdnet_mel(sr, frames, n_fft, hop_length=hop_length, mel_weights=MEL_WEIGHTS)
    # plot_mel(mel)
    # # return

    # spectogram = np.abs(librosa.stft(frames, n_fft=n_fft, hop_length=hop_length))
    # plot_spec(spectogram)
    # mel = spectogram
    # mel = tf.tensordot(MEL_WEIGHTS, mel, 1)
    # mel = tf.math.pow(mel, 2)

    # mel = mel.numpy()
    # print(mel.shape)
    # plot_mel(mel)
    # return
    # test_image = np.zeros((40,40),dtype = np.uint8)
    # test_image[10][10] = 255
    # kernel = np.ones((4, 10), np.uint8)
    # image = cv2.dilate(test_image, kernel)
    # print("NUmber of non zeros ", len(image[image>0]))

    # normed = scale_minmax(spectogram,0,255).astype(np.uint8)
    # # normed = 255 - normed
    # print(normed)
    # plt.imshow(image,origin="lower")
    # plt.show()

    # mix_file(args.file, args.mix)
    signals, noise, spectogram, frames, end = signal_noise(args.file)
    freq_filter = 1000
    total_signal = 0
    for s in signals:
        print("Sig is ", s)
        total_signal += s.length
    print("Sig percent ", 100 * total_signal / len(frames) / sr)

    # plot_mel_signals(np.abs(spectogram), signals, noise)
    # return
    # print(signals)
    # means_merge(spectogram,signals)
    # return
    # for s in signal:
    # print(s)
    # 1 / 0
    # for s in signals:
    # print(s)
    # return
    # length_per_segment = []
    # for start in range(int(end-3)+1):
    #     s_end = start + 3
    #     signal_length = 0
    #     for s in signals:
    #         if s.start < start and s.end < s_end:
    #             continue

    #         if s.start > s_end:
    #             break
    #         signal_length += min(s.end,s_end) - max(start,s.start)
    #         # print("adding ",min(s.end,s_end) -  max(start,s.start), " for ", start, s_end, " from signal ", s, " length ", s.length)
    #     length_per_segment.append(signal_length)
    #     print(f"Signal length at {start}-{s_end} is {signal_length}")
    tracks = get_tracks_from_signals(signals, end)

    length_per_segment = []
    best_segment = None
    previous_segment = None
    length_score = None
    print("End is ", end)
    for start in range(int(end - 3) + 1):
        s_end = start + 3
        signal_length = 0
        for s in tracks:
            if s.start < start and s.end < s_end:
                continue

            if s.start > s_end:
                break
            signal_length += min(s.end, s_end) - max(start, s.start)
            # print("adding ",min(s.end,s_end) -  max(start,s.start), " for ", start, s_end, " from signal ", s, " length ", s.length)
        if len(length_per_segment) > 0:
            length_score = length_per_segment[-1]
            if len(length_per_segment) == 1:
                length_score += signal_length
            else:
                length_score += (signal_length + length_per_segment[-2]) / 2
            # print("Comparing",best_segment, " to ",length_score)
            if best_segment is None or best_segment[2] < length_score:
                # print("Updating ",start-1)
                best_segment = (start - 1, signal_length, length_score)
        else:
            best_segment = (start, signal_length, signal_length)
            # print(length_score)
        print(f"Signal length at {start}-{s_end} is {signal_length}")

        length_per_segment.append(signal_length)

    print("Best signal is ", best_segment)
    for t in tracks:
        print("Track is ", t)
    plot_mel_signals(np.abs(spectogram), tracks, noise)

    # tracks = merge_again(tracks)
    # ,time_overlap_percent = 0.5, freq_overlap_percent = 0.5)

    # post_filter = []
    # tracks_sorted = sorted(tracks, key=lambda track: track.start)
    # current_track = None
    # for t in tracks_sorted:
    #     if current_track is None:
    #         current_track = t
    #         post_filter.append(current_track)

    #         continue
    #     overlap = current_track.time_overlap(t)
    #     percent_overlap = overlap / t.length
    #     percent_overlap_2 = overlap/ current_track.length

    #     f_overlap = current_track.mel_freq_overlap(t)
    #     f_percent_overlap = f_overlap / t.mel_freq_range

    #     if percent_overlap_2 > 0.5:
    #         post_filter = post_filter[:len(post_filter)-1]
    #         post_filter.append(t)
    #         current_track = t
    #     elif percent_overlap > 0.5 or (percent_overlap > 0 and f_percent_overlap > 0.5):
    #         if f_percent_overlap> 0.5:
    #             current_track.end = max(current_track.end,t.end)
    #     else:
    #         # encountered another big track shall we just make this current track and allow some small overlap???
    #         current_track = t
    #         post_filter.append(current_track)

    #     # if  percent_overlap>0.5 or percent_overlap_2> 0.5:
    #     #     print("Current track ", current_track, " over laps ", t, " with ", overlap)
    #     #     print("Track 1 over lap is ",percent_overlap, " track 2 overlap is ",percent_overlap_2)
    #     # el
    #     if overlap <= 0:
    #         current_track = t
    #         post_filter.append(current_track)
    # return

    # tracks_to_audio(tracks, spectogram, frames)
    # plot_mel_signals(np.abs(spectogram), signals, noise)
    return
    # process(args.file)
    process(args.file)
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
        self.predictions = []

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


# def plot_mel(mel, filename):
#     plt.figure(figsize=(10, 10))

#     ax = plt.subplot(1, 1, 1)
#     img = librosa.display.specshow(
#         mel, x_axis="time", y_axis="mel", sr=48000, fmax=11000, ax=ax,hop_length=281
#     )
#     plt.show()
#     # plt.savefig(f"{filename}.png", format="png")
#     plt.clf()
#     plt.close()


# def segment_overlap(first, second):
#     return (
#         (first[1] - first[0])
#         + (second[1] - second[0])
#         - (max(first[1], second[1]) - min(first[0], second[0]))
#     )


# class Signal:
#     def __init__(self, start, end, freq_start, freq_end):
#         self.start = start
#         self.end = end
#         self.freq_start = freq_start
#         self.freq_end = freq_end

#         self.mel_freq_start = mel_freq(freq_start)
#         self.mel_freq_end = mel_freq(freq_end)
#         self.predictions = []

#     def to_features(self):
#         # should 1 second be the same diff as 1000 hz? so scale a 1000 to 1
#         return np.float32([self.start,self.end,self.mel_freq_start/500,self.mel_freq_end/500])

#     def time_overlap(self, other):
#         return segment_overlap(
#             (self.start, self.end),
#             (other.start, other.end),
#         )

#     def mel_freq_overlap(self, other):
#         return segment_overlap(
#             (self.mel_freq_start, self.mel_freq_end),
#             (other.mel_freq_start, other.mel_freq_end),
#         )

#     def freq_overlap(s, s2):
#         return segment_overlap(
#             (self.mel_freq_start, self.mel_freq_end),
#             (other.mel_freq_start, other.mel_freq_end),
#         )

#     def to_array(self):
#         return [self.start, self.end, self.freq_start, self.freq_end]

#     @property
#     def mel_freq_range(self):
#         return self.mel_freq_end - self.mel_freq_start

#     @property
#     def freq_range(self):
#         return self.freq_end - self.freq_start

#     @property
#     def length(self):
#         return self.end - self.start

#     def enlarge(self, scale, min_track_length=0.7):
#         new_length = self.length * scale
#         if new_length < min_track_length:
#             new_length = min_track_length

#         extension = (new_length - self.length) / 2
#         self.start = self.start - extension
#         self.end = self.end + extension
#         self.start = max(self.start, 0)

#         new_length = (self.freq_end - self.freq_start) * scale
#         extension = (new_length - (self.freq_end - self.freq_start)) / 2
#         self.freq_start = self.freq_start - extension
#         self.freq_end = self.freq_end + extension
#         self.freq_start = max(self.freq_start, 0)
#         self.mel_freq_start = mel_freq(self.freq_start)
#         self.mel_freq_end = mel_freq(self.freq_end)

#     def merge(self, other):
#         self.start = min(self.start, other.start)
#         self.end = max(self.end, other.end)
#         self.freq_start = min(self.freq_start, other.freq_start)
#         self.freq_end = max(self.freq_end, other.freq_end)
#         self.mel_freq_start = mel_freq(self.freq_start)
#         self.mel_freq_end = mel_freq(self.freq_end)

#     def __str__(self):
#         return (
#             f"Signal: {self.start}-{self.end}  mel: {self.mel_freq_start} {self.mel_freq_end} hz:{self.freq_start} {self.freq_end}"
#         )


from scipy.signal import butter, sosfilt, sosfreqz, freqs


# NOT USING AN OF THIS
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    btype = "lowpass"
    freqs = []
    if lowcut > 0:
        btype = "bandpass"
        low = lowcut / nyq
        freqs.append(low)
    if highcut > 0:
        high = highcut / nyq
        if high < 1:
            freqs.append(high)
        else:
            btype = "highpass"
    else:
        btype = "highpass"
    if len(freqs) == 0:
        return None
    sos = butter(order, freqs, analog=False, btype=btype, output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs=48000, order=2):
    if lowcut <= 0 and highcut <= 0:
        return data
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    if sos is None:
        return data
    filtered = sosfilt(sos, data)
    return np.float32(filtered)


# def butter_bandpass(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     btype = "lowpass"
#     freqs = []
#     if lowcut > 0:
#         btype = "bandpass"
#         low = lowcut / nyq
#         freqs.append(low)
#     high = highcut / nyq
#     freqs.append(high)
#     print("Freqs are ",freqs, " sr is",fs)
#     sos = butter(order, freqs, analog=False, btype=btype, output="sos")
#     return sos


# def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
#     sos = butter_bandpass(lowcut, highcut, fs, order=order)
#     filtered = sosfilt(sos, data)
#     return filtered


if __name__ == "__main__":
    main()
