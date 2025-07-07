# Build a segment dataset for training.
# Segment headers will be extracted from a track database and balanced
# according to class. Some filtering occurs at this stage as well, for example
# tracks with low confidence are excluded.

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

# from config.config import Config
import numpy as np

from audiodataset import AudioDataset, space_signals
from audiowriter import create_tf_records
import tensorflow as tf
from tfdataset import get_dataset
import time
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import librosa
from audiomodel import get_preprocess_fn
from tfdataset import get_dataset

import soundfile as sf
import matplotlib
from custommel import mel_spec
from denoisetest import (
    space_signals,
    signal_noise_data,
    butter_bandpass_filter,
)
from identifytracks import get_tracks_from_signals, signal_noise, get_end, Signal

from plot_utils import plot_mel, plot_mel_signals, plot_spec
import matplotlib.patches as patches
import csv

# matplotlib.use("TkAgg")

CALL_LENGTH = 1


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


def preprocess_file_signals(file, seg_length, stride, hop_length, mean_sub, use_mfcc):
    signals, noise, _, _ = signal_noise(file)
    # signals = space_signals(signals)
    frames, sr = load_recording(file)
    mels = []
    n_fft = sr // 10

    for s in signals:
        print("doing singal", s)
        start = s[0]
        end = start + seg_length
        # end = s[0]
        count = 0
        while end < s[1] or count == 0:
            print("doing", start, end)
            start_sr = int(start * sr)
            end_sr = int(min(end, s[1]) * sr)

            data = frames[start_sr:end_sr]
            # if len(data) < sr * seg_length:
            #     data_2 = np.zeros((int(sr * seg_length)))
            #     data_2[: len(data)] = data
            #     data = data_2

            spectogram = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))
            mel = mel_spec(
                spectogram,
                sr,
                n_fft,
                hop_length,
                n_mels,
                50,
                11000,
                break_freq=break_freq,
            )

            half = mel[:, 75:]
            if np.amax(half) == np.amin(half):
                print("mel max is same")
                strides_per = math.ceil(seg_length / 2.0 / stride) + 1
                mels = mels[:-strides_per]
                print("remove last ", strides_per, len(mels), " end is")
                return mels, len(frames) / sr
                # 1 / 0

            mel = librosa.power_to_db(mel)

            mel = tf.expand_dims(mel, axis=2)
            start += stride
            end = start + seg_length

            count += 1
            mels.append(mel)
    return mels, len(frames) / sr


def denoise_spec(spectogram, sr):
    # And compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(spectogram)
    S_filter = librosa.decompose.nn_filter(
        S_full,
        aggregate=np.median,
        metric="cosine",
        width=int(librosa.time_to_frames(2, sr=sr)),
    )

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimium
    # with the input spectrum forces this.
    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(
        S_filter, margin_i * (S_full - S_filter), power=power
    )

    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
    return mask_v * spectogram
    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    print("mask", mask_v.shape, spectogram.shape, S_full.shape)
    stft_fore = spectogram * mask_v
    y_inv = librosa.griffinlim(np.abs(stft_fore))

    import soundfile as sf

    sf.write("foreground.wav", y_inv, sr)
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(
        librosa.amplitude_to_db(S_full[:], ref=np.max), y_axis="log", sr=sr
    )
    plt.title("Full spectrum")
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(
        librosa.amplitude_to_db(S_background[:], ref=np.max), y_axis="log", sr=sr
    )
    plt.title("Background")
    plt.colorbar()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(
        librosa.amplitude_to_db(S_foreground[:], ref=np.max),
        y_axis="log",
        x_axis="time",
        sr=sr,
    )
    plt.title("Foreground")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def show_signals(file):
    frames, sr = load_recording(file)

    s_data = frames[: sr * 3]
    print(s_data)
    n_fft = sr // 10
    hop_length = 281
    spectogram = np.abs(librosa.stft(s_data, n_fft=n_fft, hop_length=hop_length))
    denoised_stft = denoise_spec(spectogram, sr)
    # return
    signals, noise = signal_noise_data(spectogram, sr)
    signals2, noise = signal_noise_data(denoised_stft, sr)
    mel = mel_spec(denoised_stft, sr, n_fft, hop_length, 120, 50, 11000, power=1)
    # S = librosa.feature.melspectrogram(y=frames, sr=sr, power=1)

    plot_mel_signals(mel, signals, signals2)


# MIGHT BE WORTH TRYING
#
# import random
# def split_sound(clip):
#     """Returns the sound array, sample rate and
#     x_split = intervals where sound is louder than top db
#     """
#     db = librosa.core.amplitude_to_db(clip)
#     mean_db = np.abs(db).mean()
#     std_db = db.std()
#     x_split = librosa.effects.split(y=clip, top_db = mean_db - std_db)
#     return x_split
# def split_sound(clip):
#     """Returns the sound array, sample rate and
#     x_split = intervals where sound is louder than top db
#     """
#     db = librosa.core.amplitude_to_db(clip)
#     mean_db = np.abs(db).mean()
#     std_db = db.std()
#     x_split = librosa.effects.split(y=clip, top_db=mean_db + 2 * std_db)
#     return x_split


def load_samples(
    frames,
    sr,
    tracks,
    segment_length,
    stride,
    hop_length=281,
    mean_sub=False,
    use_mfcc=False,
    mel_break=1000,
    htk=True,
    n_mels=160,
    fmin=50,
    fmax=11000,
    channels=1,
    power=1,
    db_scale=False,
    normalize=True,
    pad_short_tracks=True,
):
    # pad_short_tracks = False
    filter_below = 1000
    logging.info(
        "Loading samples with length %s stride %s hop length %s and mean_sub %s mfcc %s break %s htk %s n mels %s fmin %s fmax %s",
        segment_length,
        stride,
        hop_length,
        mean_sub,
        use_mfcc,
        mel_break,
        htk,
        n_mels,
        fmin,
        fmax,
    )
    mels = []
    i = 0
    # n_fft = sr // 10
    n_fft = 4096
    # hop_length = 640  # feature frame rate of 75

    sample_size = int(sr * segment_length)
    jumps_per_stride = int(sr * stride)
    length = len(frames) / sr
    end = segment_length
    mel_samples = []
    for t in tracks:
        show_spec = False
        track_data = []
        start = 0
        end = start + segment_length

        sr_end = int(t.end * sr)
        sr_start = int(sr * t.start)

        if pad_short_tracks:
            end = min(end, t.length)
            track_frames = frames[sr_start:sr_end]
        else:
            missing = sample_size - (sr_end - sr_start)
            if missing > 0:
                offset = np.random.randint(0, missing)
                sr_start = sr_start - offset

                if sr_start <= 0:
                    sr_start = 0
                    sr_end = sr_start + sample_size
                    sr_end = min(sr_end, len(frames))
                else:
                    end_offset = sr_end + missing - offset
                    if end_offset > len(frames):
                        end_offset = len(frames)
                        sr_start = end_offset - sample_size
                        sr_start = max(sr_start, 0)
                    sr_end = end_offset
                assert sr_end - sr_start == sample_size
            # print("Track ",t , " becomes ", sr_start/sr, sr_end / sr)
            track_frames = frames[sr_start:sr_end]

        sr_start = 0
        sr_end = min(sr_end, sample_size)
        while True:
            data = track_frames[sr_start:sr_end]
            print("Data size for track ", t, data.shape)
            if len(data) != sample_size:
                extra_frames = sample_size - len(data)
                offset = np.random.randint(0, extra_frames)
                data = np.pad(data, (offset, extra_frames - offset))
            if filter_below and t.freq_end < filter_below:
                logging.info(
                    "Filter freq below %s %s %s", filter_below, t.freq_start, t.freq_end
                )
                # data = butter_bandpass_filter(data, t.freq_start, t.freq_end, sr)
                # 1/0
            if show_spec:
                print("Showing spec for ", t)
            if normalize:
                data = normalize_data(data)
                print("NORMALIZED")
            spect = get_spect(
                data,
                sr,
                hop_length,
                mean_sub,
                use_mfcc,
                mel_break,
                htk,
                n_mels,
                fmin,
                fmax,
                n_fft,
                power,
                db_scale,
                channels,
                low_pass=t.freq_start,
                high_pass=t.freq_end,
                show_spec=show_spec,
            )
            # if t.start > 24 and t.start < 34:
            #     print(spect[:, :, 0].shape, "Spec for ", sr_start / sr, sr_end / sr)
            #     plot_spec(spect[:, :, 0])

            show_spec = False
            track_data.append(spect)
            start = start + stride
            end = start + segment_length
            sr_start = int(start * sr)
            sr_end = min(int(end * sr), sr_start + sample_size)
            # always take 1 sample
            if end > t.length:
                break
        mel_samples.append(track_data)
    return mel_samples


def get_spect(
    data,
    sr,
    hop_length,
    mean_sub,
    use_mfcc,
    mel_break,
    htk,
    n_mels,
    fmin,
    fmax,
    n_fft,
    power,
    db_scale,
    channels=1,
    low_pass=None,
    high_pass=None,
    show_spec=False,
):
    data = data.copy()
    if not htk:
        mel = librosa.feature.melspectrogram(
            y=data,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            n_mels=n_mels,
        )
        print("HTK")
    else:
        butter = False
        if butter:
            print("applying butter")
            data = butter_bandpass_filter(data, low_pass, high_pass, sr)

        # data = bandpassed + noise

        # bandpassed = data
        spectogram = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))

        # bins = 1 + n_fft / 2
        # max_f = sr / 2
        # gap = max_f / bins
        # if low_pass is not None:
        #     min_bin = low_pass // gap
        #     spectogram[: int(min_bin)] = 0
        #
        # if high_pass is not None:
        #     max_bin = high_pass // gap
        #     spectogram[int(max_bin) :] = 0
        # print("F min is ",fmin)
        # fmin =200
        print("Power ?,", power, fmin, fmax)
        mel = mel_spec(
            spectogram,
            sr,
            n_fft,
            hop_length,
            n_mels,
            fmin,
            fmax,
            mel_break,
            power=power,
        )
    if show_spec:
        plot_mel(mel, fmin=fmin, fmax=fmax)
    if db_scale:
        mel = librosa.power_to_db(mel, ref=np.max)
    mel = tf.expand_dims(mel, axis=2)

    if use_mfcc:
        mfcc = librosa.feature.mfcc(
            y=data,
            sr=sr,
            hop_length=hop_length,
            htk=True,
            fmin=50,
            fmax=11000,
            n_mels=80,
        )
        mfcc = tf.image.resize_with_pad(mfcc, *mel.shape)
        mel = tf.concat((mel, mfcc), axis=0)
    # end = start + sample_size
    if mean_sub:
        mel_m = tf.reduce_mean(mel, axis=1)
        mel_m = tf.expand_dims(mel_m, axis=1)
        mel = mel - mel_m
    if channels > 1:
        mel = tf.repeat(mel, channels, axis=2)
    return mel


def preprocess_file(
    tracks, file, seg_length, stride, hop_length, mean_sub, use_mfcc, break_freq, n_mels
):
    frames, sr = load_recording(file)
    # spits = split_sound(frames[: sr * 3])
    # for s in spits:
    #     print("Split", s / sr)
    stride = 1

    length = len(frames) / sr
    end = 0
    sample_size = int(seg_length * sr)
    logging.info(
        "sr %s seg %s sample size %s stride %s hop%s mean sub %s mfcc %s  break %s mels %s",
        sr,
        seg_length,
        sample_size,
        stride,
        hop_length,
        mean_sub,
        use_mfcc,
        break_freq,
        n_mels,
    )
    # logging.info("sample is %s", length)
    mels = []
    i = 0
    n_fft = sr // 10
    sr_stride = int(stride * sr)
    start = 0
    while end < (length + stride):
        start_offset = i * sr_stride

        end = i * stride + seg_length

        if end > length:
            s_data = frames[-sample_size:]
        else:
            s_data = frames[start_offset : start_offset + sample_size]
        if len(s_data) < seg_length * sr:
            # print("data is", len(s_data) / sr)
            s_data = np.pad(s_data, (0, int(1.5 * sr)))
            # print("data is now", len(s_data) / sr)

        spectogram = np.abs(librosa.stft(s_data, n_fft=n_fft, hop_length=hop_length))

        # print(spectogram.shape)
        # spectogram[:100, :] = 0
        # spectogram = np.clip(spectogram, 0, np.mean(spectogram))

        # print(spectogram.shape)
        # a_max = np.amax(spectogram[100:, :])
        # print("max above is", a_max, " below", np.amax(spectogram[:100, :]))
        # print("clipping to ", a_max)
        # spectogram[:100, :] *= 0.5

        # spectogram[:100, :]
        mel = mel_spec(
            spectogram,
            sr,
            n_fft,
            hop_length,
            n_mels,
            50,
            11000,
            power=1,
            break_freq=break_freq,
        )
        third = int(mel.shape[1] * 1 / 3)
        half = mel[:, third:]
        if np.amax(half) == np.amin(half):
            print("mel max is same")
            strides_per = math.ceil(seg_length / 3.0 / stride) + 1
            mels = mels[:-strides_per]

            return mels, length
            # 1 / 0
        # mel = librosa.power_to_db(mel, ref=np.max)
        # break
        # if i == 10:
        #     break
        # mel2 = np.power(mel, 0.1)
        # plot_mel(mel2, i)
        # pcen_S = librosa.pcen(mel * (2**31))
        # plot_mel(mel, 0)
        # plot_mel(pcen_S, 0)
        # print("Preprocess mel")
        # plot_mel(mel, i)
        # lib_mel = librosa.feature.melspectrogram(
        #     y=s_data,
        #     sr=sr,
        #     n_fft=n_fft,
        #     hop_length=hop_length,
        #     n_mels=120,
        #     fmin=50,
        #     fmax=11000,
        #     power=1,
        # )
        # print("saveing", start)
        # plot_mel(mel, f"more-{start}")

        mel = tf.expand_dims(mel, axis=2)

        if use_mfcc:
            mfcc = librosa.feature.mfcc(
                y=s_data,
                sr=sr,
                hop_length=hop_length,
                htk=True,
                fmin=50,
                fmax=11000,
                n_mels=80,
            )
            mfcc = tf.expand_dims(mfcc, axis=2)
            plot_mfcc(mfcc)
            mfcc = tf.image.resize_with_pad(mfcc, mel.shape[0], mel.shape[1])
            plot_mfcc(mfcc[:, :, 0])
            mel = tf.concat((mel, mfcc), axis=0)
        mean_sub = False
        if mean_sub:
            mel_m = tf.reduce_mean(mel, axis=0)
            mel_m = tf.expand_dims(mel_m, axis=0)
            plot_mel(mel.numpy()[:, :, 0])

            mel_time = mel - mel_m
            a_max = tf.math.reduce_max(mel_time)
            a_min = tf.math.reduce_min(mel_time)
            m_range = a_max - a_min
            mel_time = 80 * (mel_time - a_min) / m_range
            # mel_no_ref = 80 * mel_no_ref
            mel_time -= 80
            plot_mel(mel_time.numpy()[:, :, 0])
            mel_m = tf.reduce_mean(mel, axis=1)
            mel_m = tf.expand_dims(mel_m, axis=1)
            mel_time = mel - mel_m
            a_max = tf.math.reduce_max(mel_time)
            a_min = tf.math.reduce_min(mel_time)
            m_range = a_max - a_min
            mel_time = 80 * (mel_time - a_min) / m_range
            # mel_no_ref = 80 * mel_no_ref
            mel_time -= 80
            plot_mel(mel_time.numpy()[:, :, 0])
            1 / 0
        # mean over each mel bank
        # print("mean of mel is", round(1000 * np.mean(mel), 4))
        # mel = tf.repeat(mel, 3, axis=2)
        mels.append(mel)
        # break
        i += 1
        start += stride
        # if i == 3:
        #     break
    return mels, length


def get_chirp_samples(rec_data, sr=32000, stride=1, length=5):
    start = 0

    samples = []
    while True:
        sr_s = start * sr
        sr_e = (start + length) * sr
        sr_s = int(sr_s)
        sr_e = int(sr_e)
        s = rec_data[sr_s:sr_e]
        start += stride
        if len(s) < length * sr:
            s = np.pad(s, (0, int(length * sr - len(s))))
        samples.append(s)

        if sr_e >= len(rec_data):
            break
    return np.float32(samples)


def yamn_embeddings(file, stride=1):
    import tensorflow_hub as hub

    rec_data, sr = load_recording(file, resample=16000)
    samples = get_chirp_samples(rec_data, sr=sr, stride=stride, length=3)
    # Load the model.
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    # model = hub.load("https://tfhub.dev/google/bird-vocalization-classifier/1")

    embeddings = []
    for s in samples:
        logits, embedding, _ = model(s)
        embeddings.append(embedding)
    return np.array(embeddings), len(rec_data) / sr


def chirp_embeddings(file, stride=5):
    import tensorflow_hub as hub

    rec_data, sr = load_recording(file, resample=32000)
    samples = get_chirp_samples(rec_data, sr=sr, stride=stride)
    # Load the model.
    model = hub.load("https://tfhub.dev/google/bird-vocalization-classifier/1")

    embeddings = []
    for s in samples:
        logits, embedding = model.infer_tf(s[np.newaxis, :])
        print(embedding.shape)
        embeddings.append(embedding[0])
    return np.array(embeddings), len(rec_data) / sr


def normalize_data(x):
    min_v = np.min(x, -1, keepdims=True)
    x = x - min_v
    max_v = np.max(x, -1, keepdims=True)
    x = x / max_v + 0.000001
    x = x - 0.5
    x = x * 2
    return x


def add_ebird_mappings(labels):
    from audiodataset import RELABEL

    ebird_map = {}
    for l in labels:
        ebird_map[l] = []

    with open("eBird_taxonomy_v2024.csv") as f:
        for line in f:
            split_l = line.split(",")
            model_label = None
            # 1/0
            if split_l[4].lower() in RELABEL.keys():
                lbl_check = RELABEL[split_l[4].lower()]
                if lbl_check in labels:
                    model_label = lbl_check
            if model_label is None and split_l[4].lower() in labels:
                model_label = split_l[4].lower()
            # if split_l[9].lower() in labels:
            # model_label = split_l[9].lower()
            if "kiwi" in split_l[4].lower() and "kiwi" in labels:
                model_label = "kiwi"
            if model_label is not None:
                if split_l[2].lower() not in ebird_map[model_label]:
                    ebird_map[model_label].append(split_l[2].lower())

    with open("classes.csv", newline="") as csvfile:
        dreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        i = -1
        for row in dreader:
            i += 1
            if i == 0:
                continue
            # ebird = (common, extra)
            model_label = None
            if row[1].lower() in labels:
                model_label = row[1].lower()
            if row[4].lower() in labels:
                model_label = row[4].lower()
            if "kiwi" in row[1].lower() and "kiwi" in labels:
                model_label = "kiwi"
            if model_label is not None:
                if row[2].lower() not in ebird_map[model_label]:
                    ebird_map[model_label].append(row[2].lower())
    return ebird_map


def predict_on_folder(load_model, base_dir):
    total_files = 0
    total_correct = 0
    load_model = Path(load_model)
    logging.info("Loading %s with weights %s", load_model, "val_acc")
    model = tf.keras.models.load_model(
        str(load_model),
        compile=False,
    )
    model.load_weights(load_model.parent / "val_binary_accuracy.weights.h5")

    model.summary()
    with open(load_model.parent / "metadata.txt", "r") as f:
        meta = json.load(f)
    multi = meta.get("multi_label", True)
    labels = meta.get("labels", [])
    with open(load_model.parent / "metadata.txt", "w") as f:
        json.dump(meta, f, indent=4)

    multi_label = meta.get("multi_label", True)
    segment_length = meta.get("segment_length", 3)
    segment_stride = meta.get("segment_stride", 1)
    use_mfcc = meta.get("use_mfcc", True)
    mean_sub = meta.get("mean_sub", False)
    use_mfcc = meta.get("use_mfcc", False)
    hop_length = meta.get("hop_length", 281)
    prob_thresh = meta.get("threshold", 0.7)
    model_name = meta.get("name", False)
    break_freq = meta.get("break_freq", 1000)
    n_mels = meta.get("n_mels", 160)
    normalize = meta.get("normalize", True)
    power = meta.get("power", 2)
    hop_length = 281
    morepork_i = labels.index("morepork")
    meta_files = base_dir.glob("**/*.txt")
    for metadata_file in meta_files:
        file = metadata_file.with_suffix(".m4a")
        if not file.exists():
            file = metadata_file.with_suffix(".wav")
        if not file.exists():
            file = metadata_file.with_suffix(".mp3")
        if not file.exists():
            file = metadata_file.with_suffix(".flac")
        if not file.exists():
            logging.info("Not recording for %s", metadata_file)
            continue

        with metadata_file.open("r") as f:
            # add in some metadata stats
            meta = json.load(f)

        best_track = meta["best_track"]
        track = Signal(best_track["start"], best_track["end"], 0, 15000, 0)
        frames, sr = load_recording(file)
        end = len(frames) / sr
        track.end = min(end, track.end)
        tracks = [track]

        # to debug each 3 second segment
        # start = 0
        # while start < end-2:
        #     track = Signal(start,min(end,start + 3), 0, 15000, 0)
        #     tracks.append(track)
        #     start +=1
        all_data = load_samples(
            frames,
            sr,
            tracks,
            segment_length,
            segment_stride,
            hop_length,
            mean_sub,
            mel_break=break_freq,
            n_mels=n_mels,
            normalize=normalize,
            power=power,
        )
        for data, track in zip(all_data, tracks):
            # data = np.array(data)
            assert len(data) == 1
            if "efficientnet" in model_name.lower():
                logging.info("Repeating input")
                data = np.repeat(data, 3, -1)
            prediction = model.predict(np.array(data))[0]
            result = ModelResult(model_name)
            max_p = None
            # this is for multi label
            # logging.info("%s Pred at %s Seconds %s ",track,track.start,np.round(100*prediction))
            for i, p in enumerate(prediction):
                if max_p is None or p > max_p[1]:
                    max_p = (i, p)
                if p >= prob_thresh:
                    result.labels.append(labels[i])
                    result.confidences.append(round(p * 100))

            label = best_track["tags"][0]["what"]
            if label == "morepo2":
                label = "morepork"
            label_conf = round(prediction[labels.index(label)] * 100)

            if label not in result.labels:
                logging.info(
                    "%s %s has morepork %s predictions  %s",
                    metadata_file,
                    best_track["tags"][0]["what"],
                    label_conf,
                    result.preds_tostr(),
                )
            else:
                total_correct += 1
        total_files += 1

    logging.info(
        "COrrect %s out of %s ( %s )",
        total_correct,
        total_files,
        round(100 * total_correct / total_files),
    )
    # max_i = np.argmax(predictions)
    # max_conf = predictions[max_i]


def predict_on_test(split_file, load_model, base_dir, confusion_file="confusion.png"):
    confusion_file = Path(confusion_file)
    from build import split_by_file

    dataset = AudioDataset("all", None)
    dataset.load_meta(base_dir)
    with open(split_file, "r") as t:
        # add in some metadata stats
        split_meta = json.load(t)
    _, _, test = split_by_file(dataset, split_meta)

    total_files = 0
    total_correct = 0
    load_model = Path(load_model)
    logging.info("Loading %s with weights %s", load_model, "val_acc")
    model = tf.keras.models.load_model(
        str(load_model),
        compile=False,
    )
    # model.load_weights(load_model.parent / "val_binary_accuracy.weights.h5")

    model.summary()
    with open(load_model.parent / "metadata.txt", "r") as f:
        meta = json.load(f)
    multi = meta.get("multi_label", True)
    labels = meta.get("labels", [])
    with open(load_model.parent / "metadata.txt", "w") as f:
        json.dump(meta, f, indent=4)

    multi_label = meta.get("multi_label", True)
    segment_length = meta.get("segment_length", 3)
    segment_stride = meta.get("segment_stride", 1)
    use_mfcc = meta.get("use_mfcc", True)
    mean_sub = meta.get("mean_sub", False)
    use_mfcc = meta.get("use_mfcc", False)
    hop_length = meta.get("hop_length", 281)
    prob_thresh = meta.get("threshold", 0.7)
    model_name = meta.get("name", False)
    break_freq = meta.get("break_freq", 1000)
    n_mels = meta.get("n_mels", 160)
    normalize = meta.get("normalize", True)
    extra_label_map = meta.get("extra_label_map")

    remapped_labels = meta.get("remapped_labels", {})
    power = meta.get("power", 2)
    hop_length = 281
    n_fft = 4096
    fmin = 50
    fmax = 11000
    y_true = []
    predicted = []
    from audiodataset import load_data

    for rec in test.recs.values():
        found = False
        for l in rec.human_tags:
            if l in labels:
                found = True
                break
        if not found:
            continue
        frames, sr = load_recording(rec.filename)
        file_y = []
        file_data = []
        for sample in rec.samples:

            label = sample.tags[0]
            if label in remapped_labels:
                label_i = remapped_labels[label]
                if label_i == -1:
                    if label in extra_label_map:
                        label_i = extra_label_map[label]
                    if label_i == -1:
                        logging.info("Ignoring %s", label)
                        continue
                logging.info("%s becomes %s", label, labels[label_i])
            else:
                logging.info("%s not in remapped %s", rec.filename, label)
                continue
            file_y.append(label_i)
            spec = load_data(
                test.config, sample.start, frames, sr, end=sample.end, use_padding=False
            )
            data = mel_spec(
                spec.spectogram,
                sr,
                n_fft,
                hop_length,
                n_mels,
                fmin,
                fmax,
                break_freq,
                power=power,
            )
            data = data[:, :, np.newaxis]
            file_data.append(data)
        if len(file_data) == 0:
            continue
        file_data = np.array(file_data)
        print(file_data.shape, " power is ", power)

        if "efficientnet" in model_name.lower():
            logging.info("Repeating input")
            file_data = np.repeat(file_data, 3, -1)

        predictions = model.predict(np.array(file_data))
        pred = tf.argmax(predictions, axis=1)

        # this is for multi label
        print(file_y)
        print(pred)
        y_true.extend(file_y)
        predicted.extend(pred)

    y_true = np.array(y_true)
    predicted = np.array(predicted)
    cm = confusion_matrix(y_true, predicted, labels=np.arange(len(labels)))

    from audiomodel import plot_confusion_matrix

    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(confusion_file.with_suffix(".png"), format="png")


def main():
    init_logging()
    args = parse_args()
    # predict_on_test(args.split_file, args.model, args.dir, args.confusion)
    # return
    if args.dir:
        predict_on_folder(args.model, args.dir)
        return
    frames, sr = load_recording(args.file)
    end = get_end(frames, sr)
    frames = frames[: int(sr * end)]
    signals, _ = signal_noise(frames, sr)

    tracks = get_tracks_from_signals(signals, end)
    # tracks = [track for track in tracks if track.start > 18 and track.end < 21]
    # track.start = 28.0p6436538696289
    # track.end = 31.0
    # tracks = [track]
    # for s in tracks:
    #     print("SIgnals are ", s)
    # get_speech_score(args.file)
    # show_signals(args.file)
    # return
    # db_check(args.file)
    # return
    load_model = Path(args.model)
    # test(args.file)
    # return
    logging.info("Loading %s with weights %s", load_model, "val_acc")
    model = tf.keras.models.load_model(
        str(load_model),
        # custom_objects={
        #     "hamming_loss": hamming,
        #     "top_k_categorical_accuracy": prec_at_k,
        # },
        compile=False,
    )
    model.summary()
    with open(load_model.parent / "metadata.txt", "r") as f:
        meta = json.load(f)
    multi = meta.get("multi_label", True)
    # if  1== 0 and not multi:
    #     popped = model.layers.pop()
    #     logging.info("Replacing softmax with sigmoid")
    #     popped.activation = tf.keras.activations.sigmoid
    #     model = tf.keras.models.Model(model.input, outputs=popped.output)
    #     model.summary()

    # an idea to get more details predictions
    # model.trainable = False
    #
    # x = model.layers[-1](model.layers[-3].output, training=False)
    # mid_model = tf.keras.Model(inputs=model.input, outputs=x)
    # # for l in mid_model.layers:
    # #     l.trainable = False
    # mid_model.summary()

    # model_pop(model)
    # return
    # model = tf.keras.models.load_model(str(load_model))

    # model.load_weights(load_model.parent / "val_loss.weights.h5")
    # save_dir = Path("frozen_model")

    # model.save(save_dir / load_model.parent.name/ load_model.name)
    # 1 / 0

    labels = meta.get("labels", [])
    ebird_ids = meta.get("ebird_ids", [])
    ebirds = add_ebird_mappings(labels)
    list_mappings = []
    for l in labels:
        list_mappings.append(ebirds[l])
    for label, ids in zip(labels, ebird_ids):
        print(label, "Have ", ids, " now calced ", ebirds[label])

    meta["ebird_ids"] = list_mappings
    with open(load_model.parent / "metadata.txt", "w") as f:
        json.dump(meta, f, indent=4)
    assert len(ebirds) == len(labels)

    multi_label = meta.get("multi_label", True)
    segment_length = meta.get("segment_length", 3)
    segment_stride = meta.get("segment_stride", 1)
    use_mfcc = meta.get("use_mfcc", True)
    mean_sub = meta.get("mean_sub", False)
    use_mfcc = meta.get("use_mfcc", False)
    hop_length = meta.get("hop_length", 281)
    prob_thresh = meta.get("threshold", 0.7)
    model_name = meta.get("name", False)
    break_freq = meta.get("break_freq", 1000)
    n_mels = meta.get("n_mels", 160)
    normalize = meta.get("normalize", True)
    power = meta.get("power", 2)
    fmin = meta.get("fmin", 50)
    # fmin = 1000
    fmax = meta.get("fmax", 11000)
    pad_short_tracks = meta.get("pad_short_tracks", True)

    hop_length = 281
    # print("stride is", segment_stride)
    # segment_length = 2
    # segment_stride = 0.5
    # segment_stride = 0.1
    # multi_label = True
    # labels = ["bird", "human"]
    start = 0
    if args.dataset:
        data_path = Path(args.dataset)
        meta_f = data_path.parent / "training-meta.json"
        with open(meta_f, "r") as f:
            meta = json.load(f)
        labels = meta.get("labels", [])
        filenames = tf.io.gfile.glob(f"{args.dataset}/*.tfrecord")
        print("loading", filenames)
        dataset, _ = get_dataset(
            # dir,
            filenames,
            labels,
            [],
            batch_size=32,
            image_size=(128 * 2, 61 * 2),
            augment=False,
            resample=False,
            use_species=False,
            shuffle=False,
            reshuffle=False,
            deterministic=True,
            # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
        )
        y_pred = model.predict(dataset)
        predicted_categories = np.int64(tf.argmax(y_pred, axis=1))

        true_categories = tf.concat([y for x, y in dataset], axis=0)
        true_categories = np.int64(tf.argmax(true_categories, axis=1))
        correct_other = 0
        correct_bird = 0
        total_birds = 0
        total_others = 0
        total = len(true_categories)
        for y, pred in zip(true_categories, predicted_categories):
            lbl = labels[y]
            if lbl == "bird":
                total_birds += 1
                if pred == 0:
                    correct_bird += 1
            else:
                total_others += 1
                if pred != 0:
                    correct_other += 1
            # print("Predicted", pred, " for ", y)

        print("Bird accuracy ", round(100 * correct_bird / total_birds))
        print("Other accuracy ", round(100 * correct_other / total_others))
        return
    if args.file:
        file = Path(args.file)
        if model_name == "embeddings":
            data, length = chirp_embeddings(file, segment_stride)
        elif model_name == "yamn-embeddings":
            data, length = yamn_embeddings(file, segment_stride)
        else:
            frames, sr = load_recording(file)
            # frames = butter_bandpass_filter(frames, 0, 10000, sr, order=5)

            data = load_samples(
                frames,
                sr,
                tracks,
                segment_length,
                segment_stride,
                hop_length,
                mean_sub,
                mel_break=break_freq,
                n_mels=n_mels,
                normalize=normalize,
                power=power,
                fmin=fmin,
                fmax=fmax,
                pad_short_tracks=pad_short_tracks,
            )
        # data = np.array(data)

    start = 0

    for d, t in zip(data, tracks):
        pred_counts = [0] * len(labels)
        print("Predicting", t, " samples are ", len(d))
        # print(np.array(d).shape)
        d = np.repeat(d, 3, -1)
        predictions = model.predict(np.array(d))

        previous_pred = None
        print("Predictin", t)
        for start_i, p in enumerate(predictions):
            max_p = np.argmax(p)
            conf = p[max_p]
            print("At ", start_i, " this is now ", t.start + start_i * segment_stride)
            # print("Current is ",max_p, conf, " previous was ", previous_pred)
            if conf >= prob_thresh:
                if previous_pred == max_p or len(predictions) == 1:
                    print(
                        "Pred for ", t.start + start_i * segment_stride, labels[max_p]
                    )
                previous_pred = max_p
                pred_counts[max_p] += 1
            else:
                previous_pred = None
            # print(
            #     "Pred for start ", t.start + start_i * segment_stride, np.round(p * 100)
            # )

        #     for i, percent in enumerate(p):
        #         if percent >= prob_thresh:
        #             pred_counts[i] += 1
        #             # print(
        #             #     "For track",
        #             #     t.start + start_i * segment_stride,
        #             #     " have ",
        #             #     labels[i],
        #             #     " ",
        #             #     np.round(100 * percent),
        #             # )
        # pred_labels = []
        # print("COUNTS are", pred_counts, " more than ", len(predictions) // 2)

        prediction = np.mean(predictions, axis=0)
        print("Prediction is", np.round(100 * prediction))
        for i, count in enumerate(pred_counts):
            if count > 0:
                print(f"{labels[i]}: {count}")
        # print("count pred is ", pred_labels)
        result = ModelResult(model_name)
        t.predictions.append(result)
        max_p = None
        # for l_i, p_counts in enumerate(pred_counts):
        #     if p_counts >= max(1, len(predictions) // 2):
        #         pred_labels.append(labels[l_i])
        #         result.labels.append(labels[l_i])
        #         result.confidences.append(100)
        # continue
        for i, p in enumerate(prediction):
            if max_p is None or p > max_p[1]:
                max_p = (i, p)
            # print("probably of ", labels[i], round(100 * p))
            if p >= prob_thresh:
                result.labels.append(labels[i])
                result.confidences.append(round(p * 100))
        if len(result.labels) == 0:
            # use max prediction
            result.raw_tag = labels[max_p[0]]
            result.raw_confidence = round(max_p[1] * 100)
    for t in tracks:
        print(
            "Track",
            t,
            " has prediction",
            t.predictions[0].labels,
            t.predictions[0].confidences,
            t.predictions[0].raw_tag,
            t.predictions[0].raw_confidence,
        )
    return
    print("Have ", len(signal), " possible signals")

    chirps = 0
    sorted_tracks = [
        t for t in tracks if t.label in ["bird", "kiwi", "whistler", "morepork"]
    ]
    sorted_tracks = sorted(
        sorted_tracks,
        key=lambda track: track.start,
    )
    last_end = 0
    track_index = 0
    for t in sorted_tracks:
        start = t.start
        end = t.end
        if start < last_end:
            start = last_end
            end = max(start, end)
        for s in signals:
            if ((end - start) + (s[1] - s[0])) > max(end, s[1]) - min(start, s[0]):
                # print("Have track", t, " for ", s, t.start, t.end, t.label)
                chirps += 1
            elif s[0] > start:
                break
        last_end = t.end
    gap = 0.20000001
    max_chirps = length / gap
    max_chirps = math.ceil(max_chirps)

    print("Have ", chirps, " chirps/", max_chirps)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confusion", help="Save confusion matrix for model")
    parser.add_argument("--file", help="Audio file to predict")

    parser.add_argument("--dataset", help="Dataset to predict")
    parser.add_argument("-d", "--dir", help="Directory to predict")
    parser.add_argument(
        "--split-file",
        default=None,
        help="Split the dataset using clip ids specified in this file",
    )
    parser.add_argument("model", help="Run name")
    args = parser.parse_args()
    if args.dir:
        args.dir = Path(args.dir)
    if args.split_file:
        args.split_file = Path(args.split_file)
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


def get_speech_score(file):
    """Check whether the audio contains human speech."""
    speech_filter_width = 5
    sr = 16000
    class_names = class_names_from_csv(
        "/home/gp/cacophony/chirp/models/yamnnet/assets/yamnet_class_map.csv"
    )
    audio, sr = load_recording(file, resample=sr)
    yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
    yamnet_model = hub.load(yamnet_model_handle)
    # if self.speech_filter_threshold <= 0.0:
    #   return -1.0
    # resample audio to yamnet 16kHz target.

    scores, embeddings, log_mel_spectrogram = yamnet_model(audio)
    print(scores.shape)
    for s in scores.numpy():
        # print(s)
        print(np.round(s * 100))
        max_i = s.argmax()
        print("At ", class_names[max_i], np.amax(s) * 100)
    # print(
    #     class_names[scores.numpy().mean(axis=0).argmax()]
    # )  # # Apply a low-pass filter over the yamnet speech logits.


# # This ensures that transient false positives don't ruin our day.
# speech_logits = (
#     np.convolve(speech_logits, np.ones([speech_filter_width]), 'valid') / width
# )
# return speech_logits.max()


def class_names_from_csv(csv_file):
    """Returns list of class names corresponding to score vector."""
    # Open a file: file
    import io

    with open(csv_file, mode="r") as f:
        class_map_csv_text = f.read()
        # csv_r = csv.reader(f, delimiter=",", quotechar="|")
        # for class_index, mid, display_name in csv.reader(csv_r):
        #     print("have ", display_name, class_index)

    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [
        display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)
    ]
    class_names = class_names[1:]  # Skip CSV header
    return class_names


class ModelResult:
    def __init__(self, model):
        self.model = model
        self.labels = []
        self.confidences = []
        self.raw_tag = None
        self.raw_confidence = None

    def get_meta(self):
        meta = {}
        meta["model"] = self.model
        meta["species"] = self.labels
        meta["likelihood"] = self.confidences
        # used when no actual tag
        if self.raw_tag is not None:
            meta["raw_tag"] = self.raw_tag
            meta["raw_confidence"] = self.raw_confidence
        return meta

    def preds_tostr(self):
        output = ""
        for conf, label in zip(self.confidences, self.labels):
            output = f"{output} {label}:{conf}"
        return output


# @tf.keras.utils.register_keras_serializable(package="MyLayers", name="MagTransform")
# class MagTransform(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(MagTransform, self).__init__(**kwargs)
#         self.a = self.add_weight(
#             initializer=tf.keras.initializers.Constant(value=-1.0),
#             name="a-power",
#             dtype="float32",
#             shape=(),
#             trainable=True,
#             # constraint=tf.keras.constraints.MinMaxNorm(
#             #     min_value=-2.0, max_value=1.0, rate=1.0, axis=-1
#             # ),
#         )

#     def call(self, inputs):
#         c = tf.math.pow(inputs, tf.math.sigmoid(self.a))
#         return c


if __name__ == "__main__":
    main()
