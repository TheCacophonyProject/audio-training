# Build a segment dataset for training.
# Segment headers will be extracted from a track database and balanced
# according to class. Some filtering occurs at this stage as well, for example
# tracks with low confidence are excluded.

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
import tensorflow_addons as tfa

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
from custommels import mel_spec
from denoisetest import signal_noise, space_signals, signal_noise_data
from plot_utils import plot_mel, plot_mel_signals
import matplotlib.patches as patches
import tensorflow_hub as hub
import csv

matplotlib.use("TkAgg")

CALL_LENGTH = 1


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


def preprocess_file_signals(file, seg_length, stride, hop_length, mean_sub, use_mfcc):
    signals, noise = signal_noise(file)
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
            mel = mel_spec(spectogram, sr, n_fft, hop_length, 120, 50, 11000)

            half = mel[:, 75:]
            if np.amax(half) == np.amin(half):
                print("mel max is same")
                strides_per = math.ceil(seg_length / 2.0 / stride) + 1
                mels = mels[:-strides_per]
                print("remove last ", strides_per, len(mels))
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


def preprocess_file(file, seg_length, stride, hop_length, mean_sub, use_mfcc):
    frames, sr = load_recording(file)
    # spits = split_sound(frames[: sr * 3])
    # for s in spits:
    #     print("Split", s / sr)
    length = len(frames) / sr
    end = 0
    sample_size = int(seg_length * sr)
    logging.info(
        "sr %s seg %s sample size %s stride %s hop%s mean sub %s mfcc %s",
        sr,
        seg_length,
        sample_size,
        stride,
        hop_length,
        mean_sub,
        use_mfcc,
    )
    logging.info("sample is %s", length)
    mels = []
    i = 0
    n_fft = sr // 10
    sr_stride = int(stride * sr)
    while end < (length + stride):
        start_offset = i * sr_stride

        end = i * stride + seg_length

        if end > length:
            s_data = frames[-sample_size:]
        else:
            s_data = frames[start_offset : start_offset + sample_size]
        if len(s_data) < seg_length * sr:
            print("data is", len(s_data) / sr)
            s_data = np.pad(s_data, (0, int(1.5 * sr)))
            print("data is now", len(s_data) / sr)

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
        mel = mel_spec(spectogram, sr, n_fft, hop_length, 120, 50, 11000, power=1)

        half = mel[:, 75:]
        if np.amax(half) == np.amin(half):
            print("mel max is same")
            strides_per = math.ceil(seg_length / 2.0 / stride) + 1
            mels = mels[:-strides_per]
            print("remove last ", strides_per, len(mels))
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
        # plot_mel(lib_mel, i)

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


def main():
    init_logging()
    args = parse_args()
    # get_speech_score(args.file)
    # show_signals(args.file)
    # return
    # db_check(args.file)
    # return
    load_model = Path(args.model)
    # test(args.file)
    # return
    logging.info("Loading %s with weights %s", load_model, "val_acc")
    hamming = tfa.metrics.HammingLoss(mode="multilabel", threshold=0.8)
    prec_at_k = tf.keras.metrics.TopKCategoricalAccuracy()
    model = tf.keras.models.load_model(
        str(load_model),
        # custom_objects={
        #     "hamming_loss": hamming,
        #     "top_k_categorical_accuracy": prec_at_k,
        # },
        compile=False,
    )
    model.summary()
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

    # model.load_weights(load_model / "val_binary_accuracy").expect_partial()
    # model.save(load_model / "frozen_model")
    # 1 / 0
    with open(load_model / "metadata.txt", "r") as f:
        meta = json.load(f)
    labels = meta.get("labels", [])
    multi_label = meta.get("multi_label", True)
    segment_length = meta.get("segment_length", 1.5)
    segment_stride = meta.get("segment_stride", 2)
    use_mfcc = meta.get("use_mfcc", True)
    mean_sub = meta.get("mean_sub", False)
    use_mfcc = meta.get("use_mfcc", False)
    hop_length = meta.get("hop_length", 640)
    prob_thresh = meta.get("threshold", 0.7)
    model_name = meta.get("name", False)

    hop_length = 281

    segment_stride = 3
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
            data, length = preprocess_file(
                file, segment_length, segment_stride, hop_length, mean_sub, use_mfcc
            )
        data = np.array(data)

    print("data is", data.shape, data.dtype, np.amax(data))
    predictions = model.predict(np.array(data))
    # other_pred = mid_model(np.array(data), training=False)
    # print(other_pred.shape)
    tracks = []
    start = 0
    active_tracks = {}
    for i, prediction in enumerate(predictions):
        # other = other_pred[i]
        print(
            np.sum(prediction),
            "at",
            start,
            "-",
            start + segment_length,
            np.round(prediction * 100),
        )
        # other_s = start
        # sub_labels = []
        # # think of the times interms of stft
        # # win length / sr
        #
        # downscale = 513 / 45.0
        # win_s = downscale * 4800 / 48000
        # hop_s = downscale * 281 / 48000
        # print("over lap is ", win_s - hop_s)
        # for other_i, other in enumerate(other[0]):
        #     new_labels = []
        #     for l_i, p in enumerate(other):
        #         if p >= prob_thresh:
        #             label = labels[l_i]
        #             new_labels.append(label)
        #     for sub in sub_labels:
        #         if sub not in new_labels:
        #             print(
        #                 sub,
        #                 " end ",
        #                 round(other_s, 1),
        #             )
        #     if len(new_labels) > 0:
        #         for l in new_labels:
        #             if l not in sub_labels:
        #                 print(
        #                     l,
        #                     " start at ",
        #                     round(other_s, 1),
        #                 )
        #     print(
        #         other_i,
        #         "at",
        #         round(other_s, 1),
        #         "-",
        #         round(other_s + win_s, 1),
        #         np.round(other * 100),
        #         new_labels,
        #     )
        #     sub_labels = new_labels
        #     other_s += hop_s

        if start + segment_length > length:
            print("final one")
            start = length - segment_length

        results = []
        track_labels = []
        if multi_label:
            # print("doing multi", prediction * 100)
            for l_i, p in enumerate(prediction):
                if p >= prob_thresh:
                    label = labels[l_i]
                    results.append((p, label))
                    track_labels.append(label)
        else:
            best_i = np.argmax(prediction)
            best_p = prediction[best_i]
            if best_p >= prob_thresh:
                label = labels[best_i]
                results.append((best_p, label))
                track_labels.append[label]

        specific_bird = any(
            [l for l in track_labels if l not in ["human", "noise", "bird"]]
        )
        # remove tracks that have ended
        existing_tracks = list(active_tracks.keys())
        # print("Current", track_labels, "active", existing_tracks)
        for existing in existing_tracks:
            track = active_tracks[existing]
            if track.label not in track_labels or (
                track.label == "bird" and specific_bird
            ):
                # if specific_bird:
                # pass
                # track.end = start
                # track.end = start + CALL_LENGTH
                track.end = min(start + segment_length - segment_stride, track.end)
                if track.end > length:
                    track.end = length

                # else:
                # track.end = track.end - segment_length / 2
                # dont do this straight away as if we have a small stride.
                # we could have more bird calls at later times but within track
                if start >= track.end:
                    del active_tracks[track.label]
                # print("removed", track.label)

        for r in results:
            label = r[1]

            if specific_bird and label == "bird":
                continue
            track = active_tracks.get(label, None)
            if track is None:
                t_s = start
                t_e = start + segment_length
                if start > 0:
                    t_s = start - segment_stride + segment_length - CALL_LENGTH

                if (i + 1) < len(predictions):
                    t_e = start + segment_stride + CALL_LENGTH
                t_e = max(t_s, t_e)
                # track = Track(label, t_s, t_e, r[0])
                track = Track(label, start, start + segment_length, r[0])
                track.end = min(track.end, length)
                print("Creating track", label, start, track.end)
                tracks.append(track)
                active_tracks[label] = track
            else:
                # track.end = start + segment_stride + CALL_LENGTH
                track.end = start + segment_length - segment_stride

                if track.end > length:
                    track.end = length
                print("Ending track", track.start, track.end, track.label)
                track.confidences.append(r[0])
            # else:

        # elif track is not None:
        #     track.end = start + (segment_length / 2 - segment_stride)
        #     tracks.append((track))
        #     track = None

        start += segment_stride
    for t in tracks:
        # new_start = t.start
        # new_end = t.end
        # if t.start > 0:
        #     # since previous prediction wasnt this
        #     # we could say that t.start - segment_stride + segment_length doesn't contain this species
        #     # or is mid call so for safety can add a 0.5 to it
        #     #  i.e. start is 5 - 7.5
        #     # 4.75 - 7.25 has no bird
        #     # so probably start should be  #7.25 - 0.5 for safety
        #     new_start = t.start - segment_stride + segment_length - 1
        #     # same for end
        #     # at t.end + 1 - segment _length no bird, which probably means
        #     #  i.e end is 5 (predicted on 2.5 - 5) has bird
        #     #  2.75 - 5.25 no bird
        #     #  so probably should end at 2.5 + 0.5 for safety
        # if t.end < length:
        #     new_end = t.end - segment_length + segment_stride + 1

        print(f"{t.start}-{t.end} have {t.label}")

    signals, noise = signal_noise(file)
    signals = space_signals(signals, 0.2)
    print("Have ", len(signals), " possible signals")
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
    for s in signals:
        if track_index >= len(sorted_tracks):
            break
        while track_index < len(sorted_tracks):
            t = sorted_tracks[track_index]
            start = t.start
            end = t.end
            if start < last_end:
                start = last_end
                end = max(start, end)
            # overlap
            if ((end - start) + (s[1] - s[0])) > max(end, s[1]) - min(start, s[0]):
                # print("Have track", t, " for ", s, t.start, t.end, t.label)
                if t.label in ["bird", "kiwi", "whistler", "morepork"]:
                    chirps += 1
                if end > s[1]:
                    # check next signal
                    break
            elif start > s[1]:
                break
            last_end = end
            track_index += 1
    gap = 0.20000001
    max_chirps = length / gap
    max_chirps = math.ceil(max_chirps)

    print("Have ", chirps, " chirps/", max_chirps)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confusion", help="Save confusion matrix for model")
    parser.add_argument("--file", help="Audio file to predict")
    parser.add_argument("--dataset", help="Dataset to predict")

    parser.add_argument("model", help="Run name")

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


if __name__ == "__main__":
    main()
