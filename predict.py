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

from audiodataset import AudioDataset
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
from denoisetest import signal_noise, space_signals

matplotlib.use("TkAgg")

PROB_THRESH = 0.8
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
            if len(data) < sr * seg_length:
                data_2 = np.zeros((int(sr * seg_length)))
                data_2[: len(data)] = data
                data = data_2

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

            mel = librosa.power_to_db(mel, ref=np.max)
            mel = tf.expand_dims(mel, axis=2)
            start += stride
            end = start + seg_length

            count += 1
            mels.append(mel)
    return mels, len(frames) / sr


def preprocess_file(file, seg_length, stride, hop_length, mean_sub, use_mfcc):
    frames, sr = load_recording(file)
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
        spectogram = np.abs(librosa.stft(s_data, n_fft=n_fft, hop_length=hop_length))
        mel = mel_spec(spectogram, sr, n_fft, hop_length, 120, 50, 11000)
        # mel = librosa.feature.melspectrogram(
        #     y=s_data,
        #     sr=sr,
        #     n_fft=n_fft,
        #     hop_length=hop_length,
        #     fmin=50,
        #     fmax=11000,
        #     n_mels=120,
        # )
        half = mel[:, 75:]
        if np.amax(half) == np.amin(half):
            print("mel max is same")
            strides_per = math.ceil(seg_length / 2.0 / stride) + 1
            mels = mels[:-strides_per]
            print("remove last ", strides_per, len(mels))
            return mels, length
            # 1 / 0

        mel = librosa.power_to_db(mel, ref=np.max)
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
        if mean_sub:
            mel_m = tf.reduce_mean(mel, axis=1)

            mel_m = tf.expand_dims(mel_m, axis=1)
            mel = mel - mel__m
        # mean over each mel bank
        # print("mean of mel is", round(1000 * np.mean(mel), 4))
        mel = tf.repeat(mel, 3, axis=2)
        mels.append(mel)
        i += 1
        # break
    return mels, length


def plot_mfcc(mfcc):
    plt.figure(figsize=(10, 10))

    img = librosa.display.specshow(mfcc, x_axis="time", ax=ax)
    ax = plt.subplot(1, 1, 1)
    plt.show()
    # plt.savefig(f"mel-power-{i}.png", format="png")
    plt.clf()
    plt.close()


def plot_mel(mel, i=0):
    fig = plt.figure(figsize=(10, 10))

    ax = plt.subplot(1, 1, 1)
    img = librosa.display.specshow(
        mel, x_axis="time", y_axis="mel", sr=48000, fmax=11000, ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    # plt.show()
    plt.savefig(f"mel-power-{i}.png", format="png")
    plt.clf()
    plt.close()


def main():
    init_logging()
    args = parse_args()
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
    # model = tf.keras.models.load_model(str(load_model))

    model.load_weights(load_model / "val_loss").expect_partial()
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
    hop_length = 281
    # print("stride is", segment_stride)
    # segment_length = 2
    segment_stride = 0.5
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
        data, length = preprocess_file(
            file, segment_length, segment_stride, hop_length, mean_sub, use_mfcc
        )
        data = np.array(data)

    print("data is", data.shape, data.dtype, np.amax(data))
    predictions = model.predict(np.array(data))
    tracks = []
    start = 0
    active_tracks = {}
    for prediction in predictions:
        print("at", start, np.round(prediction * 100))
        # break
        if start + segment_length > length:
            print("final one")
            start = length - segment_length
        results = []
        track_labels = []
        if multi_label:
            # print("doing multi", prediction * 100)
            for i, p in enumerate(prediction):
                if p >= PROB_THRESH:
                    label = labels[i]
                    results.append((p, label))
                    track_labels.append(label)
        else:
            best_i = np.argmax(prediction)
            best_p = prediction[best_i]
            if best_p >= PROB_THRESH:
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
                if specific_bird:
                    track.end = start
                else:
                    track.end = track.end - segment_length / 2
                del active_tracks[track.label]
                # print("removed", track.label)

        for r in results:
            label = r[1]

            if specific_bird and label == "bird":
                continue
            track = active_tracks.get(label, None)
            if track is None:
                track = Track(label, start, start + segment_length, r[0])
                tracks.append(track)
                active_tracks[label] = track
            else:
                track.end = start + segment_length
                track.confidences.append(r[0])
            # else:

        # elif track is not None:
        #     track.end = start + (segment_length / 2 - segment_stride)
        #     tracks.append((track))
        #     track = None

        start += segment_stride
    for t in tracks:
        print(f"{t.start}-{t.end} have {t.label}")


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


if __name__ == "__main__":
    main()
