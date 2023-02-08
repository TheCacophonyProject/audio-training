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
import pytz
import json
from dateutil.parser import parse as parse_date
import sys
import itertools

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

matplotlib.use("TkAgg")


def load_recording(file, resample=48000):
    frames, sr = librosa.load(str(file), sr=None)
    if resample is not None and resample != sr:
        frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
        sr = resample
    return frames, sr


seg_length = 1.5


def preprocess_file(file):
    stride = 1
    frames, sr = load_recording(file)
    length = len(frames) / sr
    end = 0
    sample_size = int(seg_length * sr)
    mels = []
    i = 0
    n_fft = sr // 10
    print(n_fft)
    sr_stride = stride * sr
    hop_length = 640  # feature frame rate of 75
    mel_all = librosa.feature.melspectrogram(
        y=frames,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=50,
        fmax=11000,
        n_mels=80,
    )
    mel_all = librosa.power_to_db(mel_all, ref=np.max)
    print("mel all is", mel_all.shape)
    mel_sample_size = int(1 + seg_length * sr / hop_length)
    jumps_per_stride = int(mel_sample_size / seg_length)
    print("jumper per stride", jumps_per_stride, sample_size)
    while end < length:
        start_offset = i * sr_stride

        end = i * stride + seg_length
        if end > length:
            print("start off set is", start_offset, " length", length, end)
            sub = frames[start_offset:]
            s_data = np.zeros(int(seg_length * sr))
            print("sub is", len(sub), len(s_data))
            s_data[0 : len(sub)] = sub
        else:
            s_data = frames[start_offset : start_offset + sample_size]
        # print("loading from ", start_offset / sr)
        # sf.write(f"test{i}.wav", s_data, sr, subtype="PCM_24")
        start = int(jumps_per_stride * (i * stride))
        mel = mel_all[:, start : start + mel_sample_size].copy()
        i += 1
        #
        # mel = librosa.feature.melspectrogram(
        #     y=s_data,
        #     sr=sr,
        #     n_fft=n_fft,
        #     hop_length=hop_length,
        #     fmin=50,
        #     fmax=11000,
        #     n_mels=80,
        # )
        # mel = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(y=s_data, sr=sr, hop_length=n_fft // 2, htk=True)
        mfcc = mfcc - np.amax(mfcc)
        mfcc /= 2

        # plot_mel(mel, i)
        # mel = mel[:, :, np.newaxis]
        # mel_mf = tf.concat((mel, mfcc), axis=0)
        # mel_mf = mel
        # mel_mf = tf.reshape(mel_mf, [*mel_mf.shape, 1])

        # image = tf.concat((mel_mf, mel_mf, mel_mf), axis=2)
        print(mel.shape)
        mel_m = tf.reduce_mean(mel, axis=1)
        # gp not sure to mean over axis 0 or 1
        mel_m = tf.expand_dims(mel_m, axis=1)
        # mean over each mel bank
        empty = np.zeros(((80, 113)))
        # print("setting mel at 0 -", mel.shape[1])
        empty[:, : mel.shape[1]] = mel
        mel = empty
        # print(mel_m.shape)
        mel = mel - mel_m
        mel = mel[:, :, np.newaxis]
        # mel = np.repeat(mel, 3, axis=2)
        # mel = tf.image.resize(mel, (128 * 2, 61 * 2))
        # image = tf.image.resize(mel, (128 * 2, 61 * 2))

        mels.append(mel)
    return mels


def plot_mel(mel, i=0):

    plt.figure(figsize=(10, 10))

    ax = plt.subplot(1, 1, 1)
    img = librosa.display.specshow(
        mel, x_axis="time", y_axis="mel", sr=48000, fmax=8000, ax=ax
    )
    plt.savefig(f"mel-power-{i}.png", format="png")
    # plt.clf()


def main():
    init_logging()
    args = parse_args()
    load_model = Path(args.model)
    logging.info("Loading %s with weights %s", load_model, "val_acc")
    model = tf.keras.models.load_model(str(load_model))

    model.load_weights(load_model / "val_auc").expect_partial()
    with open(load_model / "metadata.txt", "r") as f:
        meta = json.load(f)
    labels = meta.get("labels", [])

    # labels = ["bird", "human"]
    model_name = "inceptionv3"
    model.summary()
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
        data = preprocess_file(file)
        data = np.array(data)
        print(data.shape)
    results = model.predict(np.array(data))
    track = None
    for r in results:
        # best_i = np.argmax(r)
        # best_p = r[best_i]
        results = []
        for i, p in enumerate(r):
            if p > 0.5:
                results.append((round(100 * p), labels[i]))

        print(f"{results} {start} - {start+seg_length} ")
        start += 1
        continue
        if best_p > 0.7:
            if track is None:
                track = (best_i, start)
            elif track[0] != best_i:
                print(f"Changed pred {labels[track[0]]} {track[1]} - {start}")
                track = (best_i, start)

        elif track is not None:
            print(
                f"Low prob {best_p} - {labels[track[0]]} {track[1]} - {start + seg_length/2.0 - 1}"
            )
            track = None
        start += 1
    # species = results[0]
    if track is not None:
        print(f"Final {best_p} - {labels[track[0]]} {track[1]} - {start}")
    return
    animal = results[1]
    for s, r in zip(species, animal):

        print(
            f"{start} - {start+seg_length} Species",
            np.round(s, 1),
            " class ",
            np.round(r, 1),
        )
        start += 1


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


if __name__ == "__main__":
    main()
