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


def load_recording(file, resample=48000):
    frames, sr = librosa.load(str(file), sr=None)
    if resample is not None and resample != sr:
        frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
        sr = resample
    return frames, sr


def preprocess_file(file):
    stride = 1
    seg_length = 3
    frames, sr = load_recording(file)
    length = len(frames) / sr
    end = 0
    sample_size = seg_length * sr
    mels = []
    i = 0
    n_fft = sr // 10
    sr_stride = stride * sr
    while end < length:
        start_offset = i * sr_stride

        end = i * stride + seg_length
        i += 1
        if end > length:
            print("start off set is", start_offset, " length", length, end)
            sub = frames[start_offset:]
            s_data = np.zeros((seg_length * sr))
            print("sub is", len(sub), len(s_data))
            s_data[0 : len(sub)] = sub
        else:
            s_data = frames[start_offset : start_offset + sample_size]
        print("loading from ", start_offset / sr)
        mel = librosa.feature.melspectrogram(
            y=s_data, sr=sr, n_fft=n_fft, hop_length=n_fft // 2
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(y=s_data, sr=sr, hop_length=n_fft // 2, htk=True)
        mfcc = mfcc - np.amax(mfcc)
        mfcc /= 2

        # plot_mel(mel,i)
        # mel = mel[:, :, np.newaxis]
        # mel_mf = tf.concat((mel, mfcc), axis=0)
        # mel_mf = mel
        # mel_mf = tf.reshape(mel_mf, [*mel_mf.shape, 1])

        # image = tf.concat((mel_mf, mel_mf, mel_mf), axis=2)
        mel = mel[:, :, np.newaxis]
        mel = np.repeat(mel, 3, axis=2)
        # mel = tf.image.resize(mel, (128 * 2, 61 * 2))
        image = tf.image.resize(mel, (128 * 2, 61 * 2))

        mels.append(image.numpy())
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

    model.load_weights(load_model / "val_accuracy").expect_partial()

    labels = ["bird", "human"]
    model_name = "inceptionv3"

    start = 0
    if args.dataset:
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
    results = model.predict(np.array(data))

    for r in results:
        print(f"{start} - {start+3}  class  {np.round(r, 1)}")
        start += 1
    # species = results[0]
    return
    animal = results[1]
    for s, r in zip(species, animal):
        print(f"{start} - {start+3} Species", np.round(s, 1), " class ", np.round(r, 1))
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
