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

matplotlib.use("TkAgg")


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


seg_length = 2


def preprocess_file(file):
    stride = 0.5
    frames, sr = load_recording(file)
    length = len(frames) / sr
    end = 0
    sample_size = int(seg_length * sr)
    print("sr", sr, seg_length, sample_size)
    print("Length of recording is", length)
    mels = []
    i = 0
    n_fft = sr // 10
    print(n_fft)
    sr_stride = int(stride * sr)
    hop_length = 640  # feature frame rate of 75
    # mel_all = librosa.feature.melspectrogram(
    #     y=frames,
    #     sr=sr,
    #     n_fft=n_fft,
    #     hop_length=hop_length,
    #     fmin=50,
    #     fmax=11000,
    #     n_mels=80,
    # )
    # mel_all = librosa.power_to_db(mel_all, ref=np.max)
    # mel_sample_size = int(1 + seg_length * sr / hop_length)
    # jumps_per_stride = int(mel_sample_size / seg_length)
    while end < (length + stride):
        start_offset = i * sr_stride

        end = i * stride + seg_length
        print("start", i * stride)

        if end > length:
            s_data = frames[-sample_size:]
            # sub = frames[start_offset:]
            # s_data = np.zeros(int(seg_length * sr))
            # start_pos = np.random.randint((sr / 4))
            # start_pos = 0
            # s_data[start_pos : start_pos + len(sub)] = sub

            # s_data = np.pad(sub, int(seg_length * sr))
            # print(s_data.shape, sr, seg_length, seg_length * sr)
        else:
            s_data = frames[start_offset : start_offset + sample_size]
        # print(
        #     start_offset,
        #     sample_size,
        #     len(s_data),
        #     "data max is",
        #     np.amax(s_data),
        #     s_data.dtype,
        # )
        # print(s_data[0])
        # 1 / 0
        # for d in s_data:
        #     print(d)
        # print("end")
        mel = librosa.feature.melspectrogram(
            y=s_data,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=50,
            fmax=11000,
            n_mels=80,
        )
        mel = librosa.power_to_db(mel, ref=np.max)

        # print("loading from ", start_offset / sr)
        # sf.write(f"test{i}.wav", s_data, sr, subtype="PCM_24")
        # start = int(jumps_per_stride * (i * stride))
        # mel = mel_all[:, start : start + mel_sample_size].copy()
        i += 1
        # if i >= 60:
        # plot_mel(mel, i)
        # plot_mel(mel)
        # print("mel max is", np.amax(mel))
        mel_m = tf.reduce_mean(mel, axis=1)
        # print("Mean at ", i, " is", mel_m)
        # gp not sure to mean over axis 0 or 1
        mel_m = tf.expand_dims(mel_m, axis=1)
        # mean over each mel bank
        # print(mel.shape)
        empty = np.zeros(((80, 151)))
        empty[:, : mel.shape[1]] = mel
        mel = empty
        # mel = mel - mel_m
        mel = mel[:, :, np.newaxis]
        # print("mean of mel is", round(1000 * np.mean(mel), 4))
        mels.append(mel)
        # break
    return mels, length


def plot_mel(mel, i=0):
    plt.figure(figsize=(10, 10))

    ax = plt.subplot(1, 1, 1)
    img = librosa.display.specshow(
        mel, x_axis="time", y_axis="mel", sr=48000, fmax=11000, ax=ax
    )
    # plt.show()
    plt.savefig(f"mel-power-{i}.png", format="png")
    # plt.clf()


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
        custom_objects={
            "hamming_loss": hamming,
            "top_k_categorical_accuracy": prec_at_k,
        },
        compile=False,
    )
    # model = tf.keras.models.load_model(str(load_model))

    model.load_weights(load_model / "val_loss").expect_partial()
    with open(load_model / "metadata.txt", "r") as f:
        meta = json.load(f)
    labels = meta.get("labels", [])
    multi_label = meta.get("multi_label", True)
    segment_length = meta.get("segment_length", 1.5)
    segment_stride = meta.get("segment_stride", 2)
    # print("stride is", segment_stride)
    # segment_length = 2
    segment_stride = 0.5
    # multi_label = True
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
        data, length = preprocess_file(file)
        data = np.array(data)

    print("data is", data.shape, data.dtype, np.amax(data))
    predictions = model.predict(np.array(data))
    tracks = []
    start = 0
    active_tracks = {}
    for prediction in predictions:
        print("at", start, np.round(prediction * 100))
        # break
        if start + seg_length > length:
            print("final one")
            start = length - seg_length
        results = []
        track_labels = []
        if multi_label:
            # print("doing multi", prediction * 100)
            for i, p in enumerate(prediction):
                if p >= 0.7:
                    label = labels[i]
                    results.append((p, label))
                    track_labels.append(label)
        else:
            best_i = np.argmax(prediction)
            best_p = prediction[best_i]
            if best_p > 0.7:
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
    #
    # for r in results:
    #     # best_i = np.argmax(r)
    #     # best_p = r[best_i]
    #     results = []
    #     for i, p in enumerate(r):
    #         if p > 0.65:
    #             results.append((round(100 * p), labels[i]))
    #
    #     print(f"{results} {start} - {start+seg_length} ")
    #     start += 1
    #     continue
    #     if best_p > 0.7:
    #         if track is None:
    #             track = (best_i, start)
    #         elif track[0] != best_i:
    #             print(f"Changed pred {labels[track[0]]} {track[1]} - {start}")
    #             track = (best_i, start)
    #
    #     elif track is not None:
    #         print(
    #             f"Low prob {best_p} - {labels[track[0]]} {track[1]} - {start + seg_length/2.0 - 1}"
    #         )
    #         track = None
    #     start += 1
    # # species = results[0]
    # if track is not None:
    #     print(f"Final {best_p} - {labels[track[0]]} {track[1]} - {start}")
    # return
    # animal = results[1]
    # for s, r in zip(species, animal):
    #
    #     print(
    #         f"{start} - {start+seg_length} Species",
    #         np.round(s, 1),
    #         " class ",
    #         np.round(r, 1),
    #     )
    #     start += 1


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
