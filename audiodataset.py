import logging
import json
from pathlib import Path
from collections import namedtuple
from dateutil.parser import parse as parse_date
import soundfile as sf

import librosa
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np


SEGMENT_LENGTH = 3  # seconds
SEGMENT_STRIDE = 1  # of a second
FRAME_LENGTH = 255


REJECT_TAGS = ["unidentified", "other"]

ACCEPT_TAGS = ["bird", "morepork", "kiwi", "rain", "human", "norfolk golden whistler"]

RELABEL = {}
RELABEL["north island brown kiwi"] = "kiwi"
RELABEL["great spotted kiwi"] = "kiwi"
RELABEL["norfolk morepork"] = "morepork"


class AudioDataset:
    def __init__(self, name):
        # self.base_path = Path(base_path)
        self.name = name
        self.recs = []
        self.labels = set()
        # self.samples_by_label
        self.samples = []

    def load_meta(self, base_path):
        meta_files = Path(base_path).glob("**/*.txt")
        for f in meta_files:
            meta = load_metadata(f)
            r = Recording(meta, f)
            self.recs.append(r)
            # r.get_human_tags()
            for tag in r.human_tags:
                self.labels.add(tag)

    def get_counts(self):
        counts = {}
        for s in self.samples:
            if s.tag in counts:
                counts[s.tag] += 1
            else:
                counts[s.tag] = 0
        return counts

    def print_counts(self):
        counts = {}
        original_c = {}
        for r in self.recs:
            for track in r.tracks:
                tags = track.tags
                if len(tags) == 0:
                    continue
                elif len(tags) == 1:
                    tag = list(tags)[0]
                    if tag not in counts:
                        counts[tag] = 1
                    else:
                        counts[tag] += 1

                    tag = list(track.original_tags)[0]
                    if tag not in RELABEL:
                        continue
                    if tag not in original_c:
                        original_c[tag] = 1
                    else:
                        original_c[tag] += 1
                else:
                    logging.info(
                        "Conflicting tags %s track %s -  %s tags", r.id, track.id, tags
                    )
        logging.info("Counts from %s recordings", len(self.recs))
        for k, v in counts.items():
            logging.info("%s: %s", k, v)

        for k, v in original_c.items():
            logging.info("%s: %s used as %s", k, v, RELABEL[k])

    def print_sample_counts(self):
        counts = {}
        original_c = {}

        for track in self.samples:
            tags = track.tags
            if len(tags) == 1:
                tag = list(tags)[0]
                if tag not in counts:
                    counts[tag] = 1
                else:
                    counts[tag] += 1

                tag = list(track.original_tags)[0]
                if tag not in RELABEL:
                    continue
                if tag not in original_c:
                    original_c[tag] = 1
                else:
                    original_c[tag] += 1
            else:
                logging.info(
                    "Conflicting tags %s track %s -  %s tags", r.id, track.id, tags
                )
        logging.info("Counts from %s Samples", len(self.samples))
        for k, v in counts.items():
            logging.info("%s: %s", k, v)
        for k, v in original_c.items():
            logging.info("%s: %s used as %s", k, v, RELABEL[k])

    def add_sample(self, sample):
        self.samples.append(sample)
        for t in sample.tags:
            self.labels.add(t)

    def remove(self, sample):
        sample.rec.tracks.remove(sample)
        if sample in self.samples:
            self.samples.remove(sample)


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


def filter_track(track):
    if len(track.tags) != 1:
        return True

    tag = track.tag
    if tag in REJECT_TAGS:
        return True
    if ACCEPT_TAGS is not None and tag not in ACCEPT_TAGS:
        return True
    return False


class Recording:
    def __init__(self, metadata, filename):
        self.filename = filename.with_suffix(".m4a")
        self.metadata = metadata
        self.id = metadata.get("id")
        self.device_id = metadata.get("deviceId")
        self.group_id = metadata.get("groupId")
        self.rec_date = metadata.get("recordingDateTime")
        if self.rec_date is not None:
            self.rec_date = parse_date(self.rec_date)

        self.tracks = []
        self.human_tags = set()
        for track in metadata.get("tracks"):
            t = Track(track, self.filename, self.id, self)
            if filter_track(t):
                continue
            self.tracks.append(t)
            for tag in t.human_tags:
                self.human_tags.add(tag)
        self.sample_rate = None
        self.rec_data = None
        self.resampled = False

    def load_recording(self, resample=None):
        frames, sr = librosa.load(str(self.filename), sr=None)
        if resample is not None and resample != sr:
            frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
            sr = resample
            print("resampled to ", sr)
            self.resampled = True
        self.sample_rate = sr
        self.rec_data = frames

    def load_track_data(self):
        logging.info(
            "Have data for %s sr %s len frames %s",
            self.filename,
            self.sample_rate,
            len(frames),
        )

        base_name = self.filename.stem
        seg_frames = SEGMENT_LENGTH * sr
        for t in self.tracks:
            t_start = int(sr * t.start)
            t_end = int(sr * t.end)
            segment_count = int(max(1, (t.length - SEGMENT_LENGTH) // SEGMENT_STRIDE))
            for i in range(segment_count):
                start_offset = i * seg_frames
                # zero pad shorter
                sub = frames[t_start + start_offset : SEGMENT_LENGTH * sr]
                if len(sub) < seg_frames:
                    s_data = np.zeros((SEGMENT_LENGTH * sr))
                    s_data[0 : len(sub)] = sub
                else:
                    s_data = sub
                spectrogram = tf.signal.stft(
                    s_data,
                    frame_length=FRAME_LENGTH,
                    frame_step=FRAME_LENGTH // 2,
                    fft_length=FRAME_LENGTH,
                    pad_end=True,
                )

                spectrogram = tf.abs(spectrogram).numpy()
                spectrogram = spectrogram[..., tf.newaxis]

    @property
    def bin_id(self):
        return self.id


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


def show_s(data, id):
    fig, axes = plt.subplots(1, figsize=(12, 8))

    plot_spectrogram(data, axes)
    axes.set_title("Spectrogram")
    plt.suptitle("TIT")
    print("SHOWIN??")
    # plt.show()
    plt.savefig(f"foo-tf-2-{id}.png")


class Track:
    def __init__(self, metadata, filename, rec_id, rec):
        self.rec = rec
        self.filename = filename
        self.rec_id = rec_id
        self.start = metadata["start"]
        self.end = metadata["end"]
        self.id = metadata.get("id")
        self.automatic_tags = set()
        self.human_tags = set()
        self.automatic = metadata.get("automatic")
        self.original_tags = set()
        tags = metadata.get("tags", [])
        for tag in tags:
            what = tag.get("what")
            original = what
            if what in RELABEL:
                what = RELABEL[what]
            t = Tag(what, tag.get("confidence"), tag.get("automatic"), original)
            if t.automatic:
                self.automatic_tags.add(t.what)
            else:
                self.original_tags.add(t.original)
                self.human_tags.add(t.what)

    def get_data(self, resample=None):
        if self.rec.rec_data is None:
            self.rec.load_recording(resample)
        sr = self.rec.sample_rate
        frames = self.rec.rec_data
        seg_frames = SEGMENT_LENGTH * sr
        t_start = int(sr * self.start)
        t_end = int(sr * self.end)
        segment_count = int(max(1, (self.length - SEGMENT_LENGTH) // SEGMENT_STRIDE))
        segments = []
        for i in range(segment_count):
            start_offset = i * seg_frames
            # zero pad shorter
            sub = frames[
                t_start + start_offset : t_start + start_offset + SEGMENT_LENGTH * sr
            ]
            if len(sub) < seg_frames:
                s_data = np.zeros((SEGMENT_LENGTH * sr))
                s_data[0 : len(sub)] = sub
            else:
                s_data = sub
            spectrogram = tf.signal.stft(
                s_data,
                frame_length=FRAME_LENGTH,
                frame_step=FRAME_LENGTH // 2,
                fft_length=FRAME_LENGTH,
                pad_end=True,
            )
            spectrogram = tf.abs(spectrogram).numpy()
            segments.append(
                SpectrogramData(spectrogram, t_start + start_offset, SEGMENT_LENGTH)
            )
        return segments

    #
    # def get_human_tags(self):
    #     return set([t.what for t in self.human_tags])

    @property
    def length(self):
        return self.end - self.start

    @property
    def tags(self):
        return self.human_tags

    @property
    def tag(self):
        return list(self.human_tags)[0]

    @property
    def bin_id(self):
        return f"{self.rec_id}-{self.tag}"


SpectrogramData = namedtuple("SpectrogramData", "data start_s length")

Tag = namedtuple("Tag", "what confidence automatic original")
