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
import math
import librosa.display

import audioread.ffdec  # Use ffmpeg decoder

SEGMENT_LENGTH = 2.5  # seconds
SEGMENT_STRIDE = 1  # of a second
FRAME_LENGTH = 255


REJECT_TAGS = ["unidentified", "other", "mammal", "sheep"]

ACCEPT_TAGS = None
# # [
#     "house sparrow",
#     "bird",
#     "morepork",
#     "kiwi",
#     "rain",
#     "human",
#     "norfolk golden whistler",
# ]

RELABEL = {}
RELABEL["north island brown kiwi"] = "kiwi"
RELABEL["great spotted kiwi"] = "kiwi"
RELABEL["norfolk morepork"] = "morepork"
RELABEL["golden whistler"] = "whistler"
RELABEL["norfolk golden whistler"] = "whistler"

SAMPLE_GROUP_ID = 0


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
            r = Recording(meta, f.with_suffix(".m4a"))
            self.add_recording(r)
            self.samples.extend(r.samples)

    def add_recording(self, r):
        self.recs.append(r)
        # r.get_human_tags()
        for tag in r.human_tags:
            self.labels.add(tag)

    def get_rec_counts(self):
        counts = {}
        for s in self.samples:
            for tag in s.tags:
                if tag in counts:
                    counts[tag].add(s.tag)
                else:
                    counts[tag] = {s.rec_id}
        return counts

    def get_counts(self):
        counts = {}
        for s in self.samples:
            for tag in s.tags:
                if tag in counts:
                    counts[tag] += 1
                else:
                    counts[tag] = 0
        return counts

    def print_counts(self):
        counts = {}
        original_c = {}
        rec_counts = {}
        for r in self.recs:
            for track in r.tracks:
                tags = track.tags
                # if len(tags) == 0:
                # continue
                # allowsing multi label
                for tag, original in zip(tags, track.original_tags):
                    # elif len(tags) == 1 or ("bird" not in track.tags):
                    if tag not in counts:
                        counts[tag] = 1
                        rec_counts[tag] = {r.id}
                        print("added", tag)
                    else:
                        counts[tag] += 1
                        rec_counts[tag].add(r.id)

                    if original not in RELABEL:
                        continue

                    if original not in original_c:
                        original_c[original] = 1
                        print("adding ", original)
                        rec_counts[original] = {r.id}

                    else:
                        original_c[original] += 1
                        rec_counts[original].add(r.id)
                    # else:
                # logging.info(
                # "Conflicting tags %s track %s -  %s tags", r.id, track.id, tags
                # )
        logging.info("Counts from %s recordings", len(self.recs))
        for k, v in counts.items():
            logging.info("%s: %s ( %s )", k, v, len(rec_counts[k]))

        for k, v in original_c.items():
            logging.info(
                "%s: %s used as %s ( %s )", k, v, RELABEL[k], len(rec_counts[k])
            )

    def print_sample_counts(self):
        counts = {}
        original_c = {}
        rec_counts = {}
        for s in self.samples:
            tags = s.tags
            for tag in tags:
                # if len(tags) == 1 or "birds" not in tags:
                # tag = list(tags)[0]
                if tag not in counts:
                    counts[tag] = 1
                    rec_counts[tag] = {s.rec_id}
                else:
                    counts[tag] += 1
                    rec_counts[tag].add(s.rec_id)
                continue
                # tag = list(track.original_tags)[0]
                if tag not in RELABEL:
                    continue
                # tag = RELABEL[tag]
                if tag not in original_c:
                    original_c[tag] = 1
                    rec_counts[tag] = {s.rec_id}

                else:
                    original_c[tag] += 1
                    rec_counts[tag].add(s.rec_id)

            # else:
            #     logging.info(
            #         "Conflicting tags %s track %s -  %s tags",
            #         track.rec.id,
            #         track.id,
            #         tags,
            #     )
        logging.info("Counts from %s Samples", len(self.samples))
        for k, v in counts.items():
            logging.info("%s: %s ( %s )", k, v, len(rec_counts[k]))
        for k, v in original_c.items():
            logging.info(
                "%s: %s ( %s ) used as %s", k, v, len(rec_counts[k]), RELABEL[k]
            )

    def add_sample(self, sample):
        self.samples.append(sample)
        for t in sample.tags:
            self.labels.add(t)

    def remove(self, sample):
        # sample.rec.tracks.remove(sample)
        if sample in self.samples:
            self.samples.remove(sample)
        # if sample in sample.rec.samples:
        # print("remove from rec")
        # sample.rec.samples.remove(sample)


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


class AudioSample:
    def __init__(self, rec, tags, start, end, track_ids, group_id, bin_id=None):
        self.rec = rec
        self.rec_id = rec.id
        self.tags = list(tags)
        self.tags.sort()
        self.start = start
        self.end = end
        self.track_ids = track_ids
        self.spectogram_data = None
        self.sr = None
        self.group = group_id
        if bin_id is None:
            self.bin_id = f"{self.rec_id}"
        else:
            self.bin_id = bin_id

    @property
    def length(self):
        return self.end - self.start

    @property
    def tags_s(self):
        return "\n".join(self.tags)

    @property
    def track_id(self):
        return self.bin_id


class Recording:
    def __init__(self, metadata, filename):
        self.filename = filename
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
        self.samples = []
        self.load_samples()

    def load_samples(self):
        global SAMPLE_GROUP_ID
        SAMPLE_GROUP_ID += 1
        sorted_tracks = sorted(
            self.tracks,
            key=lambda track: track.start,
        )
        self.samples = []
        if len(sorted_tracks) == 0:
            return
        track = sorted_tracks[0]
        start = track.start
        end = start + SEGMENT_LENGTH
        i = 1
        labels = set()
        labels = labels | track.human_tags
        bin = 0
        bin_id = f"{self.id}-{bin}"
        tracks = [track.id]
        while True:
            # start = round(start, 1)
            # end = round(end, 1)
            other_tracks = []
            for t in sorted_tracks[i:]:
                # starts in this sample
                if t.start > end:
                    break
                if t.start < start:
                    s = start
                else:
                    s = t.start

                if t.end > end:
                    # possible to miss out on some of a track
                    e = end
                else:
                    e = t.end
                intersect = e - s
                if intersect > SEGMENT_STRIDE:
                    # if t.start<= start and t.end <= end:
                    other_tracks.append(t)
                    labels = labels | t.human_tags
                    tracks.append(t.id)
            self.samples.append(
                AudioSample(
                    self,
                    labels,
                    start,
                    min(track.end, end),
                    tracks,
                    SAMPLE_GROUP_ID,
                    bin_id,
                )
            )
            # print("sample length is", self.samples[-1].length)
            start += SEGMENT_STRIDE
            # print("track end is ", track.end, " and start is", start)
            if (track.end - start) < SEGMENT_LENGTH / 2:
                old_end = track.end
                track = None
                # get new track
                for z, t in enumerate(sorted_tracks[i:]):
                    # print("checking track ", t.human_tags, t.start)
                    if t.end > start:
                        if t.start > old_end:
                            pass
                            # new bin as non overlapping audio
                            # bin += 1
                            # bin_id = f"{self.id}-{bin}"
                        track = t
                        tracks = [t.id]
                        start = max(start, t.start)
                        i = i + z + 1
                        labels = set()
                        labels = labels | track.human_tags
                        break
                if track is None:
                    # got all tracks
                    break
            end = start + SEGMENT_LENGTH
        # other_tracks = [t for t in sorted_tracks[i:] if t.start<= start and t.end >= end]
        # for t in sorted_tracks:

    def load_recording(self, resample=None):
        try:
            # with open(str(self.filename), "rb") as f:
            # frames, sr = librosa.load(self.filename)
            #  librosa wont close the file properly..... go figure
            aro = audioread.ffdec.FFmpegAudioFile(self.filename)
            frames, sr = librosa.load(aro)
            aro.close()
            if resample is not None and resample != sr:
                frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
                sr = resample
                self.resampled = True
            self.sample_rate = sr
            self.rec_data = frames
        except:
            logging.error("Coult not load %s", str(self.filename), exc_info=True)
            return False
        return True

    def get_data(self, resample=None):
        global SAMPLE_GROUP_ID
        SAMPLE_GROUP_ID += 1

        # 1 / 0
        if self.rec_data is None:
            loaded = self.load_recording(resample)
            if not loaded:
                return None
        sr = self.sample_rate
        frames = self.rec_data
        for sample in self.samples:
            spectogram, mel, mfcc, s_data = load_data(sample.start, frames, sr)
            if spectogram is None:
                print("error loading")
                continue
            sample.spectogram_data = SpectrogramData(
                spectogram,
                mel,
                mfcc,
                s_data.copy(),
            )
            sample.sr = sr

        return self.samples

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
            self.add_tag(tag)

    def add_tag(self, tag):
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
        global SAMPLE_GROUP_ID
        SAMPLE_GROUP_ID += 1

        if self.rec.rec_data is None:
            loaded = self.rec.load_recording(resample)
            if not loaded:
                return None

        sr = self.rec.sample_rate
        frames = self.rec.rec_data
        if self.start is None:
            self.start = 0
        i = 0
        start_s = self.start
        samples = []
        while (start_s + SEGMENT_LENGTH / 2) < self.end or i == 0:
            spectogram, mel, mfcc, s_data = load_data(start_s, frames, sr)
            if spectogram is None:
                continue
            sample = AudioSample(
                self.rec,
                self.human_tags,
                start_s,
                start_s + SEGMENT_LENGTH,
                [self.id],
                SAMPLE_GROUP_ID,
            )
            print(mfcc.shape)
            sample.spectogram_data = SpectrogramData(
                spectogram,
                mel,
                mfcc,
                s_data.copy(),
            )
            samples.append(sample)
            print("Getting for ", start_s, self.end, i, self.end - start_s)
            start_s += SEGMENT_STRIDE
            print(mel.shape)
            i += 1
        return samples

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
        all_tags = self.tags
        tag = None
        # for t in all_tags:
        #     if t in ["bird", "human", "video-game", "other"]:
        #         tag = t

        if tag is None and len(self.human_tags) > 0:
            return list(self.human_tags)[0]
        else:
            return tag

    @property
    def bin_id(self):
        return f"{self.rec_id}-{self.tag}"


def plot_mel(mel):
    print("pltting")
    plt.figure(figsize=(10, 10))

    ax = plt.subplot(1, 1, 1)
    print
    img = librosa.display.specshow(
        mel, x_axis="time", y_axis="mel", sr=48000, fmax=8000, ax=ax
    )
    plt.savefig("mel.png", format="png")
    # plt.clf()

    power_mel = librosa.power_to_db(mel)
    plt.figure(figsize=(10, 10))

    ax = plt.subplot(1, 1, 1)
    img = librosa.display.specshow(
        power_mel, x_axis="time", y_axis="mel", sr=48000, fmax=8000, ax=ax
    )
    plt.savefig("mel-power.png", format="png")
    # plt.clf()


SpectrogramData = namedtuple("SpectrogramData", "spect mel mfcc raw raw_length")

Tag = namedtuple("Tag", "what confidence automatic original")


def load_data(
    start_s,
    frames,
    sr,
    segment_l=SEGMENT_LENGTH,
    segment_stride=SEGMENT_STRIDE,
    hop_length=640,
    n_fft=None,
    end=None,
):
    sr_stride = int(segment_stride * sr)

    if n_fft is None:
        n_fft = sr // 10
    start = start_s * sr
    start = int(start)
    if end is None:
        end = int(segment_l * sr) + start
    else:
        end = int(end * sr)
    data_length = segment_l
    try:
        # zero pad shorter
        s_data = frames[start : int(segment_l * sr + start)]
        data_length = len(s_data) / sr
        # if end > len(frames):
        #     sub = frames[start:end]
        #     s_data = np.zeros(int(segment_l * sr))
        #     # randomize zero padding location
        #     extra_frames = len(s_data) - len(sub)
        #     # offset = np.random.randint(0, extra_frames)
        #     offset = 0
        #     s_data[offset : offset + len(sub)] = sub
        #     data_length = len(sub) / sr
        # else:
        #     s_data = frames[start:end]
        if len(s_data) < int(segment_l * sr):
            sub = s_data
            data_length = len(sub) / sr
            s_data = np.zeros(int(segment_l * sr))
            # randomize zero padding location
            # extra_frames = len(s_data) - len(sub)
            # offset = np.random.randint(0, extra_frames)
            offset = 0
            s_data[offset : offset + len(sub)] = sub
        spectogram = np.abs(librosa.stft(s_data, n_fft=n_fft, hop_length=hop_length))
        # these should b derivable from spectogram but the librosa exmaples produce different results....
        mel = librosa.feature.melspectrogram(
            y=s_data,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=50,
            fmax=11000,
            n_mels=80,
        )
        mfcc = librosa.feature.mfcc(
            y=s_data,
            sr=sr,
            hop_length=hop_length,
            htk=True,
            fmin=50,
            fmax=11000,
            n_mels=80,
        )
        return spectogram, mel, mfcc, s_data, data_length
    except:
        logging.error(
            "Error getting segment  start %s lenght %s",
            start_s,
            SEGMENT_LENGTH,
            exc_info=True,
        )
    return None, None, None, None, None
