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
from custommels import mel_spec

#
# SEGMENT_LENGTH = 2.5  # seconds
# SEGMENT_STRIDE = 1  # of a second
# HOP_LENGTH = 281
# BREAK_FREQ = 1750
# HTK = True
# FMIN = 50
# FMAX = 11000
# N_MELS = 120
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


class Config:
    def __init__(self, **args):
        self.segment_length = args.get("seg_length", 3)
        self.segment_stride = args.get("stride", 1)
        self.hop_length = args.get("hop_length", 281)
        self.break_freq = args.get("break_freq", 1750)
        self.htk = not args.get("slaney", False)
        self.fmin = args.get("fmin", 50)
        self.fmax = args.get("fmax", 11000)
        self.n_mels = args.get("mels", 120)


class AudioDataset:
    def __init__(self, name, config):
        # self.base_path = Path(base_path)
        self.config = config
        self.name = name
        self.recs = []
        self.rec_keys = []
        self.labels = set()
        # self.samples_by_label
        self.samples = []

    def load_meta(self, base_path):
        meta_files = Path(base_path).glob("**/*.txt")
        for f in meta_files:
            meta = load_metadata(f)
            audio_f = f.with_suffix(".m4a")
            if not audio_f.exists():
                audio_f = f.with_suffix(".wav")
            if not audio_f.exists():
                audio_f = f.with_suffix(".mp3")
                # hack to find files, probably should look
                # at all files in dir or store file in metadata
            r = Recording(meta, audio_f, self.config)
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
                    counts[tag].add(s.rec_id)
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

    def remove_rec(self, rec):
        self.recs.remove(rec)
        for s in rec.samples:
            self.samples.remove(s)
        if rec.id in self.rec_keys:
            self.rec_keys.remove(rec.id)

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
                    else:
                        counts[tag] += 1
                        rec_counts[tag].add(r.id)

                    if original not in RELABEL:
                        continue

                    if original not in original_c:
                        original_c[original] = 1
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

    def add_sample(self, rec, sample):
        if sample.rec_id not in self.rec_keys:
            self.recs.append(rec)
            self.rec_keys.append(rec.id)
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


def get_samples(rec_frames, sample):
    end = 0
    start = 0
    while end < len(frames):
        AudioSample(
            self,
            labels,
            start,
            min(track.end, end),
            tracks,
            SAMPLE_GROUP_ID,
            bin_id,
        )
        start += SEGMENT_STRIDE
        end = start + SEGMENT_LENGTH


class AudioSample:
    def __init__(self, rec, tags, start, end, track_ids, group_id, bin_id=None):
        self.rec_id = rec.id
        self.tags = list(tags)
        self.tags.sort()
        self.start = start
        self.end = end
        self.track_ids = track_ids
        self.spectogram_data = None
        self.sr = None
        self.logits = None
        self.embeddings = None
        self.group = group_id
        self.predicted_labels = None
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
    def __init__(self, metadata, filename, config):
        self.filename = filename
        self.metadata = metadata
        self.id = metadata.get("id")
        self.device_id = metadata.get("deviceId")
        self.group_id = metadata.get("groupId")
        self.rec_date = metadata.get("recordingDateTime")
        self.signals = metadata.get("signal", [])
        self.noises = metadata.get("noise", [])
        if self.rec_date is not None:
            self.rec_date = parse_date(self.rec_date)

        self.tracks = []
        self.human_tags = set()
        for track in metadata.get("tracks", []):
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

        self.load_samples(config.segment_length, config.segment_stride)

    def recalc_tags(self):
        for track in self.tracks:
            for tag in track.human_tags:
                self.human_tags.add(tag)

    def space_signals(self, spacing=0.1):
        self.signals = space_signals(signals, spacing)

    def load_samples_old(self, segment_length, segment_stride):
        global SAMPLE_GROUP_ID
        SAMPLE_GROUP_ID += 1
        sorted_tracks = sorted(
            self.tracks,
            key=lambda track: track.start,
        )
        actual_s = segment_stride
        self.samples = []
        if len(sorted_tracks) == 0:
            return
        track = sorted_tracks[0]
        # print("have", len(sorted_tracks), "tracks")
        # for t in sorted_tracks:
        # print("Tracks is", t.filename, t.start, t.end)
        start = track.start
        end = start + segment_length
        i = 1
        labels = set()
        labels = labels | track.human_tags
        bin = 0
        bin_id = f"{self.id}-{bin}"
        tracks = [track.id]
        # print("rec-", self.id, "tracks is", track.start, track.end, track.id)
        while True:
            # logging.info("Using %s %s", start, end)
            # start = round(start, 1)
            # end = round(end, 1)
            other_tracks = []
            tracks = [track.id]
            labels = set(track.human_tags)
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
                if intersect > 0.5:
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
            if "morepork" in labels:
                segment_stride = 3.5
            else:
                segment_stride = actual_s
            # print("sample length is", self.samples[-1].length)
            start += segment_stride
            # print("track end is ", track.end, " and start is", start)
            if (track.end - start) < segment_length / 2:
                old_end = track.end
                track = None
                start = start - segment_length
                # find the next track after start
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
            end = start + segment_length
        # other_tracks = [t for t in sorted_tracks[i:] if t.start<= start and t.end >= end]
        # for t in sorted_tracks:
        # for t in self.tracks:
        # print(self.id, "have track from ", t.start, t.end)
        # for s in self.samples:
        # print(self.id, "Have sample", s.start, s.end, s.tags, self.filename)

    def load_samples(self, segment_length, segment_stride):
        self.samples = []
        global SAMPLE_GROUP_ID
        SAMPLE_GROUP_ID += 1
        sorted_tracks = sorted(
            self.tracks,
            key=lambda track: track.start,
        )
        # always take 1 one sample, but dont bother with more if they are short
        min_sample_length = segment_length * 0.7
        # can be used to seperate among train/val/test
        bin_id = f"{self.id}-0"

        actual_s = segment_stride
        for track in self.tracks:
            if "morepork" in track.human_tags:
                # sometimes long tracks with multiple calls, think this should seperate them
                segment_stride = min(segment_stride, 3.5)
            else:
                segment_stride = actual_s
            start = track.start
            end = start + segment_length
            end = min(end, track.end)
            while True:
                labels = set(track.human_tags)
                other_tracks = []
                for other_track in sorted_tracks:
                    if track == other_track:
                        continue

                    # starts in this sample
                    if other_track.start > end:
                        break
                    overlap = (
                        (end - start)
                        + (other_track.length)
                        - (max(end, other_track.end) - min(start, other_track.start))
                    )
                    min_overlap = min(0.9 * segment_length, other_track.length * 0.9)

                    # enough overlap or we engulf the track
                    if overlap >= min_overlap or (
                        overlap > 0 and end > other_track.end
                    ):
                        # if t.start<= start and t.end <= end:
                        other_tracks.append(other_track)
                        labels = labels | other_track.human_tags
                # print("new samples with tracks", other_tracks)
                other_tracks.append(track)
                self.samples.append(
                    AudioSample(
                        self,
                        labels,
                        start,
                        min(track.end, end),
                        [track.id for t in other_tracks],
                        SAMPLE_GROUP_ID,
                        bin_id,
                    )
                )
                start += segment_stride
                end = start + segment_length
                end = min(end, track.end)
                if start > track.end or (end - start) < min_sample_length:
                    break

        # other_tracks = [t for t in sorted_tracks[i:] if t.start<= start and t.end >= end]
        # for t in sorted_tracks:
        # print("FOR ", self.id)
        # for t in self.tracks:
        #     print(self.id, "have track from ", t.start, t.end)
        # for s in self.samples:
        #     print(self.id, "Have sample", s.start, s.end, s.tags, self.filename)

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
    def tags_key(self):
        tags = list(self.human_tags)
        tags.sort()
        return "-".join(tags)

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


SpectrogramData = namedtuple(
    "SpectrogramData", "spect mel stft raw raw_length pcen mel_s"
)

Tag = namedtuple("Tag", "what confidence automatic original")


def load_data(
    config,
    start_s,
    frames,
    sr,
    n_fft=None,
    end=None,
):
    segment_l = config.segment_length
    segment_stride = config.segment_stride
    sr_stride = int(segment_stride * sr)
    hop_length = config.hop_length
    fmin = config.fmin
    fmax = config.fmax
    n_mels = config.n_mels
    htk = config.htk
    break_freq = config.break_freq

    if n_fft is None:
        n_fft = sr // 10
    start = start_s * sr
    start = round(start)
    if end is None:
        end = round(segment_l * sr) + start
    else:
        end = round(end * sr)
    data_length = segment_l
    spec = None
    try:
        #  use if dont want padding
        # s_data = frames[start : int(segment_l * sr + start)]
        # zero pad shorter
        s_data = frames[start:end]

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
            s_data = np.pad(s_data, (0, int(segment_l * sr) - len(s_data)))
        assert len(s_data) == int(segment_l * sr)
        spectogram = np.abs(librosa.stft(s_data, n_fft=n_fft, hop_length=hop_length))
        #     mel = mel_spec(
        #         spectogram,
        #         sr,
        #         n_fft,
        #         hop_length,
        #         n_mels,
        #         fmin,
        #         fmax,
        #         break_freq,
        #         power=2,
        #     )
        #     mel_pcen = mel_spec(
        #         spectogram,
        #         sr,
        #         n_fft,
        #         hop_length,
        #         n_mels,
        #         fmin,
        #         fmax,
        #         break_freq,
        #         power=1,
        #     )
        #     print(mel_pcen.shape)
        # else:
        #     # these should b derivable from spectogram but the librosa exmaples produce different results....
        #     mel = librosa.feature.melspectrogram(
        #         y=s_data,
        #         sr=sr,
        #         n_fft=n_fft,
        #         hop_length=hop_length,
        #         fmin=fmin,
        #         fmax=fmax,
        #         n_mels=n_mels,
        #     )
        # mel_pcen = librosa.feature.melspectrogram(
        #     y=s_data,
        #     sr=sr,
        #     n_fft=n_fft,
        #     hop_length=hop_length,
        #     fmin=fmin,
        #     fmax=fmax,
        #     n_mels=n_mels,
        #     power=1,
        # )
        # pcen_s = librosa.pcen(mel_pcen * (2**31), sr=sr, hop_length=hop_length)
        mfcc = None
        # pcen_s = None
        # mfcc = librosa.feature.mfcc(
        #     y=s_data,
        #     sr=sr,
        #     hop_length=hop_length,
        #     htk=htk,
        #     fmin=fmin,
        #     fmax=fmax,
        #     n_mels=n_mels,
        # )
        spec = SpectrogramData(None, None, spectogram, None, data_length, None, None)
    except:
        logging.error(
            "Error getting segment  start %s lenght %s",
            start_s,
            config.segment_length,
            exc_info=True,
        )
    return spec


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
    #
    # print("spaced have", len(new_signals))
    # for s in new_signals:
    #     print(s)
    return new_signals
