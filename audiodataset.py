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
RELABEL["new zealand fantail"] = "fantail"

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
        self.break_freq = args.get("break_freq", 1000)
        self.htk = not args.get("slaney", False)
        self.fmin = args.get("fmin", 50)
        self.fmax = args.get("fmax", 11000)
        self.n_mels = args.get("mels", 160)
        self.filter_frequency = args.get("filter_freq", True)


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
            try:
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
            except:
                logging.error("Error loading %s", f, exc_info=True)

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
    def __init__(
        self,
        rec,
        tags,
        start,
        end,
        track_ids,
        group_id,
        signal_percent,
        bin_id=None,
        min_freq=None,
        max_freq=None,
    ):
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
        self.signal_percent = signal_percent
        self.group = group_id
        self.predicted_labels = None
        self.min_freq = min_freq
        self.max_freq = max_freq

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
        for track in metadata.get("Tracks", []):
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
        self.signal_percent()
        self.load_samples(config.segment_length, config.segment_stride)

    def signal_percent(self):
        freq_filter = 1000

        for t in self.tracks:
            signal_time = 0
            signals = 0
            prev_e = None
            for s in self.signals:
                if s[2] < freq_filter:
                    continue
                if ((t.end - t.start) + (s[1] - s[0])) > max(t.end, s[1]) - min(
                    t.start, s[0]
                ):
                    start = max(s[0], t.start)
                    if prev_e is not None:
                        start = max(prev_e, start)
                    end = min(s[1], t.end)
                    if start > end:
                        continue
                    signal_time += end - start
                    signals += 1
                    prev_e = end
                    if t.end < s[1]:
                        break
                if t.end < s[0]:
                    break
            if t.length > 0:
                t.signal_percent = signal_time / t.length
            else:
                t.signal_percent = 0

    def recalc_tags(self):
        for track in self.tracks:
            for tag in track.human_tags:
                self.human_tags.add(tag)

    def space_signals(self, spacing=0.1):
        self.signals = space_signals(signals, spacing)

    def load_samples(self, segment_length, segment_stride, do_overlap=True):
        self.samples = []
        global SAMPLE_GROUP_ID
        SAMPLE_GROUP_ID += 1
        sorted_tracks = sorted(
            self.tracks,
            key=lambda track: track.start,
        )
        # always take 1 one sample, but dont bother with more if they are short
        # want to sample end of tracks always
        # does this make data unfair for shorter concise tracks
        min_sample_length = segment_length - segment_stride + 1 / 20

        # can be used to seperate among train/val/test
        bin_id = f"{self.id}-0"

        for track in self.tracks:
            start = track.start
            end = start + segment_length
            end = min(end, track.end)
            # print("checking", track.start, "-", track.end, track.human_tags)
            while True:
                min_freq = track.min_freq
                max_freq = track.max_freq
                labels = set(track.human_tags)
                other_tracks = []
                if do_overlap:
                    for other_track in sorted_tracks:
                        if track == other_track:
                            continue

                        # starts in this sample
                        if other_track.start > end:
                            break
                        overlap = (
                            (end - start)
                            + (other_track.length)
                            - (
                                max(end, other_track.end)
                                - min(start, other_track.start)
                            )
                        )
                        min_overlap = min(
                            0.9 * segment_length, other_track.length * 0.9
                        )

                        # enough overlap or we engulf the track
                        if overlap >= min_overlap or (overlap >= other_track.length):
                            other_tracks.append(other_track)
                            labels = labels | other_track.human_tags
                            if min_freq is not None:
                                if other_track.min_freq is None:
                                    min_freq = None
                                else:
                                    min_freq = min(other_track.min_freq, min_freq)
                            if max_freq is not None:
                                if other_track.max_freq is None:
                                    max_freq = None
                                else:
                                    max_freq = max(other_track.max_freq, max_freq)

                other_tracks.append(track)
                self.samples.append(
                    AudioSample(
                        self,
                        labels,
                        start,
                        min(track.end, end),
                        [track.id for t in other_tracks],
                        SAMPLE_GROUP_ID,
                        track.signal_percent,
                        bin_id=bin_id,
                        min_freq=min_freq,
                        max_freq=max_freq,
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
        #     print(
        #         self.id,
        #         "Have sample",
        #         s.start,
        #         s.end,
        #         s.tags,
        #         self.filename,
        #         s.track_ids,
        #     )

    # def load_recording(self, resample=None):
    #     try:
    #         print("Loading", self.filename)
    #         # with open(str(self.filename), "rb") as f:
    #         # frames, sr = librosa.load(self.filename)
    #         #  librosa wont close the file properly..... go figure
    #         aro = audioread.ffdec.FFmpegAudioFile(self.filename)
    #         frames, sr = librosa.load(aro, sr=None)
    #         assert sr == 48000
    #         aro.close()
    #         if resample is not None and resample != sr:
    #             frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
    #             sr = resample
    #             self.resampled = True
    #         self.sample_rate = sr
    #         self.rec_data = frames
    #     except:
    #         logging.error("Coult not load %s", str(self.filename), exc_info=True)
    #         return False
    #     return True

    # def get_data(self, resample=None):
    #     global SAMPLE_GROUP_ID
    #     SAMPLE_GROUP_ID += 1
    #
    #     # 1 / 0
    #     if self.rec_data is None:
    #         loaded = self.load_recording(resample)
    #         if not loaded:
    #             return None
    #     sr = self.sample_rate
    #     frames = self.rec_data
    #     for sample in self.samples:
    #         spectogram, mel, mfcc, s_data = load_data(sample.start, frames, sr)
    #         if spectogram is None:
    #             print("error loading")
    #             continue
    #         sample.spectogram_data = SpectrogramData(
    #             spectogram,
    #             mel,
    #             mfcc,
    #             s_data.copy(),
    #         )
    #         sample.sr = sr
    #
    #     return self.samples

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


# what positoins in db are scaled by
TOP_FREQ = 48000 / 2


class Track:
    def __init__(self, metadata, filename, rec_id, rec):
        self.rec = rec
        self.filename = filename
        self.rec_id = rec_id
        self.start = metadata["start"]
        self.end = metadata["end"]
        self.id = metadata.get("id")
        positions = metadata.get("positions", [])
        self.min_freq = None
        self.max_freq = None
        if len(positions) > 0:
            y = positions[0]["y"]
            height = positions[0]["height"]
            if height != 1:
                self.min_freq = y * TOP_FREQ
                self.max_freq = height * TOP_FREQ + self.min_freq
        self.automatic_tags = set()
        self.human_tags = set()
        self.automatic = metadata.get("automatic")
        self.original_tags = set()
        self.signal_percent = None
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

    #
    # def get_data(self, resample=None):
    #     global SAMPLE_GROUP_ID
    #     SAMPLE_GROUP_ID += 1
    #
    #     if self.rec.rec_data is None:
    #         loaded = self.rec.load_recording(resample)
    #         if not loaded:
    #             return None
    #
    #     sr = self.rec.sample_rate
    #     frames = self.rec.rec_data
    #     if self.start is None:
    #         self.start = 0
    #     i = 0
    #     start_s = self.start
    #     samples = []
    #     while (start_s + SEGMENT_LENGTH / 2) < self.end or i == 0:
    #         spectogram, mel, mfcc, s_data = load_data(start_s, frames, sr)
    #         if spectogram is None:
    #             continue
    #         sample = AudioSample(
    #             self.rec,
    #             self.human_tags,
    #             start_s,
    #             start_s + SEGMENT_LENGTH,
    #             [self.id],
    #             SAMPLE_GROUP_ID,
    #         )
    #         sample.spectogram_data = SpectrogramData(
    #             spectogram,
    #             mel,
    #             mfcc,
    #             s_data.copy(),
    #         )
    #         samples.append(sample)
    #         print("Getting for ", start_s, self.end, i, self.end - start_s)
    #         start_s += SEGMENT_STRIDE
    #         print(mel.shape)
    #         i += 1
    #     return samples

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


SpectrogramData = namedtuple("SpectrogramData", "raw raw_length buttered")

Tag = namedtuple("Tag", "what confidence automatic original")


def load_data(
    config, start_s, frames, sr, n_fft=None, end=None, min_freq=None, max_freq=None
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
            extra_frames = int(segment_l * sr) - len(s_data)
            offset = np.random.randint(0, extra_frames)
            s_data = np.pad(s_data, (offset, extra_frames - offset))
        assert len(s_data) == int(segment_l * sr)

        buttered = butter_bandpass_filter(s_data, min_freq, max_freq, sr)
        spec = SpectrogramData(s_data.copy(), data_length, buttered)
    except:
        logging.error(
            "Error getting segment  start %s lenght %s",
            start_s,
            config.segment_length,
            exc_info=True,
        )
    return spec


from scipy.signal import butter, sosfilt, sosfreqz, freqs


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    btype = "lowpass"
    freqs = []
    if lowcut is not None and lowcut > 0:
        btype = "bandpass"
        low = lowcut / nyq
        freqs.append(low)
    if highcut is not None:
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


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    if lowcut is None and highcut is None or highcut <= lowcut:
        logging.warn("No freq to filter")
        return None
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    if sos is None:
        return None
    filtered = sosfilt(sos, data)
    return filtered


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
