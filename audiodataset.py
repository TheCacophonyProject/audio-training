import logging
import json
from pathlib import Path
from collections import namedtuple

from dateutil.parser import parse as parse_date
import soundfile as sf

import librosa

import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
import math
import librosa.display

import audioread.ffdec  # Use ffmpeg decoder
from custommels import mel_spec
import sys

sys.path.append("../pyAudioAnalysis")
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
MIN_TRACK_LENGTH = 1.5

LOW_SAMPLES_LABELS = ["australasian bittern", "banded dotterel", "rifleman"]
logging.info("Allow %s to have a recording over multiple datasets", LOW_SAMPLES_LABELS)


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
        self.recs = {}
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
        self.recs[r.id] = r
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
                    counts[tag] = 1
        return counts

    def remove_rec(self, rec):
        for s in rec.samples:
            self.samples.remove(s)
        if rec.id in self.recs:
            del self.recs[rec.id]

    def print_counts(self):
        counts = {}
        original_c = {}
        rec_counts = {}
        for r in self.recs.values():
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
        # make a clone of recordings so it is independent of original rec, and add correct tracks
        if sample.rec_id not in self.recs:
            cloned_rec = rec.clone()
            cloned_rec.tracks = []
            self.recs[rec.id] = cloned_rec

        tracks = [t for t in rec.tracks if t.id in sample.track_ids]
        self.recs[rec.id].samples.append(sample)
        self.recs[rec.id].add_tracks(tracks)
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
        mixed_label=None,
        low_sample=False,
    ):
        self.low_sample = low_sample
        self.mixed_label = mixed_label
        self.rec_id = rec.id
        self.tags = list(tags)
        non_bird = [t for t in tags if t not in ["noirse", "bird"]]
        if len(non_bird) > 0:
            self.first_tag = non_bird[0]
        else:
            self.first_tag = self.tags[0]
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
    def __init__(self, metadata, filename, config, load_samples=True):
        self.filename = filename
        self.metadata = metadata
        self.id = metadata.get("id")
        self.device_id = metadata.get("deviceId")
        self.group_id = metadata.get("groupId")
        self.rec_date = metadata.get("recordingDateTime")
        self.signals = metadata.get("signal", [])
        self.noises = metadata.get("noise", [])
        self.duration = metadata.get("duration")
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
        if load_samples:
            self.signal_percent()
            self.load_samples(config.segment_length, config.segment_stride)

    def add_tracks(self, tracks):
        for t in tracks:
            if len([True for existing in self.tracks if existing.id == t.id]) == 1:
                continue
            if filter_track(t):
                continue
            self.tracks.append(t)
            for tag in t.human_tags:
                self.human_tags.add(tag)

    def clone(self):
        cloned = Recording(self.metadata, self.filename, None, load_samples=False)
        return cloned

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

    def load_samples(self, segment_length, segment_stride, do_overlap=False):
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

        SEG_LEEWAY = 0.5
        MAX_TRACK_SAMPLES = 4

        max_track_length = segment_length + (MAX_TRACK_SAMPLES - 1) * segment_stride
        min_sample_length = segment_length - SEG_LEEWAY
        low_sample_rec = any([True for l in self.human_tags if l in LOW_SAMPLES_LABELS])
        # can be used to seperate among train/val/test
        bin_id = f"{self.id}-0"
        for track in self.tracks:
            # lets cap samples at 4 per track
            track_samples = (track.length - segment_length) / segment_stride
            track_samples = max(track_samples, 0)
            left_over = track_samples - int(track_samples)
            track_samples = int(track_samples) + 1

            sample_jitter = None
            sample_starts = (
                np.arange(track_samples, step=segment_stride, dtype=np.float32)
                + track.start
            )

            if track.length > max_track_length:
                # if track is long lets just take some random samples
                extra_length = track.length - max_track_length
                sample_jitter = extra_length / MAX_TRACK_SAMPLES
                sample_starts = np.random.choice(
                    sample_starts, MAX_TRACK_SAMPLES, replace=False
                ) + np.random.rand(MAX_TRACK_SAMPLES)
                left_over = 0
            # adjust start times by a random float this way we can incorporate
            # end sometimes and start othertimes

            # if a track has a little bit that will be cut off at the end
            # adjust tsample to be random start
            if left_over > 0 and track_samples == 1 and left_over < SEG_LEEWAY:
                start_jitter = np.random.rand() * left_over
                sample_starts += start_jitter
            sample_i = 1
            for start in sample_starts:
                end = start + segment_length
                end = min(end, track.end)
                if sample_i > 1:
                    if start > track.end or (end - start) < min_sample_length:
                        # dont think this will ever happen
                        break
                if (
                    left_over > 0
                    and left_over < SEG_LEEWAY
                    and sample_i == track_samples
                ):
                    # always include end
                    # this is assuming segment_stride has already include a sample
                    # with the start anyway
                    end = track.end
                    start = end - segment_length

                sample_i += 1
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
                        if overlap >= min_overlap:
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
                if low_sample_rec:
                    bin_id = f"{self.id}-{int(start // segment_length)}"
                self.samples.append(
                    AudioSample(
                        self,
                        labels,
                        start,
                        end,
                        [track.id for t in other_tracks],
                        SAMPLE_GROUP_ID,
                        track.signal_percent,
                        bin_id=bin_id,
                        min_freq=min_freq,
                        max_freq=max_freq,
                        mixed_label=track.mixed_label,
                        low_sample=low_sample_rec,
                    )
                )
                # s = self.samples[-1]
                # print(
                #     "Have sample",
                #     s.start,
                #     s.end,
                #     " from track ",
                #     track.start,
                #     track.end,
                # )
                # incase of jitter
                # start += segment_stride
                # end = start + segment_length
                # end = min(end, track.end)
                if start > track.end or (end - start) < min_sample_length:
                    break

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


def load_features(signal, sr):
    from pyAudioAnalysis import MidTermFeatures as aF

    # defaults from the pyAudio wiki
    mw = 1.0
    ms = 1.0
    sw = 0.050
    ss = 0.050
    mid_features, short_features, _ = aF.mid_feature_extraction(
        signal,
        sr,
        round(sr * mw),
        round(sr * ms),
        round(sr * sw),
        round(sr * ss),
    )

    return short_features, mid_features


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
        self.mid_features = None
        self.short_features = None
        self.mixed_label = None
        tags = metadata.get("tags", [])
        for tag in tags:
            self.add_tag(tag)

    def ensure_track_length(self, rec_duration):
        start, end = ensure_track_length(
            self.start, self.end, MIN_TRACK_LENGTH, track_end=rec_duration
        )
        self.start = start
        self.end = end

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


SpectrogramData = namedtuple(
    "SpectrogramData", "raw raw_length buttered short_features,mid_features"
)

Tag = namedtuple("Tag", "what confidence automatic original")


def load_data(
    config,
    start_s,
    frames,
    sr,
    n_fft=None,
    end=None,
    min_freq=None,
    max_freq=None,
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
        short_f = None
        mid_f = None
        try:
            short_f, mid_f = load_features(s_data, sr)
            windows = short_f.shape[1]
            if windows < 60:
                short_f = np.pad(short_f, ((0, 0), (0, 60 - windows)))
            windows = mid_f.shape[1]
            if windows < 3:
                mid_f = np.pad(mid_f, ((0, 0), (0, 3 - windows)))

            assert short_f.shape == (68, 60)
            assert mid_f.shape == (136, 3)
        except:
            logging.info("Error loading features")
        if len(s_data) < int(segment_l * sr):
            extra_frames = int(segment_l * sr) - len(s_data)
            offset = np.random.randint(0, extra_frames)
            s_data = np.pad(s_data, (offset, extra_frames - offset))
        assert len(s_data) == int(segment_l * sr)
        buttered = butter_bandpass_filter(s_data, min_freq, max_freq, sr)
        spectogram = np.abs(librosa.stft(s_data, n_fft=n_fft, hop_length=hop_length))
        if buttered is not None:
            spectogram_buttered = np.abs(
                librosa.stft(buttered, n_fft=n_fft, hop_length=hop_length)
            )
        else:
            spectogram_buttered = buttered
        spec = SpectrogramData(
            spectogram, data_length, spectogram_buttered, short_f, mid_f
        )
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


# makes sure track is a certain length otherwise adjusts start and end  randomly to make length
def ensure_track_length(start, end, min_length, track_end=None):
    current_length = end - start
    extra_length = min_length - current_length
    if extra_length <= 0:
        return start, end

    begin_pad = round(np.random.rand() * extra_length, 1)
    # (extra_length * 10) / 10
    start = start - begin_pad
    start = max(start, 0)
    end = start + min_length
    if track_end is not None:
        end = min(end, track_end)

    return start, end
