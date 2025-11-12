import logging
import json
from pathlib import Path
from collections import namedtuple

from dateutil.parser import parse as parse_date
from utils import get_label_to_ebird_map, get_ebird_id, get_ebird_ids_to_labels
import soundfile as sf

import librosa

import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
import math
import librosa.display
import scipy
import audioread.ffdec  # Use ffmpeg decoder
from custommel import mel_spec
import sys

labels_to_ebird_map = get_label_to_ebird_map()
ebird_to_labels_map = get_ebird_ids_to_labels()

DO_AUDIO_FEATURES = False
if DO_AUDIO_FEATURES:
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
REJECT_TAGS = ["unidentified", "other", "mammal"]
MAX_TRACK_SAMPLES = 4

ACCEPT_TAGS = None
# [
#     "ausbit1",
#     "ausmag2",
#     "dobplo1",
#     "eurbla",
#     "gryger1",
#     "houspa",
#     "kiwi",
#     "morepo2",
#     "nezbel1",
#     "nezfan1",
#     "rebdot1",
#     "silver3",
#     "sonthr1",
#     "tui1",
# ]
# # [
#     "house sparrow",
#     "bird",
#     "morepork",
#     "kiwi",
#     "rain",
#     "human",
#     "norfolk golden whistler",
# ]

RELABEL = {
    "mohoua novaeseelandiae": "pipipi1",
    "sackin1": "sackin3",
    "baicra1": "baicra4",
    "nibkiw1": "kiwi",
    "grskiw1": "kiwi",
    "norfolk morepork": "morepo2",
    "y01193": "y01193",
    "norfolk golden whistler": "y01193",
    "gobwhi1": "y01193",
}

# RELABEL["mohoua novaeseelandiae"] = "brown-creeper"
# RELABEL["new zealand fantail"] = "fantail"
# RELABEL["shining bronze-cuckoo"] = "shining-cuckoo"
# RELABEL["long-tailed koel"] = "long-tailed-cuckoo"
# RELABEL["masked lapwing"] = "spur-winged-plover"
# RELABEL["sacred kingfisher (new zealand)"] = "new-zealand-kingfisher"
# RELABEL["norfolk island gerygone"] = "norfolk gerygone"
# RELABEL["kelp gull"] = "southern-black-backed-gull"
# RELABEL["common myna"] = "indian-myna"
# RELABEL["baillon's crake"] = "marsh-crake"
# RELABEL["north island brown kiwi"] = "kiwi"
# RELABEL["great spotted kiwi"] = "kiwi"
# RELABEL["norfolk morepork"] = "morepork"
# RELABEL["golden whistler"] = "whistler"
# RELABEL["norfolk golden whistler"] = "whistler"
# RELABEL["golden-backed whistler"] = "whistler"

# GP TO DO should make sure labels that point to same ebird id are put together

SAMPLE_GROUP_ID = 0
MIN_TRACK_LENGTH = 1.5

# LOW_SAMPLES_LABELS = ["australasian bittern", "banded dotterel", "rifleman"]
LOW_SAMPLES_LABELS = []
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
        self.filter_frequency = args.get("filter_freq", False)


class AudioDataset:
    def __init__(self, name, config):
        if config is None:
            config = Config()
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
                if not audio_f.exists():
                    audio_f = f.with_suffix(".flac")
                    # hack to find files, probably should look
                    # at all files in dir or store file in metadata
                r = Recording(meta, audio_f, self.config, tighten_tracks=False)

                self.add_recording(r)
            except:
                logging.error("Error loading %s", f, exc_info=True)

    def add_recording(self, r):
        if r.id in self.recs:
            logging.info(
                "Already have %s in recs from %s now trying to add %s, Ignoring",
                r.id,
                self.recs[r.id].filename,
                r.filename,
            )
        self.recs[r.id] = r

        self.samples.extend(r.samples)

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

    def remove_rec(self, clip_id):
        if clip_id not in self.recs:
            return
        rec = self.recs[clip_id]
        for sample in rec.samples:
            self.remove(sample)
        del self.recs[clip_id]

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


audio_id = 0


class AudioSample:
    def __init__(
        self,
        rec,
        tags,
        text_tags,
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
        global audio_id
        self.id = audio_id
        audio_id += 1
        self.rec_id = None
        self.location = None
        if rec is not None:
            self.rec_id = rec.id
            self.location = rec.location
        self.low_sample = low_sample
        self.mixed_label = mixed_label
        self.tags = list(tags)
        self.text_tags = list(text_tags)
        # self.ebird_ids = labels_to_ebird(self.tags)
        non_bird = [t for t in tags if t not in ["noise", "bird"]]
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

    def clone(self):
        cloned = AudioSample(
            rec=None,
            tags=self.tags,
            text_tags=self.text_tags,
            start=self.start,
            end=self.end,
            track_ids=self.track_ids,
            group_id=self.group,
            signal_percent=self.signal_percent,
            bin_id=self.bin_id,
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            low_sample=self.low_sample,
        )
        cloned.rec_id = self.rec_id
        cloned.location = self.location
        if self.bin_id is None:
            self.bin_id = f"{self.rec_id}"
        return cloned

    @property
    def length(self):
        return self.end - self.start

    @property
    def tags_s(self):
        return "\n".join(self.tags)

    @property
    def text_tags_s(self):
        return "\n".join(self.text_tags)

    @property
    def track_id(self):
        return self.bin_id

    def __str__(self):
        return f"{self.rec_id}:{self.tags} - {self.start}-{self.end}"


class Recording:
    def __init__(
        self, metadata, filename, config, load_samples=True, tighten_tracks=True
    ):
        self.filename = filename
        self.metadata = metadata
        self.id = metadata.get("id")
        self.device_id = metadata.get("deviceId")
        self.group_id = metadata.get("groupId")
        self.rec_date = metadata.get("recordingDateTime")
        self.signals = metadata.get("signal", [])
        self.noises = metadata.get("noise", [])
        self.duration = metadata.get("duration")
        location = metadata.get("location")
        self.location = None
        lat = None
        lng = None
        country_code = None
        if location is not None:
            try:
                if isinstance(location, list):
                    location = location[0]
                lat = location.get("lat")
                lng = location.get("lng")
                self.location = (lat, lng)

            except:
                logging.error("Could not parse lat lng", exc_info=True)
                pass
        if self.rec_date is not None:
            self.rec_date = parse_date(self.rec_date)

        self.tracks = []
        self.human_tags = set()
        tracks_meta = metadata.get("Tracks")
        if tracks_meta is None:
            tracks_meta = metadata.get("tracks", [])

        for track in tracks_meta:
            t = Track(track, self.filename, self.id, self, tighten=tighten_tracks)
            if filter_track(t):
                continue
            self.tracks.append(t)
            for tag in t.human_tags:
                self.human_tags.add(tag)
        self.sample_rate = None
        self.rec_data = None
        self.resampled = False
        self.samples = []
        self.unused_samples = []
        self.small_strides = []
        if load_samples:
            self.signal_percent()
            self.load_samples(config.segment_length, config.segment_stride)

    def load_samples(self, segment_length, segment_stride):
        (self.samples, self.small_strides, self.unused_samples) = self.get_samples(
            segment_length, segment_stride
        )

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

    def get_samples(
        self,
        segment_length,
        segment_stride,
        do_overlap=False,
        for_label=None,
        extra_samples=True,
    ):
        logging.info(
            "Getting samples with length: %s stide: %s over: %s for: %s extra: %s",
            segment_length,
            segment_stride,
            do_overlap,
            for_label,
            extra_samples,
        )
        samples = []
        extra_small_strides = []
        unused_samples = []
        max_samples = MAX_TRACK_SAMPLES
        if len(self.samples) > 0:
            logging.debug("Loading samples when we already have samples")
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

        min_sample_length = segment_length - SEG_LEEWAY
        # low_sample_rec = any([True for l in self.human_tags if l in LOW_SAMPLES_LABELS])
        # can be used to seperate among train/val/test
        if for_label is None:
            tracks = self.tracks
        else:
            tracks = [t for t in self.tracks if for_label in t.human_tags]

        tracks = [t for t in self.tracks if not t.rms_filtered]

        bin_id = f"{self.id}-0"
        for track in tracks:
            if track.bird_track and (track.noise_track or track.animal_track):
                logging.info("SKipping track as is noise/animal and bird")
                continue
            adjusted = False
            # dont do noise tracks that happen at the same time as bird tracks
            if not track.bird_track:
                for other_track in tracks:
                    if track == other_track:
                        continue
                    overlap = segment_overlap(
                        [track.og_start, track.og_end],
                        [other_track.og_start, other_track.og_end],
                    )
                    if other_track.bird_track and overlap > 0:
                        # check track is still valid i.e. has over x seconds
                        if track.og_start > other_track.og_start:
                            track.start = other_track.og_end
                            track.end = max(track.start, track.end)
                        elif other_track.og_end > track.end:
                            track.end = other_track.og_start
                        else:
                            start_section = other_track.og_start - track.start
                            end_section = track.end - other_track.og_end
                            if start_section > end_section:
                                track.end = other_track.og_start
                            else:
                                track.start = other_track.og_end
                        track.start = min(track.og_end, track.start)
                        track.end = min(track.end, track.og_end)

                        logging.info(
                            "Rec %s Track %s overlaps a bird track %s adjusted track times to %s-%s",
                            self.id,
                            track.id,
                            other_track.id,
                            track.start,
                            track.end,
                        )
                        adjusted = True
            if adjusted and track.length < 1:
                logging.error(
                    "Skipping noise track as too short %s-%s", self.id, track.id
                )
                continue

            start_stride = segment_stride
            max_samples = (track.length - segment_length) / segment_stride
            if track.length > 3:
                max_samples += 1
            max_samples = round(max_samples)
            max_samples = max(max_samples, 1)

            track_samples = (track.length - segment_length) / segment_stride
            # if track_samples < 1:
            # logging.info("Low track samples so small stride")
            # start_stride = segment_stride / 2
            # allow an extra track with we have over 1/2 segment stride
            track_samples = round(track_samples)
            track_samples = max(track_samples, 0)
            left_over = track_samples - int(track_samples)
            track_samples = int(track_samples) + 1

            # max_track_length = segment_length + (max_samples - 1) * segment_stride

            sample_jitter = None
            sample_starts = (
                np.arange(track.length, step=start_stride, dtype=np.float32)
                + track.start
            )

            max_samples = MAX_TRACK_SAMPLES
            if track_samples > 1:
                sample_starts = (
                    sample_starts + np.random.rand(len(sample_starts)) / 2 - 0.25
                )
            if track_samples > max_samples:
                # track.length > max_track_length:
                # might be worth letting more samples for some labels

                # if track is long lets just take some random samples
                # extra_length = track.length - max_track_length
                # sample_jitter = extra_length / max_samples
                selected_samples = np.random.choice(
                    sample_starts, max_samples, replace=False
                )
                # dont think this is needed + np.random.rand(max_samples)
                left_over = 0
            else:
                selected_samples = sample_starts
            # adjust start times by a random float this way we can incorporate
            # end sometimes and start othertimes

            small_strides = (
                np.arange(track_samples, step=start_stride, dtype=np.float32)
                + track.start
                + start_stride / 2
            )

            if track_samples > 1:
                small_strides = (
                    small_strides + np.random.rand(len(small_strides)) / 2 - 0.25
                )
            # logging.info(
            #     "%s  Track times are %s-%s samples are %s num samples %s small strides %s",
            #     track.human_tags,
            #     track.start,
            #     track.end,
            #     sample_starts,
            #     track_samples,
            #     small_strides,
            # )
            # if a track has a little bit that will be cut off at the end
            # adjust tsample to be random start
            if left_over > 0 and track_samples == 1 and left_over < SEG_LEEWAY:
                start_jitter = np.random.rand() * left_over
                sample_starts += start_jitter
            sample_i = 1

            low_sample_track = any(
                [True for l in track.human_tags if l in LOW_SAMPLES_LABELS]
            )
            small_stride = False
            if extra_samples:
                all_starts = [sample_starts, small_strides]
            else:
                all_starts = [sample_starts]

            for starts in all_starts:
                # logging.info(
                #     "Loading from starts %s msmalll stride %s", starts, small_stride
                # )
                # sample_i = 0
                for start in starts:
                    # no negative starts
                    start = max(0, start)
                    used_sample = start in selected_samples and not small_stride
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
                    text_labels = set(track.human_text_tags)

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
                                text_labels = text_labels | other_track.human_text_tags
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
                    if low_sample_track:
                        # if is a low sample track allow, tracks to spread over multiple datasets
                        # might not be a good idea
                        bin_id = f"{self.id}-{track.id}"
                        # {int(start // segment_length)}"
                    sample = AudioSample(
                        self,
                        labels,
                        text_labels,
                        start,
                        end,
                        [track.id for t in other_tracks],
                        SAMPLE_GROUP_ID,
                        track.signal_percent,
                        bin_id=bin_id,
                        min_freq=min_freq,
                        max_freq=max_freq,
                        mixed_label=track.mixed_label,
                        low_sample=low_sample_track,
                    )
                    if used_sample:
                        samples.append(sample)
                    elif small_stride and extra_samples:
                        extra_small_strides.append(sample)
                    elif extra_samples:
                        unused_samples.append(sample)

                    min_sample_length = segment_length - SEG_LEEWAY

                    if start > track.end or (end - start) < min_sample_length:
                        break
                small_stride = True
                # just for first segment
                min_sample_length = 1.5
            # for s in self.samples:
            #     if track.id in s.track_ids:
            #         logging.info("USed samples are %s", s)

            # for unused in self.unused_samples:
            #     if track.id in unused.track_ids:
            #         logging.info("Not Used samples are %s", unused)
            # for unused in self.small_strides:
            #     if track.id in unused.track_ids:
            #         logging.info("SMall stride samples are %s", unused)
        return (samples, extra_small_strides, unused_samples)

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
    def __init__(self, metadata, filename, rec_id, rec, segment_length=3, tighten=True):
        self.rec = rec
        self.filename = filename
        self.rec_id = rec_id

        self.start = metadata["start"]
        self.end = metadata["end"]
        self.og_start = self.start
        self.og_end = self.end
        self.id = metadata.get("id")
        positions = metadata.get("positions", [])
        self.min_freq = metadata.get("minFreq", None)
        self.max_freq = metadata.get("maxFreq", None)

        if len(positions) > 0:
            y = positions[0]["y"]
            height = positions[0]["height"]
            if height != 1:
                if self.min_freq is None:
                    self.min_freq = y * TOP_FREQ
                if self.max_freq is None:
                    self.max_freq = height * TOP_FREQ + self.min_freq

        self.automatic_tags = set()
        self.human_tags = set()
        self.human_text_tags = set()
        self.automatic = metadata.get("automatic")
        self.original_tags = set()
        self.signal_percent = None
        self.mid_features = None
        self.short_features = None
        self.mixed_label = None
        self.rms_filtered = False
        tags = metadata.get("tags", [])
        for tag in tags:
            self.add_tag(tag)
        from birdsconfig import (
            ALL_BIRDS,
            ANIMAL_LABELS,
            NOISE_LABELS,
        )

        self.bird_track = False
        self.animal_track = False
        self.noise_track = False
        for tag in self.human_tags:
            if tag in ALL_BIRDS:
                self.bird_track = True
            if tag in ANIMAL_LABELS:
                self.animal_track = True
            elif tag in NOISE_LABELS:
                self.noise_track = True
        if tighten:
            self.tighten_track(metadata, segment_length)

    def tighten_track(self, metadata, segment_length):

        if not self.bird_track:
            # dont do anything for noisy tracks
            return
        if "upper_rms" not in metadata:
            # self.rms_filtered = True
            logging.info(
                "Missing rms %s human tag %s id is %s not filtering",
                self.filename,
                self.human_tags,
                self.id,
            )
            return
        # probably not doing anything so put very low
        MIN_STDDEV_PERCENT = 0.01
        rms_thresh = 0.00001
        rms_height = 0.001
        upper_rms = metadata["upper_rms"]
        rms_hop = metadata.get("rms_hop_length", 281)
        rms_sr = metadata.get("rms_sr", 48000)

        upper_peaks, _ = scipy.signal.find_peaks(
            upper_rms, threshold=rms_thresh / 10, height=rms_height / 10, width=2
        )
        if len(self.human_tags) == 0:
            return

        if self.bird_track:
            rms = metadata["bird_rms"]
            noise_rms = metadata["noise_rms"]
            rms = metadata["bird_rms"]
        else:
            rms = metadata["noise_rms"]
            noise_rms = metadata["bird_rms"]
        rms = np.array(rms)
        rms_peaks, rms_meta = scipy.signal.find_peaks(
            rms, threshold=rms_thresh, height=rms_height, width=2
        )
        noise_peaks, noise_meta = scipy.signal.find_peaks(
            noise_rms, threshold=rms_thresh, height=rms_height, width=2
        )
        remove_rms_noise(rms, rms_peaks, rms_meta, noise_peaks, noise_meta, upper_peaks)

        best_offset, _ = best_rms(rms, segment_length, rms_sr, rms_hop)
        start = self.start + best_offset * rms_hop / rms_sr
        end = min(start + segment_length, self.end)
        # logging.info("Track %s - %s becomes %s - %s", self.start, self.end, start, end)
        self.start = start
        self.end = end

        track_rms = rms[best_offset : int(self.end * rms_sr / rms_hop)]
        std_dev = np.std(track_rms)
        mean = np.mean(track_rms)

        percent_of_mean = std_dev / mean
        if percent_of_mean < MIN_STDDEV_PERCENT:
            logging.error(
                "RMS below std %s percent of mean %s for rec %s track at %s - %s id %s",
                std_dev,
                percent_of_mean,
                self.rec.id if self.rec is not None else "",
                self.start,
                self.end,
                self.id,
            )
            self.rms_filtered = True

    def ensure_track_length(self, rec_duration):
        start, end = ensure_track_length(
            self.start, self.end, MIN_TRACK_LENGTH, track_end=rec_duration
        )
        self.start = start
        self.end = end

    def add_tag(self, tag):
        text_label = tag.get("what")
        ebird_id = get_ebird_id(text_label, labels_to_ebird_map)

        original = ebird_id
        if ebird_id in RELABEL:
            ebird_id = RELABEL[ebird_id]

            text_label = ebird_to_labels_map.get(ebird_id, [ebird_id])[0]
        t = Tag(
            text_label, ebird_id, tag.get("confidence"), tag.get("automatic"), original
        )
        if t.automatic:
            self.automatic_tags.add(t.ebird_id)
        else:
            self.original_tags.add(t.original)
            self.human_tags.add(t.ebird_id)
            self.human_text_tags.add(text_label)

    def overlaps(self, other):
        return segment_overlap([self.start, self.end], [other.start, other.end])

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
    def freq_start(self):
        return self.min_freq

    @property
    def freq_end(self):
        return self.max_freq

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
        if len(self.human_tags) > 0:
            return list(self.human_tags)[0]
        return None

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
    "SpectrogramData", "raw spectogram raw_length buttered short_features,mid_features"
)

Tag = namedtuple("Tag", "what ebird_id confidence automatic original")


def load_data(
    config,
    start_s,
    frames,
    sr,
    n_fft=None,
    end=None,
    min_freq=None,
    max_freq=None,
    use_padding=False,
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
        n_fft = 4096  # power of 2 is best, otherwise need to know

        # i think we want to keep this the same
        # base2 = round(math.log2(sr // 10))
        # n_fft = int(math.pow(2, base2))
        # n_fft = sr // 10
    start = start_s * sr
    start = round(start)
    end_s = end
    if end is None:
        end = round(segment_l * sr) + start
    else:
        end = round(end * sr)

    # can make samples with negative start,
    if start_s < 0:
        logging.warning("Adjusting start to zero")
        start = 0
    data_length = segment_l
    spec = None
    if use_padding:
        s_data = frames[start:end]
    else:
        sr_data_l = sr * segment_l
        missing = sr_data_l - (end - start)
        if missing > 0:
            # print("Missing is ", missing / sr, " from ", start / sr, end / sr)
            offset = np.random.randint(0, missing)
            start = start - offset

            if start <= 0:
                start = 0
                end = start + sr_data_l
                end = min(end, len(frames))
            else:
                end_offset = end + missing - offset
                if end_offset > len(frames):
                    end_offset = len(frames)
                    start = end_offset - sr_data_l
                    start = max(start, 0)
                end = end_offset
            # print(
            #     "Now start is ",
            #     start / sr,
            #     " end ",
            #     end / sr,
            #     " original was ",
            #     start_s,
            #     " - ",
            #     end_s,
            #     " for length ",
            #     len(frames) / sr,
            # )
            # assert end - start == sr_data_l

        s_data = frames[start : int(segment_l * sr + start)]

    if end > len(frames) or start > len(frames):
        over_end = (end - len(frames)) / sr
        if over_end < 0.5:
            end = len(frames)
            logging.info(
                "Just out of bounds so setting to end start %s end %s frame length %s",
                start / sr,
                end / sr,
                len(frames) / sr,
            )
        else:
            logging.error(
                "Out of frame bounds start %s end %s frame length %s",
                start / sr,
                end / sr,
                len(frames) / sr,
            )
            raise Exception("Out of frame bounds")
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
    if DO_AUDIO_FEATURES:
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
            logging.info("Error loading features", exc_info=True)
    if len(s_data) < int(segment_l * sr):
        extra_frames = int(segment_l * sr) - len(s_data)
        offset = np.random.randint(0, extra_frames)
        s_data = np.pad(s_data, (offset, extra_frames - offset))
    assert len(s_data) == int(segment_l * sr)
    # buttered = butter_bandpass_filter(s_data, min_freq, max_freq, sr)
    normed = normalize_data(s_data)
    spectogram = np.abs(librosa.stft(normed, n_fft=n_fft, hop_length=hop_length))
    # if buttered is not None:
    #     spectogram_buttered = np.abs(
    #         librosa.stft(buttered, n_fft=n_fft, hop_length=hop_length)
    #     )
    # else:
    #     spectogram_buttered = buttered
    spec = SpectrogramData(s_data, spectogram, data_length, None, short_f, mid_f)
    a_max = np.amax(s_data)
    a_min = np.amin(s_data)
    if a_max == a_min:
        print("Error max is min ", a_max, a_min, start_s, end)
        logging.error(
            "Max is min %s start %s end %s data length %s ",
            a_max,
            a_min,
            start / sr,
            end / sr,
            len(frames) / sr,
        )
        raise Exception("Max is min")
    # except:
    #     logging.error(
    #         "Error getting segment  start %s lenght %s",
    #         start_s,
    #         config.segment_length,
    #         exc_info=True,
    #     )
    return spec


def normalize_data(x):
    min_v = np.min(x, -1, keepdims=True)
    x = x - min_v
    max_v = np.max(x, -1, keepdims=True)
    x = x / max_v + 0.000001
    x = x - 0.5
    x = x * 2
    return x


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


# look for peaks which occur in all 3 rms data and remove them by setting them to the average rms
def remove_rms_noise(
    rms,
    rms_peaks,
    rms_meta,
    noise_peaks,
    noise_meta,
    upper_peaks,
    sr=48000,
    hop_length=281,
):
    percent_diff = 0.55

    max_time_diff = 0.1 * sr / hop_length
    for n_i, n_p in enumerate(noise_peaks):
        rms_found = None
        rms_index = None
        upper_found = None
        for i, b_p in enumerate(rms_peaks):
            if abs(b_p - n_p) < max_time_diff:
                rms_found = b_p
                rms_index = i
                break
        if not rms_found:
            continue
        for u_p in upper_peaks:
            if abs(u_p - n_p) < max_time_diff:
                upper_found = u_p
                break
        if rms_found is not None and upper_found is not None:
            lower_bound = int(rms_meta["left_ips"][rms_index])
            upper_bound = int(rms_meta["right_ips"][rms_index])
            rms_width = upper_bound - lower_bound

            noise_lower_bound = int(noise_meta["left_ips"][n_i])
            noise_upper_bound = int(noise_meta["right_ips"][n_i])
            noise_width = noise_upper_bound - noise_lower_bound

            rms_height = rms_meta["peak_heights"][rms_index]
            noise_height = noise_meta["peak_heights"][n_i]

            width_percent = min(rms_width, noise_width) / max(rms_width, noise_width)
            height_percent = min(rms_height, noise_height) / max(
                rms_height, noise_height
            )
            # print(height_percent, width_percent)
            # print(rms_height,noise_height,"Height percent",noise_height / rms_height, " RMs ", rms_width, " noise ",noise_width,  " width ratio ",noise_width/ rms_width)
            if width_percent < percent_diff or height_percent < percent_diff:
                continue

            # logging.info("Full noise at %s  ", n_p * hop_length / sr)
            lower_bound = int(rms_meta["left_ips"][rms_index])
            upper_bound = int(rms_meta["right_ips"][rms_index])

            # upper_slice = upper_rms[0][ lower_bound: upper_bound]

            rms[lower_bound:upper_bound] = 0
    non_zero_mean = np.mean(rms[rms != 0])
    rms[rms == 0] = non_zero_mean


def best_rms(rms, segment_length=3, sr=48000, hop_length=281):
    sr = 48000
    window_size = sr * segment_length / hop_length
    window_size = int(window_size)
    first_window = np.sum(rms[:window_size])
    rolling_sum = first_window
    max_index = (0, first_window)
    for i in range(1, len(rms) - window_size):
        rolling_sum = rolling_sum - rms[i - 1] + rms[i + window_size]
        if rolling_sum > max_index[1]:
            max_index = (i, rolling_sum)
    return max_index


def segment_overlap(first, second):
    return (
        (first[1] - first[0])
        + (second[1] - second[0])
        - (max(first[1], second[1]) - min(first[0], second[0]))
    )


# def labels_to_ebird(labels):
#     ebird_ids = []
#     for lbl in labels:
#         ebird_ids.append(get_ebird_id(lbl, ebird_map))
#     logging.info("Labels %s become %s", labels, ebird_ids)
#     return ebird_ids
