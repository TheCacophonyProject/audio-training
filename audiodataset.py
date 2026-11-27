import logging
import json
from pathlib import Path
from collections import namedtuple

from dateutil.parser import parse as parse_date
from utils import (
    centre_bonus,
    get_label_to_ebird_map,
    get_ebird_id,
    get_ebird_ids_to_labels,
    prefixed_id,
    split_key,
)

# shared with birdnetcompare and perchembed so they find the same best window,
# otherdata imports it from here
from birdnetcompare import best_rms
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
HOP_LENGTH = 273
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
        self.hop_length = args.get("hop_length", 273)
        self.break_freq = args.get("break_freq", 1000)
        self.htk = not args.get("slaney", False)
        self.fmin = args.get("fmin", 50)
        self.fmax = args.get("fmax", 11000)
        self.n_mels = args.get("mels", 160)
        self.filter_frequency = args.get("filter_freq", False)
        self.tighten_tracks = not args.get("dont_tighten_tracks", False)
        self.filter_rms = not args.get("dont_filter_rms", False)
        # only use each tracks best segment (track.best_start - track.best_end)
        self.best_segment = args.get("best_segment", False)
        # use a recording's "confirmed-tracks" (write_best_windows.py) in place
        # of its "Tracks" when it has them
        self.confirmed_tracks = args.get("confirmed_tracks", False)
        # seconds a probe window where a bird scored (the original track's
        # primary_starts / secondary_starts) must overlap a sample to label it
        self.presence_overlap = args.get("presence_overlap", 1.5)
        # seeds random generation so a dataset build can be reproduced exactly
        self.seed = args.get("seed")


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
        # sorted so recordings, and the random numbers their samples use, come
        # in the same order on every run
        meta_files = sorted(Path(base_path).glob("**/*.txt"))

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
                if not audio_f.exists():
                    logging.error("No audio found for %s", f)
                    continue
                    # hack to find files, probably should look
                    # at all files in dir or store file in metadata
                # samples are made once every file is loaded, so clips of one
                # recording (e.g. tier1) are merged first and their samples see
                # overlapping tracks from the other clips
                r = Recording(meta, audio_f, self.config, load_samples=False)

                self.add_recording(r)
            except:
                logging.error("Error loading %s", f, exc_info=True)
        self.load_samples()

    def load_samples(self):
        """Makes every recording's samples from its tracks, in load order so the
        random numbers are used in the same order on every run"""
        failed = []
        for r in self.recs.values():
            if r.samples:
                continue
            try:
                r.signal_percent()
                r.load_samples(self.config.segment_length, self.config.segment_stride)
            except:
                logging.error("Error making samples for %s", r.id, exc_info=True)
                failed.append(r.id)
                continue
            self.samples.extend(r.samples)
        # as before samples were made after loading, a recording that errors
        # isn't kept
        for rec_id in failed:
            del self.recs[rec_id]

    def add_recording(self, r):
        if r.id in self.recs:
            logging.info(
                "Already have %s in recs from %s now trying to add %s, merging tracks",
                r.id,
                self.recs[r.id].filename,
                r.filename,
            )
            # e.g. tier1 clips of one recording, each clip is its own file so
            # its tracks keep their own filename and rec (the clip)
            existing = self.recs[r.id]
            before = {t.id for t in existing.tracks}
            existing.add_tracks(r.tracks)
            added = {t.id for t in existing.tracks} - before
            # a sample's own track is last in track_ids, only take samples of
            # tracks that were added so the same file loaded twice isn't doubled
            samples = [s for s in r.samples if s.track_ids and s.track_ids[-1] in added]
            existing.samples.extend(samples)
            existing.small_strides.extend(
                s for s in r.small_strides if s.track_ids and s.track_ids[-1] in added
            )
            existing.unused_samples.extend(
                s for s in r.unused_samples if s.track_ids and s.track_ids[-1] in added
            )
            self.samples.extend(samples)
            for tag in r.human_tags:
                self.labels.add(tag)
            return
        self.recs[r.id] = r

        self.samples.extend(r.samples)

        # r.get_human_tags()
        for tag in r.human_tags:
            self.labels.add(tag)

    def get_rec_counts(self, tags="tags"):
        """Recording ids per tag, tags is the sample attribute to count e.g.
        tags, secondary_tags or soft_tags"""
        counts = {}
        for s in self.samples:
            for tag in getattr(s, tags):
                if tag in counts:
                    counts[tag].add(s.rec_id)
                else:
                    counts[tag] = {s.rec_id}
        return counts

    def get_counts(self, tags="tags"):
        """Samples per tag, tags is the sample attribute to count e.g. tags,
        secondary_tags or soft_tags"""
        counts = {}

        for s in self.samples:
            for tag in getattr(s, tags):
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
            # starts empty, gets only the tracks (and their tags) of samples
            # added here
            cloned_rec = rec.clone()
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


# write_best_windows.py's tracks to keep, used in place of "Tracks" with
# Config.confirmed_tracks
CONFIRMED_KEY = "confirmed-tracks"


def tag_labels(what):
    """(ebird id, text label) of a tag's what, as Track.add_tag maps it"""
    ebird_id = get_ebird_id(what, labels_to_ebird_map)
    text_label = what
    if ebird_id in RELABEL:
        ebird_id = RELABEL[ebird_id]
        text_label = ebird_to_labels_map.get(ebird_id, [ebird_id])[0]
    return ebird_id, text_label


def presence_windows(track_meta, segment_length):
    """[(ebird id, text label, start, end)] of the probe windows where a
    "Tracks" track's primary (primary_starts) or each secondary
    (secondary_starts) scored, in the track's file times"""
    windows = []
    tags = track_meta.get("tags", [])
    primaries = [
        t["what"]
        for t in tags
        if t.get("what") and not t.get("secondary") and t.get("automatic") is False
    ]
    for what in primaries:
        ebird_id, text_label = tag_labels(what)
        for start in track_meta.get("primary_starts", []):
            windows.append((ebird_id, text_label, start, start + segment_length))
    for what, starts in (track_meta.get("secondary_starts") or {}).items():
        ebird_id, text_label = tag_labels(what)
        for start in starts:
            windows.append((ebird_id, text_label, start, start + segment_length))
    return windows


def heard_windows(track_meta, segment_length):
    """{label: [(start, end)]} from a "Tracks" track's probe_starts and
    knn_starts, None if it has neither"""
    if "probe_starts" not in track_meta and "knn_starts" not in track_meta:
        return None
    heard = {}
    for key in ("probe_starts", "knn_starts"):
        for label, starts in (track_meta.get(key) or {}).items():
            heard.setdefault(label, []).extend(
                (start, start + segment_length) for start in starts
            )
    return heard


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
        secondary_tags=None,
        soft_tags=None,
        overlaps=None,
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
        self.spectrogram_data = None
        self.sr = None
        self.logits = None
        self.embeddings = None
        self.signal_percent = signal_percent
        self.group = group_id
        self.predicted_labels = None
        self.min_freq = min_freq
        self.max_freq = max_freq
        # other birds listed somewhere in the sample's clips e.g. tier1
        # secondary labels, build moves those birdnet doesn't find in the
        # sample's time span to soft_tags
        self.secondary_tags = sorted(secondary_tags or [])
        self.soft_tags = sorted(soft_tags or [])
        # {track id: (start, end, min overlap)} of each other track whose tags
        # the sample has, the part of it in the sample in that track's own file
        # times and the overlap get_samples needed to add its tags
        self.overlaps = dict(overlaps or {})

        if bin_id is None:
            # split by the recording's split key, so recordings sharing it
            # stay together
            self.bin_id = f"{getattr(rec, 'split_key', None) or self.rec_id}"
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
            secondary_tags=self.secondary_tags,
            soft_tags=self.soft_tags,
            overlaps=self.overlaps,
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
    def __init__(self, metadata, filename, config, load_samples=True):
        self.filename = filename
        self.metadata = metadata
        # Train_001 and Train_002 ids clash so are prefixed with the dataset,
        # perchembed's recording_id is the same
        self.id = prefixed_id(filename, metadata.get("id"))
        # recordings with the same key stay in the same train, validation or
        # test set, samples' bin ids start with it
        self.split_key = split_key(filename, metadata)
        self.best_segment = config.best_segment if config is not None else False
        self.presence_overlap = (
            getattr(config, "presence_overlap", None) if config is not None else None
        )
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
        confirmed = (
            config is not None
            and getattr(config, "confirmed_tracks", False)
            and metadata.get(CONFIRMED_KEY) is not None
        )
        originals = {}
        if confirmed:
            # where every bird of the original tracks scored, so samples can
            # be labelled with birds outside the confirmed tracks
            originals = {str(t.get("id")): t for t in tracks_meta}
            tracks_meta = metadata[CONFIRMED_KEY]

        for track in tracks_meta:
            # a confirmed track with no rms data (tier1's 3s windows, cacophony
            # segments merged from several tracks) is already the window to
            # use, one with rms data (other cacophony ones) is tightened by rms
            # as Tracks are
            use_rms = not confirmed or "upper_rms" in track
            t = Track(
                track,
                self.filename,
                self.id,
                self,
                tighten=(
                    (config.tighten_tracks if config is not None else True)
                    and use_rms
                ),
                filter_rms=(
                    (config.filter_rms if config is not None else True) and use_rms
                ),
            )
            if confirmed:
                t.confirmed = True
                if t.best_start is None:
                    # not tightened, it's its own best segment
                    t.best_start = t.start
                    t.best_end = t.end
                original = originals.get(str(t.source_track))
                if original is not None:
                    t.presence = presence_windows(original, config.segment_length)
                    t.heard = heard_windows(original, config.segment_length)
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
        self.samples, self.small_strides, self.unused_samples = self.get_samples(
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
        """An empty copy of this recording: the same id, filename, metadata,
        duration, date, location and best_segment, but always with no tracks,
        no samples and no human tags. Tracks aren't rebuilt from the metadata.

        For AudioDataset.add_sample, which gives each dataset its own copy and
        adds only the original Track objects of the samples it holds (they keep
        their own filename and rec, so tracks merged from other files still
        load their audio from those files), and the tags come from those tracks"""
        no_tracks = {
            k: v for k, v in self.metadata.items() if k not in ("tracks", "Tracks")
        }
        cloned = Recording(no_tracks, self.filename, None, load_samples=False)
        cloned.metadata = self.metadata
        cloned.best_segment = self.best_segment
        cloned.presence_overlap = self.presence_overlap
        # empty from the no_tracks metadata, set so it holds whatever init does
        cloned.tracks = []
        cloned.human_tags = set()
        cloned.samples = []
        return cloned

    def signal_percent(self):
        freq_filter = 1000

        for t in self.tracks:
            signal_time = 0
            signals = 0
            prev_e = None
            # the track's own clip, a merged recording's signals are only the
            # first clip's
            signals_list = t.rec.signals if t.rec is not None else self.signals
            for s in signals_list:
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
        for_label=None,
        extra_samples=True,
    ):
        logging.debug(
            "Getting samples with length: %s stide: %s for: %s extra: %s",
            segment_length,
            segment_stride,
            for_label,
            extra_samples,
        )
        samples = []
        extra_small_strides = []
        unused_samples = []
        if len(self.samples) > 0:
            logging.debug("Loading samples when we already have samples")
        global SAMPLE_GROUP_ID
        SAMPLE_GROUP_ID += 1

        # merged recordings hold tracks from several clip files, each track's
        # start/end is relative to its own file, offset_start moves it onto the
        # original recording's timeline so overlaps are between real times
        def track_offset(t):
            return t.offset_start or 0

        sorted_tracks = sorted(
            self.tracks,
            key=lambda track: track.start + track_offset(track),
        )
        # always take 1 one sample, but dont bother with more if they are short
        # want to sample end of tracks always
        # does this make data unfair for shorter concise tracks

        SEG_LEEWAY = 0.5
        # low_sample_rec = any([True for l in self.human_tags if l in LOW_SAMPLES_LABELS])
        # can be used to seperate among train/val/test
        if for_label is None:
            tracks = self.tracks
        else:
            tracks = [t for t in self.tracks if for_label in t.human_tags]

        tracks = [t for t in tracks if not t.rms_filtered]
        # confirmed tracks' presence windows on the original recording's
        # timeline, once per source track (its confirmed tracks share them):
        # [(filename, offset_start, ebird id, text label, start, end)]
        presence = []
        seen_sources = set()
        for t in self.tracks:
            key = (t.filename, t.offset_start, t.source_track)
            if not t.presence or key in seen_sources:
                continue
            seen_sources.add(key)
            for ebird_id, text_label, w_start, w_end in t.presence:
                presence.append(
                    (
                        t.filename,
                        t.offset_start,
                        ebird_id,
                        text_label,
                        w_start + track_offset(t),
                        w_end + track_offset(t),
                    )
                )
        presence_overlap = (
            self.presence_overlap if self.presence_overlap is not None else 1.5
        )
        # bird tracks first so they own samples shared with overlapping tracks
        tracks = sorted(tracks, key=lambda t: not t.bird_track)
        # samples made so far per pool, to skip duplicates from overlapping tracks
        made = {"main": [], "small": [], "unused": []}

        for track in tracks:
            offset = track_offset(track)
            # which audio this track's samples are, for the duplicate check
            source = (track.filename, track.offset_start)
            # the track's own file, a merged recording's duration is only the
            # first file's
            duration = track.rec.duration if track.rec is not None else self.duration
            # recordings sharing a split key e.g. the same site and device stay
            # in the same one of train, validation, test
            bin_id = self.split_key
            start_stride = segment_stride
            # jitter starts by up to a quarter of the stride either way
            jitter = start_stride / 4

            track_samples = (track.length - segment_length) / segment_stride
            # if track_samples < 1:
            # logging.info("Low track samples so small stride")
            # start_stride = segment_stride / 2
            # allow an extra track with we have over 1/2 segment stride
            track_samples = max(track_samples, 0)
            # seconds of track past the last full stride, before rounding
            left_over = (track_samples - int(track_samples)) * segment_stride
            track_samples = int(round(track_samples)) + 1

            # max_track_length = segment_length + (max_samples - 1) * segment_stride

            sample_starts = (
                np.arange(track.length, step=start_stride, dtype=np.float32)
                + track.start
            )

            max_samples = MAX_TRACK_SAMPLES
            if track_samples > 1:
                sample_starts = (
                    sample_starts
                    + (np.random.rand(len(sample_starts)) * 2 - 1) * jitter
                )
            # drop starts that would make a short sample, before choosing samples
            sample_starts = filter_starts(
                sample_starts,
                track.end,
                segment_length,
                segment_length - SEG_LEEWAY,
                first_min_length=0,
            )
            if len(sample_starts) > max_samples:
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
                np.arange(track.length, step=start_stride, dtype=np.float32)
                + track.start
                + start_stride / 2
            )

            if track_samples > 1:
                small_strides = (
                    small_strides
                    + (np.random.rand(len(small_strides)) * 2 - 1) * jitter
                )
            small_strides = filter_starts(
                small_strides,
                track.end,
                segment_length,
                segment_length - SEG_LEEWAY,
                first_min_length=1.5,
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
            low_sample_track = any(
                [True for l in track.human_tags if l in LOW_SAMPLES_LABELS]
            )
            small_stride = False
            if extra_samples:
                all_starts = [sample_starts, small_strides]
            else:
                all_starts = [sample_starts]
            if self.best_segment and track.best_start is not None:
                # the best segment is the only sample for this track, end is
                # min(best_start + segment_length, track.end) so matches best_end
                sample_starts = np.array([track.best_start])
                selected_samples = sample_starts
                track_samples = 1
                left_over = 0
                all_starts = [sample_starts]

            for starts in all_starts:
                # logging.info(
                #     "Loading from starts %s msmalll stride %s", starts, small_stride
                # )
                # sample_i = 0
                # starts are already filtered by filter_starts so are >= 0 and
                # give samples of the required length
                for start_i, start in enumerate(starts):
                    used_sample = start in selected_samples and not small_stride
                    end = start + segment_length
                    end = min(end, track.end)

                    if (
                        not small_stride
                        and track_samples > 1
                        and start_i == track_samples - 1
                        and 0 < left_over < SEG_LEEWAY
                    ):
                        # always include end, the last full sample is moved to
                        # finish at the track end, single sample tracks are
                        # randomly jittered instead
                        end = track.end
                        start = end - segment_length

                    # load_data always loads segment_length of audio, so expand
                    # short samples here so overlaps include the extra audio
                    missing = segment_length - (end - start)
                    if missing > 0:
                        start = start - np.random.rand() * missing
                        end = start + segment_length
                        if start < 0:
                            start = 0
                            end = segment_length
                        if duration is not None and end > duration:
                            end = duration
                            start = max(0, end - segment_length)

                    min_freq = track.min_freq
                    max_freq = track.max_freq
                    labels = set(track.human_tags)
                    text_labels = set(track.human_text_tags)

                    other_tracks = []
                    overlaps = {}
                    # labels from overlaps this track doesn't hear
                    soft = set()

                    def heard(tag, start=start, end=end):
                        """the probe or knn heard tag in this track's own
                        audio where this sample is"""
                        if track.heard is None:
                            return True
                        return any(
                            min(end, w_end) - max(start, w_start) >= presence_overlap
                            for w_start, w_end in track.heard.get(tag, [])
                        )

                    # the sample on the original recording's timeline
                    rec_start = start + offset
                    rec_end = end + offset
                    for other_track in sorted_tracks:
                        # rms filtered tracks make no samples so shouldn't
                        # label other tracks samples either
                        if track == other_track or other_track.rms_filtered:
                            continue
                        # another file's track is only on the same timeline if
                        # both files have offsets into the original recording
                        if other_track.filename != track.filename and (
                            track.offset_start is None
                            or other_track.offset_start is None
                        ):
                            continue
                        other_offset = track_offset(other_track)
                        other_start = other_track.start + other_offset
                        other_end = other_track.end + other_offset
                        # starts in this sample, sorted by track start and the best
                        # window is inside the track so this still holds for it
                        if other_start > rec_end:
                            break
                        if (
                            self.best_segment
                            and other_track.offset_start is not None
                            and other_track.best_start is not None
                        ):
                            # a clip of a longer recording (tier1), its label only
                            # says the bird is somewhere in the clip, so only
                            # count it if its best window (where its bird is
                            # loudest) is in this sample
                            other_start = other_track.best_start + other_offset
                            other_end = other_track.best_end + other_offset
                        other_length = other_end - other_start
                        overlap = (
                            (rec_end - rec_start)
                            + other_length
                            - (max(rec_end, other_end) - min(rec_start, other_start))
                        )
                        # 0.5s of the other track in this sample, or most of it if
                        # it is shorter than that
                        min_overlap = min(0.5, other_length * 0.9)
                        if other_track.confirmed and self.best_segment:
                            # a confirmed best segment (tier1) is a 3s window
                            # the bird is somewhere in, so it needs as much
                            # overlap as a presence window. Without
                            # best_segment (cacophony) a track is the call
                            min_overlap = min(presence_overlap, other_length * 0.9)

                        # enough overlap or we engulf the track
                        if overlap >= min_overlap:
                            rejected = {
                                tag
                                for tag in other_track.human_tags
                                if tag not in labels and not heard(tag)
                            }
                            if rejected:
                                # this track doesn't hear it, the other track's
                                # bird may just be too far away or not here
                                soft |= rejected
                                labels = labels | (other_track.human_tags - rejected)
                                continue
                            other_tracks.append(other_track)
                            # where, in the other track's own file times, so
                            # it can be checked against that track's windows,
                            # and the overlap it needed
                            overlaps[other_track.id] = (
                                max(rec_start, other_start) - other_offset,
                                min(rec_end, other_end) - other_offset,
                                min_overlap,
                            )
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

                    # birds the probe scored where this sample is, from any
                    # clip on the same timeline, even outside their confirmed
                    # tracks
                    for (
                        p_file,
                        p_offset,
                        ebird_id,
                        text_label,
                        p_start,
                        p_end,
                    ) in presence:
                        if p_file != track.filename and (
                            track.offset_start is None or p_offset is None
                        ):
                            continue
                        shared = min(rec_end, p_end) - max(rec_start, p_start)
                        if shared >= presence_overlap and ebird_id not in labels:
                            if heard(ebird_id):
                                labels = labels | {ebird_id}
                                text_labels = text_labels | {text_label}
                            else:
                                soft.add(ebird_id)

                    other_tracks.append(track)
                    # a secondary bird that is a label of an overlapping
                    # track is already a label
                    secondary_tags = set(track.secondary_tags)
                    for t in other_tracks:
                        if t is track:
                            continue
                        for tag in t.secondary_tags:
                            if heard(tag):
                                secondary_tags.add(tag)
                            else:
                                soft.add(tag)
                    secondary_tags -= labels
                    soft -= labels | secondary_tags
                    # bin_id is always the recording's split key, so a
                    # recording is only ever in one of train, validation, test
                    sample = AudioSample(
                        self,
                        labels,
                        text_labels,
                        start,
                        end,
                        [t.id for t in other_tracks],
                        SAMPLE_GROUP_ID,
                        track.signal_percent,
                        bin_id=bin_id,
                        min_freq=min_freq,
                        max_freq=max_freq,
                        mixed_label=track.mixed_label,
                        low_sample=low_sample_track,
                        secondary_tags=secondary_tags,
                        overlaps=overlaps,
                        soft_tags=soft,
                    )
                    if used_sample:
                        pool = "main"
                    elif small_stride and extra_samples:
                        pool = "small"
                    elif extra_samples:
                        pool = "unused"
                    else:
                        pool = None
                    # overlapping tracks share labels, so their samples over the
                    # overlap are near duplicates, keep the first one made
                    if pool is not None and not is_duplicate_sample(
                        sample, track.id, source, made, pool, segment_length
                    ):
                        made[pool].append((sample, track.id, source))
                        if pool == "main":
                            samples.append(sample)
                            # spares from other tracks made earlier which duplicate
                            # this main sample are no longer needed
                            for spare_pool, spare_list in (
                                ("small", extra_small_strides),
                                ("unused", unused_samples),
                            ):
                                for spare in remove_duplicate_spares(
                                    sample,
                                    track.id,
                                    source,
                                    made,
                                    spare_pool,
                                    segment_length,
                                ):
                                    spare_list.remove(spare)
                        elif pool == "small":
                            extra_small_strides.append(sample)
                        else:
                            unused_samples.append(sample)
                small_stride = True
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
    def __init__(
        self,
        metadata,
        filename,
        rec_id,
        rec,
        segment_length=3,
        tighten=True,
        filter_rms=True,
    ):
        self.rec = rec
        self.filename = filename
        self.rec_id = rec_id

        self.start = metadata["start"]
        self.end = metadata["end"]
        self.og_start = self.start
        self.og_end = self.end
        # where this track's file starts in the original recording, seconds,
        # e.g. tier1 clips cut from a longer recording, None if the file is the
        # recording
        self.offset_start = metadata.get("offset_start")
        self.offset_end = metadata.get("offset_end")
        self.id = metadata.get("id")
        # the "Tracks" track a confirmed track came from, e.g. 31 for
        # 31_secondary
        self.source_track = metadata.get("source_track", self.id)
        # every "Tracks" track a confirmed track was merged from
        self.merged_tracks = metadata.get("merged_tracks") or [self.source_track]
        # [(ebird id, text label, start, end)] where the probe scored a bird
        # of the source track, in this track's file times, set for confirmed
        # tracks
        self.presence = []
        # from "confirmed-tracks", set by Recording
        self.confirmed = False
        # {label: [(start, end)]} windows of this track's own audio where the
        # probe scored the label or knn agreed with it, any label not only the
        # track's tags, set for confirmed tracks. A label from an overlap is
        # only a label if it's heard here, otherwise a soft tag. None checks
        # nothing
        self.heard = None
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
        self.best_start = None
        self.best_end = None
        # ebird ids of other birds somewhere in the clip, e.g. tier1 secondary
        # labels, kept apart as where they are in the clip is unknown
        self.secondary_tags = set()
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
        if tighten or filter_rms:
            self.tighten_track(metadata, segment_length, tighten, filter_rms)

    def tighten_track(self, metadata, segment_length, tighten, filter_rms):

        if not self.bird_track:
            # dont do anything for noisy tracks
            return
        if "upper_rms" not in metadata:
            self.rms_filtered = True and filter_rms
            logging.info(
                "Missing rms %s human tag %s id is %s  filtering? %s",
                self.filename,
                self.human_tags,
                self.id,
                self.rms_filtered,
            )
            return
        # probably not doing anything so put very low
        MIN_STDDEV_PERCENT = 0.01
        if len(self.human_tags) == 0:
            return

        start, end, rms, noise_removed, frame_length, best_offset = rms_best_segment(
            metadata, self.start, self.end, segment_length, centre_bonus(self.filename)
        )
        track_rms = rms[best_offset : best_offset + int(segment_length / frame_length)]
        if tighten:
            self.best_start = start
            self.best_end = end
        # trim track box to where the bird rms is active, any length
        # track_rms only covers the tracks own frequencies so faint calls stand out more
        span_rms = rms
        active_db = RMS_ACTIVE_DB
        if metadata.get("track_rms") is not None:
            box_rms = np.array(metadata["track_rms"])
            if len(box_rms) == len(rms):
                span_rms = box_rms
                active_db = TRACK_RMS_ACTIVE_DB
        active = rms_active_span(
            span_rms, frame_length, ignore=noise_removed, active_db=active_db
        )
        if active is not None:
            active_start, active_end = active
            # only meant to remove loose padding so never trim more than
            # RMS_MAX_TRIM_PERCENT of the track up to RMS_MAX_TRIM from either end
            max_trim = min(
                RMS_MAX_TRIM, (self.og_end - self.og_start) * RMS_MAX_TRIM_PERCENT
            )
            new_start = max(self.og_start, self.og_start + active_start)
            new_start = min(new_start, self.og_start + max_trim)
            new_end = min(self.og_end, self.og_start + active_end)
            new_end = max(new_end, self.og_end - max_trim)
            if new_end > new_start:
                logging.debug(
                    "Track %s %s - %s becomes %s - %s",
                    self.id,
                    self.start,
                    self.end,
                    new_start,
                    new_end,
                )
                self.start = new_start
                self.end = new_end

        std_dev = np.std(track_rms)
        mean = np.mean(track_rms)

        percent_of_mean = std_dev / mean
        if filter_rms and percent_of_mean < MIN_STDDEV_PERCENT:
            logging.error(
                "RMS below std %s percent of mean %s for rec %s track at %s - %s id %s",
                std_dev,
                percent_of_mean,
                self.rec.id if self.rec is not None else "",
                start,
                end,
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
        if tag.get("secondary"):
            self.secondary_tags.add(t.ebird_id)
        elif t.automatic:
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
    #         spectrogram, mel, mfcc, s_data = load_data(start_s, frames, sr)
    #         if spectrogram is None:
    #             continue
    #         sample = AudioSample(
    #             self.rec,
    #             self.human_tags,
    #             start_s,
    #             start_s + SEGMENT_LENGTH,
    #             [self.id],
    #             SAMPLE_GROUP_ID,
    #         )
    #         sample.spectrogram_data = SpectrogramData(
    #             spectrogram,
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
    "SpectrogramData", "raw spectrogram raw_length buttered short_features,mid_features"
)

Tag = namedtuple("Tag", "what ebird_id confidence automatic original")


def load_data(
    config,
    start_s,
    frames,
    sr,
    n_fft=None,
    end=None,
    use_padding=False,
    calc_spec=False,
):
    segment_l = config.segment_length
    segment_stride = config.segment_stride
    hop_length = HOP_LENGTH

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
    if len(frames) < round(segment_l * sr):
        # recording is shorter than a segment, take all of it and pad below
        use_padding = True
        start = 0
        end = len(frames)
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
    # stored peak normalised, as the model is trained and used on
    s_data = normalize_data(s_data)
    spectrogram = None
    if calc_spec:
        normed = s_data
        # mimic tfdataset.raw_to_mel tf.signal.stft, no centering / padding,
        # periodic hann window, frame length = n_fft
        spectrogram = np.abs(
            librosa.stft(
                normed,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window="hann",
                center=False,
            )
        )
    # if buttered is not None:
    #     spectrogram_buttered = np.abs(
    #         librosa.stft(buttered, n_fft=n_fft, hop_length=hop_length)
    #     )
    # else:
    #     spectrogram_buttered = buttered
    spec = SpectrogramData(s_data, spectrogram, data_length, None, short_f, mid_f)
    a_max = np.amax(s_data)
    a_min = np.amin(s_data)
    if a_max == a_min:
        print("Error max is min ", a_max, a_min, start_s, end)
        logging.error(
            "Max  %s min %s start %s end %s data length %s ",
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
    # peak normalize to [-1, 1], keeping 0 as silence
    peak = np.max(np.abs(x), -1, keepdims=True)
    peak = np.maximum(peak, 1e-6)

    return x / peak


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


# frames this many dB above the tracks background level count as active
RMS_ACTIVE_DB = 3
# threshold when using track_rms, which only covers the tracks frequencies
TRACK_RMS_ACTIVE_DB = 3
# pad active span and round out to this resolution, as rms frames (~6ms) are
# finer than we can trust and quiet call onsets can fall below the threshold
RMS_ACTIVE_PAD = 0.2
RMS_ACTIVE_RESOLUTION = 0.1
# seconds rms must stay above threshold to count as active
RMS_MIN_ACTIVE = 0.05
# max seconds trimmed from each end of a track
RMS_MAX_TRIM = 1.0
# max percent of the track trimmed from each end
RMS_MAX_TRIM_PERCENT = 0.25


def rms_active_span(
    rms, frame_length, ignore=None, active_db=RMS_ACTIVE_DB, pad=RMS_ACTIVE_PAD
):
    # returns start/end in seconds relative to rms start of first/last active frame
    rms_db = librosa.amplitude_to_db(np.asarray(rms), ref=1.0)
    if ignore is None:
        ignore = np.zeros(len(rms_db), dtype=bool)
    if ignore.all():
        return None
    floor = np.percentile(rms_db[~ignore], 20)
    above = (rms_db > floor + active_db) & ~ignore
    # only count sound lasting at least RMS_MIN_ACTIVE, narrow band rms
    # fluctuates enough for single frames to cross the threshold
    min_frames = max(1, int(round(RMS_MIN_ACTIVE / frame_length)))
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(np.int8), 0]))
    active = []
    for run_start, run_end in zip(edges[::2], edges[1::2]):
        if run_end - run_start >= min_frames:
            active.append((run_start, run_end))
    if len(active) == 0:
        return None
    active = [active[0][0], active[-1][1] - 1]
    start = active[0] * frame_length - pad
    end = (active[-1] + 1) * frame_length + pad
    start = math.floor(start / RMS_ACTIVE_RESOLUTION) * RMS_ACTIVE_RESOLUTION
    end = math.ceil(end / RMS_ACTIVE_RESOLUTION) * RMS_ACTIVE_RESOLUTION
    return start, end


def samples_match(sample, other, segment_length, offset=0, other_offset=0):
    # same labels and overlapping by more than half a segment, offsets move each
    # sample onto the original recording's timeline
    overlap = min(sample.end + offset, other.end + other_offset) - max(
        sample.start + offset, other.start + other_offset
    )
    return overlap > segment_length / 2 and set(other.tags) == set(sample.tags)


def same_audio(sample, other, segment_length, source, other_source):
    """source is (filename, offset_start). Samples in the same file are compared
    on the file's times. Samples from different files (a merged recording's
    clips) are only compared if both files have offsets, on the original
    recording's times, as overlapping clips hold the same audio. Without offsets
    different files are never duplicates"""
    filename, offset = source
    other_filename, other_offset = other_source
    if filename == other_filename:
        return samples_match(sample, other, segment_length)
    if offset is None or other_offset is None:
        return False
    return samples_match(sample, other, segment_length, offset, other_offset)


def is_duplicate_sample(sample, track_id, source, made, pool, segment_length):
    # duplicate if a sample from another track holding the same audio matches,
    # spare samples also check the main samples
    pools = [pool] if pool == "main" else [pool, "main"]
    for check_pool in pools:
        for other, other_track_id, other_source in made[check_pool]:
            if other_track_id != track_id and same_audio(
                sample, other, segment_length, source, other_source
            ):
                return True
    return False


def remove_duplicate_spares(sample, track_id, source, made, pool, segment_length):
    # remove spares in pool from other tracks which match sample, returns them
    removed = [
        other
        for other, other_track_id, other_source in made[pool]
        if other_track_id != track_id
        and same_audio(sample, other, segment_length, source, other_source)
    ]
    made[pool] = [entry for entry in made[pool] if entry[0] not in removed]
    return removed


def filter_starts(starts, track_end, segment_length, min_length, first_min_length=None):
    # remove sample starts which would give samples shorter than min_length
    # the first start is kept if its at least first_min_length so short tracks
    # still get a sample
    starts = np.maximum(starts, 0)
    lengths = np.minimum(starts + segment_length, track_end) - starts
    keep = lengths >= min_length
    if first_min_length is not None and len(starts) > 0:
        keep[0] = keep[0] or lengths[0] >= first_min_length
    return starts[keep]


def rms_best_segment(
    track_meta, track_start, track_end, segment_length=3, centre_bonus=0
):
    """The segment_length window of a bird track with the most bird rms, once
    rms noise is removed, centred windows preferred by centre_bonus (see
    best_rms). track_meta needs upper_rms, bird_rms and noise_rms starting at
    track_start. Returns start, end, the noise removed rms, which rms frames
    were noise removed, the rms frame length and the window's first frame"""
    rms_thresh = 0.00001
    rms_height = 0.001
    upper_rms = track_meta["upper_rms"]
    rms_hop = track_meta.get("rms_hop_length", 281)
    rms_sr = track_meta.get("rms_sr", 48000)

    upper_peaks, _ = scipy.signal.find_peaks(
        upper_rms, threshold=rms_thresh / 10, height=rms_height / 10, width=2
    )
    rms = np.array(track_meta["bird_rms"])
    noise_rms = track_meta["noise_rms"]
    rms_peaks, rms_meta = scipy.signal.find_peaks(
        rms, threshold=rms_thresh, height=rms_height, width=2
    )
    noise_peaks, noise_meta = scipy.signal.find_peaks(
        noise_rms, threshold=rms_thresh, height=rms_height, width=2
    )
    pre_noise_rms = rms.copy()
    remove_rms_noise(rms, rms_peaks, rms_meta, noise_peaks, noise_meta, upper_peaks)
    # removed noise is filled with the mean, so don't count it as active
    noise_removed = rms != pre_noise_rms

    # rms starts at the original track start
    frame_length = rms_hop / rms_sr
    best_offset, _ = best_rms(rms, segment_length, rms_sr, rms_hop, centre_bonus)
    start = track_start + best_offset * frame_length
    end = min(start + segment_length, track_end)
    return start, end, rms, noise_removed, frame_length, best_offset


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
