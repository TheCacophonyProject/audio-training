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

# from config.config import Config
import numpy as np

from audiodataset import (
    AudioDataset,
    RELABEL,
    Track,
    AudioSample,
    Config,
    LOW_SAMPLES_LABELS as dataset_low_samples,
)
from audiowriter import create_tf_records
import warnings
import math
from pathlib import Path
from collections import Counter
import soundfile as sf

# warnings.filterwarnings("ignore")
# remove librosa pysound warnings
# tensorflow stealing my log handler
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
        root_logger.removeHandler(handler)

MAX_TEST_BINS = None
MAX_TEST_SAMPLES = None
MIN_SAMPLES = 1
MIN_BINS = 1
LOW_SAMPLES_LABELS = ["bittern"]
VAL_PERCENT = 0.15
TEST_PERCENT = 0.05


def split_label(
    dataset, datasets, label, existing_test_count=0, max_samples=None, no_test=False
):
    # split a label from dataset such that vlaidation is 15% or MIN_BINS
    # recs = [r for r in dataset.recs if label in r.human_tags]

    samples_by_bin = {}
    total_tracks = set()
    total_tracks = 0
    sample_bins = set()
    tracks = set()
    num_samples = 0
    rec_by_id = dataset.recs
    for s in dataset.samples:
        if s.rec_id not in rec_by_id:
            logging.error("Could not find %s in dataset", s.rec_id)
            continue
        rec = rec_by_id[s.rec_id]
        if label not in rec.human_tags:
            continue
        # for s in rec.samples:
        if label in s.tags:
            sample_bins.add(s.bin_id)
            tracks = tracks | set(s.track_ids)
            num_samples += 1
        if s.bin_id in samples_by_bin:
            samples_by_bin[s.bin_id].append(s)
        else:
            samples_by_bin[s.bin_id] = [s]
    # sorted as set order changes between runs, which would change the shuffle
    sample_bins = sorted(sample_bins)
    total_tracks = len(tracks)
    # sample_bins = [sample.bin_id for sample in samples]
    if len(sample_bins) == 0:
        return
    # sample_bins duplicates
    # sample_bins = list(set(sample_bins))
    random.shuffle(sample_bins)
    train_c, validate_c, test_c = datasets

    camera_type = "validate"
    add_to = validate_c
    last_index = 0
    label_count = 0
    min_samples = MIN_SAMPLES
    min_bins = MIN_BINS
    total_bins = len(sample_bins)
    if label in LOW_SAMPLES_LABELS or total_bins < 20:
        min_bins = 1
        min_samples = 1

    if label in LOW_SAMPLES_LABELS:
        min_samples = 10
    num_validate_samples = max(num_samples * VAL_PERCENT, min_samples)

    num_test_samples = max(num_samples * TEST_PERCENT, min_samples)
    if MAX_TEST_SAMPLES is not None:
        num_test_samples = min(MAX_TEST_SAMPLES, num_test_samples)
    num_test_samples -= existing_test_count
    # should have test covered by test set
    #  VALIDATION LIMITS

    num_validate_bins = max(total_bins * VAL_PERCENT, min_bins)

    # TEST LIMITS
    num_test_bins = max(total_bins * TEST_PERCENT, min_bins)
    if MAX_TEST_BINS is not None:
        num_test_bins = min(MAX_TEST_BINS, num_test_bins)

    num_test_bins -= existing_test_count

    bin_limit = num_validate_bins
    sample_limit = num_validate_samples
    bins = set()
    print(
        label,
        "looking for val bins",
        num_validate_bins,
        "  out of bins",
        total_bins,
        "and # samples",
        num_validate_samples,
        "from total samples",
        num_samples,
        "# test tracks",
        num_test_bins,
        "# num test samples",
        num_test_samples,
    )
    logging.info("Loading Val data %s with samples %s", label, len(sample_bins))
    recs = set()
    if total_bins > 0:
        for i, sample_bin in enumerate(sample_bins):
            samples = samples_by_bin[sample_bin]
            for sample in samples:
                # not really bins but bins are by bins right now
                bins.add(sample.bin_id)
                label_count += 1
                recs.add(sample.rec_id)
                rec = rec_by_id[sample.rec_id]
                add_to.add_sample(rec, sample)
                dataset.remove(sample)
            samples_by_bin[sample_bin] = []
            last_index = i
            bin_count = len(bins)
            if label_count >= sample_limit and bin_count >= bin_limit:
                # 100 more for test
                if no_test:
                    break
                if add_to == validate_c:
                    add_to = test_c
                    camera_type = "test"
                    if num_test_samples <= 0:
                        break
                    sample_limit = num_test_samples
                    bin_limit = num_test_bins
                    label_count = 0
                    bins = set()
                    logging.info(
                        "Loading Test data %s with leftovers %s",
                        label,
                        len(sample_bins),
                    )

                else:
                    break

        sample_bins = sample_bins[last_index + 1 :]
    logging.info("Loading Train data with leftovers %s", len(sample_bins))

    camera_type = "train"
    added = 0
    for i, sample_bin in enumerate(sample_bins):
        samples = samples_by_bin[sample_bin]
        for sample in samples:
            rec = rec_by_id[sample.rec_id]
            train_c.add_sample(rec, sample)
            dataset.remove(sample)
            added += 1
        samples_by_bin[sample_bin] = []


def get_test_recorder(dataset, test_clips, after_date):
    # load test set camera from tst_clip ids and all clips after a date
    test_c = Recorder("Test-Set-Camera")
    test_samples = [
        sample
        for sample in dataset.samples
        if sample.clip_id in test_clips
        or after_date is not None
        and sample.start_time.replace(tzinfo=pytz.utc) > after_date
    ]
    for sample in test_samples:
        dataset.remove_sample(sample)
        test_c.add_sample(sample)
    return test_c


def split_by_file(dataset, split, test_clips=[], no_test=False):
    split_by_ds = split["recs"]
    datasets = []
    for name in ["train", "validation", "test"]:
        split_clips = split_by_ds[name]
        ds = AudioDataset(name, dataset.config)
        datasets.append(ds)
        logging.info("Loading %s using ids %s", name, split_clips)
        for clip_id in split_clips:
            if clip_id in dataset.recs:
                rec = dataset.recs[clip_id]
                ds.add_recording(rec)
                dataset.remove_rec(clip_id)

    return datasets


def split_randomly(dataset, datasets=None, test_clips=[], no_test=False):
    # split data randomly such that a clip is only in one dataset
    # have tried many ways to split i.e. location and cameras found this is simplest
    # and the results are the same
    if datasets is None:
        train = AudioDataset("train", dataset.config)
        train.enable_augmentation = True
        validation = AudioDataset("validation", dataset.config)
        test = AudioDataset("test", dataset.config)
        datasets = [train, validation, test]
    labels = list(dataset.labels)
    labels.sort()
    for label in labels:
        split_label(
            dataset,
            datasets,
            label,
            no_test=no_test,
            # existing_test_count=existing_test_count,
        )
    return datasets


def dataset_from_signal(args):
    config = Config(**vars(args))

    signal_dir = Path(args.dir)
    sets = ["train", "validation", "test"]
    r_id = 0
    t_id = 0
    dataset_counts = {}
    datesets = []
    all_labels = set()
    for s in sets:
        print("calculating ", s)
        set_dir = signal_dir / s
        dataset = AudioDataset(s, config)
        dataset.load_meta(set_dir)
        for r in dataset.recs.values():
            r_id += 1
            r.id = r_id
            file_name = r.filename.stem
            label_i = file_name.rindex("-")
            label = file_name[:label_i]
            r.human_tags.add(label)
            tags = [{"automatic": False, "what": label}]
            t_id += 1
            t = Track(
                {"id": t_id, "start": 0, "end": None, "tags": tags}, r.filename, r.id, r
            )
            r.tracks.append(t)
            sample = AudioSample(r, r.human_tags, 0, None, [t.id], 1)
            r.samples = [sample]
            dataset.samples.extend(r.samples)
            dataset.labels.add(label)
        dataset.print_counts()
        dataset.print_sample_counts()
        datesets.append(dataset)
        all_labels.update(dataset.labels)
        l_counts = dataset.get_rec_counts()
        # human_counts = l_counts.get("human", [])
        # human_counts = len(human_counts)
        # recs_by_label = {}
        # to_delete = []
        # for r in dataset.recs:
        #
        #     tag = r.tracks[0].tag
        #     if tag not in ["bird", "human"]:
        #         to_delete.append(r)
        #         continue
        #     if tag not in recs_by_label:
        #         recs_by_label[tag] = []
        #     recs_by_label[tag].append(r)

        # bird_recs = recs_by_label.get("bird")
        # random.shuffle(bird_recs)
        # to_remove = bird_recs[human_counts:]
        # to_remove.extend(to_delete)
        # for rec in to_remove:
        #     dataset.remove_rec(rec)
        # just save birds and humans for now and make same count
        dataset.print_counts()
        dataset.print_sample_counts()

    all_labels = list(all_labels)
    all_labels.sort()
    # all_labels = ["bird", "human"]
    for dataset in datesets:
        dataset.labels = all_labels
        dir = signal_dir / "training-data" / dataset.name
        create_tf_records(dataset, dir, dataset.labels, num_shards=100)
        r_counts = dataset.get_rec_counts()
        for k, v in r_counts.items():
            r_counts[k] = len(v)

        dataset_counts[dataset.name] = {
            "rec_counts": r_counts,
            "sample_counts": dataset.get_counts(),
        }
    meta_filename = signal_dir / "training-data" / "training-meta.json"
    meta_data = {
        "labels": all_labels,
        "type": "audio",
        "counts": dataset_counts,
        "by_label": False,
        "relabbled": RELABEL,
    }
    meta_data.update(config.__dict__)

    with open(meta_filename, "w") as f:
        json.dump(meta_data, f, indent=4)


def filter_birds(dataset, args):
    """Removes tracks that birdnetcompare puts in "Tracks with no birdnet tags"
    (no birdnet bird and no clear signal in the best rms window) and that the
    perch embedding analysis also flags. Tracks failing only one check are kept
    and logged so we can decide later how many of those to remove."""
    import csv
    from argparse import Namespace
    from birdnetcompare import NO_TAGS_SECTION, load_taxonomy, row_section, track_rows

    if args.perch_flags is None:
        logging.warning(
            "No --perch-flags so no tracks will be removed, birdnet needs perch to agree"
        )
    perch = {}
    if args.perch_flags is not None:
        with open(args.perch_flags, newline="") as f:
            perch = {row["track_id"]: row for row in csv.DictReader(f)}
        logging.info("Loaded %s perch results from %s", len(perch), args.perch_flags)

    taxonomy = load_taxonomy(Path(__file__).parent / "eBird_taxonomy_v2024.csv")
    compare_args = Namespace(
        min_conf=args.birdnet_min_conf,
        tolerance=0.0,
        segment_length=3,
        clear_signal_db=args.clear_signal_db,
    )

    def perch_details(row):
        return "doubt %s probe own prob %s knn agreement %s (perch says %s / %s)" % (
            row["doubt"],
            row["probe_own_prob"],
            row["knn_agreement"],
            row["probe_label"],
            row["knn_label"],
        )

    # per label counts of what happened to each track
    outcomes = {}
    dataset.samples = []
    for r in dataset.recs.values():
        no_birdnet = set()
        if "birdnet" in r.metadata:
            for row in track_rows(Path(r.filename), r.metadata, taxonomy, compare_args):
                if row_section(row) == NO_TAGS_SECTION:
                    no_birdnet.add(str(row["track_id"]))

        tracks_del = []
        for t in r.tracks:
            track_id = str(t.id)
            perch_row = perch.get(track_id)
            perch_flagged = perch_row is not None and perch_row["flagged"] == "True"
            label = ";".join(sorted(t.human_tags))
            if track_id in no_birdnet and perch_flagged:
                outcome = "removed"
                tracks_del.append(t)
                logging.info(
                    "Removing rec %s track %s %s, no birdnet bird or clear signal and %s",
                    r.id,
                    track_id,
                    label,
                    perch_details(perch_row),
                )
            elif track_id in no_birdnet and perch_row is None:
                outcome = "birdnet, no perch result"
                logging.info(
                    "Keeping rec %s track %s %s, no birdnet bird or clear signal but no perch result",
                    r.id,
                    track_id,
                    label,
                )
            elif track_id in no_birdnet:
                outcome = "birdnet only"
                logging.info(
                    "Keeping rec %s track %s %s, no birdnet bird or clear signal but perch agrees with label, %s",
                    r.id,
                    track_id,
                    label,
                    perch_details(perch_row),
                )
            elif perch_flagged:
                outcome = "perch only"
                logging.info(
                    "Keeping rec %s track %s %s, perch flagged %s",
                    r.id,
                    track_id,
                    label,
                    perch_details(perch_row),
                )
            else:
                outcome = "kept"
            outcomes.setdefault(label, Counter())[outcome] += 1

        if not args.filter_dry_run:
            for t in tracks_del:
                r.tracks.remove(t)
            # recalc_tags only adds so clear tags of removed tracks first
            r.human_tags = set()
            r.recalc_tags()
        r.samples = []
        r.load_samples(dataset.config.segment_length, dataset.config.segment_stride)
        dataset.samples.extend(r.samples)

    columns = ["removed", "birdnet only", "birdnet, no perch result", "perch only"]
    logging.info(
        "%sPer label tracks: %s",
        "DRY RUN, nothing removed. " if args.filter_dry_run else "",
        " / ".join(columns),
    )
    totals = Counter()
    for label, counts in sorted(outcomes.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(counts.values())
        totals.update(counts)
        totals["total"] += total
        logging.info(
            "  %s (%s tracks): %s",
            label,
            total,
            " / ".join(
                f"{counts[c]} ({100 * counts[c] / total:.1f}%)" for c in columns
            ),
        )
    if totals["total"]:
        logging.info(
            "  all (%s tracks): %s",
            totals["total"],
            " / ".join(
                f"{totals[c]} ({100 * totals[c] / totals['total']:.1f}%)"
                for c in columns
            ),
        )
    logging.info("Post filtering counts")
    dataset.print_sample_counts()


def trim_noise(dataset):
    dataset.samples = []
    # set tracks to start at first signal within the track start end and end with last signal
    for r in dataset.recs.values():
        # r.space_signals()
        tracks_del = []
        for t in r.tracks:
            offset = 0
            t_s = None
            t_e = 0
            for s in r.signals:
                if ((t.end - t.start) + (s[1] - s[0])) > max(t.end, s[1]) - min(
                    t.start, s[0]
                ):
                    if t_s is None:
                        t_s = max(t.start, s[0])

                    if t.end < s[1]:
                        t_e = t.end
                        break
                    else:
                        t_e = s[1]
                elif t_s is not None:
                    # Done
                    break
            if t_s is None:
                logging.warn("Rec %s track %s has no signal data", r.id, t.id)
                tracks_del.append(t)
            # r.tracks.remove()
            # print("track ", t.start, t.end, " now has", t_s, t_e, t.human_tags)
            t.start = t_s
            t.end = t_e
        for t in tracks_del:
            r.tracks.remove(t)
        r.recalc_tags()
        r.samples = []
        r.load_samples(dataset.config.segment_length, dataset.config.segment_stride)
        dataset.samples.extend(r.samples)


# Try add some more samples for underpresented labels
# Do this by not limiting long tracks to only 4 samples
# And by allowing samples that start at strides of 0.5 seconds instead of just
# one second
# If still very low will repeat samples
def undersample_ds(dataset):
    lbl_counts = dataset.get_counts()
    # if "bird" in lbl_counts:
    # del lbl_counts["bird"]
    # if "noise" in lbl_counts:
    # del lbl_counts["noise"]
    counts = list(lbl_counts.values())
    counts.sort(reverse=True)
    if len(counts) <= 1:
        return
    target_i = min(len(counts) - 1, 8)
    target_count = counts[target_i]
    target_count = target_count * 3 / 4
    # median = np.mean(counts)

    logging.info("COunts are %s", counts)
    extra_samples = {}
    high_samples = []
    # seeded from the global state so --seed reproduces it
    rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))

    for lbl, count in lbl_counts.items():
        extra_samples[lbl] = 0
        print(lbl, "Count ", count, target_count)
        if count > target_count:
            extra_samples[lbl] = count - target_count
            high_samples.append(lbl)
    logging.info(
        "Remove extra samples for %s target count %s",
        extra_samples,
        target_count,
    )
    # dataset.samples = []
    for lbl in high_samples:
        remove_chance = extra_samples[lbl] / lbl_counts[lbl]
        # print(f"Chance to remove {lbl} is {remove_chance}")
        recs = list(dataset.recs.values())
        random.shuffle(recs)
        total_bird = 0

        for rec in recs:
            samples_by_track = {}
            # should try remove samples from tracks with multiple samples
            rec_samples = []
            for sample in rec.samples:
                if lbl in sample.tags:
                    total_bird += 1
                    track_samples = samples_by_track.setdefault(sample.track_id, [])
                    track_samples.append(sample)
                    chance = rng.random()
                    # print("Checking against ",chance)
                    if chance < remove_chance:
                        # print(f"Remoing {lbl} sample")
                        dataset.remove(sample)
                    else:
                        rec_samples.append(sample)
                else:
                    rec_samples.append(sample)

            rec.samples = rec_samples
        print(f"Total {lbl} samples are {total_bird}")


# Try add some more samples for underpresented labels
# Do this by not limiting long tracks to only 4 samples
# And by allowing samples that start at strides of 0.5 seconds instead of just
# one second
# If still very low will repeat samples
def oversample_ds(original_ds, dataset, max_repeats=1):
    lbl_counts = dataset.get_counts()

    if "bird" in lbl_counts:
        del lbl_counts["bird"]
    if "noise" in lbl_counts:
        del lbl_counts["noise"]
    counts = list(lbl_counts.values())
    counts.sort(reverse=True)
    if len(counts) <= 1:
        return
    target_i = min(len(counts) - 1, 8)
    target_count = counts[target_i]
    # median = np.mean(counts)

    logging.info("COunts are %s", counts)
    extra_samples = {}
    low_samples = []
    for lbl, count in lbl_counts.items():
        extra_samples[lbl] = 0
        if count < target_count:
            extra_samples[lbl] = target_count - count
            low_samples.append(lbl)
    logging.info(
        "Try get extra samples for %s target count %s",
        extra_samples,
        target_count,
    )
    # dataset.samples = []
    for lbl in low_samples:
        unused_samples = {}
        small_stride_samples = {}
        used_samples = {}
        for clip_id, rec in original_ds.recs.items():
            if rec.id not in dataset.recs:
                continue
            for sample in rec.samples:
                if lbl in sample.tags:
                    used_samples[sample.id] = sample
            for unused in rec.unused_samples:
                if lbl in unused.tags:
                    unused_samples[unused.id] = unused
            for unused in rec.small_strides:
                if lbl in unused.tags:
                    small_stride_samples[unused.id] = unused
        # np.random.shuffle(extra_samples)
        logging.info(
            "Unused samples %s length %s missing %s",
            lbl,
            len(unused_samples),
            extra_samples[lbl],
        )
        selected_samples = np.random.choice(
            list(unused_samples.values()),
            int(min(len(unused_samples), extra_samples[lbl])),
            replace=False,
        )
        extra_samples[lbl] -= len(selected_samples)
        logging.info("Adding unused %s", len(selected_samples))

        for sample in selected_samples:
            sample.low_sample = True
            rec = original_ds.recs[sample.rec_id]
            rec.unused_samples.remove(sample)

            dataset.recs[sample.rec_id].samples.append(sample)
            dataset.samples.append(sample)

        # small stride smaples
        selected_samples = np.random.choice(
            list(small_stride_samples.values()),
            int(min(len(small_stride_samples), extra_samples[lbl])),
            replace=False,
        )
        extra_samples[lbl] -= len(selected_samples)
        logging.info("Adding small stride %s", len(selected_samples))
        for sample in selected_samples:
            sample.low_sample = True
            rec = original_ds.recs[sample.rec_id]
            rec.small_strides.remove(sample)

            dataset.recs[sample.rec_id].samples.append(sample)
            dataset.samples.append(sample)

        extra_samples[lbl] -= len(selected_samples)

        if extra_samples[lbl] > target_count / 2:
            repeat_samples = []
            repeat_small_strides = []
            repeat_unused_samples = []
            for rec in dataset.recs.values():
                if lbl not in rec.human_tags:
                    continue
                (samples, small_strides, unused_samples) = rec.get_samples(
                    dataset.config.segment_length,
                    dataset.config.segment_stride,
                    for_label=lbl,
                )
                repeat_samples.extend(samples)
                repeat_unused_samples.extend(unused_samples)
                repeat_small_strides.extend(small_strides)
                # logging.info(
                #     "Loaded extra sets for %s got %s and %s and %s",
                #     lbl,
                #     len(samples),
                #     len(small_strides),
                #     len(unused_samples),
                # )
            sample_index = 0

            sample_sets = [repeat_samples, repeat_small_strides, repeat_unused_samples]
            repeat = 0
            if len(repeat_samples) == 0:
                continue
            while extra_samples[lbl] >= 1 and (
                max_repeats is None or repeat / 3 < max_repeats
            ):
                logging.info("Running on %s repeat %s", lbl, repeat)
                sample_index = repeat % 3
                sample_set = sample_sets[sample_index]
                repeat += 1
                selected_samples = np.random.choice(
                    list(sample_set),
                    int(min(len(sample_set), extra_samples[lbl])),
                    replace=False,
                )
                extra_samples[lbl] -= len(selected_samples)
                logging.info(
                    "Adding %s for %s from sample index set %s missing %s",
                    len(selected_samples),
                    lbl,
                    sample_index,
                    extra_samples[lbl],
                )
                for sample in selected_samples:
                    sample.low_sample = True
                    dataset.recs[sample.rec_id].samples.append(sample)
                    dataset.samples.append(sample)


def main():
    init_logging()
    args = parse_args()
    if args.seed is None:
        # still pick and log a seed so any build can be reproduced
        args.seed = random.SystemRandom().randint(0, 2**31 - 1)
    logging.info("Using seed %s", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # print(args, args.__dict__)
    config = Config(**vars(args))
    # SEGMENT_LENGTH = args.seg_length
    # SEGMENT_STRIDE = args.stride
    # HOP_LENGTH = args.hop_length
    # BREAK_FREQ = args.break_freq
    # HTK = not args.slaney
    # FMIN = args.fmin
    # FMAX = args.fmax
    # N_MELS = args.mels
    if args.signal:
        dataset_from_signal(args)
        return
    # config = load_config(args.config_file)
    dataset = AudioDataset("all", config)
    dataset.load_meta(args.dir)
    if args.plot_signal:
        logging.info("Plotting signals")
        from otherdata import plot_signal

        plot_signal(dataset, Path(args.dir))
        return

    dataset.print_sample_counts()

    if args.filter_birds:
        filter_birds(dataset, args)
    # for r in dataset.recs:
    #     if "whistler" not in r.human_tags:
    #         print(r.id, " missing", r.human_tags)
    # trim_noise(dataset)
    # return
    # dataset.load_meta()
    # return
    datasets = None
    if args.split_file:
        logging.info("Splitting by %s", args.split_file)
        with open(args.split_file, "r") as t:
            # add in some metadata stats
            meta = json.load(t)
        datasets = split_by_file(dataset, meta)
        # else:
        logging.info("Remaining samples")
        dataset.print_sample_counts()

    logging.info("Splitting randomly")
    datasets = split_randomly(dataset, datasets=datasets, no_test=args.no_test)
    logging.info("Remaining samples")
    dataset.print_sample_counts()

    all_labels = set()
    for d in datasets:
        logging.info("")
        logging.info("%s Dataset", d.name)
        d.print_sample_counts()

        all_labels.update(d.labels)

    # for rec in dataset.recs.values():
    #     for s in rec.samples:
    #         logging.info("USed samples are %s", s)

    #     for unused in rec.unused_samples:
    #         logging.info("Not Used samples are %s", unused)
    if args.balance:
        logging.info("Balancing datasets")
        undersample_ds(datasets[0])
        undersample_ds(datasets[1])

        oversample_ds(dataset, datasets[0], max_repeats=5)
        oversample_ds(dataset, datasets[1])

        logging.info("After balance")
        for d in datasets[:2]:
            logging.info("")
            logging.info("%s Dataset", d.name)
            d.print_sample_counts()

            all_labels.update(d.labels)
    all_labels = list(all_labels)
    all_labels.sort()
    for d in datasets:
        d.labels = all_labels
        print("setting all labels", all_labels)
    validate_datasets(datasets)
    base_dir = args.out_dir
    if args.create_signal_wavs:
        record_dir = os.path.join(base_dir, "signal-data/")
        for dataset in datasets:
            dir = os.path.join(record_dir, dataset.name)

            print("Saving signal")
            create_signal_data(dataset, Path(dir), datasets[0].labels)
            # r_counts = dataset.get_rec_counts()
        return
    record_dir = os.path.join(base_dir, "training-data/")
    print("saving to", record_dir)
    # return
    dataset_counts = {}
    dataset_recs = {}
    for dataset in datasets:
        dir = os.path.join(record_dir, dataset.name)
        r_counts = dataset.get_rec_counts()
        for k, v in r_counts.items():
            r_counts[k] = len(v)
        dataset_recs[dataset.name] = list(dataset.recs.keys())
        dataset_counts[dataset.name] = {
            "rec_counts": r_counts,
            "sample_counts": dataset.get_counts(),
        }
        create_tf_records(dataset, dir, datasets[0].labels, num_shards=100)

        # dataset.saveto_numpy(os.path.join(base_dir))
    # dont need dataset anymore just need some meta
    meta_filename = f"{base_dir}/training-data/training-meta.json"
    meta_data = {
        # "segment_length": SEGMENT_LENGTH,
        # "segment_stride": SEGMENT_STRIDE,
        # "hop_length": HOP_LENGTH,
        # "n_mels": N_MELS,
        # "fmin": FMIN,
        # "fmax": FMAX,
        # "break_freq": BREAK_FREQ,
        # "htk": HTK,
        "labels": datasets[0].labels,
        "type": "audio",
        "counts": dataset_counts,
        "by_label": False,
        "relabbled": RELABEL,
    }
    meta_data.update(config.__dict__)
    with open(meta_filename, "w") as f:
        json.dump(meta_data, f, indent=4)

    # recording ids in each dataset, kept under "recs" so this file can be
    # passed straight to --split-file to rebuild the same split
    split_filename = f"{base_dir}/training-data/training-split.json"
    with open(split_filename, "w") as f:
        json.dump({"seed": config.seed, "recs": dataset_recs}, f, indent=4)


def validate_datasets(datasets):
    train, validation, test = datasets
    train_tracks = [s.bin_id for s in train.samples]
    val_tracks = [s.bin_id for s in validation.samples]
    test_tracks = [s.bin_id for s in test.samples]

    for t in train_tracks:
        assert t not in val_tracks and t not in test_tracks

    for t in val_tracks:
        assert t not in test_tracks

    #  make sure all tags from a recording are only in one dataset
    train_tracks = [f"{s.rec_id}" for s in train.samples if not s.low_sample]
    val_tracks = [f"{s.rec_id}" for s in validation.samples if not s.low_sample]
    test_tracks = [f"{s.rec_id}" for s in test.samples if not s.low_sample]
    for t in train_tracks:
        assert t not in val_tracks and t not in test_tracks

    for t in val_tracks:
        assert t not in test_tracks


def create_signal_data(dataset, output_path, labels):
    if output_path.is_dir():
        logging.info("Clearing dir %s", output_path)
        for child in output_path.glob("*"):
            if child.is_file():
                child.unlink()
    output_path.mkdir(parents=True, exist_ok=True)
    recs = dataset.recs.values()
    np.random.shuffle(recs)
    audio_data = {}
    print("recs are", len(recs))
    sr = 48000
    for r in recs:
        r.space_signals()
        loaded = r.load_recording(resample=sr)
        if not loaded:
            continue
        for t in r.tracks:
            track_data = []
            # print("Checking", t, r.signals)
            for s in r.signals:
                if ((t.end - t.start) + (s[1] - s[0])) > max(t.end, s[1]) - min(
                    t.start, s[0]
                ):
                    # print("signal at ", t.start, t.end, s)
                    pre_sig = s[0] - t.start
                    t_e = min(s[1], t.end) * sr
                    t_s = max(s[0], t.start) * sr
                    t_e = math.ceil(t_e)
                    t_s = math.floor(t_s)
                    # print("getting data from", len(r.rec_data), t_s, t_e)
                    track_data.extend(r.rec_data[t_s:t_e])
                elif s[0] > t.start:
                    break
            key = t.tags_key
            if key in audio_data:
                offset = len(audio_data[key][1])
                audio_data[key][1].extend(track_data)
                meta = audio_data[key][2]["recs"]
                rec_meta = meta.setdefault(r.id, {})
                rec_meta[t.id] = [offset, offset + len(track_data)]
            else:
                audio_data[key] = (
                    1,
                    track_data,
                    {
                        "recs": {r.id: {t.id: [0, len(track_data)]}},
                    },
                )
        r.rec_data = None
        # print("adding data", len(track_data), key)
        save_data(audio_data, output_path, min_seconds=10)
    save_data(audio_data, output_path, min_seconds=None)


def save_data(audio_data, output_dir, sr=48000, min_seconds=10):
    for l in audio_data.keys():
        data = audio_data[l]
        if len(data) == 0:
            continue
        if min_seconds is None or len(data[1]) > sr * min_seconds:
            name = output_dir / f"{l}-{data[0]}.wav"
            sf.write(str(name), data[1], sr)
            name = output_dir / f"{l}-{data[0]}.txt"

            with open(name, "w") as f:
                json.dump(data[2], f, indent=4)
            print("Saving", name)
            # data[0] += 1
            # data[1] = []
            # data[2] = {"recs": {}}
            audio_data[l] = (data[0] + 1, [], {"recs": {}})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="Dir to load")
    parser.add_argument("--no-test", action="count", help="NO test set")
    parser.add_argument("--signal", action="count", help="Load signal data")
    parser.add_argument(
        "--create-signal-wavs", action="count", help="Create signal wavs"
    )

    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument("-m", "--mels", default=160, help="Number of mels to use")
    parser.add_argument("-b", "--break-freq", default=1000, help="Break freq to use")
    parser.add_argument(
        "--slaney", action="count", help="Use slaney or htk (htk for custom break freq)"
    )
    parser.add_argument("--hop-length", default=281, help="Number of hops to use")
    parser.add_argument("--fmin", default=100, help="Min freq")
    parser.add_argument("--fmax", default=11000, help="Max Freq")
    parser.add_argument(
        "--seg-length", default=3, type=int, help="Segment length in seconds"
    )
    parser.add_argument("--stride", default=1, type=float, help="Segment stride")
    parser.add_argument(
        "out_dir", default="/data/audio-data", help="Directory to place files in"
    )
    parser.add_argument(
        "--filter-freq",
        default=True,
        action="count",
        help="Filter frequency of tracks",
    )
    parser.add_argument(
        "--dont-tighten-tracks",
        default=False,
        action="store_true",
        help="Dont Tighten tracks to the best 3 seconds",
    )
    parser.add_argument(
        "--dont-filter-rms",
        default=False,
        action="store_true",
        help="Filter tracks which the rms isnt high enogh",
    )

    parser.add_argument(
        "--balance",
        default=False,
        action="count",
        help="Try balance labels by oversampling low sample classess (doesn't seem to be effective)",
    )
    parser.add_argument(
        "--plot-signal",
        default=False,
        action="count",
        help="Plot signal percent",
    )

    parser.add_argument(
        "--filter-birds",
        action="store_true",
        help="Remove tracks with no birdnet bird or clear signal that perch also flags",
    )
    parser.add_argument(
        "--filter-dry-run",
        action="store_true",
        help="With --filter-birds only log what would be removed",
    )
    parser.add_argument(
        "--perch-flags",
        default=None,
        help="perch-flags.csv from perchembed.py --analyse",
    )
    parser.add_argument(
        "--birdnet-min-conf",
        type=float,
        default=0.1,
        help="Ignore birdnet detections below this confidence",
    )
    parser.add_argument(
        "--clear-signal-db",
        type=float,
        default=12,
        help="Minimum signal strength (db above background) for a clear signal",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random generation, saved in training-meta.json. A random "
        "seed is picked and logged if not given",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Split the dataset using clip ids specified in this file",
    )
    args = parser.parse_args()
    args.dir = Path(args.dir)
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
