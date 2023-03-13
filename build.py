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

from audiodataset import AudioDataset, RELABEL, SEGMENT_LENGTH, SEGMENT_STRIDE
from audiowriter import create_tf_records
import warnings

# warnings.filterwarnings("ignore")
# remove librosa pysound warnings

MAX_TEST_TRACKS = 10
MAX_TEST_SAMPLES = 10
MIN_SAMPLES = 10
MIN_TRACKS = 10
LOW_SAMPLES_LABELS = []


def split_label(
    dataset, datasets, label, existing_test_count=0, max_samples=None, no_test=False
):
    # split a label from dataset such that vlaidation is 15% or MIN_TRACKS
    # recs = [r for r in dataset.recs if label in r.human_tags]

    samples_by_bin = {}
    total_tracks = set()
    total_tracks = 0
    sample_bins = set()
    tracks = set()
    num_samples = 0
    for s in dataset.samples:
        if label not in s.rec.human_tags:
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
    sample_bins = list(sample_bins)
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
    min_t = MIN_SAMPLES

    if label in LOW_SAMPLES_LABELS:
        min_t = 10
    num_validate_samples = max(num_samples * 0.15, min_t)
    num_test_samples = (
        min(MAX_TEST_SAMPLES, max(num_samples * 0.05, min_t)) - existing_test_count
    )
    # should have test covered by test set

    min_t = MIN_TRACKS

    if label in LOW_SAMPLES_LABELS or total_tracks < 20:
        min_t = 1

    num_validate_tracks = max(total_tracks * 0.15, min_t)
    num_test_tracks = (
        min(MAX_TEST_TRACKS, max(total_tracks * 0.05, min_t)) - existing_test_count
    )
    track_limit = num_validate_tracks
    sample_limit = num_validate_samples
    tracks = set()
    print(
        label,
        "looking for val tracks",
        num_validate_tracks,
        "  out of tracks",
        total_tracks,
        "and # samples",
        num_validate_samples,
        "from total samples",
        num_samples,
        "# test tracks",
        num_test_tracks,
        "# num test samples",
        num_test_samples,
    )
    recs = set()
    if total_tracks > 5:
        for i, sample_bin in enumerate(sample_bins):
            samples = samples_by_bin[sample_bin]
            for sample in samples:
                # not really tracks but bins are by tracks right now
                tracks.add(sample.bin_id)
                label_count += 1
                recs.add(sample.rec_id)
                add_to.add_sample(sample)
                dataset.remove(sample)
            samples_by_bin[sample_bin] = []
            last_index = i
            track_count = len(tracks)
            if label_count >= sample_limit and track_count >= track_limit:
                # 100 more for test
                if no_test:
                    break
                if add_to == validate_c:
                    add_to = test_c
                    camera_type = "test"
                    if num_test_samples <= 0:
                        break
                    sample_limit = num_test_samples
                    track_limit = num_test_tracks
                    label_count = 0
                    tracks = set()

                else:
                    break

        sample_bins = sample_bins[last_index + 1 :]

    camera_type = "train"
    added = 0
    for i, sample_bin in enumerate(sample_bins):
        samples = samples_by_bin[sample_bin]
        for sample in samples:
            train_c.add_sample(sample)
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


def split_randomly(dataset, test_clips=[], no_test=False):
    # split data randomly such that a clip is only in one dataset
    # have tried many ways to split i.e. location and cameras found this is simplest
    # and the results are the same
    train = AudioDataset("train")
    train.enable_augmentation = True
    validation = AudioDataset("validation")
    test = AudioDataset("test")

    for label in dataset.labels:
        split_label(
            dataset,
            (train, validation, test),
            label,
            no_test=no_test
            # existing_test_count=existing_test_count,
        )

    return train, validation, test


def main():
    init_logging()
    args = parse_args()
    # config = load_config(args.config_file)
    dataset = AudioDataset("all")
    dataset.load_meta(args.dir)
    # dataset.load_meta()
    dataset.print_counts()
    datasets = split_randomly(dataset, no_test=args.no_test)
    dataset.print_counts()
    all_labels = set()
    for d in datasets:
        logging.info("%s Dataset", d.name)
        d.print_sample_counts()

        all_labels.update(d.labels)
    all_labels = list(all_labels)
    all_labels.sort()
    for d in datasets:
        d.labels = all_labels
        print("setting all labels", all_labels)
    validate_datasets(datasets)
    base_dir = "."
    record_dir = os.path.join(base_dir, "training-data/")
    print("saving to", record_dir)
    dataset_counts = {}
    for dataset in datasets:
        dir = os.path.join(record_dir, dataset.name)
        create_tf_records(dataset, dir, datasets[0].labels, num_shards=100)
        r_counts = dataset.get_rec_counts()
        for k, v in r_counts.items():
            r_counts[k] = len(v)

        dataset_counts[dataset.name] = {
            "rec_counts": r_counts,
            "sample_counts": dataset.get_counts(),
        }

        # dataset.saveto_numpy(os.path.join(base_dir))
    # dont need dataset anymore just need some meta
    meta_filename = f"{base_dir}/training-data/training-meta.json"
    meta_data = {
        "segment_length": SEGMENT_LENGTH,
        "segment_stride": SEGMENT_STRIDE,
        "labels": datasets[0].labels,
        "type": "audio",
        "counts": dataset_counts,
        "by_label": False,
        "relabbled": RELABEL,
    }
    with open(meta_filename, "w") as f:
        json.dump(meta_data, f, indent=4)


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
    train_tracks = [f"{s.bin_id}" for s in train.samples]
    val_tracks = [f"{s.bin_id}" for s in validation.samples]
    test_tracks = [f"{s.bin_id}" for s in test.samples]
    for t in train_tracks:
        assert t not in val_tracks and t not in test_tracks

    for t in val_tracks:
        assert t not in test_tracks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="Dir to load")
    parser.add_argument("--no-test", action="count", help="NO test set")

    parser.add_argument("-c", "--config-file", help="Path to config file to use")
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
