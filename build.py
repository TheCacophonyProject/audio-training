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

from audiodataset import AudioDataset, RELABEL, Track, AudioSample, Config
from audiowriter import create_tf_records
import warnings
import math
from pathlib import Path
import soundfile as sf

# warnings.filterwarnings("ignore")
# remove librosa pysound warnings

MAX_TEST_BINS = None
MAX_TEST_SAMPLES = None
MIN_SAMPLES = 10
MIN_BINS = 10
LOW_SAMPLES_LABELS = []
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
    rec_by_id = {}
    for r in dataset.recs:
        rec_by_id[r.id] = r
    for s in dataset.samples:
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
    num_validate_samples = max(num_samples * VAL_PERCENT, min_t)
    num_test_samples = max(num_samples * TEST_PERCENT, min_t)
    if MAX_TEST_SAMPLES is not None:
        num_test_samples = min(MAX_TEST_SAMPLES, num_test_samples)
    num_test_samples -= existing_test_count
    # should have test covered by test set
    #  VALIDATION LIMITS
    min_t = MIN_BINS
    total_bins = len(sample_bins)
    if label in LOW_SAMPLES_LABELS or total_bins < 20:
        min_t = 1

    num_validate_bins = max(total_bins * VAL_PERCENT, min_t)

    # TEST LIMITS
    num_test_bins = max(total_bins * TEST_PERCENT, min_t)
    if MAX_TEST_BINS is not None:
        num_test_bins = min(MAX_TEST_BINS, num_test_bins)

    num_test_bins -= existing_test_count
    if label == "rifleman":
        num_validate_bins = 2
        num_validate_samples = 2
        num_test_bins = 1
        num_test_samples = 1

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
    recs = set()
    if total_bins > 5:
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

                else:
                    break

        sample_bins = sample_bins[last_index + 1 :]

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


def split_randomly(dataset, test_clips=[], no_test=False):
    # split data randomly such that a clip is only in one dataset
    # have tried many ways to split i.e. location and cameras found this is simplest
    # and the results are the same
    train = AudioDataset("train", dataset.config)
    train.enable_augmentation = True
    validation = AudioDataset("validation", dataset.config)
    test = AudioDataset("test", dataset.config)

    for label in dataset.labels:
        split_label(
            dataset,
            (train, validation, test),
            label,
            no_test=no_test
            # existing_test_count=existing_test_count,
        )

    return train, validation, test


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
        for r in dataset.recs:
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


def filter_birds(dataset):
    dataset.samples = []
    freq_filter = 1000
    logging.info("Filtering unclear birds")
    # set tracks to start at first signal within the track start end and end with last signal
    for r in dataset.recs:
        # r.space_signals()
        tracks_del = []
        for t in r.tracks:
            offset = 0
            if "bird" not in t.human_tags:
                continue
            signal_time = 0
            signals = 0
            prev_e = None
            for s in r.signals:
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
                    # logging.info(
                    #     "Adding singal %s for track %s-%s overlap signal time is %s",
                    #     s,
                    #     t.start,
                    #     t.end,
                    #     signal_time,
                    # )
                    prev_e = end
                    if t.end < s[1]:
                        break
                if t.end < s[0]:
                    break
            # logging.info(
            #     "Total signals %s total signal time is %s for a track starting at  %s - %s percent signal %s",
            #     signals,
            #     signal_time,
            #     t.start,
            #     t.end,
            #     round(100 * signal_time / t.length),
            # )
            signal_percent = signal_time / t.length
            if signal_percent < 0.1:
                logging.warn(
                    "Filtering rec %s track %s ( At %s) because has signal time %s from %s signals",
                    r.id,
                    t.id,
                    t.start,
                    signal_percent,
                    signals,
                )
            # if t_s is None:
            #     logging.warn("Rec %s track %s has no signal data", r.id, t.id)
            #     tracks_del.append(t)

        for t in tracks_del:
            r.tracks.remove(t)
        r.recalc_tags()
        r.samples = []
        r.load_samples(dataset.config.segment_length, dataset.config.segment_stride)
        dataset.samples.extend(r.samples)


def trim_noise(dataset):
    dataset.samples = []
    # set tracks to start at first signal within the track start end and end with last signal
    for r in dataset.recs:
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


def main():
    init_logging()
    args = parse_args()
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
    filter_birds(dataset)
    return
    # for r in dataset.recs:
    #     if "whistler" not in r.human_tags:
    #         print(r.id, " missing", r.human_tags)
    # trim_noise(dataset)
    # return
    # dataset.load_meta()
    # return
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
    base_dir = "./yamnet-data"
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


def create_signal_data(dataset, output_path, labels):
    if output_path.is_dir():
        logging.info("Clearing dir %s", output_path)
        for child in output_path.glob("*"):
            if child.is_file():
                child.unlink()
    output_path.mkdir(parents=True, exist_ok=True)
    recs = dataset.recs
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
    parser.add_argument("-m", "--mels", default=120, help="Number of mels to use")
    parser.add_argument("-b", "--break-freq", default=1750, help="Break freq to use")
    parser.add_argument(
        "--slaney", action="count", help="Use slaney or htk (htk for custom break freq)"
    )
    parser.add_argument("--hop-length", default=281, help="Number of hops to use")
    parser.add_argument("--fmin", default=50, help="Min freq")
    parser.add_argument("--fmax", default=11000, help="Max Freq")
    parser.add_argument("--seg-length", default=3, help="Segment length in seconds")
    parser.add_argument("--stride", default=1, help="Segment stride")

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
