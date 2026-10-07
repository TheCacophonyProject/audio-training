import sys
import math
import argparse
import matplotlib.pyplot as plt

import tensorflow as tf
from functools import partial
import numpy as np
import time
import json
import logging
import librosa
import librosa.display
from custommel import mel_f
from pathlib import Path
from birdsconfig import (
    ANIMAL_LABELS,
    ALL_BIRDS,
    RELABEL_MAP,
    BIRD_TRAIN_LABELS,
    EXTRA_LABELS,
    OTHER_LABELS,
    HUMAN_LABELS,
    NOISE_LABELS,
)


BIRD_PATH = []
NOISE_PATH = []
NZ_BOX = [166.509144322, -34.4506617165, 178.517093541, -46.641235447]

AUTOTUNE = tf.data.AUTOTUNE


insect = None
fp = None
HOP_LENGTH = 281
N_MELS = 160
SR = 48000
BREAK_FREQ = 1000
NFFT = 4096
MEL_WEIGHTS = mel_f(48000, N_MELS, 500, 11000, NFFT, BREAK_FREQ)
MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)

FMIN = 100
FMAX = 11000

MOREPORK_MAX = 1200


def set_merge_labels(new_merge):
    global RELABEL_MAP
    RELABEL_MAP = new_merge
    logging.info("Set merge %s", new_merge)


def set_specific_by_count(meta):
    counts = meta["counts"]
    training = counts["train"]["sample_counts"]
    training_rec = counts["train"]["rec_counts"]

    validation = counts["validation"]["sample_counts"]
    if "ebird_ids" in meta:
        ebird_ids = meta["ebird_ids"]
    else:
        ebird_ids = meta["labels"]
    # just for until dataset is build with ebird ids
    # ebird_id_to_labels = {}
    # for label, ebird_id in zip(meta["labels"], ebird_ids):
    #     if ebird_id in ebird_id_to_labels:
    #         ebird_id_to_labels[ebird_id].append(label)
    #     else:
    #         ebird_id_to_labels[ebird_id] = [label]
    # #        logging.info("Adding %s for %s",ebird_id,label)

    dataset_lbls = ["train samples", "rec counts", "validation sample"]
    count_dics = [training, training_rec, validation]
    # for v, lbl_list in ebird_id_to_labels.items():
    #     for k in lbl_list:
    #         for dataset_lbl, dataset in zip(dataset_lbls, count_dics):
    #             if k in dataset:
    #                 if v not in dataset:
    #                     dataset[v] = 0
    #                 total_count = dataset[k]
    #                 if v in dataset:
    #                     total_count += dataset[v]
    #                 dataset[k] = total_count
    #                 logging.info(
    #                     "%s - Adding samples of %s to  %s for a total of %s",
    #                     dataset_lbl,
    #                     k,
    #                     v,
    #                     total_count,
    #                 )

    #                 if v in dataset:
    #                     dataset[v] = total_count
    #                 logging.info(
    #                     "%s Setting total of %s to %s", dataset_lbl, v, total_count
    #                 )

    # set counts to be counts of all merged labels
    for k, v in RELABEL_MAP.items():
        for dataset_lbl, dataset in zip(dataset_lbls, count_dics):
            if k in dataset:
                if v not in dataset:
                    dataset[v] = 0
                total_count = dataset[k]
                if v in dataset:
                    total_count += dataset[v]
                dataset[k] = total_count
                logging.info(
                    "%s - Adding samples of %s to  %s for a total of %s",
                    dataset_lbl,
                    k,
                    v,
                    total_count,
                )

                if v in dataset:
                    dataset[v] = total_count
                    logging.info(
                        "%s Setting total of %s to %s", dataset_lbl, v, total_count
                    )

    labels_with_data = []
    for label, count in training.items():
        rec_count = training_rec[label]
        if label not in validation:
            continue
        val_count = validation[label]
        if count > 50 and rec_count > 50 and val_count > 2:
            labels_with_data.append(label)
            if label not in ALL_BIRDS and label not in RELABEL_MAP:
                logging.info("Have data for %s but not included ", label)
            if label in ALL_BIRDS and label not in BIRD_TRAIN_LABELS:
                BIRD_TRAIN_LABELS.append(label)
                logging.info(
                    "Using %s because have data samples: %s and recs %s val samples %s:",
                    label,
                    count,
                    rec_count,
                    not val_count,
                )


def get_excluded_labels(labels):
    excluded_labels = []
    for l in labels:
        if l in NOISE_LABELS and "noise" in labels:
            continue
        elif l in HUMAN_LABELS and "human" in labels:
            continue
        elif l in OTHER_LABELS and "other" in labels:
            continue
        elif l not in BIRD_TRAIN_LABELS and l not in EXTRA_LABELS:
            excluded_labels.append(l)
    for k, v in RELABEL_MAP.items():
        if v not in excluded_labels and k in excluded_labels:
            excluded_labels.remove(k)
    return excluded_labels


DIMENSIONS = (160, 188)

mel_s = (N_MELS, 513)
sftf_s = (2049, 188)
mfcc_s = (20, 188)
# [mels, time, channels] so image width is time
DIMENSIONS = (mel_s[0], mel_s[1], 1)
YAMNET_EMBEDDING_SHAPE = (6, 1024)
EMBEDDING_SHAPE = (1280,)


import random

# labels to apply weighting to if  y true is generic "bird"
NZ_BIRD_LOSS_WEIGHTING = []
BIRD_WEIGHTING = []
SPECIFIC_BIRD_MASK = []


def load_dataset(filenames, num_labels, labels, args, has_ebird=True):
    deterministic = args.get("deterministic", False)
    if not deterministic:
        logging.info("Shuffling filenames")
        random.shuffle(filenames)
    read_record = args.get("read_record", read_tfrecord)
    #
    #     image_size,
    #     labeled=True,
    #     augment=False,
    #     preprocess_fn=None,
    #     include_features=False,
    #     only_features=False,
    #     one_hot=True,
    # ):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = (
        deterministic  # disable order, increase speed
    )
    dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=AUTOTUNE, compression_type="GZIP"
    )

    # dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4)
    # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order

    augment = args.get("augment", False)
    preprocess_fn = args.get("preprocess_fn")
    one_hot = args.get("one_hot", True)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    # dataset = dataset.filter(filter_short)

    # set up weighting based on specific and generic birds
    global NZ_BIRD_LOSS_WEIGHTING, BIRD_WEIGHTING, SPECIFIC_BIRD_MASK, GENERIC_BIRD_MASK
    SPECIFIC_BIRD_MASK = np.zeros(num_labels)
    BIRD_WEIGHTING = np.zeros(num_labels)
    NZ_BIRD_LOSS_WEIGHTING = np.zeros(num_labels)
    GENERIC_BIRD_MASK = np.zeros(num_labels)
    if "rifleman" in labels:
        NZ_BIRD_LOSS_WEIGHTING[labels.index("rifleman")] = 1
    if "bird" in labels:
        NZ_BIRD_LOSS_WEIGHTING[labels.index("bird")] = 1
        BIRD_WEIGHTING[labels.index("bird")] = 1
        GENERIC_BIRD_MASK[labels.index("bird")] = 1

    for i, l in enumerate(labels):
        if (l in ALL_BIRDS or l in BIRD_TRAIN_LABELS) and l != "bird":
            SPECIFIC_BIRD_MASK[i] = 1
    SPECIFIC_BIRD_MASK = tf.constant(SPECIFIC_BIRD_MASK, dtype=tf.float32)
    BIRD_WEIGHTING = tf.constant(BIRD_WEIGHTING, dtype=tf.float32)
    NZ_BIRD_LOSS_WEIGHTING = tf.constant(NZ_BIRD_LOSS_WEIGHTING, dtype=tf.float32)
    GENERIC_BIRD_MASK = tf.constant(GENERIC_BIRD_MASK, dtype=tf.float32)

    dataset = dataset.map(
        partial(
            read_record,
            num_labels=num_labels,
            augment=augment,
            preprocess_fn=preprocess_fn,
            one_hot=one_hot,
            mean_sub=args.get("mean_sub", False),
            add_noise=args.get("add_noise", False),
            no_bird=args.get("no_bird", False),
            embeddings=args.get("embeddings", False),
            filter_freq=args.get("filter_freq", False),
            random_butter=args.get("random_butter", 0),
            only_features=args.get("only_features", False),
            features=args.get("features", False),
            multi=args.get("multi_label", True),
            load_raw=args.get("load_raw", True),
            model_name=args.get("model_name", "badwinner2"),
            has_ebird=has_ebird,
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )

    # if args.get("filter_bad", False):
    #     logging.info("Filtering bad")
    #     dataset = dataset.filter(lambda x, y: not filter_bad_tracks(x, y, labels))
    if args.get("only_features", False):
        logging.info("Removing Nan")

        filter_nan = lambda x, y: filter_nan_samples(x)
    else:
        if args.get("features"):
            logging.info("Removing feature Nan")

            filter_nan = (
                lambda x, y: not tf.reduce_any(tf.math.is_nan(x[0]))
                and tf.math.count_nonzero(x[1]) > 0
                and tf.math.count_nonzero(x[2]) > 0
            )
        else:
            logging.info("Removing Nan")

            filter_nan = lambda x, y: filter_nan_samples(x)

            # filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x))
    dataset = dataset.filter(filter_nan)
    if args.get("one_hot", True):
        filter_excluded = lambda x, y: not tf.math.equal(tf.math.count_nonzero(y[0]), 0)
    else:
        filter_excluded = lambda x, y: tf.math.greater(y[0], -1)

    dataset = dataset.filter(filter_excluded)

    # mel conversion is done after filtering so it only runs on kept samples
    if (
        read_record is read_tfrecord
        and not args.get("load_raw", True)
        and not args.get("embeddings", False)
        and not args.get("only_features", False)
    ):
        dataset = dataset.map(
            partial(
                mel_map,
                features=args.get("features", False),
                pcen=args.get("pcen", False),
            ),
            num_parallel_calls=AUTOTUNE,
            deterministic=deterministic,
        )
    return dataset


def mel_map(x, y, features=False, pcen=False):
    if features:
        x = (spectrogram_to_mel(x[0], pcen), *x[1:])
    else:
        x = spectrogram_to_mel(x, pcen)
    return x, y


@tf.function
def filter_nan_samples(features):
    is_nan_mask = tf.math.is_nan(features)
    inf_mask = tf.math.is_inf(features)
    has_nan = tf.math.reduce_any(tf.math.logical_or(is_nan_mask, inf_mask))
    return tf.math.logical_not(has_nan)


def get_distribution(dataset, num_labels, batched=True, one_hot=True):
    dist = np.zeros((num_labels), dtype=np.float32)
    counts = None
    total = 0
    # only pull the labels out, so x is never copied into python
    dataset = dataset.map(
        lambda x, y: y[0] if isinstance(y, tuple) else y,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if not batched:
        # iterating element by element is slow, count in chunks instead
        dataset = dataset.batch(1024)
    for y in dataset.as_numpy_iterator():
        if one_hot:
            y = y.reshape(-1, y.shape[-1])
            batch_counts = np.count_nonzero(y, axis=0)
            total += len(y)
        else:
            y = np.asarray(y).reshape(-1).astype(np.int64)
            total += len(y)
            y = y[(y >= 0) & (y < num_labels)]
            batch_counts = np.bincount(y, minlength=num_labels)
        counts = batch_counts if counts is None else counts + batch_counts
    if counts is None:
        return dist, 0
    n = min(num_labels, len(counts))
    dist[:n] = counts[:n]
    return dist, total


def get_remappings(
    labels, excluded_labels, keep_excluded_in_extra=True, use_generic_bird=False
):
    extra_label_map = {}
    # remapped = {}
    re_dic = {}
    new_labels = labels.copy()
    # for l in new_labels:

    for excluded in excluded_labels:
        if excluded in new_labels:
            new_labels.remove(excluded)

    merge_v = list(RELABEL_MAP.values())
    logging.info("New labels are %s", new_labels)
    for k, v in RELABEL_MAP.items():
        if k in new_labels and v not in new_labels:
            if v in ["noise", "human"]:
                excluded_labels.append(k)
                logging.info("Excluding %s as not using noise human", k)
            else:
                new_labels.append(v)
    new_labels.sort()
    for label in RELABEL_MAP.keys():
        if label in new_labels and label not in merge_v:
            new_labels.remove(label)
            logging.info(
                "Removing from labels %s as is merged into another label", label
            )
    for l in labels:
        if l in excluded_labels:
            re_dic[l] = -1
            logging.info("Excluding %s", l)
        else:
            if l in RELABEL_MAP and RELABEL_MAP[l] in new_labels:
                logging.info("Re labeiling %s as %s", l, RELABEL_MAP[l])
                re_dic[l] = new_labels.index(RELABEL_MAP[l])
            else:
                re_dic[l] = new_labels.index(l)
    if not use_generic_bird:
        re_dic["bird"] = -1

    if not keep_excluded_in_extra:
        labels = new_labels

    for l in labels:
        if l in excluded_labels:
            continue
        remap_label = None

        if l in NOISE_LABELS:
            if "noise" in new_labels:
                remap_label = "noise"
        elif l in HUMAN_LABELS:
            if "human" in new_labels:
                remap_label = "human"
        elif l in OTHER_LABELS:
            if "other" in new_labels:
                remap_label = "other"
        elif l in ALL_BIRDS:
            if not use_generic_bird or l == "bird":
                continue
            if l in BIRD_TRAIN_LABELS:
                extra_label_map[l] = new_labels.index("bird")
            else:
                remap_label = "bird"
        else:
            continue
        if l == remap_label or remap_label is None:
            continue
        re_dic[l] = new_labels.index(remap_label)
        # del remapped[l]
    return (extra_label_map, re_dic, new_labels)


bird_i = None
noise_i = None
bird_mask = None


def get_dataset(dir, labels, global_epoch=None, **args):
    global FMAX, MEL_WEIGHTS, FMIN, NFFT, BREAK_FREQ, N_MELS
    if args.get("n_mels"):
        N_MELS = args.get("n_mels")

        MEL_WEIGHTS = mel_f(48000, N_MELS, FMIN, FMAX, NFFT, BREAK_FREQ)
        MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)
        logging.info("Set mels to %s", N_MELS)
    if args.get("fmin") is not None:
        FMIN = args.get("fmin", FMIN)
        FMAX = args.get("fmax", FMAX)
        logging.info("Using fmin and fmax %s %s", FMIN, FMAX)

        MEL_WEIGHTS = mel_f(48000, N_MELS, FMIN, FMAX, NFFT, BREAK_FREQ)
        MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)
    if args.get("n_fft") is not None:
        print(NFFT, "IS")
        NFFT = args.get("n_fft")
        logging.info("NFFT %s", NFFT)
        if NFFT < 2048:
            N_MELS = 96
            global DIMENSIONS
            mel_s = (N_MELS, 513)
            DIMENSIONS = (mel_s[0], mel_s[1], 1)
            logging.info("Lower mels as nfft is to low %s", N_MELS)
        MEL_WEIGHTS = mel_f(48000, N_MELS, FMIN, FMAX, NFFT, BREAK_FREQ)
        MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)
    if args.get("break_freq") is not None:
        BREAK_FREQ = args.get("break_freq")
        logging.info("Applied break freq %s", BREAK_FREQ)
        MEL_WEIGHTS = mel_f(48000, N_MELS, FMIN, FMAX, NFFT, BREAK_FREQ)
        MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)
    # always load raw audio so augmentations run on the waveform, raw_to_mel
    # then builds the mel (magnitude for PCEN, power for MagTransform)
    args["load_raw"] = True
    pcen = args.get("pcen", False)
    dataset, noise_dataset, remapped, epoch_size, labels, extra_label_dic = (
        get_a_dataset(dir, labels, args)
    )
    args["epoch_size"] = epoch_size
    deterministic = args.get("deterministic", False)
    logging.info("Mapping raw to mel pcen %s", pcen)
    if args.get("augment", False):
        if noise_dataset is not None:
            noise_class_id = labels.index("noise")
            # pair each batch with a batch of noise clips
            dataset = tf.data.Dataset.zip((dataset, noise_dataset))
            dataset = dataset.map(
                lambda batch, noise: (
                    tf_batch_mix_real_noise(
                        batch[0], noise, batch[1], noise_class_id=noise_class_id
                    ),
                    batch[1],
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        # Apply the roll on-the-fly during your training dataset pipeline
        dataset = dataset.map(
            lambda audio, label: (tf_batch_audio_roll(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        logging.info("Mixing up")

        dataset = dataset.map(
            lambda x, y: selective_batch_mixup(x, y, mixup_prob=0.7, alpha=0.4),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic,
        )

    if args.get("debug"):
        logging.info("Not mapping to mel")
    else:
        dataset = dataset.map(
            lambda x, y: raw_to_mel(x, y, pcen=pcen),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic,
        )

    if args.get("augment", False):
        dataset = dataset.map(
            lambda x, y: (tf_batch_spec_augment(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset, remapped, epoch_size, labels, extra_label_dic


def set_remapped_extra(remap, extra_l):

    global extra_label_map
    global remapped_y
    extra_label_map = extra_l
    remapped_y = remap


def load_noise_dataset(dir, num_labels, labels, args, noise_dirs):
    """Endless shuffled raw noise clips, for mixing into training audio.

    Loads the records in <noise dir>/<split name> for each noise dir, as written
    by build.py --noise-dataset
    """
    filenames = []
    for noise_dir in noise_dirs:
        noise_files = tf.io.gfile.glob(str(Path(noise_dir) / dir.name / "*.tfrecord"))
        logging.info(
            "Loading noise files %s count: %s", noise_files[:1], len(noise_files)
        )
        filenames.extend(noise_files)
    if len(filenames) == 0:
        raise ValueError(f"No noise records found in {noise_dirs} for {dir.name}")
    noise_dataset = load_dataset(filenames, num_labels, labels, args)
    # Drop labels, we only need raw wave arrays
    noise_dataset = noise_dataset.map(
        lambda wave, label: wave, num_parallel_calls=tf.data.AUTOTUNE
    )
    return noise_dataset.shuffle(buffer_size=1000).repeat()


def get_a_dataset(dir, labels, args):

    extra_label_dic = args.get("extra_label_map")
    remapped = args.get("remapped_labels", [])
    excluded_labels = args.get("excluded_labels", [])
    use_generic_bird = args.get("use_generic_bird", False)
    global extra_label_map
    global remapped_y
    if extra_label_dic is None:
        extra_label_dic, remapped, labels = get_remappings(
            labels, excluded_labels, use_generic_bird=use_generic_bird
        )
        for k, v in remapped.items():

            logging.info("Remapping %s to %s", k, " nothing " if v == -1 else labels[v])
        logging.info(
            "Remapped %s extra mapping %s new labels %s Use gen bird %s",
            remapped,
            extra_label_dic,
            labels,
            use_generic_bird,
        )
    else:
        logging.info(
            "Load with predefined extra label dic %s remapped %s excluded %s",
            extra_label_dic,
            remapped,
            excluded_labels,
        )
    remapped_y = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(remapped.keys())),
            values=tf.constant(list(remapped.values())),
        ),
        default_value=tf.constant(-1),
        name="remapped_y",
    )

    global bird_i
    global noise_i
    global human_i
    if "bird" in labels:
        bird_i = labels.index("bird")
    if "noise" in labels:
        noise_i = labels.index("noise")
    if "human" in labels:
        human_i = labels.index("human")

    # extra tags, since we have multi label problem, morepork is a bird and morepork
    # cat is a cat but also "noise"
    # extra_label_map["-10"] = -10
    if len(extra_label_dic) == 0:
        # seems to need something
        extra_label_dic["nonsense"] = 1
    extra_label_map = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(extra_label_dic.keys())),
            values=tf.constant(list(extra_label_dic.values())),
        ),
        default_value=tf.constant(-1),
        name="extra_label_map",
    )

    num_labels = len(labels)
    datasets = []
    logging.info("Loading tf records from %s", dir)
    filenames = tf.io.gfile.glob(str(dir / "*.tfrecord"))

    if args.get("second_dir") is not None:
        second_dir = Path(args.get("second_dir"))

        second_filenames = tf.io.gfile.glob(str(second_dir / "*.tfrecord"))
        logging.info(
            "Loading second files %s count: %s",
            second_filenames[:1],
            len(second_filenames),
        )

        filenames.extend(second_filenames)
        # datasets.append(second_ds)
    else:
        logging.info("Not using second dataset")

    dataset_3 = None
    if args.get("human_dir") is not None:
        second_dir = Path(args.get("human_dir"))

        second_filenames = tf.io.gfile.glob(str(second_dir / "*.tfrecord"))
        # this wont work with mix up unless you choose the same files
        random.shuffle(second_filenames)
        reduce_by = 0.03
        second_files = int(len(second_filenames) * reduce_by)
        second_filenames = second_filenames[:second_files]

        logging.info(
            "Loading human files %s count: %s",
            second_filenames[:1],
            len(second_filenames),
        )
        filenames.extend(second_filenames)

    if args.get("extra_datasets") is not None:
        for dataset in args["extra_datasets"]:

            dataset_dir = Path(dataset) / dir.name

            extra_files = tf.io.gfile.glob(str(dataset_dir / "*.tfrecord"))
            random.shuffle(extra_files)

            logging.info(
                "Loading extra ds files %s count: %s",
                extra_files[:1],
                len(extra_files),
            )
            filenames.extend(extra_files)

    logging.info("Loading %s files from %s", len(filenames), dir)

    dataset = load_dataset(filenames, num_labels, labels, args)
    datasets.append(dataset)

    xeno_files = Path("/data/audio-data/xenocanto/xeno-training-data/")
    xeno_files = xeno_files / dir.name
    do_xeno = False
    if do_xeno and xeno_files.exists() and dir.name != "test":
        logging.info("Xeno files %s", xeno_files)
        xeno_files = tf.io.gfile.glob(str(xeno_files / "*.tfrecord"))
        logging.info("Loading xeno files %s", xeno_files)
        filenames.extend(xeno_files)

    # if args.get("multi_label", True):
    #     # not sure if this is needed at all
    #     if not args.get("one_hot", True):
    #         bird_mask = tf.constant(bird_i, dtype=tf.float32)
    #         bird_filter = lambda x, y: tf.math.equal(y[0], bird_mask)
    #         others_filter = lambda x, y: not tf.math.equal(y[0], bird_mask)
    #     else:
    #         bird_mask = np.zeros(num_labels, dtype=bool)
    #         bird_mask[bird_i] = 1
    #         bird_mask = tf.constant(bird_mask)
    #         bird_filter = lambda x, y: tf.math.reduce_all(
    #             tf.math.equal(tf.cast(y[0], tf.bool), bird_mask)
    #         )
    #         others_filter = lambda x, y: not tf.math.reduce_all(
    #             tf.math.equal(tf.cast(y[0], tf.bool), bird_mask)
    #         )
    #     if not args.get("use_bird_tags", False):
    #         logging.info("Filtering out bird tags without specific bird")
    #         for i, ds in enumerate(datasets):
    #             datasets[i] = ds.filter(others_filter)

    deterministic = args.get("deterministic", False)

    if args.get("debug"):
        if len(datasets) > 0:
            # logging.info("Adding second dataset with weights [0.6,0.4]")
            dataset = tf.data.Dataset.sample_from_datasets(
                datasets,
                # weights=[0.6, 0.4],
                stop_on_empty_dataset=False,
                rerandomize_each_iteration=args.get("rerandomize_each_iteration", True),
            )
        else:
            dataset = datasets[0]
        if args.get("debug_bird"):
            logging.info("Debugging on %s", args.get("debug_bird"))
            debug_i = labels.index(args.get("debug_bird"))
            debug_mask = np.zeros(num_labels, dtype=np.float32)
            debug_mask[debug_i] = 1

            debug_mask = tf.constant(debug_mask, dtype=tf.float32)
            debug_filter = lambda x, y: tf.math.reduce_all(
                tf.math.equal(y[0], debug_mask)
            )
            dataset = dataset.filter(debug_filter)
        if args.get("signal_less_than") is not None:
            logging.info(
                "Getting tracks with signal less than %s", args.get("signal_less_than")
            )
            less_than = lambda x, y: tf.math.less(y[2], args.get("signal_less_than"))
            dataset = dataset.filter(less_than)
        batch_size = args.get("batch_size", None)
        if args.get("cache", False):
            dataset = dataset.cache()

        if batch_size is not None:
            dataset = dataset.batch(batch_size, drop_remainder=False)
        if args.get("load_raw", False):
            logging.info("Normalizing input")
            dataset = dataset.map(lambda x, y: normalize(x, y))
        logging.info("Returning debug data")

        return dataset, None, remapped, None, labels, extra_label_dic

    if not args.get("load_all_y", False):
        if args.get("loss_fn") == "WeightedCrossEntropy":
            logging.info("Mapping possiblebirds")
            dataset = dataset.map(
                lambda x, y: (x, (y[0], y[5])),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=deterministic,
            )
        else:
            for i, ds in enumerate(datasets):
                datasets[i] = ds.map(
                    lambda x, y: (x, y[0]),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=deterministic,
                )
            # dataset = dataset.map(
            #     lambda x, y: (x, y[0]),
            #     num_parallel_calls=tf.data.AUTOTUNE,
            #     deterministic=deterministic,
            # )
            # if dataset_2 is not None:
            #     dataset_2 = dataset_2.map(
            #         lambda x, y: (x, y[0]),
            #         num_parallel_calls=tf.data.AUTOTUNE,
            #         deterministic=deterministic,
            #     )

    # try caching just our dataset
    if args.get("cache", False) or dir.name != "train":
        logging.info("Caching to mem")
        datasets[0] = datasets[0].cache()
    if args.get("shuffle", True):
        for i, ds in enumerate(datasets):
            datasets[i] = ds.shuffle(
                4096, reshuffle_each_iteration=args.get("reshuffle", True)
            )

    if len(datasets) > 0:
        # logging.info("Adding second dataset with weights [0.6,0.4]")
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets,
            # weights=[0.6, 0.4],
            stop_on_empty_dataset=False,
            rerandomize_each_iteration=args.get("rerandomize_each_iteration", True),
        )
    else:
        dataset = datasets[0]
    logging.info("Loss fn is %s", args.get("loss_fn"))

    epoch_size = args.get("epoch_size")
    dist = None
    if epoch_size is None:
        dist, epoch_size = get_distribution(
            dataset, num_labels, batched=False, one_hot=args.get("one_hot", True)
        )

    if dist is not None:
        for l, d in zip(labels, dist):
            logging.info(f" for {l} have {d}")

    noise_dataset = None
    if args.get("augment", False) and args.get("noise_dirs"):
        noise_dataset = load_noise_dataset(
            dir, num_labels, labels, args, args["noise_dirs"]
        )

    batch_size = args.get("batch_size", None)
    if batch_size is not None:
        dataset = dataset.batch(
            batch_size, drop_remainder=args.get("drop_remainder", False)
        )
        if noise_dataset is not None:
            noise_dataset = noise_dataset.batch(
                batch_size, drop_remainder=args.get("drop_remainder", False)
            )

    # dont think using this anymore GP
    if args.get("weight_specific", False):
        weighting = np.ones((num_labels), dtype=np.float32)
        weighting[bird_i] = 0.8
        weighting = tf.constant(weighting)
        specific_mask = np.zeros((num_labels), dtype=np.float32)
        for i, l in enumerate(labels):
            if l in SPECIFIC_BIRD_LABELS and l != "bird":
                specific_mask[i] = 1
        print(
            "for labels", labels, " have ", specific_mask, " weighting bird", weighting
        )
        specific_mask = tf.constant(specific_mask)

        rest_weighting = tf.constant(tf.ones(num_labels))
        dataset = dataset.map(
            lambda x, y: weight_specific(
                x, y, num_labels, weighting, specific_mask, rest_weighting
            )
        )

    return dataset, noise_dataset, remapped, epoch_size, labels, extra_label_dic


import tensorflow as tf


@tf.function
def tf_batch_mix_real_noise(
    bird_batch,
    noise_batch,
    label_batch,
    noise_class_id=0,
    max_noise_ratio=0.3,
    execution_probability=0.7,
):
    """
    Blends real environmental noise clips into your primary batch,
    automatically bypassing files whose one-hot labels mark them as background noise.

    Parameters:
    - bird_batch (tf.Tensor): Target audio waveforms, shape [batch, time_samples].
    - noise_batch (tf.Tensor): Background noise clips to inject, shape [batch, time_samples].
    - label_batch (tf.Tensor): One-hot label matrix, shape [batch, num_classes].
    - noise_class_id (int): The column index representing the "Noise/Background" class.
    """
    shape_tensor = tf.shape(bird_batch)
    batch_size, time_samples = tf.split(shape_tensor, num_or_size_splits=2)
    batch_size = tf.squeeze(batch_size)
    # the last batch can be smaller than the noise batch
    noise_batch = noise_batch[:batch_size]

    # 1. Standard randomized probability mask for the batch items
    random_probs = tf.random.uniform(shape=[batch_size], minval=0.0, maxval=1.0)
    wants_noise_mask = tf.less(random_probs, execution_probability)

    # 2/3. Skip clips whose only label is noise (argmax would misfire on multi-label
    # rows, e.g. bird + noise)
    label_batch = tf.cast(label_batch, tf.float32)
    noise_col = label_batch[:, noise_class_id]
    is_not_already_noise = tf.logical_not(
        tf.logical_and(
            noise_col > 0, tf.equal(tf.reduce_sum(label_batch, axis=-1), noise_col)
        )
    )

    # 4. Combine both conditions: Only mix if it rolls 'True' AND it isn't already a noise clip
    should_noise_mask = tf.logical_and(wants_noise_mask, is_not_already_noise)

    # 5. Determine unique scaling intensities per item
    noise_scales = tf.random.uniform(
        shape=[batch_size], minval=0.01, maxval=max_noise_ratio
    )
    final_scales = tf.where(should_noise_mask, noise_scales, tf.zeros([batch_size]))

    # 6. Broadcast scales across the time dimension and mix the signals
    expanded_scales = tf.expand_dims(final_scales, axis=-1)
    scaled_noise = tf.multiply(noise_batch, expanded_scales)
    mixed_batch = tf.add(bird_batch, scaled_noise)

    # maybe can just be clipped
    # 7. Rescale anything that went over 1.0 rather than hard clipping the peaks
    peaks = tf.reduce_max(tf.abs(mixed_batch), axis=-1, keepdims=True)
    output_batch = mixed_batch / tf.maximum(peaks, 1.0)

    return output_batch


def selective_batch_mixup(audio_batch, label_batch, mixup_prob=0.7, alpha=0.4):
    """
    Applies mixup to a batch only a certain percentage of the time.
    """
    # 1. Roll a random number between 0.0 and 1.0 for the whole batch
    random_roll = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)

    # 2. Define what happens if we DO mix up
    def apply_mixup():
        batch_size = tf.shape(audio_batch)[0]

        # Sample independent lambdas
        gamma_1 = tf.random.gamma(shape=[batch_size, 1], alpha=alpha)
        gamma_2 = tf.random.gamma(shape=[batch_size, 1], alpha=alpha)
        lam = gamma_1 / (gamma_1 + gamma_2)

        # Reshape for broadcasting
        lam_audio = tf.reshape(lam, [batch_size, -1])

        # Create pairs by rolling the batch
        audio_shuffled = tf.roll(audio_batch, shift=1, axis=0)
        label_shuffled = tf.roll(label_batch, shift=1, axis=0)

        # Mix and peak normalize audio
        mixed_audio = lam_audio * audio_batch + (1.0 - lam_audio) * audio_shuffled
        peaks = tf.reduce_max(tf.abs(mixed_audio), axis=-1, keepdims=True)
        mixed_audio = mixed_audio / tf.maximum(peaks, 1e-6)

        # Mix labels
        mixed_labels = lam * label_batch + (1.0 - lam) * label_shuffled
        return mixed_audio, mixed_labels

    def skip_mixup():
        return audio_batch, label_batch

    return tf.cond(random_roll < mixup_prob, apply_mixup, skip_mixup)


# @tf.function
# def mix_up(ds_one, ds_two, global_epoch, alpha=0.4, chance=0.25, single_label=True):
#     # Unpack two datasets
#     images_one, labels_one = ds_one
#     images_two, labels_two = ds_two

#     # go down 0.05 every 5 epochs
#     step = global_epoch.value() // 5
#     logging.info("NOt decreasing aug change")
#     # chance = chance - 0.05 * tf.cast(step, tf.float32)
#     batch_size = tf.keras.ops.shape(images_one)[0]
#     l = sample_beta_distribution(batch_size, alpha, alpha)
#     aug_chance = tf.random.uniform((batch_size,))
#     # non augmented samples keep ds_one unchanged (l=1)
#     l = tf.where(aug_chance < chance, l, tf.ones_like(l))
#     x_l = tf.keras.ops.reshape(l, (batch_size, 1))
#     y_l = tf.keras.ops.reshape(l, (batch_size, 1))

#     images = images_one * x_l + images_two * (1 - x_l)
#     if single_label:
#         logging.info("Mixing up on single label, so taking maximum label")
#         y_l = tf.cast(y_l > 0.5, dtype=tf.float32)

#     labels = labels_one * y_l + labels_two * (1 - y_l)
#     # possible_labels = tf.clip_by_value(labels_one[1] + labels_two[1], 0, 1)
#     return (images, labels)


import tensorflow as tf


@tf.function
def tf_batch_audio_roll(audio_batch, max_shift_pct=0.5, execution_probability=0.7):
    """
    Performs independent randomized circular audio rolls across a batch,
    but only applies it to a specific percentage of the items.

    Parameters:
    - audio_batch (tf.Tensor): Shape [batch_size, time_samples]
    - max_shift_pct (float): Max shift allowance as a fraction of audio length.
    - execution_probability (float): Probability (0.0 to 1.0) of applying the roll.
    """
    batch_size = tf.shape(audio_batch)[0]
    time_samples = tf.shape(audio_batch)[1]

    # 1. Determine which individual items in the batch get the augmentation
    # Generates uniform random numbers [0, 1) for each batch item
    random_probs = tf.random.uniform(shape=[batch_size], minval=0.0, maxval=1.0)
    # Boolean mask: True if we should roll, False if we keep original
    should_roll_mask = random_probs < execution_probability

    # 2. Calculate unique random shift amounts for EVERY item
    max_shift_samples = tf.cast(
        tf.cast(time_samples, tf.float32) * max_shift_pct, tf.int32
    )
    shift_amounts = tf.random.uniform(
        shape=[batch_size],
        minval=-max_shift_samples,
        maxval=max_shift_samples,
        dtype=tf.int32,
    )

    # 3. Compute rolled indices via broadcasting
    time_indices = tf.range(time_samples, dtype=tf.int32)
    new_indices = (
        time_indices[tf.newaxis, :] - shift_amounts[:, tf.newaxis]
    ) % time_samples

    # 4. Gather the rolled version of the entire batch
    rolled_batch = tf.gather(audio_batch, new_indices, batch_dims=1)

    # 5. Conditionally select: If True -> take rolled, If False -> take original
    # We expand the mask dimension to [batch_size, 1] so it broadcasts across time_samples
    output_batch = tf.where(should_roll_mask[:, tf.newaxis], rolled_batch, audio_batch)

    return output_batch


@tf.function
def tf_batch_spec_augment(
    spectrogram_batch,
    max_freq_mask_pixels=8,
    max_time_mask_pixels=20,
    num_masks=2,
    execution_probability=0.8,
):
    """
    Applies SpecAugment (Frequency and Time Masking) to a batch of
    [batch, mels, time, channels] spectrograms.
    """
    tf.debugging.assert_rank(spectrogram_batch, 4)

    # Unpack dimensions dynamically without using any integer index brackets
    shape_tensor = tf.shape(spectrogram_batch)
    batch_size, freq_bins, time_steps, channels = tf.split(
        shape_tensor, num_or_size_splits=4
    )

    # Cast to scalar integers to drive random ranges and loop sizes safely
    batch_size = tf.squeeze(batch_size)
    freq_bins = tf.squeeze(freq_bins)
    time_steps = tf.squeeze(time_steps)

    # Decide which items in the batch get masked
    random_probs = tf.random.uniform(shape=[batch_size], minval=0.0, maxval=1.0)
    should_mask_mask = tf.less(random_probs, execution_probability)

    augmented_batch = tf.identity(spectrogram_batch)

    for _ in range(num_masks):
        # --- FREQUENCY BOUNDS ---
        freq_mask_heights = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=max_freq_mask_pixels, dtype=tf.int32
        )
        freq_start_positions = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=freq_bins - max_freq_mask_pixels,
            dtype=tf.int32,
        )

        # --- TIME BOUNDS ---
        time_mask_widths = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=max_time_mask_pixels, dtype=tf.int32
        )
        time_start_positions = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=time_steps - max_time_mask_pixels,
            dtype=tf.int32,
        )

        # --- COORDINATE GRIDS ---
        freq_indices = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(tf.range(freq_bins), axis=0), axis=-1),
            axis=-1,
        )
        time_indices = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(tf.range(time_steps), axis=0), axis=1),
            axis=-1,
        )

        f_start = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(freq_start_positions, axis=-1), axis=-1),
            axis=-1,
        )
        f_end = f_start + tf.expand_dims(
            tf.expand_dims(tf.expand_dims(freq_mask_heights, axis=-1), axis=-1), axis=-1
        )

        t_start = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(time_start_positions, axis=-1), axis=-1),
            axis=-1,
        )
        t_end = t_start + tf.expand_dims(
            tf.expand_dims(tf.expand_dims(time_mask_widths, axis=-1), axis=-1), axis=-1
        )

        # --- MASK LOGIC (Bracket-free boolean comparisons) ---
        is_freq_masked = tf.logical_and(
            tf.greater_equal(freq_indices, f_start), tf.less(freq_indices, f_end)
        )
        is_time_masked = tf.logical_and(
            tf.greater_equal(time_indices, t_start), tf.less(time_indices, t_end)
        )

        # --- COMBINE AND APPLY ---
        apply_mask_item = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(should_mask_mask, axis=-1), axis=-1), axis=-1
        )
        final_mask = tf.logical_and(
            tf.logical_or(is_freq_masked, is_time_masked), apply_mask_item
        )

        augmented_batch = tf.where(
            final_mask, tf.zeros_like(augmented_batch), augmented_batch
        )

    return augmented_batch


def spectrogram_to_mel(spectrogram, pcen=False):
    """Takes a (2049, 513) stft magnitude and returns a (mels, 513, 3) mel spectrogram.

    With pcen the mels are left as magnitude for the PCEN layer in the model,
    otherwise they are power, for the MagTransform layer in the model.
    """
    # conver to power
    if not pcen:
        spectrogram = tf.math.pow(spectrogram, 2)
        # if doing mix up needs to happen here

    # [mels, time]
    spectrogram = tf.tensordot(MEL_WEIGHTS, spectrogram, 1)
    print("Spect shape is ", spectrogram.shape)
    # power db
    spectrogram = tf.expand_dims(spectrogram, axis=-1)

    # spectrogram = tf.math.log10(spectrogram+tf.keras.backend.epsilon())
    # no db or normalizing, the model compresses with PCEN or MagTransform
    # spectrogram = power_to_db(spectrogram)
    # spectrogram = normalize_acoustic_fixed(spectrogram)
    if pcen:
        logging.info("Doing PCEN leaving spect as magnitude")
    else:
        # spectrogram = power_to_root_compressed(spectrogram)

        logging.info("Leaving spect as power for MagTransform")
    logging.info("Shape is %s ", spectrogram.shape)
    # if not pcen and "efficientnet" in model_name:
    #     logging.info("Repeating last dim for efficient net")
    spectrogram = tf.repeat(spectrogram, 3, 2)
    return spectrogram


@tf.function
def read_tfrecord(
    example,
    num_labels,
    augment=False,
    preprocess_fn=None,
    one_hot=True,
    mean_sub=False,
    add_noise=False,
    no_bird=False,
    embeddings=False,
    filter_freq=False,
    random_butter=0,
    only_features=False,
    features=False,
    multi=True,
    load_raw=True,
    model_name="badwinner2",
    global_epoch=None,
    has_ebird=True,
):
    tfrecord_format = {"audio/class/text": tf.io.FixedLenFeature((), tf.string)}
    if has_ebird:
        tfrecord_format["audio/class/ebird"] = tf.io.FixedLenFeature((), tf.string)

    tfrecord_format["audio/rec_id"] = tf.io.FixedLenFeature((), tf.string)
    tfrecord_format["audio/track_id"] = tf.io.FixedLenFeature((), tf.string)
    tfrecord_format["audio/low_sample"] = tf.io.FixedLenFeature((), tf.int64)
    tfrecord_format["audio/start_s"] = tf.io.FixedLenFeature((), tf.float32)

    tfrecord_format["audio/lat"] = tf.io.FixedLenFeature((), tf.float32)
    tfrecord_format["audio/lng"] = tf.io.FixedLenFeature((), tf.float32)

    if embeddings:
        logging.info("Loading embeddings")

        tfrecord_format["embedding"] = tf.io.FixedLenFeature(
            EMBEDDING_SHAPE, tf.float32
        )

    elif not only_features:
        logging.info("Loading sft audio")

        if load_raw:
            tfrecord_format["audio/raw"] = tf.io.FixedLenFeature(
                (48000 * 3), tf.float32
            )
        else:
            tfrecord_format["audio/spectrogram"] = tf.io.FixedLenFeature(
                (2049 * 513), tf.float32
            )
        if filter_freq:
            tfrecord_format["audio/buttered"] = tf.io.FixedLenFeature(
                (2049 * 513), tf.float32, default_value=tf.zeros((2049 * 513))
            )

    if features or only_features:
        tfrecord_format["audio/short_f"] = tf.io.FixedLenFeature(
            (68 * 60), tf.float32, default_value=tf.zeros((68 * 60))
        )
        tfrecord_format["audio/mid_f"] = tf.io.FixedLenFeature(
            (136 * 3), tf.float32, default_value=tf.zeros((136 * 3))
        )

    tfrecord_format["audio/signal_percent"] = tf.io.FixedLenFeature((), tf.float32)

    example = tf.io.parse_single_example(example, tfrecord_format)
    low_sample = tf.cast(example["audio/low_sample"], tf.int64)
    start_s = tf.cast(example["audio/start_s"], tf.float32)

    # else:
    #     print("Labels was ",labels)
    #     labels = tf.reduce_max(labels)
    #     print("Labels becomes ",labels)
    embed_preds = None
    if load_raw:
        spectrogram = example["audio/raw"]

    elif embeddings:
        spectrogram = example["embedding"]

    elif not only_features:
        buttered = example["audio/buttered"] if filter_freq else None
        if filter_freq and tf.math.count_nonzero(buttered) > 0:
            if random_butter > 0:
                rand = tf.random.uniform((), 0, 1)
                # do butter pass 3/5ths of the time
                spectrogram = tf.cond(
                    rand <= random_butter,
                    lambda: tf.identity(example["audio/buttered"]),
                    lambda: tf.identity(example["audio/spectrogram"]),
                )
            else:
                logging.info("USing buttered")
                spectrogram = example["audio/buttered"]
        else:

            spectrogram = example["audio/spectrogram"]
        spectrogram = tf.reshape(spectrogram, (2049, 513))

    if features or only_features:
        short_f = example["audio/short_f"]
        mid_f = example["audio/mid_f"]
        # if only_features:
        # spectrogram = tf.concat((short_f, mid_f), axis=0)
        #     # mid_f = tf.reshape(mid_f, (136, 3))
        #     # short_f = tf.reshape(short_f, (68, 60))

        #     # raw = (short_f, mid_f)
        # else:
        mid_f = tf.reshape(mid_f, (136, 3))
        short_f = tf.reshape(short_f, (68, 60))
        if only_features:
            print("ONLY FEATURES")
            spectrogram = (short_f, mid_f)
        else:
            spectrogram = (spectrogram, short_f, mid_f)
        # raw = tf.expand_dims(raw, axis=0)
    if augment:
        logging.info("Augmenting")
    # label = tf.cast(example["audio/class/label"], tf.int32)
    if has_ebird:
        label = tf.cast(example["audio/class/ebird"], tf.string)
    else:
        label = tf.cast(example["audio/class/text"], tf.string)

    split_labels = tf.strings.split(label, sep="\n")
    global remapped_y, extra_label_map
    labels = remapped_y.lookup(split_labels)
    extra = extra_label_map.lookup(split_labels)
    if multi:

        labels = tf.concat([labels, extra], axis=0)
    if one_hot:
        label = tf.reduce_max(tf.one_hot(labels, num_labels, dtype=tf.int32), axis=0)
        if not multi:
            logging.info("Choosing only one label as not multi")
            if tf.math.count_nonzero(label) == 0:
                # if all normal labels are excluded choose an extra one
                label = tf.reduce_max(
                    tf.one_hot(extra, num_labels, dtype=tf.int32), axis=0
                )
            if tf.math.count_nonzero(label) == 0:
                label = tf.zeros(num_labels, dtype=tf.int32)
            else:
                max_l = tf.argmax(label)
                label = tf.one_hot(max_l, num_labels, dtype=tf.int32)
        if embed_preds is not None:
            embed_preds = tf.reduce_max(
                tf.one_hot(embed_preds, num_labels, dtype=tf.int32), axis=0
            )
    else:
        # pretty sure this is wrong but never used
        logging.error("Don't think non one hot works please check")
        label = labels
    signal_percent = 0.0
    if no_bird:
        logging.info("no bird")
        # dont use bird or noise label from mixed ones
        if one_hot:
            no_bird_mask = np.ones(num_labels, dtype=bool)
            no_bird_mask[bird_i] = 0
            no_bird_mask = tf.constant(no_bird_mask)
            label = tf.cast(label, tf.bool)
            label = tf.math.logical_and(label, no_bird_mask)
            no_noise_mask = np.ones(num_labels, dtype=bool)
            no_noise_mask[noise_i] = 0
            no_noise_mask = tf.constant(no_noise_mask)
            label = tf.math.logical_and(label, no_noise_mask)
        else:
            print("Not doing no bird as not implemeneted")
        label = tf.cast(label, tf.int32)
    signal_percent = example["audio/signal_percent"]

    label = tf.cast(label, tf.float32)
    possible_labels = tf.ones(label.shape, tf.float32)

    lat = example["audio/lat"]
    lng = example["audio/lng"]
    # if label has no specific bird in it and has generic bird, weight differently
    if not tf.math.reduce_any(
        tf.math.logical_and(
            tf.cast(label, tf.bool), tf.cast(SPECIFIC_BIRD_MASK, tf.bool)
        )
    ) and tf.math.reduce_any(
        tf.math.logical_and(
            tf.cast(label, tf.bool), tf.cast(GENERIC_BIRD_MASK, tf.bool)
        )
    ):
        if lat == 0 or lng == 0:
            possible_labels = NZ_BIRD_LOSS_WEIGHTING
        elif (
            lat <= NZ_BOX[1]
            and lat >= NZ_BOX[3]
            and lng >= NZ_BOX[0]
            and lng <= NZ_BOX[2]
        ):
            possible_labels = NZ_BIRD_LOSS_WEIGHTING
        else:
            possible_labels = BIRD_WEIGHTING

    return spectrogram, (
        label,
        embed_preds,
        signal_percent,
        # min_freq,
        # max_freq,
        example["audio/rec_id"],
        example["audio/track_id"],
        possible_labels,
        low_sample,
        start_s,
        tf.cast(example["audio/class/text"], tf.string),
    )


def class_func(features, label):
    label = tf.argmax(label)
    return label


from collections import Counter


def calc_mean():
    datasets = ["training-data"]
    filenames = []
    labels = set()
    for d in datasets:
        # file = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/training-meta.json"
        file = f"./{d}/training-meta.json"
        with open(file, "r") as f:
            meta = json.load(f)
        labels.update(meta.get("labels", []))
        # print("loaded labels", labels)
        # species_list = ["bird", "human", "rain", "other"]

        # filenames = tf.io.gfile.glob(f"./training-data/validation/*.tfrecord")
        filenames.extend(tf.io.gfile.glob(f"./{d}/test/*.tfrecord"))
        filenames.extend(tf.io.gfile.glob(f"./{d}/train/*.tfrecord"))

        filenames.extend(tf.io.gfile.glob(f"./{d}/validation/*.tfrecord"))
    labels.add("bird")
    labels.add("noise")
    labels = list(labels)

    labels.sort()
    resampled_ds, remapped = get_dataset(
        # dir,
        filenames,
        labels,
        batch_size=32,
        augment=False,
        resample=False,
        # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
    )
    resampled_ds = resampled_ds.unbatch()
    data = [x for x, y in resampled_ds]
    data = np.array(data)

    # print(np.mean(data, axis=0).shape)
    # print(np.std(data, axis=0).shape)
    # mel_m = tf.reduce_mean(data, axis=0)
    zvals = {"mean": np.mean(data, axis=0), "std": np.std(data, axis=0)}
    zvals["mean"] = zvals["mean"].tolist()
    zvals["std"] = zvals["std"].tolist()

    with open("zvalues.txt", "w") as f:
        json.dump(zvals, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="Model to load and do preds",
    )
    parser.add_argument(
        "dir",
        help="Dataset dir",
    )
    parser.add_argument(
        "--only-features", default=False, action="count", help="Train on features"
    )
    parser.add_argument(
        "--multi-label", type=str2bool, default=False, help="Multi label"
    )
    parser.add_argument(
        "--use_bird_tags",
        default=False,
        action="count",
        help="Use tracks of generic bird tags ( without specific birds) in training",
    )
    args = parser.parse_args()
    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def debug_labels(dataset, labels):
    mapped = {}
    for l in labels:
        mapped[l] = set()
    for x, batch_y in dataset:
        y_true = batch_y[0]
        y_label = batch_y[-1]
        for y_t, y_l in zip(y_true, y_label):
            text_label = y_l.numpy().decode("utf8")
            summed = np.sum(y_t)
            if summed > 1:
                print("Multiple labels", y_t, text_label)
                continue
            max_i = tf.argmax(y_t)
            lbl = labels[max_i]
            mapped[lbl].add(text_label)
    for k, v in mapped.items():
        print(f"{k} = {v}")
    # print("Mapped labels are ",mapped)


# test stuff
def main():
    init_logging()
    args = parse_args()
    d = Path(args.dir)

    filenames = []
    labels = set()
    tf_dir = Path(d)
    # file = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/training-meta.json"
    file = tf_dir.parent / "training-meta.json"
    with file.open("r") as f:
        meta = json.load(f)
    if "ebird_ids" in meta:
        labels.update(meta.get("ebird_ids", []))
    else:
        labels.update(meta.get("labels", []))

        # labels.add("bird")
    labels.add("noise")
    # labels = list(labels)
    set_specific_by_count(meta)
    excluded_labels = get_excluded_labels(labels)
    remapped_labels = None
    extra_label_map = None
    labels = list(labels)
    labels.sort()
    fmin = FMIN
    fmax = FMAX
    if args.model:
        file = Path(args.model).parent / "metadata.txt"
        with file.open("r") as f:
            model_meta = json.load(f)
        labels = model_meta["labels"]
        excluded_labels = model_meta.get("excluded_labels")
        remapped_labels = model_meta.get("remapped_labels")
        extra_label_map = model_meta.get("extra_label_map")
        fmin = model_meta.get("fmin", FMIN)
        fmax = model_meta.get("fmax", FMAX)
    elif args.only_features:
        merge_labels = {}
        excluded_labels = []
        for l in labels:
            if l == "bird":
                continue
            if l in BIRD_TRAIN_LABELS or l in ALL_BIRDS:
                print("Setting", l, " to bird")
                merge_labels[l] = "bird"
            elif l in ANIMAL_LABELS:
                merge_labels[l] = "animal"
            elif l == "insect":
                continue
                # merge_labels[l] = "insect"
            elif l in NOISE_LABELS:
                merge_labels[l] = "noise"
            elif l in HUMAN_LABELS:
                merge_labels[l] = "human"
        set_merge_labels(merge_labels)
        args.use_bird_tags = True

    global_epoch = tf.Variable(0, name="global_epoch", trainable=True, dtype=tf.int32)

    dataset, remapped, _, labels, _ = get_dataset(
        tf_dir,
        labels,
        deterministic=True,
        batch_size=32,
        excluded_labels=excluded_labels,
        remapped_labels=remapped_labels,
        extra_label_map=extra_label_map,
        multi_label=args.multi_label,
        use_bird_tags=args.use_bird_tags,
        load_all_y=True,
        shuffle=False,
        load_raw=False,
        n_fft=4096,
        fmin=fmin,
        fmax=fmax,
        # MOREPORK_MAX,
        only_features=args.only_features,
        debug=True,
        model_name="mymodel",
        use_generic_bird=False,
        cache=True,
        global_epoch=global_epoch,
        augment=False,
        # signal_less_than = 0.1
    )
    # return
    # for epoch in range(5):
    #     global_epoch.assign(epoch)
    #     print("Global epoch assigned",global_epoch.value())
    #     for x, y in dataset:
    #         epoch_batch = y[-1]
    #         print("Epoch is ", epoch, epoch_batch[0].numpy())
    # debug_labels(dataset, labels)
    # return
    for batch_x, batch_y in dataset:
        recs = batch_y[3]
        tracks = batch_y[4]
        for x, rec, track in zip(batch_x, recs, tracks):
            data_ok = np.all(x >= -1.00002) and np.all(x <= 1.000002)
            has_nan = np.any(np.isnan(x))
            has_inf = np.any(np.isinf(x))
            a_max = np.amax(x)
            a_min = np.amin(x)
            if not data_ok or has_nan or has_inf:
                # print(x)
                x = x.numpy()
                logging.info(
                    "Bad data for rec %s track %s less than 1 %s over 1 %s max %s min %s",
                    rec,
                    track,
                    x[np.where(x < -1.000002)],
                    x[np.where(x > 1.000002)],
                    a_max,
                    a_min,
                )
                logging.info("Has nan %s has infinity", has_nan, has_inf)

            if a_max == a_min:
                logging.info(
                    "Max = Min for rec %s track %s max %s min %s",
                    rec,
                    track,
                    a_max,
                    a_min,
                )

    return
    preds = None
    if args.model is not None:
        model_path = Path(args.model)
        if model_path.is_dir():
            model_path = model_path / f"{model_path.stem}.keras"
        model = tf.keras.models.load_model(
            str(model_path),
            compile=False,
        )
        model.load_weights(model_path.parent / "val_categorical_accuracy.weights.h5")
        logging.info("LOading model with val acc %s", args.model)
        model.trainable = False
        model.summary()
        # true_categories = [y[0] if isinstance(y, tuple) else y for x, y in dataset]
        # true_categories = tf.concat(true_categories, axis=0)
        preds = model.predict(
            dataset.map(
                lambda x, y: (x, y[0]),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
        )
        index = 0
        total_more = 0
        correct = 0
        for _, y_b in dataset:

            recs = y_b[3]
            tracks = y_b[4]
            starts = y_b[7]
            y_true = y_b[0]
            for y, rec, track, start in zip(y_true, recs, tracks, starts):
                pred = preds[index]
                max_i = tf.argmax(pred)
                conf = pred[max_i]
                p_label = labels[max_i]
                y_true = tf.argmax(y)
                y_label = labels[y_true]

                index += 1
                if y_label != tf.constant("morepork"):
                    continue

                total_more += 1
                rec = rec.numpy().decode("utf8")
                track = track.numpy().decode("utf8")
                start = start.numpy()
                if max_i != y_true:
                    print(
                        f"{y_label} predicted as {p_label} with {round(100*conf)}% id: {rec}-{track} at {start}"
                    )
                else:
                    correct += 1
        # filenames.extend(tf.io.gfile.glob(f"{d}/test/**/*.tfrecord"))
        print("Correct out of total ", correct, " / ", total_more)
    # return
    global NZ_BIRD_LOSS_WEIGHTING, BIRD_WEIGHTING, SPECIFIC_BIRD_MASK, GENERIC_BIRD_MASK

    # dist, _ = get_distribution(dataset, len(labels), batched=True, one_hot=True)
    # print("Dist is ", dist)
    # for l, d in zip(labels, dist):
    #     print(f"{l} has {d}")

    # return
    for e in range(1):
        batch = 0
        global_epoch.assign(e)
        print("EPOCH", e)
        for x, y in dataset:
            batch += 1
            show_batch(
                x,
                y,
                labels,
                batch_i=batch,
                preds=(
                    preds[(batch - 1) * 32 : 32 * batch] if preds is not None else None
                ),
            )
        break


import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError("getsize() does not take argument of type: " + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                if isinstance(obj, np.ndarray):
                    size += obj.nbytes
                else:
                    size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size * 0.000001


def show_batch(image_batch, label_batch, labels, batch_i=0, preds=None):
    recs = None
    tracks = None
    starts = None
    prob_thresh = 0.7
    if isinstance(label_batch, tuple):
        signal_batch = label_batch[2]
        recs = label_batch[3]
        tracks = label_batch[4]
        starts = label_batch[7]
        label_batch = label_batch[0]
        # print("IS t uple")
        # 1/0
    # min_freq = label_batch[3]
    # max_freq = label_batch[4]
    # recs = label_batch[3]
    # tracks = label_batch[4]
    # label_batch = label_batch[0]
    fig = plt.figure(figsize=(30, 30))
    plt.subplots_adjust(hspace=0.6)
    print("images in batch", len(image_batch), len(label_batch))
    num_images = len(image_batch)
    i = 0
    for n in range(num_images):
        predicted = ""

        if preds is not None:
            pred = preds[n]
            best_labels = np.argwhere(pred > prob_thresh).ravel()
            for lbl in best_labels:
                predicted = f"{predicted} {labels[lbl]}"
        # print("Y is ", label_batch[n])
        lbl = []
        for l_i, l in enumerate(label_batch[n]):
            if l > 0:
                lbl.append(labels[l_i])
        p = n
        ax = plt.subplot(num_images // 3 + 1, 3, p + 1)
        ax.get_xaxis().set_visible(False)

        i += 1
        # plot_spec(image_batch[n][:, :, 0], ax)
        # # plt.imshow(np.uint8(image_batch[n]))
        plot_title = f"{lbl}"
        if recs is not None:
            track = tracks[n].numpy().decode("utf8")
            rec = recs[n].numpy().decode("utf8")
            start_s = np.round(starts[n].numpy(), 1)
            signal_percent = np.round(signal_batch[n].numpy(), 1)
            plot_title = f"{plot_title} - {rec}:{track} at {start_s:.1f} sig {signal_percent:.1f}"
        plt.title(f"{plot_title}\n{predicted}")
        img = image_batch[n]
        ax.imshow(img)

        # plot_mel(image_batch[n][:, :, 0], ax)
        # np.save(f"dataset-images/batch-{batch_i}-{rec}-{start_s:.1f}.npy",image_batch[n])
    plt.savefig(f"dataset-images/batch-{batch_i}.png")
    # plt.show()


def plot_mfcc(mfccs, ax):
    img = librosa.display.specshow(mfccs.numpy(), x_axis="time", ax=ax)


def plot_mel(mel, ax):
    # power = librosa.db_to_power(mel.numpy())
    img = librosa.display.specshow(
        mel.numpy(),
        # librosa.power_to_db(mel.numpy(),ref=np.max),
        x_axis="time",
        y_axis="mel",
        sr=48000,
        fmax=11000,
        fmin=100,
        ax=ax,
        hop_length=HOP_LENGTH,
    )


def plot_spec(spec, ax):
    img = librosa.display.specshow(
        spec.numpy(), sr=48000, y_axis="log", x_axis="time", ax=ax
    )
    ax.set_title("Power spectrogram")
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")


def plot_spectrogram(spectrogram, ax):
    spectrogram = spectrogram[:, :, 0]
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.numpy().T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def init_logging():
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """

    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


# WEIGHT getting bird wrong less than getting specific specis wrong
# idea is  to insentivise learning specific birds
@tf.function
def weight_specific(x, y, num_labels, weighting, specific_mask, rest_weighting):
    # mask for all specifics
    specifics = tf.tensordot(y, specific_mask, 1)
    if specifics.shape == 0:
        return x, y
    specifics = tf.expand_dims(specifics, 1)
    bird_weighted = specifics * weighting
    rest = specifics - 1
    rest = tf.math.abs(rest)
    rest_weighting = tf.ones(num_labels)
    rest_weight = rest * rest_weighting
    mask = rest_weight + bird_weighted
    return x, y * mask


def get_weighting(dataset, labels):
    # weighting = {}
    # for i in range(len(labels)):
    #     if labels[i] in ["bird", "human"]:
    #         weighting[i] = 0.5
    #     elif labels[i] == "noise":
    #         weighting[i] = 0.9
    #     else:
    #         weighting[i] = 2
    # return weighting
    # excluded_labels = []
    dont_weigh = []
    # for l in labels:
    #     if l in ["human", "bird", "noise", "whistler", "morepork", "kiwi"]:
    #         continue
    #     dont_weigh.append(l)
    num_labels = len(labels)
    dist, _ = get_distribution(dataset, num_labels)
    for l, d in zip(labels, dist):
        print(l, "  : ", d)
    zeros = dist[dist == 0]
    non_zero_labels = num_labels - len(zeros)
    total = 0
    for d, l in zip(dist, labels):
        if l not in dont_weigh:
            total += d
    # total = np.sum(dist)
    weights = {}
    for i in range(num_labels):
        if labels[i] in dont_weigh:
            weights[i] = 1
        elif dist[i] == 0:
            weights[i] = 0
        else:
            weights[i] = (1 / dist[i]) * (total / non_zero_labels)
            # cap the weights
            weights[i] = min(weights[i], 4)
            weights[i] = max(weights[i], 0.25)
            # min(weight)
        print("WEights for ", labels[i], weights[i])
    return weights


from scipy.signal import butter, sosfilt, sosfreqz, freqs


# NOT USING AN OF THIS
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    btype = "lowpass"
    freqs = []
    if lowcut > 0:
        btype = "bandpass"
        low = lowcut / nyq
        freqs.append(low)
    if highcut > 0:
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


# @tf.function
# def raw_to_mel( x, features=False):
#     if features:
#         raw = x[2]
#     else:
#         raw = x

#     stft = tf.signal.stft(
#         raw,
#         4800,
#         HOP_LENGTH,
#         fft_length=4800,
#         window_fn=tf.signal.hann_window,
#         pad_end=True,
#         name=None,
#     )
#     stft = tf.transpose(stft, [1, 0])
#     stft = tf.math.abs(stft)
#     # stft = tf.reshape(stft, [2049, mel_s[1]])
#     image = tf.tensordot(MEL_WEIGHTS, stft, 1)
#     image = tf.expand_dims(image, axis=2)
#     if features:
#         x = (x[0], x[1], image)
#     else:
#         x = image
#     return x, y


@tf.function
def butter_bitterns(mask, input, y):
    logging.info("Butter bitterns %s", mask)
    logging.info("Input is %s", input.shape)
    if tf.math.reduce_all(tf.math.equal(y[0], mask)):
        shape = input.shape
        input = butter_function(input, 100, 500, 10)
        input.set_shape(shape)

    logging.info("Input now is %s", input.shape)

    return input, y


@tf.function
def normalize_std(data):

    mean = tf.reduce_mean(data)
    stddev = tf.math.reduce_std(data)

    # Add a small epsilon to avoid division by zero
    stddev = stddev + tf.keras.backend.epsilon()

    # 2. Standardize the data
    return (data - mean) / stddev


# normalize between 0 and 1
@tf.function
def normalize_minmax(data):

    max_v = tf.reduce_max(data)
    min_v = tf.reduce_min(data)
    return 2 * ((data - min_v) / (max_v - min_v)) - 1


@tf.function
def normalize_acoustic_fixed(data):
    # Lock the boundaries to absolute decibel limits
    max_v = 0.0  # Because librosa.power_to_db(..., ref=np.max) caps peaks at 0
    min_v = -80.0  # Standard bioacoustic dynamic range floor

    # Clip the data first so any ultra-quiet outliers don't break the math
    data_clipped = tf.clip_by_value(data, min_v, max_v)

    # Scale exactly to the [-1, 1] range
    return 2 * ((data_clipped - min_v) / (max_v - min_v)) - 1


def power_to_root_compressed(mel):
    # Apply a fixed power-law compression (equivalent to BirdNET's initial state)
    # 0.25 (quad-root) or 0.33 (cube-root) squashes the dynamic range perfectly
    compressed_mel = tf.math.pow(mel, 0.25)
    return compressed_mel


# equipvalent of librosa.power_to_db
@tf.function
def power_to_db(mel):
    ref_v = tf.reduce_max(mel)
    amin = 1e-10
    mel = 10.0 * tf.keras.ops.log10(tf.math.maximum(amin, mel))
    mel -= 10.0 * tf.keras.ops.log10(tf.math.maximum(amin, ref_v))
    mel = tf.math.maximum(mel, tf.reduce_max(mel) - 80)
    return mel


@tf.function
def normalize(input, y):

    if isinstance(input, tuple):
        x = input[0]
        print("GOt tuple input")
    else:
        x = input
    # peak normalize to [-1, 1], keeping 0 as silence
    peak = tf.math.reduce_max(tf.math.abs(x), -1, keepdims=True)
    peak = tf.math.maximum(peak, 1e-6)
    x = tf.math.divide(x, peak)
    if isinstance(input, tuple):
        print("Returning tuple")
        return (x, input[1], input[2]), y
    else:
        return x, y


@tf.function
def raw_to_mel(x, y, pcen=False):
    """Raw audio batch to [batch, mels, time, 3].

    With pcen the mels are left as magnitude for the PCEN layer in the model,
    otherwise they are power, for the MagTransform layer in the model.
    """
    if isinstance(x, tuple):
        raw = x[0]
    else:
        raw = x

    global FMIN, FMAX
    # if FMIN >50:
    fmin = 0
    fmax = 0
    if FMIN != 50:
        fmin = FMIN
    if FMAX != 11000:
        fmax = FMAX
    # logging.info("Applying butter %s %s", fmin, fmax)
    # not needed if using mel freq bin fmin and fmax
    # raw =  buttr_efunction(raw,fmin,fmax)

    stft = tf.signal.stft(
        raw,
        NFFT,
        HOP_LENGTH,
        fft_length=NFFT,
        window_fn=tf.signal.hann_window,
        pad_end=True,
        name=None,
    )
    print(
        "Using hop length ",
        HOP_LENGTH,
        " and nfft ",
        NFFT,
        " STFT shape is ",
        stft.shape,
        N_MELS,
    )
    stft = tf.transpose(stft, [0, 2, 1])
    stft = tf.math.abs(stft)
    if not pcen:
        # convert to power
        stft = tf.math.pow(stft, 2)
    batch_size = tf.keras.ops.shape(raw)[0]

    weights = tf.expand_dims(MEL_WEIGHTS, 0)
    weights = tf.repeat(weights, batch_size, 0)
    # [batch, mels, time] to match stored spectrograms
    image = tf.keras.backend.batch_dot(weights, stft)
    image = tf.expand_dims(image, axis=3)
    image = tf.repeat(image, 3, 3)

    if isinstance(x, tuple):
        x = (image, x[1], x[2])
    else:
        x = image
    return x, y


@tf.function
def butter_function(x, lowcut, highcut, order=2):
    x = tf.numpy_function(
        butter_bandpass_filter, [x, lowcut, highcut, 48000, order], tf.float32
    )
    return x


def butter_bandpass_filter(data, lowcut, highcut, fs=48000, order=2):
    if lowcut <= 0 and highcut <= 0:
        return data
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    if sos is None:
        return data
    filtered = sosfilt(sos, data)
    return np.float32(filtered)


def apply_pcen(x):
    x = librosa.pcen(x * (2**31), sr=48000, hop_length=281)
    return np.float32(x)


def pcen_function(x, y):
    x = tf.squeeze(x, 2)
    x = tf.numpy_function(apply_pcen, [x], tf.float32)

    x = tf.expand_dims(x, axis=2)
    return x, y


# def resample(dataset, new_labels, dist):
#     logging.info("RESAMPLING")
#     # seems the only way to get even distribution
#     label_ds = []
#     for i, l in enumerate(new_labels):
#         if dist[i] == 0:
#             continue
#         l_mask = np.zeros((len(new_labels)))
#         l_mask[i] = 1
#         # mask = tf.constant(mask, dtype=tf.float32)

#         l_filter = lambda x, y: tf.math.reduce_all(tf.math.equal(y, l_mask))
#         l_dataset = dataset.filter(l_filter)
#         l_dataset = l_dataset.shuffle(40096, reshuffle_each_iteration=True)

#         label_ds.append(l_dataset)
#     dataset = tf.data.Dataset.sample_from_datasets(
#         label_ds,
#         # weights=[1 / len(new_labels)] * len(new_labels),
#         stop_on_empty_dataset=True,
#         rerandomize_each_iteration=True,
#     )
#     return dataset


def filter_signal(x, y):
    return tf.math.greater(y[2], 0.0)
    # return tf.math.equal(tf.argmax(y[0]), 1)


# an attempt to filter out bad tracks by filtering out bird tracks
# that are predicted as noise or human
def filter_bad_tracks(x, y, labels):
    logging.info("Filtering bad tracks")
    # filtering before batching
    actual = y[0]
    # actual = tf.expand_dims(actual, 0)

    bad_preds = y[1]
    # bad_preds = tf.expand_dims(bad_preds, 0)
    # is bird
    num_labels = len(labels)
    bird_mask_2 = np.zeros(num_labels, dtype=bool)
    bird_mask_2[labels.index("bird")] = 1
    bird_mask_2 = tf.constant(bird_mask_2)
    label = tf.cast(actual, tf.bool)
    bird_l = tf.math.reduce_any(tf.math.logical_and(label, bird_mask_2))
    # , axis=1)

    noise_mask = np.zeros(num_labels, dtype=bool)
    bad_preds = tf.cast(bad_preds, tf.bool)
    # empty_l = tf.math.reduce_any(
    #     tf.math.logical_or(bad_preds, tf.constant(noise_mask)), axis=1
    # )
    # empty_l = tf.math.logical_not(empty_l)
    # dont filter on emtpy could be legit
    noise_mask[labels.index("noise")] = 1
    noise_mask[labels.index("human")] = 1

    # noise_mask[labels.index("human")] = 1

    noise_mask = tf.constant(noise_mask)
    # any that were predicted as noise or human
    pred_l = tf.math.reduce_any(tf.math.logical_and(bad_preds, noise_mask))  # , axis=1)

    # pred_l = tf.math.logical_or(pred_l, empty_l)
    pred_bird = tf.math.reduce_any(
        tf.math.logical_and(bad_preds, bird_mask_2)
    )  # , axis=1)

    # must be not bird and noise or human
    pred_bird = tf.logical_not(pred_bird)
    pred_l = tf.logical_and(pred_bird, pred_l)
    print(actual, " vs", bad_preds)
    print("leaves with following", tf.math.logical_and(bird_l, pred_l))
    # if a bird must have been predicted as a bird

    return tf.math.logical_and(bird_l, pred_l)


def filter_short(example):
    tfrecord_format = {"audio/raw_length": tf.io.FixedLenFeature((), tf.float32)}
    example = tf.io.parse_single_example(example, tfrecord_format)
    raw_length = tf.cast(example["audio/raw_length"], tf.float32)
    return raw_length != 2.0


if __name__ == "__main__":
    main()
