import sys
import math
import matplotlib.pyplot as plt

import tensorflow as tf
from functools import partial
import numpy as np
import time
import json
import logging
import librosa
import librosa.display
from custommels import mel_f
from pathlib import Path
import tensorflow_io as tfio
from audiomentations import AddBackgroundNoise, PolarityInversion, Compose
import soundfile as sf


BIRD_PATH = []
NOISE_PATH = []


MERGE_LABELS = {
    "house sparrow": "sparrow",
    "new zealand fantail": "fantail",
    "australian magpie": "magpie",
}

# seed = 1341
# tf.random.set_seed(seed)
# np.random.seed(seed)
AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64
NOISE_LABELS = ["insect", "wind", "vehicle", "dog", "rain", "static", "noise", "cat"]
SPECIFIC_BIRD_LABELS = [
    "bird",
    "whistler",
    "kiwi",
    "morepork",
    "rifleman",
    "sparrow",
    "fantail",
]
GENERIC_BIRD_LABELS = [
    "new zealand fantail",
    "australian magpie",
    "bellbird",
    "bird",
    "black noddy",
    "blackbird",
    "california quail",
    "canada goose",
    "common starling",
    "crimson rosella",
    "dunnock",
    "fantail",
    "grey warbler",
    "goldfinch",
    "house sparrow",
    "kiwi",
    "little owl",
    "magpie",
    "morepork",
    "norfolk gerygone",
    "norfolk parrot",
    "norfolk robin",
    "north island robin",
    "parakeet",
    "red-crowned parakeet",
    "rifleman",
    "robin",
    "sacred kingfisher",
    "silvereye",
    "slender-billed white-eye",
    "song thrush",
    "sooty tern",
    "sparrow",
    "spur-winged plover",
    "starling",
    "thrush",
    "tui",
    "whistler",
    "white tern",
]
EXTRA_LABELS = ["rooster", "frog", "insect", "human", "noise"]
OTHER_LABELS = []


# JUST FOR HUMAN OR NOT MODEL
# NOISE_LABELS.extend(SPECIFIC_BIRD_LABELS)
# NOISE_LABELS.extend(GENERIC_BIRD_LABELS)
# NOISE_LABELS.extend(OTHER_LABELS)
# keep_excluded_in_extra = False


def set_specific_by_count(meta):
    counts = meta["counts"]
    training = counts["train"]["sample_counts"]
    training_rec = counts["train"]["rec_counts"]

    validation = counts["validation"]["sample_counts"]

    # set counts to be counts of all merged labels
    for k, v in MERGE_LABELS.items():
        for dataset in [counts, training, training_rec, validation]:
            if k in dataset:
                total_count = dataset[k]
                if v in dataset:
                    total_count += dataset[v]
                dataset[k] = total_count
                logging.info("Adjusting count %s to %s", k, total_count)

                if v in dataset:
                    dataset[v] = total_count
                    logging.info("Adjusting count %s to %s", v, total_count)

    labels_with_data = []
    for label, count in training.items():
        rec_count = training_rec[label]
        if label not in validation:
            continue
        val_count = validation[label]
        if count > 100 and rec_count > 50 and val_count > 2:
            labels_with_data.append(label)
            if label not in GENERIC_BIRD_LABELS:
                logging.info("Have data for %s but not included ", label)
            if label in GENERIC_BIRD_LABELS and label not in SPECIFIC_BIRD_LABELS:
                SPECIFIC_BIRD_LABELS.append(label)
                logging.info(
                    "Using %s because have data samples: %s and recs %s val samples %s:",
                    label,
                    count,
                    rec_count,
                    val_count,
                )


def get_excluded_labels(labels):
    excluded_labels = []
    for l in labels:
        # FOR HUMAN MODEL
        # if l not in ["human", "noise"]:
        #     excluded_labels.append(l)
        # continue

        if l not in SPECIFIC_BIRD_LABELS and l not in EXTRA_LABELS:
            excluded_labels.append(l)
    return excluded_labels


# signals = Path("./signal-data/train")
# wavs = list(signals.glob("*.wav"))
# for w in wavs:
#     if "bird" in w.stem:
#         BIRD_PATH.append(w)
#     else:
#         for noise in NOISE_LABELS:
#             if noise in w.stem:
#                 NOISE_PATH.append(w)
#                 break
# BIRD_LABELS = ["bird"]
# NOISE_LABELS = []
# NOISE_PATH = NOISE_PATH[:2]
# BIRD_PATH = BIRD_PATH[:2]
# NOISE_LABELS = []
insect = None
fp = None
HOP_LENGTH = 281
N_MELS = 160
SR = 48000
BREAK_FREQ = 1000
MEL_WEIGHTS = mel_f(48000, N_MELS, 50, 11000, 4800, BREAK_FREQ)
MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)
DIMENSIONS = (160, 188)

mel_s = (N_MELS, 513)
sftf_s = (2401, 188)
mfcc_s = (20, 188)
DIMENSIONS = (*mel_s, 1)
YAMNET_EMBEDDING_SHAPE = (6, 1024)
EMBEDDING_SHAPE = 1280
# TEST STUFF to blockout frequencies
# mel_bins = librosa.mel_frequencies(128, fmax=48000 / 2)
# human_lowest = np.where(mel_bins < 60)[-1][-1]
# human_max = np.where(mel_bins > 180)[0][0]

#
# # 60-180hz
# human_mel = (human_lowest, human_max)
# human_mask = np.zeros((mel_s), dtype=bool)
# human_mask[human_mel[0] : human_mel[0] + human_mel[1]] = 1

# 600-1200
# frequency_min = 600
# frequency_max = 1200
# more_lower = np.where(mel_bins < 600)[-1][-1]
# more_max = np.where(mel_bins > 1200)[0][0]
#
#
# morepork_mel = (more_lower, more_max)
#
# morepork_mask = np.zeros((mel_s), dtype=bool)
# morepork_mask[morepork_mel[0] : morepork_mel[0] + morepork_mel[1]] = 1
#
# with open(str("zvalues.txt"), "r") as f:
#     zvals = json.load(f)
#
# zvals["mean"] = np.array(zvals["mean"])
# zvals["std"] = np.array(zvals["std"])
Z_NORM = False
# Z_NORM = True
import random


def load_dataset(filenames, num_labels, labels, args):
    random.shuffle(filenames)
    #
    #     image_size,
    deterministic = args.get("deterministic", False)
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

    image_size = args["image_size"]
    labeled = args.get("labeled", True)
    augment = args.get("augment", False)
    preprocess_fn = args.get("preprocess_fn")
    one_hot = args.get("one_hot", True)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    # dataset = dataset.filter(filter_short)
    dataset = dataset.map(
        partial(
            read_tfrecord,
            num_labels=num_labels,
            image_size=image_size,
            labeled=labeled,
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
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    if args.get("filter_bad", False):
        dataset = dataset.filter(lambda x, y: not filter_bad_tracks(x, y, labels))
    if not args.get("only_features", False):
        if args.get("features"):
            filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x[2]))
        else:
            filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x))
        dataset = dataset.filter(filter_nan)

        filter_excluded = lambda x, y: not tf.math.equal(tf.math.count_nonzero(y[0]), 0)
    else:
        filter_excluded = lambda x, y: tf.math.greater(y[0], -1)

    dataset = dataset.filter(filter_excluded)
    return dataset


def filter_short(example):
    tfrecord_format = {"audio/raw_length": tf.io.FixedLenFeature((), tf.float32)}
    example = tf.io.parse_single_example(example, tfrecord_format)
    raw_length = tf.cast(example["audio/raw_length"], tf.float32)
    return raw_length != 2.0


def preprocess(data):
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return tf.keras.applications.inception_v3.preprocess_input(x), y


def get_distribution(dataset, num_labels, batched=True, one_hot=True):
    true_categories = [y for x, y in dataset]
    dist = np.zeros((num_labels), dtype=np.float32)
    if len(true_categories) == 0:
        return dist

    if len(true_categories) == 0:
        return dist
    if batched:
        true_categories = tf.concat(true_categories, axis=0)
    if len(true_categories) == 0:
        return dist
    classes = []
    if one_hot:
        for y in true_categories:
            non_zero = tf.where(y).numpy()
            classes.extend(non_zero.flatten())
            classes = np.array(classes)
    else:
        classes = np.array(true_categories)
    c = Counter(list(classes))
    for i in range(num_labels):
        dist[i] = c[i]
    return dist


def get_remappings(
    labels, excluded_labels, keep_excluded_in_extra=True, use_generic_bird=True
):
    extra_label_map = {}
    # remapped = {}
    re_dic = {}
    new_labels = labels.copy()
    for excluded in excluded_labels:
        if excluded in labels:
            new_labels.remove(excluded)
    for l in labels:
        if l in excluded_labels:
            re_dic[l] = -1
            # remapped[l] = []
            logging.info("Excluding %s", l)
        else:
            if l in MERGE_LABELS and MERGE_LABELS[l] in labels:
                logging.info("Re labeiling %s as %s", l, MERGE_LABELS[l])
                re_dic[l] = new_labels.index(MERGE_LABELS[l])
            else:
                re_dic[l] = new_labels.index(l)
            # remapped[l] = [l]
            # values.append(new_labels.index(l))
    if not use_generic_bird:
        re_dic["bird"] = -1
    re_dic["slender-billed white-eye"] = -1

    master_keys = []
    master_values = []
    if not keep_excluded_in_extra:
        labels = new_labels
    for l in labels:
        print("Remapping", l)
        if l in NOISE_LABELS:
            if "noise" in new_labels:
                remap_label = "noise"
                extra_label_map[l] = new_labels.index("noise")
            continue
        elif l in OTHER_LABELS:
            if "other" in new_labels:
                extra_label_map[l] = new_labels.index("other")
            continue
        elif l in SPECIFIC_BIRD_LABELS:
            if not use_generic_bird:
                continue
            if "bird" in new_labels:
                if l != "bird":
                    extra_label_map[l] = new_labels.index("bird")
            # or l == "human":
            continue
        elif l in GENERIC_BIRD_LABELS:
            if not use_generic_bird:
                continue
            remap_label = "bird"
            if l != "bird":
                extra_label_map[l] = new_labels.index("bird")
        else:
            continue
        if l == remap_label:
            continue
        if l in excluded_labels:
            continue
        # remapped[remap_label].append(l)
        re_dic[l] = new_labels.index(remap_label)
        # del remapped[l]
    return (extra_label_map, re_dic, new_labels)


bird_i = None
noise_i = None
bird_mask = None


def get_dataset(filenames, labels, **args):
    excluded_labels = args.get("excluded_labels", [])
    use_generic_bird = args.get("use_generic_bird", True)

    global extra_label_map
    global remapped_y

    extra_label_map, remapped, labels = get_remappings(
        labels, excluded_labels, use_generic_bird=use_generic_bird
    )

    remapped_y = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(remapped.keys())),
            values=tf.constant(list(remapped.values())),
        ),
        default_value=tf.constant(-1),
        name="remapped_y",
    )
    logging.info(
        "Remapped %s extra mapping %s new labels %s Use gen bird %s",
        remapped,
        extra_label_map,
        labels,
        use_generic_bird,
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
    extra_label_map = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(extra_label_map.keys())),
            values=tf.constant(list(extra_label_map.values())),
        ),
        default_value=tf.constant(-1),
        name="extra_label_map",
    )

    # print("keys", keys, " values", values)
    # 1 / 0
    num_labels = len(labels)
    dataset = load_dataset(filenames, num_labels, labels, args)
    if args.get("only_features", False):
        bird_mask = tf.constant(bird_i, dtype=tf.float32)
        bird_filter = lambda x, y: tf.math.equal(y[0], bird_mask)
        others_filter = lambda x, y: not tf.math.equal(y[0], bird_mask)
    else:
        bird_mask = np.zeros(num_labels, dtype=bool)
        bird_mask[bird_i] = 1
        bird_mask = tf.constant(bird_mask)
        bird_filter = lambda x, y: tf.math.reduce_all(
            tf.math.equal(tf.cast(y[0], tf.bool), bird_mask)
        )
        others_filter = lambda x, y: not tf.math.reduce_all(
            tf.math.equal(tf.cast(y[0], tf.bool), bird_mask)
        )
    bird_dataset = dataset.filter(bird_filter)
    if args.get("filter_signal") is not None:
        logging.info("Filtering signal by percent 0.1")
        bird_dataset = bird_dataset.filter(filter_signal)

    dataset = dataset.filter(others_filter)
    datasets = [dataset]
    # may perform better without adding generics birds but sitll having generic label
    if use_generic_bird:
        logging.info("Not adding generic bird tags as found performas better")
        # datasets.append(bird_dataset)
    if args.get("filenames_2") is not None:
        logging.info("Loading second files %s", args.get("filenames_2")[:1])
        second = args.get("filenames_2")
        # bird_c = dist[labels.index("bird")]

        args["no_bird"] = True
        # added bird noise to human recs but it messes model, so dont use for now
        dataset_2 = load_dataset(second, len(labels), labels, args)

        datasets.append(dataset_2)
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets,
            stop_on_empty_dataset=args.get("stop_on_empty", True),
            rerandomize_each_iteration=args.get("rerandomize_each_iteration", True),
        )
        # for i, d in enumerate(dist):
        # dist[i] += dist_2[i]
    else:
        logging.info("Not using second")
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets,
            stop_on_empty_dataset=args.get("stop_on_empty", True),
            rerandomize_each_iteration=args.get("rerandomize_each_iteration", True),
        )
    # logging.info("Filtering freq %s", args.get("filter_freq", False))
    # filter freq done in writing stage to speed up
    logging.info("args %s", args)

    if not args.get("only_features", False):
        dataset = dataset.map(
            lambda x, y: raw_to_mel(x, y, args.get("features", False))
        )

    dataset = dataset.map(lambda x, y: (x, y[0]))
    resample_data = args.get("resample", False)
    if resample_data:
        logging.info("Resampling data")
        dataset = resample(dataset, labels, dist)
    pcen = args.get("pcen", False)
    if pcen:
        logging.info("Taking PCEN")
        dataset = dataset.map(lambda x, y: pcen_function(x, y))

    # dist = get_distribution(dataset, num_labels, batched=False,one_hot=not args.get("only_features"))
    # epoch_size = np.sum(dist)
    # tf complains about running out of data if i dont specify the size????
    # dataset = dataset.take(epoch_size)
    batch_size = args.get("batch_size", None)
    dataset = dataset.cache()
    if args.get("shuffle", True) and args.get("only_features") == False:
        dataset = dataset.shuffle(
            4096, reshuffle_each_iteration=args.get("reshuffle", True)
        )
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

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
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset, remapped, 0


def filter_signal(x, y):
    return tf.math.greater(y[2], 0.1)
    # return tf.math.equal(tf.argmax(y[0]), 1)


# an attempt to filter out bad tracks by filtering out bird tracks
# that are predicted as noise or human
def filter_bad_tracks(x, y, labels):
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
    dist = get_distribution(dataset, num_labels)
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


@tf.function
def raw_to_mel(x, y, features=False):
    print("FEATURES", features)
    if features:
        raw = x[2]
        print("Using x[2]", raw.shape)
    else:
        raw = x

    stft = tf.signal.stft(
        raw,
        4800,
        HOP_LENGTH,
        fft_length=4800,
        window_fn=tf.signal.hann_window,
        pad_end=True,
        name=None,
    )
    stft = tf.transpose(stft, [1, 0])
    stft = tf.math.abs(stft)
    # stft = tf.reshape(stft, [2401, mel_s[1]])
    image = tf.tensordot(MEL_WEIGHTS, stft, 1)
    image = tf.expand_dims(image, axis=2)
    if features:
        x = (x[0], x[1], image)
    else:
        x = image
    return x, y


def butter_function(x, lowcut, highcut):
    x = tf.numpy_function(butter_bandpass_filter, [x, lowcut, highcut], tf.float32)
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


@tf.function
def read_tfrecord(
    example,
    image_size,
    num_labels,
    labeled,
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
):
    bird_l = tf.constant(["bird"])
    # tf_more_mask = tf.constant(morepork_mask)
    # tf_human_mask = tf.constant(human_mask)
    tfrecord_format = {"audio/class/text": tf.io.FixedLenFeature((), tf.string)}
    # "audio/sftf": tf.io.FixedLenFeature([sftf_s[0] * sftf_s[1]], dtype=tf.float32),
    # "audio/mel": tf.io.FixedLenFeature([mel_s[0] * mel_s[1]], dtype=tf.float32),
    # "audio/mfcc": tf.io.FixedLenFeature([mfcc_s[0] * mfcc_s[1]], dtype=tf.float32),
    # "audio/class/label": tf.io.FixedLenFeature((), tf.int64),
    # "audio/length": tf.io.FixedLenFeature((), tf.int64),
    # "audio/sftf_w": tf.io.FixedLenFeature((), tf.int64),
    # "audio/sftf_h": tf.io.FixedLenFeature((), tf.int64),
    # "audio/mel_w": tf.io.FixedLenFeature((), tf.int64),
    # "audio/mel_h": tf.io.FixedLenFeature((), tf.int64),
    # "audio/mfcc_w": tf.io.FixedLenFeature((), tf.int64),
    # "audio/raw": tf.io.FixedLenFeature(
    #     [
    #         120000,
    #     ],
    #     dtype=tf.float32,
    # ),
    tfrecord_format["audio/rec_id"] = tf.io.FixedLenFeature((), tf.string)
    tfrecord_format["audio/track_id"] = tf.io.FixedLenFeature((), tf.string)
    if embeddings:
        logging.info("Loading embeddings")

        tfrecord_format["embedding"] = tf.io.FixedLenFeature(
            EMBEDDING_SHAPE, tf.float32
        )

    elif not only_features:
        logging.info("Loading sft audio")

        tfrecord_format["audio/raw"] = tf.io.FixedLenFeature((48000 * 3), tf.float32)

        tfrecord_format["audio/buttered"] = tf.io.FixedLenFeature(
            (48000 * 3), tf.float32, default_value=tf.zeros((48000 * 3))
        )

        # tfrecord_format["audio/raw"] = tf.io.FixedLenFeature(
        #     (2401, mel_s[1]), tf.float32
        # )
        # tfrecord_format["audio/min_freq"] = tf.io.FixedLenFeature((), tf.float32)
        # tfrecord_format["audio/max_freq"] = tf.io.FixedLenFeature((), tf.float32)
    if features or only_features:
        tfrecord_format["audio/short_f"] = tf.io.FixedLenFeature((68 * 60), tf.float32)
        tfrecord_format["audio/mid_f"] = tf.io.FixedLenFeature((136 * 3), tf.float32)

    tfrecord_format["audio/signal_percent"] = tf.io.FixedLenFeature((), tf.float32)

    example = tf.io.parse_single_example(example, tfrecord_format)
    # raw = example["audio/raw"]

    label = tf.cast(example["audio/class/text"], tf.string)
    split_labels = tf.strings.split(label, sep="\n")
    global remapped_y, extra_label_map
    labels = remapped_y.lookup(split_labels)
    if multi:
        extra = extra_label_map.lookup(split_labels)
        labels = tf.concat([labels, extra], axis=0)
    else:
        labels = tf.reduce_max(labels)
    embed_preds = None
    if embeddings:
        image = example["embedding"]

    elif not only_features:
        buttered = example["audio/buttered"] if filter_freq else None
        if filter_freq and tf.math.count_nonzero(buttered) > 0:
            if random_butter > 0:
                rand = tf.random.uniform((), 0, 1)
                # do butter pass 3/5ths of the time
                raw = tf.cond(
                    rand <= random_butter,
                    lambda: tf.identity(example["audio/buttered"]),
                    lambda: tf.identity(example["audio/raw"]),
                )
            else:
                logging.info("USing buttered")
                raw = example["audio/buttered"]
        else:
            raw = example["audio/raw"]

    if features or only_features:
        short_f = example["audio/short_f"]
        mid_f = example["audio/mid_f"]
        if only_features:
            raw = tf.concat((short_f, mid_f), axis=0)
        else:
            mid_f = tf.reshape(mid_f, (136, 3))
            short_f = tf.reshape(short_f, (68, 60))
            raw = (short_f, mid_f, raw)
        # raw = tf.expand_dims(raw, axis=0)
    if augment:
        logging.info("Augmenting")
    if mean_sub:
        print("Subbing mean")
        mel_m = tf.reduce_mean(mel, axis=1)
        # gp not sure to mean over axis 0 or 1
        mel_m = tf.expand_dims(mel_m, axis=1)
        # mean over each mel bank
        mel = mel - mel_m
    if labeled:
        # label = tf.cast(example["audio/class/label"], tf.int32)

        if one_hot and not only_features:
            label = tf.reduce_max(
                tf.one_hot(labels, num_labels, dtype=tf.int32), axis=0
            )
            if embed_preds is not None:
                embed_preds = tf.reduce_max(
                    tf.one_hot(embed_preds, num_labels, dtype=tf.int32), axis=0
                )
        else:
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

        return raw, (
            label,
            embed_preds,
            signal_percent,
            # min_freq,
            # max_freq,
            example["audio/rec_id"],
            example["audio/track_id"],
        )

    return image


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
        image_size=DIMENSIONS,
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


# test stuff
def main():
    init_logging()

    # return
    datasets = ["other-training-data", "training-data", "chime-training-data"]
    datasets = ["training-data"]
    datasets = ["./audio-data/training-data"]
    filenames = []
    labels = set()
    for d in datasets:
        # file = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/training-meta.json"
        file = f"{d}/training-meta.json"
        with open(file, "r") as f:
            meta = json.load(f)
        labels.update(meta.get("labels", []))
        # print("loaded labels", labels)
        # species_list = ["bird", "human", "rain", "other"]

        # filenames = tf.io.gfile.glob(f"./training-data/validation/*.tfrecord")
        filenames.extend(tf.io.gfile.glob(f"{d}/validation/*.tfrecord"))
    labels.add("bird")
    labels.add("noise")
    labels = list(labels)
    set_specific_by_count(meta)
    excluded_labels = get_excluded_labels(labels)
    labels.sort()

    filenames_2 = tf.io.gfile.glob(f"./flickr-training-data/validation/*.tfrecord")
    # dir = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/validation"
    # weights = [0.5] * len(labels)
    start = time.time()
    resampled_ds, remapped, _ = get_dataset(
        # dir,
        filenames,
        labels,
        use_generic_bird=False,
        batch_size=32,
        image_size=DIMENSIONS,
        augment=False,
        resample=False,
        excluded_labels=excluded_labels,
        stop_on_empty=False,
        filter_freq=True,
        random_butter=0.9,
        only_features=True,
        multi_label=False
        # filenames_2=filenames_2
        # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
    )
    dist = get_distribution(resampled_ds, len(labels), batched=True, one_hot=False)
    for l, d in zip(labels, dist):
        print(f"{l} has {d}")
    for e in range(1):
        for x, y in resampled_ds:
            print(x.shape)

            print("took", time.time() - start)
            show_batch(x, y, labels)
            print("X batch of ", x.shape, " Has memory of ", getsize(np.array(x)), "MB")


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


def show_batch(image_batch, label_batch, labels):
    # min_freq = label_batch[3]
    # max_freq = label_batch[4]
    # recs = label_batch[3]
    # tracks = label_batch[4]
    # label_batch = label_batch[0]
    fig = plt.figure(figsize=(20, 20))
    print("images in batch", len(image_batch), len(label_batch))
    num_images = len(image_batch)
    i = 0
    for n in range(num_images):
        lbl = []
        for l_i, l in enumerate(label_batch[n]):
            if l == 1:
                lbl.append(labels[l_i])
        p = n
        i += 1
        ax = plt.subplot(num_images // 3 + 1, 3, p + 1)
        # plot_spec(image_batch[n][:, :, 0], ax)
        # # plt.imshow(np.uint8(image_batch[n]))
        spc = None
        plt.title(f"{lbl} ({spc}")
        img = image_batch[n]
        plot_mel(image_batch[n][:, :, 0], ax)

    plt.show()


def plot_mfcc(mfccs, ax):
    img = librosa.display.specshow(mfccs.numpy(), x_axis="time", ax=ax)


def plot_mel(mel, ax):
    # power = librosa.db_to_power(mel.numpy())
    img = librosa.display.specshow(
        mel.numpy(),
        x_axis="time",
        y_axis="mel",
        sr=48000,
        fmax=11000,
        fmin=50,
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


if __name__ == "__main__":
    main()
