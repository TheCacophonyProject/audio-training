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


# seed = 1341
# tf.random.set_seed(seed)
# np.random.seed(seed)
AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64
NOISE_LABELS = ["wind", "vehicle", "dog", "rain", "static", "noise", "cat"]
SPECIFIC_BIRD_LABELS = ["whistler", "kiwi", "morepork", "bird"]
GENERIC_BIRD_LABELS = [
    "australian magpie",
    "bellbird",
    "bird",
    "blackbird",
    "california quail",
    "canada goose",
    "common starling",
    "crimson rosella",
    "fantail",
    "grey warbler",
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

OTHER_LABELS = ["chicken", "rooster", "frog", "insect"]
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
N_MELS = 120
SR = 48000
BREAK_FREQ = 1750
MEL_WEIGHTS = mel_f(48000, N_MELS, 50, 11000, 4800, BREAK_FREQ)
MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)
DIMENSIONS = (160, 188)

mel_s = (120, 428)
sftf_s = (2401, 188)
mfcc_s = (20, 188)
DIMENSIONS = mel_s
mel_bins = librosa.mel_frequencies(128, fmax=48000 / 2)
human_lowest = np.where(mel_bins < 60)[-1][-1]
human_max = np.where(mel_bins > 180)[0][0]


# 60-180hz
human_mel = (human_lowest, human_max)
human_mask = np.zeros((mel_s), dtype=bool)
human_mask[human_mel[0] : human_mel[0] + human_mel[1]] = 1

# 600-1200
# frequency_min = 600
# frequency_max = 1200
more_lower = np.where(mel_bins < 600)[-1][-1]
more_max = np.where(mel_bins > 1200)[0][0]


morepork_mel = (more_lower, more_max)

morepork_mask = np.zeros((mel_s), dtype=bool)
morepork_mask[morepork_mel[0] : morepork_mel[0] + morepork_mel[1]] = 1

with open(str("zvalues.txt"), "r") as f:
    zvals = json.load(f)

zvals["mean"] = np.array(zvals["mean"])
zvals["std"] = np.array(zvals["std"])
Z_NORM = False
# Z_NORM = True


def load_dataset(filenames, num_labels, args):
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
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)

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
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x[1]))
    dataset = dataset.filter(filter_nan)

    filter_excluded = lambda x, y: not tf.math.equal(tf.math.count_nonzero(y), 0)
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


def get_distribution(dataset, batched=True):
    true_categories = [y for x, y in dataset]
    if len(true_categories) == 0:
        return None
    num_labels = len(true_categories[0])
    dist = np.zeros((num_labels), dtype=np.float32)

    if len(true_categories) == 0:
        return dist
    if batched:
        true_categories = tf.concat(true_categories, axis=0)
    if len(true_categories) == 0:
        return dist
    classes = []
    for y in true_categories:
        non_zero = tf.where(y).numpy()
        classes.extend(non_zero.flatten())
    classes = np.array(classes)

    c = Counter(list(classes))
    for i in range(num_labels):
        dist[i] = c[i]
    return dist


def get_remappings(labels, excluded_labels, keep_excluded_in_extra=True):
    extra_label_map = {}
    remapped = {}
    re_dic = {}
    new_labels = labels.copy()
    for excluded in excluded_labels:
        if excluded in labels:
            new_labels.remove(excluded)
    for l in labels:
        if l in excluded_labels:
            re_dic[l] = -1
            remapped[l] = []
            logging.info("Excluding %s", l)
        else:
            re_dic[l] = new_labels.index(l)
            remapped[l] = [l]
            # values.append(new_labels.index(l))

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
            if l != "bird":
                extra_label_map[l] = new_labels.index("bird")
            # or l == "human":
            continue
        elif l == "human":
            # if "noise" in new_labels:
            #     extra_label_map[l] = new_labels.index("noise")

            continue
        elif l in GENERIC_BIRD_LABELS:
            remap_label = "bird"
            if l != "bird":
                extra_label_map[l] = new_labels.index("bird")
        else:
            continue
        if l == remap_label:
            continue
        if l in excluded_labels:
            continue
        remapped[remap_label].append(l)
        re_dic[l] = new_labels.index(remap_label)
        del remapped[l]
    return (extra_label_map, re_dic, new_labels)


bird_i = None
noise_i = None


def get_dataset(filenames, labels, **args):
    excluded_labels = args.get("excluded_labels", [])

    global extra_label_map
    global remapped_y

    extra_label_map, remapped, labels = get_remappings(labels, excluded_labels)
    remapped_y = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(remapped.keys())),
            values=tf.constant(list(remapped.values())),
        ),
        default_value=tf.constant(-1),
        name="remapped_y",
    )
    logging.info(
        "Remapped %s extra mapping %s new labels %s", remapped, extra_label_map, labels
    )
    global bird_i
    global noise_i
    bird_i = labels.index("bird")
    noise_i = labels.index("noise")

    # extra tags, since we have multi label problem, morepork is a bird and morepork
    # cat is a cat but also "noise"
    extra_label_map["-10"] = -10
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
    dataset = load_dataset(filenames, len(labels), args)
    bird_mask = np.zeros(num_labels, dtype=bool)
    bird_mask[bird_i] = 1
    bird_mask = tf.constant(bird_mask)
    bird_filter = lambda x, y: tf.math.reduce_all(
        tf.math.equal(tf.cast(y, tf.bool), bird_mask)
    )
    bird_dataset = dataset.filter(bird_filter)
    others_filter = lambda x, y: not tf.math.reduce_all(
        tf.math.equal(tf.cast(y, tf.bool), bird_mask)
    )
    dataset = dataset.filter(others_filter)

    other_dist = get_distribution(dataset, batched=False)
    for i, d in enumerate(other_dist):
        logging.info("Non Bird Have %s for %s", d, labels[i])
    bird_dist = get_distribution(bird_dataset, batched=False)
    for i, d in enumerate(bird_dist):
        logging.info("Bird D Have %s for %s", d, labels[i])
    # dist = get_distribution(dataset, batched=False)

    resample_data = args.get("resample", True)
    non_bird_c = np.sum(other_dist)
    if args.get("filenames_2") is not None:
        second = args.get("filenames_2")
        # bird_c = dist[labels.index("bird")]

        args["no_bird"] = True
        # added bird noise to human recs but it messes model, so dont use for now
        dataset_2 = load_dataset(second, len(labels), args)
        # dataset = dataset.take(min(np.sum(dist_2), 5000))

        # if not resample_data:
        # bird_c = bird_c - dist[labels.index("human")]
        # dataset_2 = dataset_2.take(bird_c)
        logging.info("Taking %s", non_bird_c)
        bird_dataset = bird_dataset.take(non_bird_c)
        # logging.info("concatenating second dataset %s", second[0])
        # dist = get_distribution(dataset_2, batched=False)
        # for i, d in enumerate(dist_2):
        # logging.info("Second dataset pre taking have %s for %s", d, labels[i])
        dataset = tf.data.Dataset.sample_from_datasets(
            [bird_dataset, dataset, dataset_2],
            # stop_on_empty_dataset=args.get("stop_on_empty", True),
            rerandomize_each_iteration=True,
        )
        # for i, d in enumerate(dist):
        # dist[i] += dist_2[i]
    else:
        dataset = tf.data.Dataset.sample_from_datasets(
            [bird_dataset, dataset],
            stop_on_empty_dataset=args.get("stop_on_empty", True),
            rerandomize_each_iteration=True,
        )
    resample_data = args.get("resample", True)
    if resample_data:
        logging.info("Resampling data")
        dataset = resample(dataset, labels, dist)
    # if args.get("shuffle", True):
    #     dataset = dataset.shuffle(
    #         4096, reshuffle_each_iteration=args.get("reshuffle", True)
    #     )
    # tf refues to run if epoch sizes change so we must decide a costant epoch size even though with reject res
    # it will chang eeach epoch, to ensure this take this repeat data and always take epoch_size elements
    # epoch_size = len([0 for x, y in dataset])

    dist = get_distribution(dataset, batched=False)
    for i, d in enumerate(dist):
        logging.info("Have %s for %s", d, labels[i])
    epoch_size = np.sum(dist)
    logging.info("Setting dataset size to %s", epoch_size)
    # if not args.get("only_features", False):
    # dataset = dataset.repeat(2)
    scale_epoch = args.get("scale_epoch", None)
    if scale_epoch:
        epoch_size = epoch_size // scale_epoch
    # dataset = dataset.take(epoch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    batch_size = args.get("batch_size", None)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    if args.get("shuffle", True):
        dataset = dataset.shuffle(
            4096, reshuffle_each_iteration=args.get("reshuffle", True)
        )

    return dataset, remapped, epoch_size


def get_weighting(dataset, labels):
    weighting = {}
    for i in range(len(labels)):
        weighting[i] = 1
    return weighting
    excluded_labels = []
    dont_weigh = []
    for l in labels:
        if l in ["human", "bird", "noise", "whistler", "morepork", "kiwi"]:
            continue
        dont_weigh.append(l)
    num_labels = len(labels)
    dist = get_distribution(dataset)
    zeros = dist[dist == 0]
    non_zero_labels = num_labels - len(zeros)
    total = 0
    for d, l in zip(dist, labels):
        if l not in dont_weigh:
            total += d
    # total = np.sum(dist)
    weights = {}
    for i in range(num_labels):
        weights[i] = 1
        continue

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


def resample(dataset, labels, og):
    target_dist = np.empty((len(labels)), dtype=np.float32)
    target_dist[:] = 1 / len(labels)

    rej = dataset.rejection_resample(
        class_func=class_func,
        target_dist=target_dist,
    )
    dataset = rej.map(lambda extra_label, features_and_label: features_and_label)
    return dataset


def resample_old(dataset, labels):
    excluded_labels = []
    # excluded_labels = ["human", "bird", "morepork", "kiwi"]
    num_labels = len(labels)
    true_categories = [y for x, y in dataset]
    if len(true_categories) == 0:
        return None
    true_categories = np.int64(tf.argmax(true_categories, axis=1))
    c = Counter(list(true_categories))
    print("COUNTER", c)
    dist = np.empty((num_labels), dtype=np.float32)
    target_dist = np.empty((num_labels), dtype=np.float32)
    for i in range(num_labels):
        if labels[i] in excluded_labels:
            dist[i] = 0
            logging.info("Excluding %s for %s", c[i], labels[i])

        else:
            dist[i] = c[i]

            logging.info("Have %s for %s", dist[i], labels[i])
    # GP TESTING Just to remove some labels
    dist = dist / np.sum(dist)

    zeros = dist[dist == 0]
    non_zero_labels = num_labels - len(zeros)
    target_dist[:] = 1 / non_zero_labels

    dist = dist / np.sum(dist)
    dist_max = np.max(dist)
    # really this is what we want but when the values become too small they never get sampled
    # so need to try reduce the large gaps in distribution
    # can use class weights to adjust more, or just throw out some samples
    max_range = target_dist[0] / 2
    for i in range(num_labels):
        if dist[i] == 0:
            target_dist[i] = 0
        # elif dist_max - dist[i] > (max_range * 2):
        # target_dist[i] = dist[i]
        # print("adjusting for ", labels[i])
        target_dist[i] = max(0, target_dist[i])
    target_dist = target_dist / np.sum(target_dist)
    rej = dataset.rejection_resample(
        class_func=class_func,
        target_dist=target_dist,
    )
    dataset = rej.map(lambda extra_label, features_and_label: features_and_label)
    return dataset


def mel_from_raw(raw):
    # # if add_noise:
    # #     logging.info("Adding noise to dataset")
    # #     rand_i = tf.random.uniform(shape=[1])[0]
    # #     if tf.math.greater(rand_i, 0.5):
    # #         if tf.math.greater(rand_i, 0.75):
    # #             raw = tf.numpy_function(
    # #                 apply_bird,
    # #                 inp=[raw],
    # #                 Tout=tf.float32,
    # #                 name="apply_pipeline",
    # #             )
    # #
    # #             # make sure bird in labels
    # #             extra = remapped_y.lookup(bird_l)
    # #             labels = tf.concat([labels, extra], axis=0)
    # #
    # #         else:
    # #             # noise noise
    # #             raw = tf.numpy_function(
    # #                 apply_noise,
    # #                 inp=[raw],
    # #                 Tout=tf.float32,
    # #                 name="apply_pipeline",
    # #             )
    # # augment = AddBackgroundNoise(
    # #     sounds_path=[NOISE_PATH],
    # #     min_snr_in_db=3.0,
    # #     max_snr_in_db=30.0,
    # #     noise_transform=PolarityInversion(),
    # #     p=1,
    # # )
    #
    # # raw = augment(raw)
    #
    # # make sure bird in labels
    # # extra = remapped_y.lookup(bird_l)
    # labels = tf.concat([labels, extra], axis=0)
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
    # if you want power
    # stft = tf.math.square(stft)
    mel = tf.tensordot(MEL_WEIGHTS, stft, 1)
    # mel = tfio.audio.dbscale(mel, top_db=80)

    mel = tf.expand_dims(mel, 2)
    return mel


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
):
    bird_l = tf.constant(["bird"])
    tf_more_mask = tf.constant(morepork_mask)
    tf_human_mask = tf.constant(human_mask)
    tfrecord_format = {
        # "audio/sftf": tf.io.FixedLenFeature([sftf_s[0] * sftf_s[1]], dtype=tf.float32),
        # "audio/mel": tf.io.FixedLenFeature([mel_s[0] * mel_s[1]], dtype=tf.float32),
        # "audio/mfcc": tf.io.FixedLenFeature([mfcc_s[0] * mfcc_s[1]], dtype=tf.float32),
        # "audio/class/label": tf.io.FixedLenFeature((), tf.int64),
        "audio/class/text": tf.io.FixedLenFeature((), tf.string),
        # "audio/length": tf.io.FixedLenFeature((), tf.int64),
        "audio/raw": tf.io.FixedLenFeature((2401, 428), tf.float32),
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
    }

    example = tf.io.parse_single_example(example, tfrecord_format)
    # raw = example["audio/raw"]
    label = tf.cast(example["audio/class/text"], tf.string)
    labels = tf.strings.split(label, sep="\n")
    global remapped_y, extra_label_map
    extra = extra_label_map.lookup(labels)
    labels = remapped_y.lookup(labels)
    labels = tf.concat([labels, extra], axis=0)
    stft = example["audio/raw"]
    stft = tf.reshape(stft, [2401, 428])
    mel = tf.tensordot(MEL_WEIGHTS, stft, 1)
    # mel =
    # mel = example["audio/mel"]
    # mel = tf.reshape(mel, [*mel_s])
    # # put mel into ref db
    # a_max = tf.math.reduce_max(mel)
    # a_min = tf.math.reduce_min(mel)
    # m_range = a_max - a_min
    # mel = 80 * (mel - a_min) / m_range
    # # mel_no_ref = 80 * mel_no_ref
    # mel -= 80
    mel = tf.expand_dims(mel, axis=2)
    # mel = tf.repeat(mel, 3, axis=2)
    if augment:
        logging.info("Augmenting")
        # tf.random.uniform()
        # mel = tfio.audio.freq_mask(mel, param=10)
        # mel = tfio.audio.time_mask(mel, param=10)
    if mean_sub:
        print("Subbing mean")
        mel_m = tf.reduce_mean(mel, axis=1)
        # gp not sure to mean over axis 0 or 1
        mel_m = tf.expand_dims(mel_m, axis=1)
        # mean over each mel bank
        mel = mel - mel_m
    #
    if Z_NORM:
        print("Subbing znorm")
        mel = (mel - zvals["mean"]) / zvals["std"]
    image = mel
    if preprocess_fn is not None:
        logging.info("Preprocessing with %s", preprocess_fn)
        raise Exception("Done preprocess for audio")
        # image = preprocess_fn(image)

    if labeled:
        # label = tf.cast(example["audio/class/label"], tf.int32)

        if one_hot:
            label = tf.reduce_max(
                tf.one_hot(labels, num_labels, dtype=tf.int32), axis=0
            )
        if no_bird:
            logging.info("no bird")
            no_bird_mask = np.ones(num_labels, dtype=np.bool)
            no_bird_mask[bird_i] = 0
            no_bird_mask = tf.constant(no_bird_mask)
            label = tf.cast(label, tf.bool)
            label = tf.math.logical_and(label, no_bird_mask)
            no_noise_mask = np.ones(num_labels, dtype=np.bool)
            no_noise_mask[noise_i] = 0
            no_noise_mask = tf.constant(no_noise_mask)
            label = tf.math.logical_and(label, no_noise_mask)

            label = tf.cast(label, tf.int32)

        return image, label

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
    print(labels)
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
    # calc_mean()
    # return
    datasets = ["other-training-data", "training-data", "chime-training-data"]
    datasets = ["training-data"]
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
        filenames.extend(tf.io.gfile.glob(f"./{d}/train/*.tfrecord"))
    labels.add("bird")
    labels.add("noise")
    labels = list(labels)

    labels.sort()
    filenames_2 = tf.io.gfile.glob(f"./flickr-training-data/train/*.tfrecord")
    # dir = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/validation"
    # weights = [0.5] * len(labels)
    resampled_ds, remapped, _ = get_dataset(
        # dir,
        filenames,
        labels,
        batch_size=32,
        image_size=DIMENSIONS,
        augment=False,
        resample=False,
        # filenames_2=filenames_2
        # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
    )
    # print(get_distribution(resampled_ds))
    # ing2D()(x)
    # for e in range(2):
    #     print("epoch", e)
    #     true_categories = tf.concat([y for x, y in resampled_ds], axis=0)
    #     # true_categories = np.int64(true_categories)
    #     true_categories = np.int64(tf.argmax(true_categories, axis=1))
    #     c = Counter(list(true_categories))
    #     print("epoch is size", len(true_categories))
    #     for i in range(len(labels)):
    #         print("after have", labels[i], c[i])

    # return
    print("looping")
    for e in range(1):
        for x, y in resampled_ds:
            show_batch(x, y, None, labels, None)

            # show_batch(x, y[0], y[1], labels, species_list)


def show_batch(image_batch, label_batch, species_batch, labels, species):
    # mfcc = image_batch[1]
    # sftf = image_batch[1]
    # image_batch = image_batch[0]
    fig = plt.figure(figsize=(20, 20))
    # mfcc = image_batch[2]
    image_batch = image_batch
    print("images in batch", len(image_batch), len(label_batch))
    num_images = len(image_batch)
    # rows = int(math.ceil(math.sqrt(num_images)))
    i = 0
    for n in range(num_images):
        # print(image_batch[n].numpy().shape)
        # print(image_batch[n])
        # return
        lbl = []
        for l_i, l in enumerate(label_batch[n]):
            if l == 1:
                lbl.append(labels[l_i])
        # print(label_batch[n][label_batch[n] == 1], "BB")
        # lbl = labels[np.argmax(label_batch[n])]
        # if lbl != "morepork":
        # continue
        # if rec_batch[n] != 1384657:
        # continue
        # print("showing", image_batch[n].shape, sftf[n].shape)
        p = n
        i += 1
        ax = plt.subplot(num_images // 3 + 1, 3, p + 1)
        # plot_spec(image_batch[n][:, :, 0], ax)
        # # plt.imshow(np.uint8(image_batch[n]))
        spc = None
        if species_batch is not None:
            spc = species[np.argmax(species_batch[n])]
        plt.title(f"{lbl} ({spc}")
        # # plt.axis("off")
        # ax = plt.subplot(num_images, 3, p + 1)
        img = image_batch[n]
        print("image is", img.shape)
        plot_mel(image_batch[n][:, :, 0], ax)
        # plot_mel(image_batch[n][:, :, 0], ax)

        #
        # ax = plt.subplot(num_images, 3, p + 2)
        # plot_mel(image_batch[n][:, :, 1], ax)
        # plt.title(f"{lbl} ({spc} more")
        #
        # ax = plt.subplot(num_images, 3, p + 3)
        # plot_mel(image_batch[n][:, :, 2], ax)power_to_db
        # plt.title(f"{lbl} ({spc} all")

        # plt.imshow(np.uint8(image_batch[n]))
        # plt.title(f"{lbl} ({spc} - {rec_batch[n]}) mel")
        # plt.axis("off")
        # print(image_batch[1][n].shape)
        # ax = plt.subplot(num_images, 3, p + 3)
        # plot_mfcc(image_batch[n][:, :, 0], ax)
        # plt.title(labels[np.argmax(label_batch[n])] + " mfcc")
        # plt.axis("off")
        # name = Path(".") / f"{n}.wav"
        # print(image_batch[n].shape)
        # sf.write(str(name), image_batch[n], 48000)
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
        hop_length=201,
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
