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
from custommel import mel_f
from pathlib import Path
import tensorflow_io as tfio
from audiomentations import AddBackgroundNoise, PolarityInversion, Compose
import soundfile as sf

BIRD_PATH = []
NOISE_PATH = []
NZ_BOX = [166.509144322, -34.4506617165, 178.517093541, -46.641235447]

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
    "australasian bittern",
    "banded dotterel",
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



insect = None
fp = None
HOP_LENGTH = 281
N_MELS = 160
SR = 48000
BREAK_FREQ = 1000
NFFT = 4096
MEL_WEIGHTS = mel_f(48000, N_MELS, 50, 11000, NFFT, BREAK_FREQ)
MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)

FMIN=50
FMAX = 11000
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
# ALTERNATIVE WEIGHTS FOR DUAL MODEL
# N_MELS = 96

# MEL_WEIGHTS = mel_f(48000, N_MELS, 0, 3000, 2048, BREAK_FREQ)
# MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)

# MEL_WEIGHTS_2 = mel_f(48000, N_MELS, 500, 15000, 1024, BREAK_FREQ)
# MEL_WEIGHTS_2 = tf.constant(MEL_WEIGHTS_2)

# MEL_WEIGHTS = tf.expand_dims(MEL_WEIGHTS, 0)

# MEL_WEIGHTS = tf.expand_dims(MEL_WEIGHTS, 0)

DIMENSIONS = (160, 188)

mel_s = (N_MELS, 513)
sftf_s = (2049, 188)
mfcc_s = (20, 188)
DIMENSIONS = (*mel_s, 1)
YAMNET_EMBEDDING_SHAPE = (6, 1024)
EMBEDDING_SHAPE = (1280,)
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

# labels to apply weighting to if  y true is generic "bird"
NZ_BIRD_LOSS_WEIGHTING = []
BIRD_WEIGHTING = []
SPECIFIC_BIRD_MASK = []


def load_dataset(filenames, num_labels, labels, args):
    random.shuffle(filenames)
    read_record = args.get("read_record", read_tfrecord)
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

    # set up weighting based on specific and generic birds
    global NZ_BIRD_LOSS_WEIGHTING, BIRD_WEIGHTING, SPECIFIC_BIRD_MASK, GENERIC_BIRD_MASK
    SPECIFIC_BIRD_MASK = np.zeros(num_labels)
    BIRD_WEIGHTING = np.zeros(num_labels)
    NZ_BIRD_LOSS_WEIGHTING = np.zeros(num_labels)
    GENERIC_BIRD_MASK = np.zeros(num_labels)
    if "rifleman" in labels:
        NZ_BIRD_LOSS_WEIGHTING[labels.index("rifleman")] = 1
    NZ_BIRD_LOSS_WEIGHTING[labels.index("bird")] = 1
    BIRD_WEIGHTING[labels.index("bird")] = 1
    GENERIC_BIRD_MASK[labels.index("bird")] = 1

    for i, l in enumerate(labels):
        if (l in GENERIC_BIRD_LABELS or l in SPECIFIC_BIRD_LABELS) and l != "bird":
            SPECIFIC_BIRD_MASK[i] = 1
    SPECIFIC_BIRD_MASK = tf.constant(SPECIFIC_BIRD_MASK, dtype=tf.float32)
    BIRD_WEIGHTING = tf.constant(BIRD_WEIGHTING, dtype=tf.float32)
    NZ_BIRD_LOSS_WEIGHTING = tf.constant(NZ_BIRD_LOSS_WEIGHTING, dtype=tf.float32)
    GENERIC_BIRD_MASK = tf.constant(GENERIC_BIRD_MASK, dtype=tf.float32)

    dataset = dataset.map(
        partial(
            read_record,
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
            load_raw=args.get("load_raw", True),
            model_name=args.get("model_name", "badwinner2"),
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    
    if args.get("filter_bad", False):
        logging.info("Filtering bad")
        dataset = dataset.filter(lambda x, y: not filter_bad_tracks(x, y, labels))
    if not args.get("only_features", False):
        logging.info("Removing Nan")
        if args.get("features"):
            filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x[2]))
        else:
            filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x))
        dataset = dataset.filter(filter_nan)
    if args.get("one_hot", True):
        filter_excluded = lambda x, y: not tf.math.equal(tf.math.count_nonzero(y[0]), 0)
    else:
        filter_excluded = lambda x, y: tf.math.greater(y[0], -1)

    dataset = dataset.filter(filter_excluded)
    return dataset


def get_distribution(dataset, num_labels, batched=True, one_hot=True):
    true_categories = [y[0] if isinstance(y, tuple) else y for x, y in dataset]
    dist = np.zeros((num_labels), dtype=np.float32)
    if len(true_categories) == 0:
        return dist, 0

    if len(true_categories) == 0:
        return dist, 0
    if batched:
        true_categories = tf.concat(true_categories, axis=0)
    if len(true_categories) == 0:
        return dist, 0
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
    return dist, len(true_categories)


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


def get_dataset(dir, labels, **args):
    global FMAX,MEL_WEIGHTS,FMIN,NFFT,BREAK_FREQ,N_MELS

    if args.get("fmin") is not None:
        FMIN = args["fmin"]
        FMAX = args["fmax"]
        logging.info("Using fmin and fmax %s %s",FMIN,FMAX)

        MEL_WEIGHTS = mel_f(48000, N_MELS, FMIN, FMAX, NFFT, BREAK_FREQ)
        MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)    
    if args.get("n_fft") is not None:
        print(NFFT,"IS")
        NFFT = args.get("n_fft")
        logging.info("NFFT %s",NFFT)
        # N_MELS = 96
        MEL_WEIGHTS = mel_f(48000, N_MELS, FMIN, FMAX, NFFT, BREAK_FREQ)
        MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)
    if args.get("break_freq") is not None:
        BREAK_FREQ = args.get("break_freq")
        logging.info("Applied break freq %s",BREAK_FREQ)
        MEL_WEIGHTS = mel_f(48000, N_MELS, FMIN, FMAX, NFFT, BREAK_FREQ)
        MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)       
    ds_first, remapped, epoch_size, labels, extra_label_dic = get_a_dataset(
        dir, labels, args
    )
    args["epoch_size"] = epoch_size
    deterministic = args.get("deterministic",False)
    if args.get("load_raw", True):
        logging.info("Mapping raw to mel")
        if args.get("augment", False):
            logging.info("Mixing up")

            # Cannot get this to work
            # STFT is so slow to calculate on the fly mayaswell ust make an augmented dataset

            if True:
                ds_second, _, _, _, _ = get_a_dataset(dir, labels, args)
            else:
                pass
                # dataset = ds_first.repeat(2)
                # ds_fist = dataset.take(epoch_size)
                # ds_second = dataset.take(epoch_size)
                # ds_second =
            train_ds = tf.data.Dataset.zip((ds_first, ds_second))

            dataset = train_ds.map(
                lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.5),num_parallel_calls=tf.data.AUTOTUNE,deterministic=deterministic)
            
            # dataset = dataset.map(lambda x, y: mix_up(x, y, dataset2))

            # doing mix up
        else:
            dataset = ds_first
        if args.get("debug"):
            logging.info("Not mapping to mel")
        elif args.get("model_name") == "dual-badwinner2":
            dataset = dataset.map(lambda x, y: raw_to_mel_dual(x, y),num_parallel_calls=tf.data.AUTOTUNE,deterministic=deterministic)
        else:
            dataset = dataset.map(lambda x, y: raw_to_mel(x, y),num_parallel_calls=tf.data.AUTOTUNE,deterministic=deterministic)

    else:
        dataset = ds_first
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset, remapped, epoch_size, labels, extra_label_dic


def set_remapped_extra(remap, extra_l):

    global extra_label_map
    global remapped_y
    extra_label_map = extra_l
    remapped_y = remap


def get_a_dataset(dir, labels, args):

    extra_label_dic = args.get("extra_label_map")
    remapped = args.get("remapped_labels", [])
    excluded_labels = args.get("excluded_labels", [])
    use_generic_bird = args.get("use_generic_bird", True)

    global extra_label_map
    global remapped_y
    if extra_label_dic is None:
        extra_label_dic, remapped, labels = get_remappings(
            labels, excluded_labels, use_generic_bird=use_generic_bird
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
    logging.info(
        "Remapped %s extra mapping %s new labels %s Use gen bird %s",
        remapped,
        extra_label_dic,
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
    if len(extra_label_dic)== 0:
        # seems to need something
        extra_label_dic["nonsense"]=1
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

    xeno_files = Path("/data/audio-data/xenocanto/xeno-training-data/")
    xeno_files = xeno_files / dir.name
    if xeno_files.exists():
        logging.info("Xeno files %s",xeno_files)
        xeno_files = tf.io.gfile.glob(str(xeno_files / "*.tfrecord"))
        logging.info("Loading xeno files %s",xeno_files)
        filenames.extend(xeno_files)

    lbl_dataset = load_dataset(filenames, num_labels, labels, args)
    logging.info("Loading %s files from %s", len(filenames), dir)
    datasets.append(lbl_dataset)
    for lbl_dir in dir.iterdir():
        if not lbl_dir.is_dir():
            continue
        filenames = tf.io.gfile.glob(str(lbl_dir / "*.tfrecord"))

        lbl_dataset = load_dataset(filenames, num_labels, labels, args)
        logging.info("Loading %s files from %s", len(filenames), lbl_dir)
        datasets.append(lbl_dataset)

    # may perform better without adding generics birds but sitll having generic label
    dataset_2 = None

    if args.get("filenames_2") is not None:
        logging.info("Loading second files %s", args.get("filenames_2")[:1])
        second = args.get("filenames_2")

        #   dont think no bird is needed
        # bird_c = dist[labels.index("bird")]
        # args["no_bird"] = True
        # added bird noise to human recs but it messes model, so dont use for now

        dataset_2 = load_dataset(second, len(labels), labels, args)

    else:
        logging.info("Not using second dataset")

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        logging.info("Stopping on empty? %s", args.get("resample", False))
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets,
            # stop_on_empty_dataset=False,
            stop_on_empty_dataset=args.get(
                "stop_on_empty", args.get("resample", False)
            ),
            rerandomize_each_iteration=args.get("rerandomize_each_iteration", True),
        )

    if args.get("no_low_samples", False):
        logging.info("Filtering out low samples")
        no_low_samples_filter = lambda x, y: tf.math.equal(
            y[6], tf.constant(0, dtype=tf.int64)
        )
        dataset = dataset.filter(no_low_samples_filter)

    if not args.get("one_hot", True):
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
    if not args.get("use_bird_tags", False):
        logging.info("Filtering out bird tags without specific bird")
        dataset = dataset.filter(others_filter)
    # bird_dataset = dataset.filter(bird_filter)
    # if args.get("filter_signal") is not None:
    #     logging.info("Filtering signal by percent 0.1")
    #     bird_dataset = bird_dataset.filter(filter_signal)
    dataset = dataset.cache()
    if args.get("shuffle", True):
        dataset = dataset.shuffle(
            4096, reshuffle_each_iteration=args.get("reshuffle", True)
        )
    if dataset_2 is not None:
        logging.info("Adding second dataset with weights [0.6,0.4]")
        dataset = tf.data.Dataset.sample_from_datasets(
            [dataset, dataset_2],
            weights=[0.6, 0.4],
            stop_on_empty_dataset=True,
            rerandomize_each_iteration=args.get("rerandomize_each_iteration", True),
        )

    logging.info("Loss fn is %s", args.get("loss_fn"))
    deterministic = args.get("deterministic", False)

    if args.get("debug"):
        specific_track = lambda x, y: tf.math.equal(y[4], tf.constant("2436458"))
        dataset = dataset.filter(specific_track)
        batch_size = args.get("batch_size", None)
        dataset = dataset.cache()

        if batch_size is not None:
            dataset = dataset.batch(batch_size, drop_remainder=False)
        if args.get("load_raw",False):
            logging.info("Normalizing input")
            dataset = dataset.map(lambda x, y: normalize(x, y))
        logging.info("Returning debug data")
        return dataset, remapped, None, labels, extra_label_dic


    if args.get("loss_fn") == "WeightedCrossEntropy":
        logging.info("Mapping possiblebirds")
        dataset = dataset.map(lambda x, y: (x, (y[0], y[5])),num_parallel_calls=tf.data.AUTOTUNE,deterministic=deterministic)
    else:
        dataset = dataset.map(lambda x, y: (x, y[0]),num_parallel_calls=tf.data.AUTOTUNE,deterministic=deterministic)
    epoch_size = args.get("epoch_size")
    dist = None
    if epoch_size is None:
        dist, epoch_size = get_distribution(
            dataset, num_labels, batched=False, one_hot=args.get("one_hot", True)
        )

    pcen = args.get("pcen", False)
    if pcen:
        logging.info("Taking PCEN")
        dataset = dataset.map(lambda x, y: pcen_function(x, y))
    if dist is not None:
        for l, d in zip(labels, dist):
            logging.info(f" for {l} have {d}")
    # tf because of sample from datasets
    dataset = dataset.repeat(2)
    dataset = dataset.take(epoch_size)
    batch_size = args.get("batch_size", None)
    # dataset = dataset.cache()

    # dont think we need this iwth interleave
    # if args.get("shuffle", True):
    #     dataset = dataset.shuffle(
    #         40096, reshuffle_each_iteration=args.get("reshuffle", True)
    #     )

    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=False)

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
    
    if args.get("load_raw",False):
        logging.info("Normalizing input")
        dataset = dataset.map(lambda x, y: normalize(x, y))


    return dataset, remapped, epoch_size, labels, extra_label_dic


@tf.function
def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


# https://keras.io/examples/vision/mixup/


@tf.function
def mix_up(ds_one, ds_two, alpha=0.5):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    # batch_size = 32
    batch_size = tf.keras.ops.shape(images_one)[0]

    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.keras.ops.reshape(l, (batch_size, 1))
    y_l = tf.keras.ops.reshape(l, (batch_size, 1))

    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    # possible_labels = tf.clip_by_value(labels_one[1] + labels_two[1], 0, 1)
    return (images, labels)
# (labels, possible_labels))


# @tf.function
# def mix_up(x, y, ds2):
#     p = 0.5
#     if tf.random.uniform((), 0, 1) < p:

#         print(x2, y2)
#         alpha = tf.random.uniform((), 0, 1)
#         x = alpha * x + (1 - alpha) * x2
#         # y = tf.clip_by_value(tf.math.logical_or(y, y2[0]), 0, 1)
#     return x, y

# for item in x:
#     print(item)
# data = x.as_numpy_iterator()
# p = 0.5
# indices = tf.range(data.shape[0])
# shuffled_indices = tf.random.shuffle(indices)
# alpha = tf.random.uniform(data.shape[0])
# for i in range(data.shape[0]):
#     if tf.random.uniform((), 0, 1) < p:
#         data[i] = alpha[i] * x[i] + (1 - alpha[i]) * x[shuffled_indices[i]]
#         data[i] = tf.clip_by_value(y[i] + y[shuffled_indices[i]], 0, 1)
# return data, y


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
    load_raw=True,
    model_name="badwinner2",
):
    tfrecord_format = {"audio/class/text": tf.io.FixedLenFeature((), tf.string)}
    tfrecord_format["audio/rec_id"] = tf.io.FixedLenFeature((), tf.string)
    tfrecord_format["audio/track_id"] = tf.io.FixedLenFeature((), tf.string)
    tfrecord_format["audio/low_sample"] = tf.io.FixedLenFeature((), tf.int64)
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
            tfrecord_format["audio/raw"] = tf.io.FixedLenFeature((48000 * 3), tf.float32)
        else:
            tfrecord_format["audio/spectogram"] = tf.io.FixedLenFeature(
                (2049 * 513), tf.float32
            )
        if filter_freq:
            tfrecord_format["audio/buttered"] = tf.io.FixedLenFeature(
                (2049 * 513), tf.float32, default_value=tf.zeros((2049 * 513))
            )

    if features or only_features:
        tfrecord_format["audio/short_f"] = tf.io.FixedLenFeature((68 * 60), tf.float32)
        tfrecord_format["audio/mid_f"] = tf.io.FixedLenFeature((136 * 3), tf.float32)

    tfrecord_format["audio/signal_percent"] = tf.io.FixedLenFeature((), tf.float32)

    example = tf.io.parse_single_example(example, tfrecord_format)
    low_sample = tf.cast(example["audio/low_sample"], tf.int64)

    label = tf.cast(example["audio/class/text"], tf.string)
    split_labels = tf.strings.split(label, sep="\n")
    global remapped_y, extra_label_map
    labels = remapped_y.lookup(split_labels)
    if multi:
        extra = extra_label_map.lookup(split_labels)
        labels = tf.concat([labels, extra], axis=0)
    # else:
    #     print("Labels was ",labels)
    #     labels = tf.reduce_max(labels)
    #     print("Labels becomes ",labels)
    embed_preds = None
    if load_raw:
        spectogram = example["audio/raw"]

    elif embeddings:
        spectogram = example["embedding"]

    elif not only_features:
        buttered = example["audio/buttered"] if filter_freq else None
        if filter_freq and tf.math.count_nonzero(buttered) > 0:
            if random_butter > 0:
                rand = tf.random.uniform((), 0, 1)
                # do butter pass 3/5ths of the time
                spectogram = tf.cond(
                    rand <= random_butter,
                    lambda: tf.identity(example["audio/buttered"]),
                    lambda: tf.identity(example["audio/spectogram"]),
                )
            else:
                logging.info("USing buttered")
                spectogram = example["audio/buttered"]
        else:

            spectogram = example["audio/spectogram"]
        spectogram = tf.reshape(spectogram, (2049, 513))
        # spectogram = tf.math.pow(spectogram,2)
        spectogram = tf.tensordot(MEL_WEIGHTS, spectogram, 1)
        spectogram = tf.expand_dims(spectogram, axis=-1)
        if model_name == "efficientnetb0":
            spectogram = tf.repeat(spectogram, 3, 2)

        print("Loaded spect ", spectogram.shape)
    if features or only_features:
        short_f = example["audio/short_f"]
        mid_f = example["audio/mid_f"]
        if only_features:
            spectogram = tf.concat((short_f, mid_f), axis=0)
            # mid_f = tf.reshape(mid_f, (136, 3))
            # short_f = tf.reshape(short_f, (68, 60))

            # raw = (short_f, mid_f)
        else:
            mid_f = tf.reshape(mid_f, (136, 3))
            short_f = tf.reshape(short_f, (68, 60))
            spectogram = (short_f, mid_f, spectogram)
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

        if one_hot:
            label = tf.reduce_max(
                tf.one_hot(labels, num_labels, dtype=tf.int32), axis=0
            )
            if not multi:
                logging.info("Choosing only one label as not multi")
                max_l = tf.argmax(label)
                label = tf.one_hot(max_l, num_labels, dtype=tf.int32)

            if embed_preds is not None:
                embed_preds = tf.reduce_max(
                    tf.one_hot(embed_preds, num_labels, dtype=tf.int32), axis=0
                )
        else:
            # pretty sure this is wrong but never used
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
        return spectogram, (
            label,
            embed_preds,
            signal_percent,
            # min_freq,
            # max_freq,
            example["audio/rec_id"],
            example["audio/track_id"],
            possible_labels,
            low_sample,
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


    # batch_data = []
    # batch_data.append(np.zeros(5)-1)
    # batch_data = np.array(batch_data)
    # normalized = normalize(batch_data,None)
    # print("Normalized becomes ")
    # for n in normalized:
    #     print(n)
    # return
    # return
    datasets = ["other-training-data", "training-data", "chime-training-data"]
    datasets = ["training-data"]
    dataset_dirs = ["./audio-training/training-data"]
    # dataset_dirs = ["./augmented-training"]

    filenames = []
    labels = set()
    datasets = []
    for d in dataset_dirs:
        tf_dir = Path(d)
        # file = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/training-meta.json"
        file = f"{d}/training-meta.json"
        with open(file, "r") as f:
            meta = json.load(f)
        labels.update(meta.get("labels", []))
        labels.add("bird")
        labels.add("noise")
        labels = list(labels)
        set_specific_by_count(meta)
        excluded_labels = get_excluded_labels(labels)
        labels.sort()
        # print("loaded labels", labels)
        # species_list = ["bird", "human", "rain", "other"]

        # filenames = tf.io.gfile.glob(f"./training-data/validation/*.tfrecord")

        resampled_ds, remapped, _, labels,_ = get_dataset(
            tf_dir / "train",
            # filenames,
            labels,
            # use_generic_bird=False,
            deterministic=True,
            batch_size=32,
            image_size=DIMENSIONS,
            augment=False,
            resample=False,
            excluded_labels=excluded_labels,
            # stop_on_empty=True,
            filter_freq=False,
            random_butter=0.9,
            only_features=False,
            multi_label=False,
            load_raw=True,
            n_fft=4096,
            fmin=500,
            fmax=15000,
            use_bird_tags=False,
            debug=True,
            shuffle=False,
            multi=True,
            # filenames_2=filenames_2
            # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
        )
        for batch_x , batch_y in resampled_ds:
            print("data is ", len(batch_x))
            recs = batch_y[3]
            tracks = batch_y[4]
            for x,rec,track in zip(batch_x,recs,tracks):
                data_ok = np.all(x>=-1) and np.all(x<=1.000002)
                a_max = np.amax(x)
                a_min = np.amin(x)
                rec = rec.numpy().decode("utf8")
                track = track.numpy().decode("utf8")
                if not data_ok:
                    # print(x)
                    x = x.numpy()
                    logging.info("Bad data for rec %s track %s less than -1 %s over 1 %s max %s min %s", rec,track, x[np.where(x <-1)], x[np.where(x >1.000002)],a_max,a_min)
                

                if a_max == a_min:
                    logging.info("Max = Min for rec %s track %s max %s min %s", rec,track, a_max,a_min)

        return
        break
        # filenames.extend(tf.io.gfile.glob(f"{d}/test/**/*.tfrecord"))
    print("labels are ", labels)
    global NZ_BIRD_LOSS_WEIGHTING, BIRD_WEIGHTING, SPECIFIC_BIRD_MASK, GENERIC_BIRD_MASK
    print("GENERIC_BIRD_MASK", GENERIC_BIRD_MASK)
    print("SPECIFIC", SPECIFIC_BIRD_MASK)
    dist, _ = get_distribution(resampled_ds, len(labels), batched=True, one_hot=True)
    print("Dist is ", dist)
    for l, d in zip(labels, dist):
        print(f"{l} has {d}")
    # testing loss function results
    # from audiomodel import WeightedCrossEntropy

    # custom_loss = WeightedCrossEntropy(labels)
    # for x, y in resampled_ds:
    #     for y_true, y_possible in zip(y[0], y[1]):
    #         pred = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)
    #         y_true = tf.expand_dims(y_true, 0)
    #         y_possible = tf.expand_dims(y_possible, 0)

    #         loss = custom_loss.call([y_true, y_possible], pred)
    #         print(
    #             "for y  ",
    #             y_true,
    #             " with pred ",
    #             pred,
    #             " loss is ",
    #             loss,
    #             " and possible",
    #             y_possible,
    #         )
    #         print("")
    # return
    # for e in range(2):
    #     start = time.time()

    #     for x, y in resampled_ds:
    #         pass
    #     print("Epoch took ",time.time() - start)
    # return
    for e in range(1):
        for x, y in resampled_ds:
            print(x.shape)

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
    print("labl batch", label_batch[0])
    i = 0
    for n in range(num_images):
        # print("Y is ", label_batch[n])
        lbl = []
        for l_i, l in enumerate(label_batch[n]):
            if l > 0:
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
        # librosa.power_to_db(mel.numpy(),ref=np.max),
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
def raw_to_mel_dual(x, y, features=False):

    raw = x
    raw_2 = tf.compat.v1.identity(raw)
    raw =  butter_function(raw,0,3000)
    stft = tf.signal.stft(
        raw,
        2048,
        278,
        fft_length=2048,
        window_fn=tf.signal.hann_window,
        # pad_end=True,
        name=None,
    )
   
    stft = tf.transpose(stft, [0, 2, 1])
    stft = tf.math.abs(stft)
    print("STFT becomes ",stft.shape)
    batch_size = tf.keras.ops.shape(x)[0]

    weights = tf.expand_dims(MEL_WEIGHTS, 0)
    weights = tf.repeat(weights, batch_size, 0)
    image = tf.keras.backend.batch_dot(weights, stft)
    image = tf.expand_dims(image, axis=3)

    raw2 = butter_function(raw_2,500,15000)

    stft = tf.signal.stft(
        raw,
        1024,
        280,
        fft_length=1024,
        window_fn=tf.signal.hann_window,
        # pad_end=True,
        name=None,
    )
   
    stft = tf.transpose(stft, [0, 2, 1])
    stft = tf.math.abs(stft)
    batch_size = tf.keras.ops.shape(x)[0]

    weights = tf.expand_dims(MEL_WEIGHTS_2, 0)
    weights = tf.repeat(weights, batch_size, 0)
    image_2 = tf.keras.backend.batch_dot(weights, stft)
    image_2 = tf.expand_dims(image_2, axis=3)


    # x =  tf.keras.layers.Concatenate()([image,image_2])
    return (image,image_2), y

@tf.function
def normalize(input,y):
    min_v  = tf.math.reduce_min(input,-1,keepdims=True)
    input = tf.math.subtract(input,min_v)
    max_v = tf.math.reduce_max(input,-1,keepdims=True)
    input = tf.math.divide(input,max_v) + 0.000001
    input = tf.math.subtract(input,0.5)
    input = tf.math.multiply(input,2) 
    return input,y

@tf.function
def raw_to_mel(x, y, features=False):
    if features:
        raw = x[2]
    else:
        raw = x
        
    global FMIN,FMAX
    if FMIN >50:
        logging.info("Applying butter %s %s",FMIN,FMAX)
        raw =  butter_function(raw,FMIN,FMAX)
    
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
    )

    stft = tf.transpose(stft, [0, 2, 1])
    stft = tf.math.abs(stft)
    batch_size = tf.keras.ops.shape(x)[0]

    weights = tf.expand_dims(MEL_WEIGHTS, 0)
    weights = tf.repeat(weights, batch_size, 0)
    image = tf.keras.backend.batch_dot(weights, stft)
    image = tf.expand_dims(image, axis=3)


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
    return tf.math.greater(y[2], 0.1)
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
