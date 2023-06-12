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
from plot_utils import plot_mel
from collections import Counter

BIRD_PATH = []
NOISE_PATH = []


# seed = 1341
# tf.random.set_seed(seed)
# np.random.seed(seed)
AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64
NOISE_LABELS = ["wind", "vehicle", "dog", "rain", "static", "noise", "cat"]
SPECIFIC_BIRD_LABELS = ["whistler", "kiwi", "morepork", "bird", "rifleman"]
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


# JUST FOR HUMAN OR NOT MODEL
# NOISE_LABELS.extend(SPECIFIC_BIRD_LABELS)
# NOISE_LABELS.extend(GENERIC_BIRD_LABELS)
# NOISE_LABELS.extend(OTHER_LABELS)
# keep_excluded_in_extra = False


def get_excluded_labels(labels):
    excluded_labels = []
    for l in labels:
        # FOR HUMAN MODEL
        # if l not in ["human", "noise"]:
        #     excluded_labels.append(l)
        # continue

        if l not in SPECIFIC_BIRD_LABELS and l not in ["noise", "human"]:
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
N_MELS = 120
SR = 48000
BREAK_FREQ = 1750
MEL_WEIGHTS = mel_f(48000, N_MELS, 50, 11000, 4800, BREAK_FREQ)
MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)
DIMENSIONS = (160, 188)

mel_s = (120, 513)
sftf_s = (2401, 188)
mfcc_s = (20, 188)
DIMENSIONS = mel_s
YAMNET_EMBEDDING_SHAPE = (6, 1024)
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


def load_dataset(filenames, num_labels, labels, args):
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
            all_human=args.get("all_human", False),
            embeddings=args.get("embeddings", False),
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )


    filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x))
    dataset = dataset.filter(filter_nan)

    filter_excluded = lambda x, y: not tf.math.equal(tf.math.count_nonzero(y[0]), 0)
    dataset = dataset.filter(filter_excluded)
    return dataset


def preprocess(data):
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return tf.keras.applications.inception_v3.preprocess_input(x), y


def get_distribution(dataset, num_labels, batched=True):
    true_categories = [y[0] for x, y in dataset]
    dist = np.zeros((num_labels), dtype=np.float32)
    if len(true_categories) == 0:
        return dist
    num_labels = len(true_categories[0])

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
bird_mask = None


def get_only(dataset,labels,label):
    label_i = labels.index(label)
    mask = np.zeros(len(labels), dtype=bool)
    mask[label_i] = 1
    mask = tf.constant(mask)
    filter = lambda x, y: tf.math.reduce_any(
        tf.math.logical_and(tf.cast(y[0], tf.bool), mask)
    )
    return dataset.filter(filter)

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
    if args.get("only",None) is not None:
        logging.info("Filtering only %s",args.get("only") )
        dataset = get_only(dataset,labels,args.get("only"))
    dist = get_distribution(dataset, num_labels, batched=False)
    for i, d in enumerate(dist):
        logging.info("Have %s for %s", d, labels[i])
    #
    # dataset = tf.data.Dataset.sample_from_datasets(
    #     [bird_dataset, dataset],
    #     stop_on_empty_dataset=args.get("stop_on_empty", True),
    #     rerandomize_each_iteration=args.get("rerandomize_each_iteration", False),
    # )

    dataset = dataset.cache()
    if args.get("shuffle", True):
        dataset = dataset.shuffle(
            4096, reshuffle_each_iteration=args.get("reshuffle", True)
        )
    batch_size = args.get("batch_size", None)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset, remapped, 0



@tf.function
def mel_from_raw(raw):
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
    all_human=False,
    embeddings=False,
):
    bird_l = tf.constant(["bird"])
    tfrecord_format = {
        "audio/class/text": tf.io.FixedLenFeature((), tf.string),
        "audio/track_id": tf.io.FixedLenFeature((), tf.string),
        "audio/rec_id": tf.io.FixedLenFeature((), tf.string),
        "audio/start_s": tf.io.FixedLenFeature((), tf.float32),
    }

    if embeddings:
        logging.info("Loading embeddings")
        tfrecord_format["embedding"] = tf.io.FixedLenFeature(
            YAMNET_EMBEDDING_SHAPE, tf.float32
        )

    else:

        tfrecord_format["audio/raw"] = tf.io.FixedLenFeature(
            (2401, mel_s[1]), tf.float32
        )
        # tfrecord_format["audio/embed_predictions"] = tf.io.FixedLenFeature(
        #     (), tf.string
        # )

    example = tf.io.parse_single_example(example, tfrecord_format)
    # raw = example["audio/raw"]
    track_id = tf.cast(example["audio/track_id"], tf.string)
    rec_id = tf.cast(example["audio/rec_id"], tf.string)
    start_s = tf.cast(example["audio/start_s"], tf.float32)


    label = tf.cast(example["audio/class/text"], tf.string)
    labels = tf.strings.split(label, sep="\n")
    global remapped_y, extra_label_map
    extra = extra_label_map.lookup(labels)
    labels = remapped_y.lookup(labels)
    labels = tf.concat([labels, extra], axis=0)
    embed_preds = None
    if embeddings:
        image = example["embedding"]
    else:
        # embed_preds = tf.cast(example["audio/embed_predictions"], tf.string)
        # embed_preds = tf.strings.split(embed_preds, sep=",")
        # extra_e = extra_label_map.lookup(embed_preds)
        # embed_preds = remapped_y.lookup(embed_preds)
        # embed_preds = tf.concat([embed_preds, extra_e], axis=0)

        stft = example["audio/raw"]
        stft = tf.reshape(stft, [2401, mel_s[1]])
        image = tf.tensordot(MEL_WEIGHTS, stft, 1)
        image = tf.expand_dims(image, axis=2)

    if augment:
        logging.info("Augmenting")
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
            if embed_preds is not None:
                embed_preds = tf.reduce_max(
                    tf.one_hot(embed_preds, num_labels, dtype=tf.int32), axis=0
                )


        label = tf.cast(label, tf.float32)

        return image, (label, embed_preds,rec_id,track_id,start_s)

    return image


def class_func(features, label):
    label = tf.argmax(label)
    return label


# test stuff
def main():
    init_logging()
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
    labels.add("bird")
    labels.add("noise")
    labels = list(labels)
    excluded_labels = get_excluded_labels(labels)
    labels.sort()

    resampled_ds, remapped, _ = get_dataset(
        # dir,
        filenames,
        labels,
        batch_size=32,
        image_size=DIMENSIONS,
        augment=False,
        resample=False,
        excluded_labels=excluded_labels,
        stop_on_empty=False,
        filter_bad=True,
        only="human",
        # filenames_2=filenames_2
        # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
    )
    for l in excluded_labels:
        labels.remove(l)
    print("looping")
    for e in range(1):
        for x, y in resampled_ds:
            y_true = y[0]
            rec_ids = y[2]
            track_ids=y[3]
            starts= y[4]
            for mel,y_t,rec,track,start in zip(x,y_true,rec_ids,track_ids,starts):
                rec = rec.numpy().decode("utf-8")
                track = track.numpy().decode("utf-8")
                lbl = []
                for l_i, l in enumerate(y_t):
                    if l == 1:
                        lbl.append(labels[l_i])
                samples_lbls = "-".join(lbl)
                file = f"./mels/{samples_lbls}-{rec}-{track}-{start}"
                print(f"Have", samples_lbls,"R:" ,rec, " T: ",track, "START: ",start.numpy())
                plot_mel(mel.numpy()[:,:,0],file)



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
#
#
# def plot_mel(mel, ax):
#     # power = librosa.db_to_power(mel.numpy())
#     img = librosa.display.specshow(
#         mel.numpy(),
#         x_axis="time",
#         y_axis="mel",
#         sr=48000,
#         fmax=11000,
#         fmin=50,
#         ax=ax,
#         hop_length=201,
#     )


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
