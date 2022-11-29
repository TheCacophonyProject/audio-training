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
import tensorflow_io as tfio

# seed = 1341
# tf.random.set_seed(seed)
# np.random.seed(seed)
AUTOTUNE = tf.data.AUTOTUNE
# IMAGE_SIZE = [256, 256]
# BATCH_SIZE = 64

insect = None
fp = None

DIMENSIONS = (128, 1134)

mel_s = (128, 61)
sftf_s = (2401, 61)
mfcc_s = (20, 61)

mel_bins = librosa.mel_frequencies(128, fmax=48000 / 2)
human_lowest = np.where(mel_bins < 60)[-1][-1]
human_max = np.where(mel_bins > 180)[0][0]

print("Human lowest", human_lowest, human_max)

# 60-180hz
human_mel = (human_lowest, human_max)
human_mask = np.zeros((mel_s), dtype=np.bool)
human_mask[human_mel[0] : human_mel[0] + human_mel[1]] = 1

# 600-1200
# frequency_min = 600
# frequency_max = 1200
more_lower = np.where(mel_bins < 600)[-1][-1]
more_max = np.where(mel_bins > 1200)[0][0]


morepork_mel = (more_lower, more_max)
print("more bins", morepork_mel)

morepork_mask = np.zeros((mel_s), dtype=np.bool)
morepork_mask[morepork_mel[0] : morepork_mel[0] + morepork_mel[1]] = 1

print(morepork_mask[morepork_mel[0] : morepork_mel[0] + morepork_mel[1]].shape)


def load_dataset(filenames, num_labels, num_species, args):
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
    dataset = tf.data.TFRecordDataset(filenames)

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
    dataset = dataset.map(
        partial(
            read_tfrecord,
            num_labels=num_labels,
            num_species=num_species,
            image_size=image_size,
            labeled=labeled,
            augment=augment,
            preprocess_fn=preprocess_fn,
            one_hot=one_hot,
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )
    filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x[1]))
    dataset = dataset.filter(filter_nan)
    return dataset


#
def preprocess(data):
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return tf.keras.applications.inception_v3.preprocess_input(x), y


def get_distribution(dataset):
    true_categories = tf.concat([y[0] for x, y in dataset], axis=0)
    num_labels = len(true_categories[0])
    if len(true_categories) == 0:
        return None
    true_categories = np.int64(tf.argmax(true_categories, axis=1))
    c = Counter(list(true_categories))
    dist = np.empty((num_labels), dtype=np.float32)
    for i in range(num_labels):
        dist[i] = c[i]
    return dist


def get_dataset(base_dir, labels, species_list, **args):
    #     batch_size,
    #     image_size,
    #     reshuffle=True,
    #     deterministic=False,
    #     labeled=True,
    #     augment=False,
    #     resample=True,
    #     preprocess_fn=None,
    #     mvm=False,
    #     scale_epoch=None,
    #     only_features=False,
    #     one_hot=True,
    # ):
    # species_list = ["bird", "human", "rain"]
    num_species = len(species_list)

    num_labels = len(labels)
    global remapped_y
    remapped = {}
    global species_y
    species = {}

    keys = []
    values = []
    s_values = []
    for l in labels:
        remapped[l] = [l]
        keys.append(labels.index(l))
        values.append(labels.index(l))
        if l == "human":
            s_values.append(species_list.index("human"))
        elif l == "rain":
            s_values.append(species_list.index("rain"))
        elif l == "other":
            s_values.append(species_list.index("other"))

        else:
            s_values.append(species_list.index("bird"))

    species_y = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(s_values),
        ),
        default_value=tf.constant(-1),
        name="remapped_species",
    )
    remapped_y = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values),
        ),
        default_value=tf.constant(-1),
        name="remapped_y",
    )
    filenames = tf.io.gfile.glob(f"{base_dir}/*.tfrecord")
    dataset = load_dataset(filenames, num_labels, num_species, args)
    resample_data = args.get("resample", True)
    if resample_data:
        logging.info("Resampling data")
        dataset = resample(dataset, labels)
    if args.get("shuffle", True):
        dataset = dataset.shuffle(
            4096, reshuffle_each_iteration=args.get("reshuffle", True)
        )
    # tf refues to run if epoch sizes change so we must decide a costant epoch size even though with reject res
    # it will chang eeach epoch, to ensure this take this repeat data and always take epoch_size elements
    epoch_size = len([0 for x, y in dataset])
    logging.info("Setting dataset size to %s", epoch_size)
    if not args.get("only_features", False):
        dataset = dataset.repeat(2)
    scale_epoch = args.get("scale_epoch", None)
    if scale_epoch:
        epoch_size = epoch_size // scale_epoch
    dataset = dataset.take(epoch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    batch_size = args.get("batch_size", None)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    dist = get_distribution(dataset)
    for i, d in enumerate(dist):
        logging.info("Have %s for %s", d, labels[i])

    return dataset, remapped


def resample(dataset, labels):
    excluded_labels = ["morepork", "kiwi"]
    num_labels = len(labels)
    true_categories = [y[0] for x, y in dataset]
    if len(true_categories) == 0:
        return None
    true_categories = np.int64(tf.argmax(true_categories, axis=1))
    c = Counter(list(true_categories))
    dist = np.empty((num_labels), dtype=np.float32)
    target_dist = np.empty((num_labels), dtype=np.float32)
    for i in range(num_labels):
        if labels[i] in excluded_labels:
            dist[i] = 0
            logging.info("Excluding %s for %s", c[i], labels[i])

        else:
            dist[i] = c[i]

            logging.info("Have %s for %s", dist[i], labels[i])
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
    print(target_dist)
    rej = dataset.rejection_resample(
        class_func=class_func,
        target_dist=target_dist,
    )
    dataset = rej.map(lambda extra_label, features_and_label: features_and_label)
    return dataset


def read_tfrecord(
    example,
    image_size,
    num_labels,
    num_species,
    labeled,
    augment=False,
    preprocess_fn=None,
    one_hot=True,
):
    tf_more_mask = tf.constant(morepork_mask)
    tf_human_mask = tf.constant(human_mask)
    tfrecord_format = {
        "audio/sftf": tf.io.FixedLenFeature([sftf_s[0] * sftf_s[1]], dtype=tf.float32),
        "audio/mel": tf.io.FixedLenFeature([mel_s[0] * mel_s[1]], dtype=tf.float32),
        "audio/mfcc": tf.io.FixedLenFeature([mfcc_s[0] * mfcc_s[1]], dtype=tf.float32),
        "audio/class/label": tf.io.FixedLenFeature((), tf.int64),
        "audio/length": tf.io.FixedLenFeature((), tf.int64),
        # "audio/rec_id": tf.io.FixedLenFeature((), tf.int64),
        "audio/start_s": tf.io.FixedLenFeature(1, tf.float32),
        "audio/sftf_w": tf.io.FixedLenFeature((), tf.int64),
        "audio/sftf_h": tf.io.FixedLenFeature((), tf.int64),
        "audio/mel_w": tf.io.FixedLenFeature((), tf.int64),
        "audio/mel_h": tf.io.FixedLenFeature((), tf.int64),
        "audio/mfcc_w": tf.io.FixedLenFeature((), tf.int64),
        "audio/mfcc_h": tf.io.FixedLenFeature((), tf.int64),
    }

    example = tf.io.parse_single_example(example, tfrecord_format)

    audio_data = example["audio/sftf"]
    mel = example["audio/mel"]
    mfcc = example["audio/mfcc"]
    # mel = tf.expand_dims(mel, axis=2)

    audio_data = tf.reshape(audio_data, [*sftf_s, 1])

    mel = tf.reshape(mel, [*mel_s])
    if augment:
        logging.info("Augmenting")
        mel = tfio.audio.freq_mask(mel, param=10)
        mel3 = tfio.audio.time_mask(mel, param=10)
    mel = tf.expand_dims(mel, axis=2)

    # print(mel.shape)
    # mel_h = tf.experimental.numpy.copy(mel)
    # # print(mel_h.shape)
    # mel_h = mel_h[human_mel[0] : human_mel[0] + human_mel[1]]
    # mel_more = tf.experimental.numpy.copy(mel)
    # mel_more = mel_more[morepork_mel[0] : morepork_mel[0] + morepork_mel[1]]
    #
    # # or
    # # full scale zero out other values
    # # mel_h = tf.experimental.numpy.copy(mel)
    # # mel_h = tf.reshape(mel_h, [*mel_s])
    # # mel_h = tf.math.multiply(mel_h, tf_human_mask)
    # mel_h = tf.expand_dims(mel_h, axis=2)
    # #
    # # mel_more = tf.experimental.numpy.copy(mel)
    # # mel_more = tf.reshape(mel_more, [*mel_s])
    # # mel_more = tf.math.multiply(mel_more, tf_more_mask)
    # mel_more = tf.expand_dims(mel_more, axis=2)
    # mel = tf.expand_dims(mel, axis=2)
    #
    # # mfcc = tf.reshape(mfcc, [*mfcc_s])
    # # mfcc_max = tf.math.reduce_max(mfcc)
    # # mfcc = tf.math.subtract(mfcc, mfcc_max)
    # # mfcc = tf.math.divide(mfcc, 2)
    # length = example["audio/length"]
    # start = example["audio/start_s"]
    # # image = tf.image.grayscale_to_rgb(audio_data)
    #
    # # if we want mfcc, i think that we would want to normalized both specs first
    # # spec_mf = tf.concat((audio_data, mfcc), axis=0)
    # # mel_mf = tf.concat((mel, mfcc), axis=0)
    #
    # # mel_mf = mel
    # # mel_mf = tf.reshape(mel_mf, [*mel_mf.shape, 1])
    # # image = tf.concat((mel_mf, mel_mf, mel_mf), axis=2)
    #
    #
    # mel_more = tf.image.resize(mel_more, (128, 61))
    # mel_h = tf.image.resize(mel_h, (128, 61))
    # image = tf.concat((mel_h, mel_more, mel), axis=2)
    image = tf.concat((mel, mel, mel), axis=2)

    image = tf.image.resize(image, (128 * 2, 61 * 2))

    # image = data_augmentation(image)
    if preprocess_fn is not None:
        logging.info("Preprocessing with %s", preprocess_fn)
        raise Exception("Done preprocess for audio")
        # image = preprocess_fn(image)

    if labeled:
        label = tf.cast(example["audio/class/label"], tf.int32)
        global remapped_y
        label = remapped_y.lookup(label)
        global species_y
        species = species_y.lookup(label)
        print("num species", num_species)

        if one_hot:
            label = tf.one_hot(label, num_labels)
            species = tf.one_hot(species, num_species)

        # return image, label
        return image, (label, species)

    return image


def class_func(features, label):
    label = tf.argmax(label[0])
    return label


from collections import Counter

# test stuff
def main():
    init_logging()
    # file = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/training-meta.json"
    file = f"./other-training/training-meta.json"
    with open(file, "r") as f:
        meta = json.load(f)
    labels = meta.get("labels", [])
    species_list = ["bird", "human", "rain", "other"]

    datasets = []
    # dir = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/validation"
    # weights = [0.5] * len(labels)
    resampled_ds, remapped = get_dataset(
        # dir,
        f"./other-training-data/validation",
        labels,
        species_list,
        batch_size=32,
        image_size=DIMENSIONS,
        augment=False,
        resample=False,
        # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
    )
    # print(get_distribution(resampled_ds))
    #
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
    for e in range(2):
        for x, y in resampled_ds:
            print(len(x), len(y))
            show_batch(x, y[0], y[1], labels, species_list)


def show_batch(image_batch, label_batch, species_batch, labels, species):
    # mfcc = image_batch[1]
    # sftf = image_batch[1]
    # image_batch = image_batch[0]
    plt.figure(figsize=(200, 200))
    # mfcc = image_batch[2]
    image_batch = image_batch
    print("images in batch", len(image_batch), len(label_batch))
    num_images = 6
    # rows = int(math.ceil(math.sqrt(num_images)))
    i = 0
    for n in range(num_images):
        lbl = labels[np.argmax(label_batch[n])]
        # if lbl != "morepork":
        # continue
        # if rec_batch[n] != 1384657:
        # continue
        # print("showing", image_batch[n].shape, sftf[n].shape)
        p = i * 3
        i += 1
        print("setting", p + 1)
        ax = plt.subplot(num_images, 3, p + 1)
        # plot_spec(image_batch[n][:, :, 0], ax)
        # # plt.imshow(np.uint8(image_batch[n]))
        spc = species[np.argmax(species_batch[n])]
        plt.title(f"{lbl} ({spc} human")
        # # plt.axis("off")
        print(image_batch[n].shape)
        # ax = plt.subplot(num_images, 3, p + 1)
        plot_mel(image_batch[n][:, :, 0], ax)

        ax = plt.subplot(num_images, 3, p + 2)
        plot_mel(image_batch[n][:, :, 1], ax)
        plt.title(f"{lbl} ({spc} more")

        ax = plt.subplot(num_images, 3, p + 3)
        plot_mel(image_batch[n][:, :, 2], ax)
        plt.title(f"{lbl} ({spc} all")

        # plt.imshow(np.uint8(image_batch[n]))
        # plt.title(f"{lbl} ({spc} - {rec_batch[n]}) mel")
        # plt.axis("off")
        # print(image_batch[1][n].shape)
        # ax = plt.subplot(num_images, 3, p + 3)
        # plot_mfcc(image_batch[n][:, :, 0], ax)
        # plt.title(labels[np.argmax(label_batch[n])] + " mfcc")
        # plt.axis("off")

    plt.show()


def plot_mfcc(mfccs, ax):
    img = librosa.display.specshow(mfccs.numpy(), x_axis="time", ax=ax)


def plot_mel(mel, ax):
    # power = librosa.db_to_power(mel.numpy())
    img = librosa.display.specshow(
        mel.numpy(), x_axis="time", y_axis="mel", sr=48000, fmax=48000 / 2, ax=ax
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
