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
mfcc_s = (20, 1134)


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
    true_categories = tf.concat([y for x, y in dataset], axis=0)
    num_labels = len(true_categories[0])
    if len(true_categories) == 0:
        return None
    true_categories = np.int64(tf.argmax(true_categories, axis=1))
    c = Counter(list(true_categories))
    dist = np.empty((num_labels), dtype=np.float32)
    for i in range(num_labels):
        dist[i] = c[i]
    return dist


def get_dataset(base_dir, labels, **args):
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
    num_labels = len(labels)
    global remapped_y
    remapped = {}
    keys = []
    values = []
    for l in labels:
        remapped[l] = [l]
        keys.append(labels.index(l))
        values.append(labels.index(l))

    remapped_y = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values),
        ),
        default_value=tf.constant(-1),
        name="remapped_y",
    )
    filenames = tf.io.gfile.glob(f"{base_dir}/*.tfrecord")
    dataset = load_dataset(filenames, num_labels, args)
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
    excluded_labels = ["bird"]
    num_labels = len(labels)
    true_categories = [y for x, y in dataset]
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
        elif dist_max - dist[i] > (max_range * 2):
            target_dist[i] = dist[i]

        target_dist[i] = max(0, target_dist[i])
    target_dist = target_dist / np.sum(target_dist)

    if "sheep" in labels:
        sheep_i = labels.index("sheep")
        target_dist[sheep_i] = 0
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
    labeled,
    augment=False,
    preprocess_fn=None,
    one_hot=True,
):
    tfrecord_format = {
        "audio/sftf": tf.io.FixedLenFeature([sftf_s[0] * sftf_s[1]], dtype=tf.float32),
        "audio/mel": tf.io.FixedLenFeature([mel_s[0] * mel_s[1]], dtype=tf.float32),
        "audio/mfcc": tf.io.FixedLenFeature([mfcc_s[0] * mfcc_s[1]], dtype=tf.float32),
        "audio/class/label": tf.io.FixedLenFeature((), tf.int64),
        "audio/length": tf.io.FixedLenFeature((), tf.int64),
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

    audio_data = tf.reshape(
        audio_data, [example["audio/sftf_w"], example["audio/sftf_h"], 1]
    )
    mel = tf.reshape(mel, [example["audio/mel_h"], example["audio/mel_w"], 1])
    mfcc = tf.reshape(mfcc, [example["audio/mfcc_h"], example["audio/mfcc_w"], 1])

    length = example["audio/length"]
    start = example["audio/start_s"]
    # image = tf.image.grayscale_to_rgb(audio_data)

    # if we want mfcc, i think that we would want to normalized both specs first
    # spec_mf = tf.concat((audio_data, mfcc), axis=0)
    # mel_mf = tf.concat((mel, mfcc), axis=0)
    # image = tf.concat((spec_mf, mel_mf, mel_mf), axis=2)

    image = tf.concat((audio_data, audio_data, audio_data), axis=2)
    # if augment:
    #     logging.info("Augmenting")
    #     image = data_augmentation(image)
    if preprocess_fn is not None:
        logging.info("Preprocessing with %s", preprocess_fn)
        image = preprocess_fn(image)

    if labeled:
        label = tf.cast(example["audio/class/label"], tf.int32)
        global remapped_y
        label = remapped_y.lookup(label)
        if one_hot:
            label = tf.one_hot(label, num_labels)
        # return image, label
        return (audio_data, mel, mfcc), label

    return image


def class_func(features, label):
    label = tf.argmax(label)
    return label


from collections import Counter

# test stuff
def main():
    init_logging()
    # file = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/training-meta.json"
    file = f"./training-meta.json"
    with open(file, "r") as f:
        meta = json.load(f)
    labels = meta.get("labels", [])
    datasets = []
    # dir = "/home/gp/cacophony/classifier-data/thermal-training/cp-training/validation"
    # weights = [0.5] * len(labels)
    resampled_ds, remapped = get_dataset(
        # dir,
        f"./training-data/validation",
        labels,
        batch_size=32,
        image_size=DIMENSIONS,
        augment=True,
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
            show_batch(x, y, labels)


def show_batch(image_batch, label_batch, labels):
    plt.figure(figsize=(200, 200))
    mel = image_batch[1]
    mfcc = image_batch[2]
    image_batch = image_batch[0]
    print("images in batch", len(image_batch), len(label_batch))
    num_images = min(len(label_batch), 10)
    # rows = int(math.ceil(math.sqrt(num_images)))
    for n in range(num_images):
        # print("showing", image_batch[n])
        p = n * 3
        ax = plt.subplot(num_images, 3, p + 1)
        plot_spec(image_batch[n][:, :, 0], ax)
        # plt.imshow(np.uint8(image_batch[n]))
        plt.title(labels[np.argmax(label_batch[n])] + " sftf")
        # plt.axis("off")

        ax = plt.subplot(num_images, 3, p + 2)
        plot_mel(mel[n][:, :, 0], ax)
        # plt.imshow(np.uint8(image_batch[n]))
        plt.title(labels[np.argmax(label_batch[n])] + " mel")
        # plt.axis("off")
        # print(image_batch[1][n].shape)
        ax = plt.subplot(num_images, 3, p + 3)
        plot_mfcc(mfcc[n][:, :, 0], ax)
        # plt.title(labels[np.argmax(label_batch[n])] + " mfcc")
        # plt.axis("off")

    plt.show()


def plot_mfcc(mfccs, ax):
    img = librosa.display.specshow(mfccs.numpy(), x_axis="time", ax=ax)


def plot_mel(mel, ax):
    power = librosa.db_to_power(mel.numpy())
    img = librosa.display.specshow(
        power, x_axis="time", y_axis="mel", sr=48000, fmax=8000, ax=ax
    )


def plot_spec(spec, ax):
    img = librosa.display.specshow(spec.numpy(), y_axis="log", x_axis="time", ax=ax)
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
