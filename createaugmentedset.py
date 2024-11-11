import argparse
from pathlib import Path
from tfdataset import (
    get_a_dataset,
    get_excluded_labels,
    set_specific_by_count,
    get_remappings,
    load_dataset,
    set_remapped_extra,
)
import json
import numpy as np
import tensorflow as tf
import logging
import tfrecord_util
from custommels import mel_f
import sys
import librosa


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "directory",
        help="Directory of tf records to augment",
    )
    parser.add_argument(
        "out",
        help="Directory of tf records to augment",
    )
    args = parser.parse_args()
    args.directory = Path(args.directory)
    args.out = Path(args.out)

    return args


DIMENSIONS = (160, 188)
N_MELS = 160
BREAK_FREQ = 1000
MEL_WEIGHTS = mel_f(48000, N_MELS, 50, 11000, 4800, BREAK_FREQ)
MEL_WEIGHTS = tf.constant(MEL_WEIGHTS)


def init_logging():
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """

    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    args = parse_args()
    init_logging()
    file = args.directory.parent / "training-meta.json"
    with open(file, "r") as f:
        meta = json.load(f)
    labels = set(meta.get("labels", []))
    labels.add("bird")
    labels.add("noise")
    labels = list(labels)
    set_specific_by_count(meta)
    excluded_labels = get_excluded_labels(labels)
    labels.sort()

    dataset, remapped, labels = get_a_dataset(
        args.directory,
        # filenames,
        labels,
        read_record=read_all_tfrecord,
        # use_generic_bird=False,
        use_bird_tags=True,
        batch_size=None,
        image_size=DIMENSIONS,
        augment=False,
        resample=False,
        excluded_labels=excluded_labels,
        filter_freq=False,
        random_butter=0.9,
        only_features=False,
        multi_label=True,
    )
    dataset2, remapped, labels = get_a_dataset(
        args.directory,
        # filenames,
        labels,
        read_record=read_all_tfrecord,
        # use_generic_bird=False,
        use_bird_tags=True,
        batch_size=None,
        image_size=DIMENSIONS,
        augment=False,
        resample=False,
        excluded_labels=excluded_labels,
        filter_freq=False,
        random_butter=0.9,
        only_features=False,
        multi_label=True,
    )

    # zipped = tf.data.Dataset.zip((dataset, dataset2))

    # mixed = zipped.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.5))
    write(dataset, dataset2, args.out)
    print("Finished mix up")


def write(ds_one, ds_two, output_path):
    if output_path.is_dir():
        logging.info("Clearing dir %s", output_path)
        for child in output_path.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                for subc in child.iterdir():
                    if subc.is_file():
                        subc.unlink()
                child.rmdir()
    output_path.mkdir(parents=True, exist_ok=True)
    writer_i = 0
    records_per_file = 1000

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    writer = tf.io.TFRecordWriter(
        str(output_path / f"{writer_i}.tfrecord"), options=options
    )
    written = 0
    for ds_one, ds_two in zip(ds_one, ds_two):
        x, y = mix_up(ds_one, ds_two)
        if written % records_per_file == 0 and written > 0:

            print("Closing writer and starting a new one ", writer_i)
            writer_i += 1
            writer.close()
            writer = tf.io.TFRecordWriter(
                str(output_path / f"{writer_i}.tfrecord"), options=options
            )
        written += 1
        spectogram = np.abs(librosa.stft(x.numpy(), n_fft=4800, hop_length=HOP_LENGTH))
        # print("Spect shape is ", spectogram.shape)
        # 1 / 0
        tf_example, _ = create_tf_example(x, y, spectogram)
        writer.write(tf_example.SerializeToString())

    print("Written", written)
    writer.close()


def get_a_dataset(dir, labels, **args):

    global extra_label_map
    global remapped_y
    excluded_labels = args.get("excluded_labels", [])
    use_generic_bird = args.get("use_generic_bird", True)

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
    set_remapped_extra(remapped_y, extra_label_map)

    num_labels = len(labels)
    datasets = []
    logging.info("Loading tf records from %s", dir)
    filenames = tf.io.gfile.glob(str(dir / "*/*.tfrecord"))

    dataset = load_dataset(filenames, num_labels, labels, args)

    # may perform better without adding generics birds but sitll having generic label
    dataset_2 = None

    if args.get("filenames_2") is not None:
        logging.info("Loading second files %s", args.get("filenames_2")[:1])
        second = args.get("filenames_2")

        dataset_2 = load_dataset(second, len(labels), labels, args)

    else:
        logging.info("Not using second dataset")

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

    # if dataset_2 is not None:
    #     logging.info("Adding second dataset with weights [0.6,0.4]")
    #     dataset = tf.data.Dataset.sample_from_datasets(
    #         [dataset, dataset_2],
    #         weights=[0.6, 0.4],
    #         stop_on_empty_dataset=True,
    #         rerandomize_each_iteration=args.get("rerandomize_each_iteration", True),
    #     )

    batch_size = args.get("batch_size", None)

    # dont think we need this iwth interleave

    if batch_size is not None:
        dataset = dataset.batch(
            batch_size,
        )

    return dataset, remapped, labels


@tf.function
def read_all_tfrecord(
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
):
    feature_dict = {
        "audio/lat": tf.io.FixedLenFeature((), tf.float32),
        "audio/lng": tf.io.FixedLenFeature((), tf.float32),
        "audio/rec_id": tf.io.FixedLenFeature((), tf.string),
        "audio/track_id": tf.io.FixedLenFeature((), tf.string),
        "audio/sample_rate": tf.io.FixedLenFeature((), tf.int64),
        "audio/min_freq": tf.io.FixedLenFeature((), tf.float32),
        "audio/max_freq": tf.io.FixedLenFeature((), tf.float32),
        "audio/length": tf.io.FixedLenFeature((), tf.float32),
        # "audio/signal_percent": tf.io.FixedLenFeature((), tf.float32),
        "audio/raw_length": tf.io.FixedLenFeature((), tf.float32),
        "audio/start_s": tf.io.FixedLenFeature((), tf.float32),
        "audio/class/text": tf.io.FixedLenFeature((), tf.string),
        "audio/raw": tf.io.FixedLenFeature((144000), tf.float32),
    }

    example = tf.io.parse_single_example(example, feature_dict)

    label = tf.cast(example["audio/class/text"], tf.string)
    split_labels = tf.strings.split(label, sep="\n")
    global remapped_y, extra_label_map
    labels = remapped_y.lookup(split_labels)
    if multi:
        extra = extra_label_map.lookup(split_labels)
        labels = tf.concat([labels, extra], axis=0)
    else:
        labels = tf.reduce_max(labels)
    raw = example["audio/raw"]

    if one_hot:
        label = tf.reduce_max(tf.one_hot(labels, num_labels, dtype=tf.int32), axis=0)

    else:
        label = labels
    signal_percent = 0.0

    label = tf.cast(label, tf.float32)

    lat = example["audio/lat"]
    lng = example["audio/lng"]
    rec_id = example["audio/rec_id"]
    track_id = example["audio/track_id"]
    sample_rate = example["audio/sample_rate"]
    min_freq = example["audio/min_freq"]
    max_freq = example["audio/max_freq"]
    length = example["audio/length"]
    raw_length = example["audio/raw_length"]
    start_s = example["audio/start_s"]
    text = example["audio/class/text"]

    return raw, (
        label,
        lat,
        lng,
        rec_id,
        track_id,
        sample_rate,
        min_freq,
        max_freq,
        length,
        raw_length,
        start_s,
        text,
    )

    return image


def create_tf_example(x, y, spectogram):
    """Converts image and annotations to a tf.Example proto.

        Args:
          image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
            u'width', u'date_captured', u'flickr_url', u'id']
          image_dir: directory containing the image files.
          bbox_annotations:
            list of dicts with keys: [u'segmentation', u'area', u'iscrowd',
              u'image_id', u'bbox', u'category_id', u'id'] Notice that bounding box
              coordinates in the official COCO dataset are given as [x, y, width,
              height] tuples using absolute coordinates where x, y represent the
              top-left (0-indexed) corner.  This function converts to the format
              expected by the Tensorflow Object Detection API (which is which is
              [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
          size).
          category_index: a dict containing COCO category information keyed by the
            'id' field of each category.  See the label_map_util.create_category_index
            function.
          caption_annotations:
            list of dict with keys: [u'id', u'image_id', u'str'].
          include_masks: Whether to include instance segmentations masks
            (PNG encoded) in the result. default: False.

        Returns:
          example: The converted tf.Example
          num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
          ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    (
        label,
        lat,
        lng,
        rec_id,
        track_ids,
        sample_rate,
        min_freq,
        max_freq,
        length,
        raw_length,
        start_s,
        text,
        first_percent,
    ) = y
    feature_dict = {
        "audio/label": tfrecord_util.float_list_feature(np.float32(label)),
        "audio/lat": tfrecord_util.float_feature(lat),
        "audio/lng": tfrecord_util.float_feature(lng),
        "audio/rec_id": tfrecord_util.bytes_feature(str(rec_id).encode("utf8")),
        "audio/track_id": tfrecord_util.bytes_feature(track_ids.encode("utf8")),
        "audio/sample_rate": tfrecord_util.int64_feature(sample_rate),
        "audio/min_freq": tfrecord_util.float_feature(min_freq),
        "audio/max_freq": tfrecord_util.float_feature(max_freq),
        "audio/length": tfrecord_util.float_feature(length),
        "audio/signal_percent": tfrecord_util.float_feature(0),
        "audio/raw_length": tfrecord_util.float_feature(raw_length),
        "audio/start_s": tfrecord_util.float_feature(start_s),
        "audio/first_percent": tfrecord_util.float_feature(first_percent),
        "audio/class/text": tfrecord_util.bytes_feature(text.encode("utf8")),
        "audio/spectogram": tfrecord_util.float_list_feature(
            np.float32(spectogram.ravel())
        ),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.5):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two

    # batch_size = 32
    first_percent = tf.random.uniform((), 0.2, 0.8)
    # second_percent = tf.random.uniform((), 0, 1)

    images = images_one * first_percent + images_two * (1 - first_percent)
    labels = labels_one[0] * first_percent + labels_two[0] * (1 - first_percent)
    # labels = labels_one[0] + labels_two[0]
    labels = tf.clip_by_value(labels, 0, 1)
    (
        label,
        lat,
        lng,
        rec_id,
        track_id,
        sample_rate,
        min_freq,
        max_freq,
        length,
        raw_length,
        start_s,
        text,
    ) = labels_one

    (
        second_label,
        second_lat,
        second_lng,
        second_rec_id,
        second_track_id,
        second_sample_rate,
        second_min_freq,
        second_max_freq,
        second_length,
        second_raw_length,
        second_start_s,
        second_text,
    ) = labels_two

    text = tf.strings.split(text, sep="\n")
    second_text = tf.strings.split(second_text, sep="\n")
    text = tf.concat([text, second_text], axis=0)

    # text = "\n".join(map(str, text.numpy()))
    total = []
    for item in text.numpy():
        total.append(item.decode("utf8"))
    text = "\n".join(map(str, total))

    track_id = tf.strings.split(track_id, sep=" ")
    second_track_id = tf.strings.split(second_track_id, sep=" ")
    track_id = tf.concat([track_id, second_track_id], axis=0)
    # track_id = tf.strings.join(track_id.numpy(), separator=" ")
    # track_id = "\n".join(map(str, track_id.numpy()))
    total = []
    for item in track_id.numpy():
        total.append(item.decode("utf8"))
    track_id = "\n".join(map(str, total))
    return (
        images,
        (
            labels,
            -1,
            -1,
            rec_id,
            track_id,
            sample_rate,
            min_freq,
            max_freq,
            length,
            raw_length,
            start_s,
            text,
            first_percent,
        ),
    )


HOP_LENGTH = 281


@tf.function
def raw_to_mel(x, y, features=False):
    stft = tf.signal.stft(
        x,
        4800,
        HOP_LENGTH,
        fft_length=4800,
        window_fn=tf.signal.hann_window,
        pad_end=True,
        name=None,
    )
    stft = tf.transpose(stft, [1, 0])
    stft = tf.math.abs(stft)

    image = tf.tensor_dot(MEL_WEIGHTS, stft)
    # image = tf.expand_dims(image, axis=2)

    return image, y


if __name__ == "__main__":
    main()
