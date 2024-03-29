# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw COCO 2017 dataset to TFRecord.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --image_dir="${TRAIN_IMAGE_DIR}" \
      --image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
      --object_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --caption_annotations_file="${CAPTION_ANNOTATIONS_FILE}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100
"""
from PIL import Image
from pathlib import Path

import collections
import hashlib
import io
import json
import multiprocessing
import os
from multiprocessing import Process, Queue

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image, ImageOps
import audioread.ffdec  # Use ffmpeg decoder

import tensorflow as tf
import tfrecord_util
import librosa

from audiodataset import load_data, SpectrogramData
from multiprocessing import Pool
import tensorflow_hub as hub

import psutil


def create_tf_example(sample, labels):
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
    data = sample.spectogram_data
    # audio_data = librosa.amplitude_to_db(data.spect)
    # mel = librosa.power_to_db(data.mel, ref=np.max)
    # mel = data.mel
    tags = sample.tags_s
    track_ids = " ".join(map(str, sample.track_ids))
    feature_dict = {
        "audio/rec_id": tfrecord_util.bytes_feature(str(sample.rec_id).encode("utf8")),
        "audio/track_id": tfrecord_util.bytes_feature(track_ids.encode("utf8")),
        "audio/sample_rate": tfrecord_util.int64_feature(sample.sr),
        "audio/min_freq": tfrecord_util.float_feature(
            -1 if sample.min_freq is None else sample.min_freq
        ),
        "audio/max_freq": tfrecord_util.float_feature(
            -1 if sample.max_freq is None else sample.max_freq
        ),
        "audio/length": tfrecord_util.float_feature(sample.length),
        "audio/signal_percent": tfrecord_util.float_feature(
            0 if sample.signal_percent is None else sample.signal_percent
        ),
        "audio/raw_length": tfrecord_util.float_feature(data.raw_length),
        "audio/start_s": tfrecord_util.float_feature(sample.start),
        "audio/class/text": tfrecord_util.bytes_feature(tags.encode("utf8")),
        "audio/raw": tfrecord_util.float_list_feature(np.float32(data.raw.ravel())),
    }
    if data.buttered is not None:
        feature_dict["audio/buttered"] = tfrecord_util.float_list_feature(
            np.float32(data.buttered.ravel())
        )
    if sample.predicted_labels is not None:
        predicted_labels = ",".join(sample.predicted_labels)
        pred_dic = {
            "audio/embed_predictions": tfrecord_util.bytes_feature(
                predicted_labels.encode("utf8"),
            ),
        }
        feature_dict.update(pred_dic)

    if sample.embeddings is not None:
        pred_dic = {
            EMBEDDING: tfrecord_util.float_list_feature(sample.embeddings.ravel()),
            LOGITS: tfrecord_util.float_list_feature(sample.logits.ravel()),
            EMBEDDING_SHAPE: (
                tfrecord_util.int64_list_feature(sample.embeddings.shape),
            ),
        }
        feature_dict.update(pred_dic)
        print("Adding embeddings", sample.embeddings.shape)
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


EMBEDDING = "embedding"
RAW_AUDIO = "raw_audio"
RAW_AUDIO_SHAPE = "raw_audio_shape"
LOGITS = "logits"
EMBEDDING_SHAPE = "embedding_shape"


def create_tf_embed(sample, labels):
    tags = sample.tags_s
    track_ids = " ".join(map(str, sample.track_ids))
    feature_dict = {
        "audio/rec_id": tfrecord_util.bytes_feature(str(sample.rec_id).encode("utf8")),
        "audio/track_id": tfrecord_util.bytes_feature(track_ids.encode("utf8")),
        "audio/sample_rate": tfrecord_util.int64_feature(sample.sr),
        "audio/length": tfrecord_util.float_feature(sample.length),
        "audio/start_s": tfrecord_util.float_feature(sample.start),
        "audio/class/text": tfrecord_util.bytes_feature(tags.encode("utf8")),
        EMBEDDING: tfrecord_util.float_list_feature(sample.embeddings.ravel()),
        LOGITS: tfrecord_util.float_list_feature(sample.logits.ravel()),
        EMBEDDING_SHAPE: (tfrecord_util.int64_list_feature(sample.embeddings.shape),),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


config = None
labels = None
writer = None
base_dir = None
saved = 0
writer_i = 0
model = None
embedding_model = None
embedding_labels = None

DO_EMBEDDING = False

#
# def worker_init(c, l, d):
#     global config
#     global labels
#     global base_dir
#     labels = l
#     config = c
#     base_dir = d
#     assign_writer()
#
#     if DO_EMBEDDING:
#         global embedding_model
#         global embedding_labels
#         global model
#         global embedding_model
#         # Load the model.
#         model = hub.load("https://tfhub.dev/google/bird-vocalization-classifier/1")
#         embedding_model = tf.keras.models.load_model("./embedding_model")
#         meta_file = "./embedding_model/metadata.txt"
#         with open(str(meta_file), "r") as f:
#             meta_data = json.load(f)
#
#         embedding_labels = meta_data.get("labels")


def process_job(queue, labels, config, base_dir):
    import gc

    # Load the model.
    model = None

    embedding_model = None
    embedding_labels = None
    if DO_EMBEDDING:
        model = hub.load("https://tfhub.dev/google/bird-vocalization-classifier/1")

    pid = os.getpid()

    writer_i = 1
    name = f"{writer_i}-{pid}.tfrecord"

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    writer = tf.io.TFRecordWriter(str(base_dir / name), options=options)
    i = 0
    saved = 0
    while True:
        i += 1
        rec = queue.get()
        try:
            if rec == "DONE":
                writer.close()
                break
            else:
                saved += save_data(
                    rec,
                    writer,
                    model,
                    embedding_model,
                    base_dir,
                    config,
                    embedding_labels,
                    config.filter_frequency,
                )
                del rec
                if saved > 500:
                    logging.info("Closing old writer")
                    writer.close()
                    writer_i += 1
                    name = f"{writer_i}-{pid}.tfrecord"
                    logging.info("Opening %s", name)
                    saved = 0
                    writer = tf.io.TFRecordWriter(str(base_dir / name), options=options)
                if i % 10 == 0:
                    logging.info("Clear gc")
                    gc.collect()
        except:
            logging.error("Process_job error %s", rec.filename, exc_info=True)


def close_writer(empty=None):
    global writer
    if writer is not None:
        logging.info("Closing old writer")
        writer.close()


def assign_writer():
    close_writer()
    pid = os.getpid()
    global writer_i
    writer_i += 1
    w = name = f"{writer_i}-{pid}.tfrecord"
    logging.info("assigning writer %s", w)
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    global writer
    writer = tf.io.TFRecordWriter(str(base_dir / name), options=options)


def save_data(
    rec,
    writer,
    model,
    embedding_model,
    base_dir,
    config,
    embedding_labels,
    filter_frequency,
):
    resample = 48000
    try:
        aro = audioread.ffdec.FFmpegAudioFile(rec.filename)
        orig_frames, sr = librosa.load(aro, sr=None)
        aro.close()
    except:
        logging.error("Error loading rec %s ", rec.filename, exc_info=True)
        try:
            aro.close()
        except:
            pass
        return 0
    try:
        if DO_EMBEDDING:
            frames32 = librosa.resample(orig_frames, orig_sr=sr, target_sr=32000)

        # hack to handle getting new samples without knowing length until load
        if resample is not None and resample != sr:
            frames = librosa.resample(orig_frames, orig_sr=sr, target_sr=resample)
            sr = resample
        else:
            frames = orig_frames
        orig_frames = None
        for t in rec.tracks:
            if t.end is None:
                # logging.info(
                #     "Track end is none so setting to rec length %s", len(frames) / sr
                # )
                t.end = len(frames) / sr
        # rec.tracks[0].end = len0(frames) / sr
        print(config.filter_frequency)
        rec.load_samples(
            config.segment_length,
            config.segment_stride,
            do_overlap=not config.filter_frequency,
        )
        samples = rec.samples
        rec.sample_rate = resample
        for i, sample in enumerate(samples):
            try:
                min_freq = sample.min_freq
                max_freq = sample.max_freq
                spec = load_data(
                    config,
                    sample.start,
                    frames,
                    sr,
                    end=sample.end,
                    min_freq=min_freq,
                    max_freq=max_freq,
                )
                if DO_EMBEDDING:
                    start = sample.start * 32000
                    start = round(start)
                    end = round(sample.end * 32000)
                    if (end - start) > 32000 * config.segment_length:
                        end = start + 32000 * config.segment_length
                    data = frames32[start:end]
                    data = np.pad(data, (0, 32000 * 5 - len(data)))
                    logits, embeddings = model.infer_tf(data[np.newaxis, :])
                    sample.logits = logits.numpy()[0]
                    sample.embeddings = embeddings.numpy()[0]
                logging.info("Mem %s", psutil.virtual_memory()[2])
                # print("mel is", mel.shape)
                # print("adjusted start is", sample.start, " becomes", sample.start - start)
                if spec is None:
                    logging.warn("error loading spec for %s", rec.id)
                    continue
                # data[i] = spec
                sample.spectogram_data = spec
                sample.sr = resample
            except:
                logging.error("Error %s ", rec.id, exc_info=True)
            tf_example, num_annotations_skipped = create_tf_example(sample, labels)
            writer.write(tf_example.SerializeToString())
            del sample
        saved = len(samples)
        del samples
        del frames
        del orig_frames
    except:
        logging.error("Got error %s", rec.filename, exc_info=True)
        print("ERRR return None")
        return 0
    del rec

    logging.info("Total Saved %s", saved)
    return saved


def save_embeddings(rec):
    global writer
    resample = 32000
    tf_examples = []
    try:
        aro = audioread.ffdec.FFmpegAudioFile(rec.filename)
        frames, sr = librosa.load(aro, sr=None)
        aro.close()
    except:
        logging.error("Error loading rec %s ", rec.filename, exc_info=True)
        try:
            aro.close()
        except:
            pass
        return None

    try:
        # hack to handle getting new samples without knowing length until load
        if resample is not None and resample != sr:
            frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
            sr = resample
        for t in rec.tracks:
            if t.end is None:
                logging.info(
                    "Track end is none so setting to rec length %s", len(frames) / sr
                )
                t.end = len(frames) / sr
        # rec.tracks[0].end = len0(frames) / sr
        rec.load_samples(config.segment_length, config.segment_stride)
        samples = rec.samples
        rec.sample_rate = resample
        for i, sample in enumerate(samples):
            try:
                start = sample.start * sr
                start = round(start)
                end = round(sample.end * sr)
                s_data = frames[start:end]
                data_length = len(s_data) / sr
                if len(s_data) < int(config.segment_length * sr):
                    s_data = np.pad(
                        s_data, (0, config.segment_length * sr - len(s_data))
                    )

                sample.sr = resample
                sample.spectogram_data = s_data
            except:
                logging.error("Error %s ", rec.id, exc_info=True)
        get_embeddings(samples)
        for s in samples:
            print(
                "embeddings",
                s.embeddings.shape,
                s.logits.shape,
                s.embeddings.dtype,
                s.logits.dtype,
            )
            tf_example, num_annotations_skipped = create_tf_embed(sample, labels)
            writer.write(tf_example.SerializeToString())
        global saved
        saved += len(samples)
        del rec
        del samples
        # samples = None
        # rec = None
    except:
        logging.error("Got error %s", rec.filename, exc_info=True)
        print("ERRR return None")
        return None

    logging.info("Total Saved %s", saved)
    if saved > 200:
        assign_writer()


def get_embeddings(samples):
    # model = models.TaxonomyModelTF(32000,"./models/chirp-model/", 5.0, 5.0)
    input = np.array([s.spectogram_data for s in samples])
    logging.info("Getting embeddings %s", len(samples))
    for s in samples:
        logits, embeddings = model.infer_tf(s.spectogram_data[np.newaxis, :])
        s.logits = logits.numpy()[0]
        s.embeddings = embeddings.numpy()[0]
    # return logits, embeddings


def create_tf_records(dataset, output_path, labels, num_shards=1, cropped=True):
    output_path = Path(output_path)
    if output_path.is_dir():
        logging.info("Clearing dir %s", output_path)
        for child in output_path.glob("*"):
            if child.is_file():
                child.unlink()
    output_path.mkdir(parents=True, exist_ok=True)
    samples = dataset.recs
    samples = sorted(
        samples,
        key=lambda sample: sample.id,
    )
    np.random.shuffle(samples)

    num_labels = len(labels)
    logging.info("writing to output path: %s for %s samples", output_path, len(samples))
    logging.info("labels are %s", labels)
    num_processes = 8
    total_recs = len(samples)
    try:
        job_queue = Queue()
        processes = []
        for i in range(num_processes):
            p = Process(
                target=process_job,
                args=(job_queue, labels, dataset.config, output_path),
            )
            processes.append(p)
            p.start()
        for s in samples:
            job_queue.put(s)

        logging.info("Processing %d", job_queue.qsize())
        for i in range(len(processes)):
            job_queue.put(("DONE"))
        for process in processes:
            try:
                process.join()
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt, terminating.")
                for process in processes:
                    process.terminate()
                exit()
        logging.info("Saved %s", len(samples))

    except:
        logging.error("Error saving track info", exc_info=True)

    for r in dataset.recs:
        r.rec_data = None
        for s in r.samples:
            s.spectogram_data = None
    logging.info("Finished writing")


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
