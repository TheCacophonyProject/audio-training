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

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image, ImageOps

import tensorflow as tf
import tfrecord_util
import librosa


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
    audio_data = librosa.amplitude_to_db(data.spect, ref=np.max)
    mel = librosa.power_to_db(data.mel, ref=np.max)
    tags = sample.tags_s
    track_ids = " ".join(map(str, sample.track_ids))
    print("sample tags is ", sample.tags)
    feature_dict = {
        "audio/rec_id": tfrecord_util.bytes_feature(str(sample.rec_id).encode("utf8")),
        "audio/track_id": tfrecord_util.bytes_feature(track_ids.encode("utf8")),
        "audio/sample_rate": tfrecord_util.int64_feature(sample.rec.sample_rate),
        "audio/length": tfrecord_util.float_feature(sample.length),
        "audio/start_s": tfrecord_util.float_feature(sample.start),
        "audio/class/text": tfrecord_util.bytes_feature(tags.encode("utf8")),
        # "audio/class/label": tfrecord_util.int64_feature(labels.index(tags.tag)),
        "audio/sftf": tfrecord_util.float_list_feature(audio_data.ravel()),
        "audio/mel": tfrecord_util.float_list_feature(mel.ravel()),
        "audio/mfcc": tfrecord_util.float_list_feature(data.mfcc.ravel()),
        "audio/sftf_w": tfrecord_util.int64_feature(audio_data.shape[1]),
        "audio/sftf_h": tfrecord_util.int64_feature(audio_data.shape[0]),
        "audio/mel_w": tfrecord_util.int64_feature(mel.shape[1]),
        "audio/mel_h": tfrecord_util.int64_feature(mel.shape[0]),
        "audio/mfcc_h": tfrecord_util.int64_feature(data.mfcc.shape[1]),
        "audio/mfcc_w": tfrecord_util.int64_feature(data.mfcc.shape[0]),
        "audio/raw": tfrecord_util.float_list_feature(np.float32(data.raw)),
        "audio/raw_l": tfrecord_util.int64_feature(len(data.raw)),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


def create_tf_records(dataset, output_path, labels, num_shards=1, cropped=True):

    output_path = Path(output_path)
    if output_path.is_dir():
        logging.info("Clearing dir %s", output_path)
        for child in output_path.glob("*"):
            if child.is_file():
                child.unlink()
    output_path.mkdir(parents=True, exist_ok=True)
    samples = dataset.samples
    # keys = list(samples.keys())
    np.random.shuffle(samples)

    total_num_annotations_skipped = 0
    num_labels = len(labels)
    # pool = multiprocessing.Pool(4)
    logging.info("writing to output path: %s for %s samples", output_path, len(samples))
    writers = []
    lbl_counts = [0] * num_labels
    # lbl_counts[l] = 0
    logging.info("labels are %s", labels)

    writers = []
    for i in range(num_shards):
        name = f"%05d-of-%05d.tfrecord" % (i, num_shards)
        writers.append(tf.io.TFRecordWriter(str(output_path / name)))

    load_first = 200
    try:
        count = 0
        while len(samples) > 0:
            local_set = samples[:load_first]
            samples = samples[load_first:]
            loaded = []
            recs = set()
            for sample in local_set:
                if sample.rec_id not in recs:
                    sample.rec.get_data(resample=48000)
                    recs.add(sample.rec_id)
            # loaded.append(sample)
            # for sample in local_set:
            #     data = None
            #     try:
            #         data = sample.get_data(resample=48000)
            #     except:
            #         logging.error("Error getting data for %s", sample, exc_info=True)
            #     if data is None:
            #         continue
            #     for d in data:
            #         loaded.append((d, sample))

            loaded = np.array(local_set, dtype=object)
            np.random.shuffle(local_set)

            for sample in local_set:
                try:
                    tf_example, num_annotations_skipped = create_tf_example(
                        sample, labels
                    )
                    total_num_annotations_skipped += num_annotations_skipped
                    # do this by group where group is a track_id
                    # (possibly should be a recording id where we have more data)
                    # means we can KFold our dataset files if we want
                    writer = writers[sample.group % num_shards]
                    writer.write(tf_example.SerializeToString())
                    # print("saving example", [count % num_shards])
                    count += 1
                    if count % 100 == 0:
                        logging.info("saved %s", count)
                    # count += 1
                except Exception as e:
                    logging.error("Error saving ", exc_info=True)
            # break
    except:
        logging.error("Error saving track info", exc_info=True)
    for writer in writers:
        writer.close()
    for r in dataset.recs:
        r.rec_data = None
    logging.info(
        "Finished writing, skipped %d annotations.", total_num_annotations_skipped
    )
