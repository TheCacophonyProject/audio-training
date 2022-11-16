# Build a segment dataset for training.
# Segment headers will be extracted from a track database and balanced
# according to class. Some filtering occurs at this stage as well, for example
# tracks with low confidence are excluded.

import argparse
import os
import random
import datetime
import logging
import pickle
import pytz
import json
from dateutil.parser import parse as parse_date
import sys

# from config.config import Config
import numpy as np

from audiodataset import AudioDataset
from audiowriter import create_tf_records
import tensorflow as tf
from tfdataset import get_dataset
import time

class AudioModel:
    VERSION = 1.0
    def __init__(self):
        self.checkpoint_folder = "./train/checkpoints"
        self.log_dir = "./train/logs"
        self.data_dir = "."
        self.model_name = "inceptionv3"
        self.batch_size = 32
        self.validation = None
        self.test = None
        self.train = None
        self.remapped = None
        self.input_shape = (378, 128)
        self.preprocess_fn = None
        self.learning_rate = 0.01
        self.load_meta()
        self.load_datasets(self.data_dir, self.labels, self.input_shape)
        self.build_model(len(self.labels))
        self.model.summary()

    def load_meta(self):
        file = f"{self.data_dir}/training-meta.json"
        with open(file, "r") as f:
            meta = json.load(f)
        self.labels = meta.get("labels", [])

    def train_model(self, run_name="test",epochs = 1):
        checkpoints = self.checkpoints(run_name)

        history = self.model.fit(
            self.train,
            validation_data=self.validation,
            epochs=epochs,
            shuffle=False,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    self.log_dir, write_graph=True, write_images=True
                ),
                *checkpoints,
            ],  # log metricslast_stats
        )

        history = history.history
        test_accuracy = None
        test_files = os.path.join(self.data_dir, "test")

        if len(test_files) > 0:
            if self.test is None:
                self.test, remapped = get_dataset(
                    # dir,
                    f"{self.data_dir}/training-data/test",
                    self.labels,
                    batch_size=self.batch_size,
                    image_size=self.input_shape,
                    preprocess_fn=self.preprocess_fn
                )
            if self.test:
                test_accuracy = self.model.evaluate(self.test)

        self.save(run_name, history=history, test_results=test_accuracy)


    def save(self, run_name=None, history=None, test_results=None):
        # create a save point
        if run_name is None:
            run_name = self.params.model_name
        self.model.save(os.path.join(self.checkpoint_folder, run_name))
        self.save_metadata(run_name, history, test_results)

    def save_metadata(self, run_name=None, history=None, test_results=None):
        #  save metadata
        if run_name is None:
            run_name = self.params.model_name
        model_stats = {}
        model_stats["name"] = self.model_name
        model_stats["labels"] = self.labels
        # model_stats["hyperparams"] = self.params
        model_stats["training_date"] = str(time.time())
        model_stats["version"] = self.VERSION
        if self.remapped is not None:
            model_stats["remapped"] = self.remapped

        if history:
            json_history = {}
            for key, item in history.items():
                if isinstance(item, list) and isinstance(item[0], np.floating):
                    json_history[key] = [float(i) for i in item]
                else:
                    json_history[key] = item
            model_stats["history"] = json_history
        if test_results:
            model_stats["test_loss"] = test_results[0]
            model_stats["test_acc"] = test_results[1]
        run_dir = os.path.join(self.checkpoint_folder, run_name)
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        json.dump(
            model_stats,
            open(
                os.path.join(run_dir, "metadata.txt"),
                "w",
            ),
            indent=4,
            cls=MetaJSONEncoder,
        )
    def build_model(self, num_labels):
        input = tf.keras.Input(shape=(*self.input_shape, 3), name="input")
        base_model, self.preprocess_fn = self.get_base_model((*self.input_shape, 3))
        x = base_model(input, training=True)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        birds = tf.keras.layers.Dense(
            num_labels, activation="softmax", name="prediction"
        )(x)
        self.model = tf.keras.models.Model(input, outputs=birds)

        self.model.compile(
            optimizer=optimizer(lr = self.learning_rate),
            loss=loss(),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
            ],
        )

    def checkpoints(self, run_name):
        val_loss = os.path.join(self.checkpoint_folder, run_name, "val_loss")

        checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(
            val_loss,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
        )
        val_acc = os.path.join(self.checkpoint_folder, run_name, "val_acc")

        checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
            val_acc,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        )

        val_precision = os.path.join(self.checkpoint_folder, run_name, "val_recall")

        checkpoint_recall = tf.keras.callbacks.ModelCheckpoint(
            val_precision,
            monitor="val_recall",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        )
        earlyStopping = tf.keras.callbacks.EarlyStopping(
            patience=22, monitor="val_accuracy"
        )
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", verbose=1
        )
        return [earlyStopping, checkpoint_acc, checkpoint_loss, reduce_lr_callback]

    def load_datasets(self, base_dir, labels, shape, test=False):
        self.train, remapped = get_dataset(
            # dir,
            f"{base_dir}/training-data/train",
            labels,
            batch_size=self.batch_size,
            image_size=self.input_shape,
            augment=True,
            resample=True,
            preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
        )
        self.validation,_ = get_dataset(
            # dir,
            f"{base_dir}/training-data/validation",
            labels,
            batch_size=self.batch_size,
            image_size=self.input_shape,
            resample=True,
                preprocess_fn=self.preprocess_fn
        )
        if test:
            self.test, _ = get_dataset(
                # dir,
                f"{base_dir}/training-data/test",
                labels,
                batch_size=batch_size,
                image_size=self.input_shape,
                preprocess_fn=self.preprocess_fn
            )

    def get_base_model(self, input_shape, weights="imagenet"):
        pretrained_model = self.model_name
        if pretrained_model == "resnet":

            return (
                tf.keras.applications.ResNet50(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.resnet.preprocess_input,
            )
        elif pretrained_model == "resnetv2":
            return (
                tf.keras.applications.ResNet50V2(
                    weights=weights, include_top=False, input_shape=input_shape
                ),
                tf.keras.applications.resnet_v2.preprocess_input,
            )
        elif pretrained_model == "resnet152":
            return (
                tf.keras.applications.ResNet152(
                    weights=weights, include_top=False, input_shape=input_shape
                ),
                tf.keras.applications.resnet.preprocess_input,
            )
        elif pretrained_model == "vgg16":
            return (
                tf.keras.applications.VGG16(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.vgg16.preprocess_input,
            )
        elif pretrained_model == "vgg19":
            return (
                tf.keras.applications.VGG19(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.vgg19.preprocess_input,
            )
        elif pretrained_model == "mobilenet":
            return (
                tf.keras.applications.MobileNetV2(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.mobilenet_v2.preprocess_input,
            )
        elif pretrained_model == "densenet121":
            return (
                tf.keras.applications.DenseNet121(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.densenet.preprocess_input,
            )
        elif pretrained_model == "inceptionresnetv2":
            return (
                tf.keras.applications.InceptionResNetV2(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.inception_resnet_v2.preprocess_input,
            )
        elif pretrained_model == "inceptionv3":
            return (
                tf.keras.applications.InceptionV3(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                tf.keras.applications.inception_v3.preprocess_input,
            )
        elif pretrained_model == "efficientnetb5":
            return (
                tf.keras.applications.EfficientNetB5(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                None,
            )
        elif pretrained_model == "efficientnetb0":
            return (
                tf.keras.applications.EfficientNetB0(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                None,
            )
        elif pretrained_model == "efficientnetb1":
            return (
                tf.keras.applications.EfficientNetB1(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                ),
                None,
            )
        raise Exception("Could not find model" + pretrained_model)

    def get_preprocess_fn(self):
        pretrained_model = self.model_name
        if pretrained_model == "resnet":
            return tf.keras.applications.resnet.preprocess_input

        elif pretrained_model == "resnetv2":
            return tf.keras.applications.resnet_v2.preprocess_input

        elif pretrained_model == "resnet152":
            return tf.keras.applications.resnet.preprocess_input

        elif pretrained_model == "vgg16":
            return tf.keras.applications.vgg16.preprocess_input

        elif pretrained_model == "vgg19":
            return tf.keras.applications.vgg19.preprocess_input

        elif pretrained_model == "mobilenet":
            return tf.keras.applications.mobilenet_v2.preprocess_input

        elif pretrained_model == "densenet121":
            return tf.keras.applications.densenet.preprocess_input

        elif pretrained_model == "inceptionresnetv2":
            return tf.keras.applications.inception_resnet_v2.preprocess_input
        elif pretrained_model == "inceptionv3":
            return tf.keras.applications.inception_v3.preprocess_input
        logging.warn(
            "pretrained model %s has no preprocessing function", pretrained_model
        )
        return None


def loss(smoothing=0):
    softmax = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=smoothing,
    )

    return softmax


def optimizer(lr=None, decay=None):
    if decay is not None:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=100000,
            decay_rate=decay,
            staircase=True,
        )
    else:
        learning_rate = lr  # setup optimizer
    if learning_rate:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    return optimizer


def main():
    init_logging()
    args = parse_args()
    am = AudioModel()
    am.train_model()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="Dir to load")

    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    args = parser.parse_args()
    return args


def init_logging():
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """

    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


class MetaJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SegmentType):
            return obj.name
        return json.JSONEncoder.default(self, obj)
if __name__ == "__main__":
    main()
