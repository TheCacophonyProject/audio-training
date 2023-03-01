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

# import pytz
import json
from dateutil.parser import parse as parse_date
import sys
import itertools

# from config.config import Config
import numpy as np

from audiodataset import AudioDataset
from audiowriter import create_tf_records
import tensorflow as tf
from tfdataset import get_dataset, mel_s, get_weighting, NOISE_LABELS, BIRD_LABELS
import time
from pathlib import Path
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt

import badwinner
from sklearn.model_selection import KFold
import tensorflow_addons as tfa

training_dir = "training-data"
other_training_dir = "training-data"


def precision_at_k(y_true, y_pred):
    K = 3
    print(y_pred)
    print(y_true)
    return tf.compat.v1.metrics.average_precision_at_k(
        tf.cast(y_true, tf.float32), y_pred, K
    )


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
        self.input_shape = mel_s
        self.preprocess_fn = None
        self.learning_rate = 0.01
        self.segment_length = None
        self.segment_stride = None
        self.load_meta()

    def load_meta(self):
        file = f"{self.data_dir}/{training_dir}/training-meta.json"
        with open(file, "r") as f:
            meta = json.load(f)
        self.labels = meta.get("labels", [])
        if "bird" not in self.labels:
            self.labels.append("bird")
        if "noise" not in self.labels:
            self.labels.append("noise")
        self.labels.sort()
        self.segment_length = meta.get("segment_length", 3)
        self.segment_stride = meta.get("segment_stride", 1.5)

    def load_weights(self, weights_file):
        logging.info("Loading %s", weights_file)
        self.model.load_weights(weights_file).expect_partial()

    def cross_fold_train(self, run_name="test", epochs=2):
        datasets = ["other-training-data", "training-data", "chime-training-data"]
        datasets = ["training-data"]
        labels = set()
        filenames = []
        for d in datasets:
            # filenames = tf.io.gfile.glob(f"{base_dir}/{training_dir}/train/*.tfrecord")
            filenames.extend(tf.io.gfile.glob(f"{self.data_dir}/{d}/train/*.tfrecord"))
            filenames.extend(
                tf.io.gfile.glob(f"{self.data_dir}/{d}/validation/*.tfrecord")
            )

            file = f"{self.data_dir}/{d}/training-meta.json"
            with open(file, "r") as f:
                meta = json.load(f)
            labels.update(meta.get("labels", []))
        labels = list(labels)
        labels.sort()
        self.labels = labels
        filenames = np.array(filenames)
        test_percent = 0.2
        test_i = int(test_percent * len(filenames))
        print("Using this many test files ", test_i)
        self.test, _ = get_dataset(
            # dir,
            filenames[:test_i],
            labels,
            batch_size=self.batch_size,
            image_size=self.input_shape,
            resample=False,
            reshuffle=False,
            shuffle=False,
            deterministic=True,
            # preprocess_fn=self.preprocess_fn,
        )
        filenames = filenames[test_i:]
        skf = KFold(n_splits=5, shuffle=True)
        fold = 0
        results = {}
        for train_index, test_index in skf.split(filenames):
            fold += 1
            self.train, remapped = get_dataset(
                # dir,
                filenames[train_index],
                labels,
                batch_size=self.batch_size,
                image_size=self.input_shape,
                augment=False,
                resample=False,
                # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
            )
            self.validation, remapped = get_dataset(
                # dir,
                filenames[test_index],
                labels,
                batch_size=self.batch_size,
                image_size=self.input_shape,
                augment=False,
                resample=False,
                # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
            )
            # self.load_datasets(self.data_dir, self.labels, self.species, self.input_shape)
            self.build_model(len(self.labels))
            class_weights = get_weighting(self.train, self.labels)
            logging.info("Weights are %s", class_weights)
            history = self.model.fit(
                self.train,
                validation_data=self.validation,
                epochs=epochs,
                shuffle=False,
                class_weight=class_weights
                # callbacks=[
                #     tf.keras.callbacks.TensorBoard(
                #         self.log_dir, write_graph=True, write_images=True
                #     ),
                #     # *checkpoints,
                # ],  # log metricslast_stats
            )
            logging.info("Finished fold %s", fold)

            true_categories = [y for x, y in self.test]
            true_categories = tf.concat(true_categories, axis=0)
            true_categories = tf.argmax(true_categories, axis=1)

            predictions = self.model.predict(self.test)
            predicted_categories = np.int64(tf.argmax(predictions, axis=1))
            test_results_acc = {}
            for i, l in enumerate(remapped.keys()):
                y_mask = true_categories == i
                predicted_y = predicted_categories[y_mask]
                correct = np.sum(predicted_y == true_categories[y_mask])
                count = np.sum(y_mask)
                logging.info(
                    "%s accuracy %s / %s - %s %%",
                    l,
                    correct,
                    count,
                    round(100 * correct / max(count, 1)),
                )
                test_results_acc[l] = round(100 * correct / max(count, 1))
            correct = np.sum(predicted_categories == true_categories)
            logging.info(
                "Total accuracy %s / %s - %s %%",
                correct,
                len(predicted_categories),
                round(100 * correct / len(predicted_categories)),
            )
            test_results_acc["All"] = round(100 * correct / len(predicted_categories))

            val_history = history.history["val_accuracy"]
            best_val = np.amax(val_history)
            test_results_acc["val"] = best_val
            results[fold] = test_results_acc
        for k, v in results.items():
            logging.info("For fold %s", k)
            for key, item in v.items():
                logging.info("Got %s %s", key, item)
                # if isinstance(item, list) and isinstance(item[0], np.floating):
                #     json_history[key] = [float(i) for i in item]
                # else:
                #     json_history[key] = item

    def train_model(self, run_name="test", epochs=15, weights=None, multi_label=False):
        self.load_datasets(self.data_dir, self.labels, self.input_shape)
        self.build_model(len(self.labels), multi_label=multi_label)

        if weights is not None:
            self.load_weights(weights)
        # 1 / 0
        # load weights for bird or not then change last layer for more species
        # x = tf.keras.layers.Dense(len(self.labels), activation="softmax")(
        #     self.model.layers[-2].output
        # )
        # self.model = tf.keras.models.Model(self.model.input, outputs=x)
        # self.model.compile(
        #     optimizer=optimizer(lr=self.learning_rate),
        #     loss=loss(multi_label),
        #     metrics=[
        #         "accuracy",
        #         # tf.keras.metrics.AUC(),
        #         # tf.keras.metrics.Recall(),
        #         # tf.keras.metrics.Precision(),
        #     ],
        # )
        self.model.summary()
        checkpoints = self.checkpoints(run_name)
        # self.model.save(os.path.join(self.checkpoint_folder, run_name))
        # return
        class_weights = get_weighting(self.train, self.labels)
        logging.info("Weights are %s", class_weights)
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
            class_weight=class_weights,
        )

        history = history.history
        test_accuracy = None
        test_files = os.path.join(self.data_dir, "test")
        #
        # if len(test_files) > 0:
        #     if self.test is None:
        #         self.test, remapped = get_dataset(
        #             # dir,
        #             f"{self.data_dir}/training-data/test",
        #             self.labels,
        #             self.species,
        #             batch_size=self.batch_size,
        #             image_size=self.input_shape,
        #             preprocess_fn=self.preprocess_fn,
        #             reshuffle=False,
        #             shuffle=False,
        #             resample=False,
        #             deterministic=True,
        #         )
        #     if self.test:
        #         test_accuracy = self.model.evaluate(self.test)

        self.save(
            run_name,
            history=history,
            test_results=test_accuracy,
            multi_label=multi_label,
        )

    def save(self, run_name=None, history=None, test_results=None, multi_label=False):
        # create a save point
        if run_name is None:
            run_name = self.params.model_name
        self.model.save(os.path.join(self.checkpoint_folder, run_name))
        self.save_metadata(run_name, history, test_results, multi_label)
        if self.test is not None:
            confusion(self.model, self.labels, self.test, run_name)

    def save_metadata(
        self, run_name=None, history=None, test_results=None, multi_label=False
    ):
        #  save metadata
        if run_name is None:
            run_name = self.params.model_name
        model_stats = {}
        model_stats["name"] = self.model_name
        model_stats["labels"] = self.labels
        model_stats["multi_label"] = multi_label
        model_stats["segment_stride"] = self.segment_stride
        model_stats["segment_length"] = self.segment_length

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

    def build_model(self, num_labels, bad=True, multi_label=False):
        if bad:
            self.model = badwinner.build_model(
                self.input_shape, None, num_labels, multi_label=multi_label
            )
        else:
            norm_layer = tf.keras.layers.Normalization()
            norm_layer.adapt(data=self.train.map(map_func=lambda spec, label: spec))
            input = tf.keras.Input(shape=(*self.input_shape, 3), name="input")
            base_model, self.preprocess_fn = self.get_base_model((*self.input_shape, 3))
            x = norm_layer(input)
            x = base_model(x, training=True)

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            birds = tf.keras.layers.Dense(
                num_labels, activation="softmax", name="prediction"
            )(x)

            outputs = [birds]
            self.model = tf.keras.models.Model(input, outputs=outputs)

        if multi_label:
            acc = tf.metrics.binary_accuracy
        else:
            acc = tf.metrics.categorical_accuracy

        hamming = tfa.metrics.HammingLoss(mode="multilabel", threshold=0.8)

        prec_at_k = tf.keras.metrics.TopKCategoricalAccuracy()
        self.model.compile(
            optimizer=optimizer(lr=self.learning_rate),
            loss=loss(multi_label),
            metrics=[
                acc,  #
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
                hamming,
                # f1,
                prec_at_k,
            ],
        )

    def checkpoints(self, run_name):
        metrics = [
            "val_loss",
            "val_binary_accuracy",
            "val_top_k_categorical_accuracy",
            "val_precision",
            "val_auc",
            "val_recall",
            "val_hamming_loss",
        ]
        checks = []
        for m in metrics:
            m_dir = os.path.join(self.checkpoint_folder, run_name, m)
            if "loss" in m:
                mode = "auto"
            else:
                mode = "max"
            m_check = tf.keras.callbacks.ModelCheckpoint(
                m_dir,
                monitor=m,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode=mode,
            )
            checks.append(m_check)
        earlyStopping = tf.keras.callbacks.EarlyStopping(
            patience=22,
            monitor="val_precision",
        )
        checks.append(earlyStopping)
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_precision", verbose=1
        )
        checks.append(reduce_lr_callback)

        return checks

    def load_datasets(self, base_dir, labels, shape, test=False):
        datasets = ["other-training-data", "training-data", "chime-training-data"]
        datasets = ["training-data"]
        labels = set()
        filenames = []
        for d in datasets:
            # filenames = tf.io.gfile.glob(f"{base_dir}/{training_dir}/train/*.tfrecord")
            filenames.extend(tf.io.gfile.glob(f"{base_dir}/{d}/train/*.tfrecord"))
            file = f"{base_dir}/{d}/training-meta.json"
            with open(file, "r") as f:
                meta = json.load(f)
            labels.update(meta.get("labels", []))
        labels = list(labels)
        labels.sort()
        self.labels = labels
        if "bird" not in self.labels:
            self.labels.append("bird")
        if "noise" not in self.labels:
            self.labels.append("noise")
        self.labels.sort()
        logging.info("Loading train")
        excluded_labels = []
        for l in self.labels:
            if l not in BIRD_LABELS and l not in ["noise", "human"]:
                excluded_labels.append(l)

        logging.info("labels are %s Excluding %s", self.labels, excluded_labels)
        self.train, remapped = get_dataset(
            # dir,
            filenames,
            self.labels,
            batch_size=self.batch_size,
            image_size=self.input_shape,
            augment=False,
            resample=False,
            excluded_labels=excluded_labels,
            # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
        )
        filenames = []
        for d in datasets:
            # filenames = tf.io.gfile.glob(f"{base_dir}/{training_dir}/train/*.tfrecord")
            filenames.extend(tf.io.gfile.glob(f"{base_dir}/{d}/validation/*.tfrecord"))
        logging.info("Loading Val")
        self.validation, _ = get_dataset(
            # dir,
            filenames,
            self.labels,
            batch_size=self.batch_size,
            image_size=self.input_shape,
            resample=False,
            excluded_labels=excluded_labels,
            # preprocess_fn=self.preprocess_fn,
        )
        if test:
            logging.info("Loading test")
            self.test, _ = get_dataset(
                # dir,
                f"{base_dir}/{training_dir}/test",
                self.labels,
                batch_size=batch_size,
                image_size=self.input_shape,
                excluded_labels=excluded_labels,
                # preprocess_fn=self.preprocess_fn,
            )
        self.remapped = remapped
        for l in excluded_labels:
            self.labels.remove(l)

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


def get_preprocess_fn(pretrained_model):
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
    logging.warn("pretrained model %s has no preprocessing function", pretrained_model)
    return None


def loss(multi_label=False, smoothing=0):
    if multi_label:
        logging.info("Using binary")
        softmax = tf.keras.losses.BinaryCrossentropy(
            label_smoothing=smoothing,
        )
    else:
        logging.info("Using cross")
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


def confusion(model, labels, dataset, filename="confusion.png"):
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5])
    true_categories = [y for x, y in dataset]
    true_categories = tf.concat(true_categories, axis=0)
    y_true = []
    for y in true_categories:
        non_zero = tf.where(y).numpy()
        y_true.append(list(non_zero.flatten()))
    y_true = y_true

    true_categories = np.int64(tf.argmax(true_categories, axis=1))
    y_pred = model.predict(dataset)

    predicted_categories = []
    for pred in y_pred:
        cur_preds = []
        for i, p in enumerate(pred):
            if p > 0.7:
                cur_preds.append(i)
        predicted_categories.append(cur_preds)

    for i, l in enumerate(labels):
        print("for ", l)
        lbl_count = 0
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        neg_c = 0
        for y, p in zip(y_true, predicted_categories):
            if i in y:
                lbl_count += 1
                if i in p:
                    tp += 1
                else:
                    fp += 1
            else:
                neg_c += 1
                if i in p:
                    fn += 1
                else:
                    tn += 1

        print("Have", lbl_count)
        if lbl_count == 0:
            continue
        print(
            "{}( {}%)\t{}( {}% )".format(
                tp, round(100 * tp / lbl_count), fp, round(100 * fp / lbl_count)
            )
        )
        print(
            "{}( {}%)\t{}( {}% )".format(
                tp, round(100 * tn / neg_c), fp, round(100 * fn / neg_c)
            )
        )
    y_true = mlb.fit_transform(y_true)
    predicted_categories = mlb.fit_transform(predicted_categories)
    cms = multilabel_confusion_matrix(
        y_true, predicted_categories, labels=np.arange(len(labels))
    )
    # Log the confusion matrix as an image summary.
    for i, cm in enumerate(cms):
        figure = plot_confusion_matrix(cm, class_names=[labels[i], "not"])
        logging.info("Saving confusion to %s", filename)
        plt.savefig(f"{labels[i]}-{filename}", format="png")


#
#
# def confusion(model, labels, dataset, filename="confusion.png"):
#     true_categories = [y for x, y in dataset]
#     true_categories = tf.concat(true_categories, axis=0)
#     y_true = []
#     for y in true_categories:
#         non_zero = tf.where(y).numpy()
#         y_true.append(non_zero.flatten())
#     y_true = np.array(y_true)
#
#     true_categories = np.int64(tf.argmax(true_categories, axis=1))
#     y_pred = model.predict(dataset)
#
#     predicted_categories = []
#     for pred in y_pred:
#         cur_preds = []
#         for i, p in enumerate(pred):
#             if p > 0.7:
#                 cur_preds.append(i)
#         predicted_categories.append(cur_preds)
#     predicted_categories = np.array(predicted_categories)
#     print(y_true, predicted_categories)
#     cm = multiconfusion_matrix(
#         true_categories, predi3cted_categories, labels=np.arange(len(labels))
#     )
#     # Log the confusion matrix as an image summary.
#     figure = plot_confusion_matrix(cm, class_names=labels)
#     logging.info("Saving confusion to %s", filename)
#     plt.savefig(filename, format="png")


# from tensorflow examples
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cm = np.nan_to_num(cm)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def main():
    init_logging()
    args = parse_args()
    print(args)
    if args.confusion is not None:
        load_model = Path("./train/checkpoints") / args.name
        logging.info("Loading %s with weights %s", load_model, "val_acc")
        hamming = tfa.metrics.HammingLoss(mode="multilabel", threshold=0.8)
        prec_at_k = tf.keras.metrics.TopKCategoricalAccuracy()
        model = tf.keras.models.load_model(
            str(load_model),
            custom_objects={
                "hamming_loss": hamming,
                "top_k_categorical_accuracy": prec_at_k,
            },
            compile=False,
        )

        model.load_weights(load_model / "val_hamming_loss").expect_partial()

        meta_file = load_model / "metadata.txt"
        print("Meta", meta_file)
        with open(str(meta_file), "r") as f:
            meta_data = json.load(f)
        labels = meta_data.get("labels")
        model_name = meta_data.get("name")
        preprocess = get_preprocess_fn(model_name)
        dataset, _ = get_dataset(
            tf.io.gfile.glob(f"./training-data/test/*.tfrecord"),
            labels,
            image_size=mel_s,
            shuffle=False,
            resample=False,
            deterministic=True,
            reshuffle=False,
            batch_size=64,
        )

        hamming = tfa.metrics.HammingLoss(mode="multilabel", threshold=0.8)

        prec_at_k = tf.keras.metrics.TopKCategoricalAccuracy()
        model.compile(
            optimizer=optimizer(lr=1),
            loss=loss(True),
            metrics=[
                # acc,  #
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
                hamming,
                # f1,
                prec_at_k,
            ],
        )
        model.evaluate(dataset)

        if dataset is not None:
            confusion(model, labels, dataset, args.confusion)

    else:
        am = AudioModel()
        if args.cross:
            am.cross_fold_train(run_name=args.name)
        else:
            args.multi = args.multi == 1
            am.train_model(
                run_name=args.name, weights=args.weights, multi_label=args.multi
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confusion", help="Save confusion matrix for model")
    parser.add_argument("-w", "--weights", help="Weights to use")
    parser.add_argument("--cross", action="count", help="Cross fold val")
    parser.add_argument("--multi", action="count", help="Multi label")

    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument("name", help="Run name")

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
