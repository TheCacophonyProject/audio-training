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
from tfdataset import (
    get_dataset,
    DIMENSIONS,
    get_weighting,
    NOISE_LABELS,
    SPECIFIC_BIRD_LABELS,
    get_excluded_labels,
    YAMNET_EMBEDDING_SHAPE,
)

# from tfdatasetembeddings import get_dataset, DIMENSIONS
import time
from pathlib import Path
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt

import badwinner
import badwinner2
from sklearn.model_selection import KFold
import tensorflow_addons as tfa
import math

from resnet import wr_resnet

#
# num_residual_units = 2
# momentum = 0.9
# l2_weight = 0.0001
# k = 2
# lr_step_epoch = 100
# lr_decay = 0.1


class AudioModel:
    VERSION = 1.0

    def __init__(
        self,
        model_name="badwinner2",
        data_dir="/data/audio-data",
        second_data_dir=None,
        training_dir="./train",
    ):
        print(data_dir, second_data_dir is None)
        self.training_dir = Path(training_dir)
        self.checkpoint_folder = self.training_dir / "checkpoints"
        self.log_dir = self.training_dir / "logs"
        self.data_dir = data_dir
        self.second_data_dir = second_data_dir
        self.model_name = model_name
        self.batch_size = 32
        self.validation = None
        self.test = None
        self.train = None
        self.remapped = None
        self.input_shape = DIMENSIONS
        if model_name == "embeddings":
            self.input_shape = YAMNET_EMBEDDING_SHAPE

        self.preprocess_fn = None
        self.learning_rate = 0.01
        self.mean_sub = False
        self.training_data_meta = None
        self.loss_fn = None
        self.load_meta()

    def load_meta(self):
        file = self.data_dir / "training-meta.json"
        with open(file, "r") as f:
            meta = json.load(f)
        self.labels = meta.get("labels", [])
        if "bird" not in self.labels:
            self.labels.append("bird")
        if "noise" not in self.labels:
            self.labels.append("noise")
        self.labels.sort()
        self.training_data_meta = meta

    def load_weights(self, weights_file):
        logging.info("Loading %s", weights_file)
        self.model.load_weights(weights_file).expect_partial()

    def cross_fold_train(self, run_name="test", epochs=15, multi=True):
        datasets = ["other-training-data", "training-data", "chime-training-data"]
        datasets = ["./training-data"]
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
        if "bird" not in self.labels:
            self.labels.append("bird")
        if "noise" not in self.labels:
            self.labels.append("noise")
        excluded_labels = get_excluded_labels(self.labels)
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
            excluded_labels=excluded_labels,
            mean_sub=self.mean_sub
            # preprocess_fn=self.preprocess_fn,
        )
        filenames = filenames[test_i:]
        skf = KFold(n_splits=5, shuffle=True)
        fold = 0
        results = {}
        og_labels = labels.copy()
        for l in excluded_labels:
            labels.remove(l)
        for train_index, test_index in skf.split(filenames):
            fold += 1
            self.train, remapped = get_dataset(
                # dir,
                filenames[train_index],
                og_labels,
                batch_size=self.batch_size,
                image_size=self.input_shape,
                augment=False,
                resample=False,
                excluded_labels=excluded_labels,
                mean_sub=self.mean_sub
                # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
            )
            self.validation, remapped = get_dataset(
                # dir,
                filenames[test_index],
                og_labels,
                batch_size=self.batch_size,
                image_size=self.input_shape,
                augment=False,
                resample=False,
                excluded_labels=excluded_labels,
                mean_sub=self.mean_sub
                # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
            )
            # self.load_datasets(self.data_dir, self.labels, self.species, self.input_shape)
            self.build_model(len(labels), multi_label=multi)
            class_weights = get_weighting(self.train, self.labels)
            logging.info("Weights are %s", class_weights)
            cm_dir = self.checkpoint_folder / run_name
            cm_dir.mkdir(parents=True, exist_ok=True)
            history = self.model.fit(
                self.train,
                validation_data=self.validation,
                epochs=epochs,
                shuffle=False,
                class_weight=class_weights,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        cm_dir / "val_loss",
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=True,
                        mode="min",
                    )
                ]
                # callbacks=[
                #     tf.keras.callbacks.TensorBoard(
                #         self.log_dir, write_graph=True, write_images=True
                #     ),
                #     # *checkpoints,
                # ],  # log metricslast_stats
            )
            logging.info("Finished fold %s", fold)
            self.model.load_weights(str(cm_dir / "val_loss")).expect_partial()
            true_categories = [y for x, y in self.test]
            true_categories = tf.concat(true_categories, axis=0)
            y_true = []
            for y in true_categories:
                non_zero = tf.where(y).numpy()
                y_true.append(list(non_zero.flatten()))
            y_true = y_true

            predictions = self.model.predict(self.test)
            # predicted_categories = np.int64(tf.argmax(predictions, axis=1))
            threshold = 0.5
            predicted_categories = []
            for pred in predictions:
                cur_preds = []
                for i, p in enumerate(pred):
                    if p > threshold:
                        cur_preds.append(i)
                predicted_categories.append(cur_preds)

            test_results_acc = {}
            correct = 0
            total = 0
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
                        total += 1
                        lbl_count += 1
                        if i in p:
                            tp += 1
                            correct += 1
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
                        tp, round(100 * tp / (tp + fp)), fp, round(100 * fp / (tp + fp))
                    )
                )
                print(
                    "{}( {}%)\t{}( {}% )".format(
                        fn, round(100 * fn / (tn + fn)), tn, round(100 * tn / (tn + fn))
                    )
                )
                accuracy = round(100 * tp / (tp + fp))
                test_results_acc[l] = [[tp, fp], [fn, tn]]
            # for i, l in enumerate(remapped.keys()):
            #
            #     y_mask = true_categories == i
            #     predicted_y = predicted_categories[y_mask]
            #     correct = np.sum(predicted_y == true_categories[y_mask])
            #     count = np.sum(y_mask)
            #     logging.info(
            #         "%s accuracy %s / %s - %s %%",
            #         l,
            #         correct,
            #         count,
            #         round(100 * correct / max(count, 1)),
            #     )
            #     test_results_acc[l] = round(100 * correct / max(count, 1))
            # correct = np.sum(predicted_categories == true_categories)
            logging.info(
                "Total accuracy %s / %s - %s %%",
                correct,
                total,
                round(100 * correct / total),
            )
            test_results_acc["%Correct"] = round(100 * correct / total)

            val_history = history.history["val_binary_accuracy"]
            best_val = np.amax(val_history)
            test_results_acc["val_acc"] = best_val

            loss_history = history.history["val_loss"]
            best_loss = np.amin(loss_history)
            test_results_acc["val_loss"] = best_loss
            results[fold] = test_results_acc
        for k, v in results.items():
            logging.info("For fold %s", k)
            for key, item in v.items():
                logging.info("Got %s %s", key, item)
                # if isinstance(item, list) and isinstance(item[0], np.floating):
                #     json_history[key] = [float(i) for i in item]
                # else:
                #     json_history[key] = item

    def train_model(self, run_name="test", epochs=100, weights=None, multi_label=False):
        self.log_dir = self.log_dir / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.load_datasets(self.labels, self.input_shape, test=True)
        self.num_classes = len(self.labels)
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
        model_stats.update(self.training_data_meta)
        model_stats["name"] = self.model_name
        model_stats["labels"] = self.labels
        model_stats["multi_label"] = multi_label
        model_stats["mean_sub"] = self.mean_sub
        model_stats["loss_fn"] = self.loss_fn

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

    def build_model(self, num_labels, multi_label=False):
        if self.model_name == "badwinner2":
            logging.info("Building bad winner2")
            self.model = badwinner2.build_model(
                self.input_shape, None, num_labels, multi_label=multi_label
            )
        elif self.model_name == "badwinner":
            logging.info("Building bad winner")
            raise Exception("Dont use bad winner use badwinner 2")
            self.model = badwinner.build_model(
                self.input_shape, None, num_labels, multi_label=multi_label
            )
        elif self.model_name == "embeddings":
            self.model = get_linear_model(self.input_shape, len(self.labels))
        elif self.model_name == "wr-resnet":
            self.model = wr_resnet.WRResNet(
                self.input_shape,
                len(self.labels),
            )
        else:
            norm_layer = tf.keras.layers.Normalization()
            norm_layer.adapt(data=self.train.map(map_func=lambda spec, label: spec))
            input = tf.keras.Input(shape=(*self.input_shape, 1), name="input")

            base_model, self.preprocess_fn = self.get_base_model((*self.input_shape, 1))
            x = norm_layer(input)
            x = base_model(x)
            # , training=True)
            base_model.summary()

            # x = tf.keras.layers.GlobalAveragePooling2D()(x)
            activation = "softmax"
            if multi_label:
                activation = "sigmoid"
            logging.info("Using %s activation", activation)
            birds = tf.keras.layers.Dense(
                num_labels, activation=activation, name="prediction"
            )(x)

            outputs = [birds]
            self.model = tf.keras.models.Model(input, outputs=outputs)

        if multi_label:
            acc = tf.metrics.binary_accuracy
        else:
            acc = tf.metrics.categorical_accuracy
        loss_fn = loss(multi_label)
        self.loss_fn = "BCE"
        # loss_fn.__name__
        self.model.compile(
            optimizer=optimizer(lr=self.learning_rate),
            loss=loss_fn,
            metrics=[
                acc,  #
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
                tf.keras.losses.Huber(),
            ],
        )

    def checkpoints(self, run_name):
        metrics = [
            "val_loss",
            "val_binary_accuracy",
            "val_precision",
            "val_auc",
            "val_recall",
            "huber_loss",
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
            monitor="val_loss",
            mode="min",
        )
        checks.append(earlyStopping)
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", verbose=1, mode="min"
        )
        checks.append(reduce_lr_callback)
        print(str(self.log_dir / "cm"))
        file_writer_cm = tf.summary.create_file_writer(str(self.log_dir / "cm"))
        cm_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: log_confusion_matrix(
                epoch, logs, self.model, self.validation, file_writer_cm, self.labels
            )
        )
        hist_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_hist_weights(self.model, file_writer_cm)
        )
        checks.append(hist_callback)
        return checks

    def load_datasets(self, labels, shape, test=False):
        logging.info(
            "Loading datasets with %s and secondary dir %s ",
            self.data_dir,
            self.second_data_dir,
        )
        labels = set()
        filenames = []
        second_filenames = None
        if self.second_data_dir is not None:
            second_filenames = tf.io.gfile.glob(
                f"{str(self.second_data_dir)}/train/*.tfrecord"
            )
        datasets = ["."]
        for d in datasets:
            # filenames = tf.io.gfile.glob(f"{base_dir}/{training_dir}/train/*.tfrecord")
            filenames.extend(
                tf.io.gfile.glob(f"{str(self.data_dir)}/{d}/train/*.tfrecord")
            )
            file = f"{self.data_dir}/{d}/training-meta.json"
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
        # if "other" not in self.labels:
        # self.labels.append("other")
        self.labels.sort()
        logging.info("Loading train")
        excluded_labels = get_excluded_labels(self.labels)

        logging.info("labels are %s Excluding %s", self.labels, excluded_labels)
        self.train, remapped, epoch_size = get_dataset(
            # dir,
            filenames,
            self.labels,
            batch_size=self.batch_size,
            image_size=self.input_shape,
            augment=False,
            resample=False,
            excluded_labels=excluded_labels,
            mean_sub=self.mean_sub,
            deterministic=True,
            shuffle=False,
            filenames_2=second_filenames,
            embeddings=self.model_name == "embeddings"
            # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
        )
        self.num_train_instance = epoch_size
        filenames = []
        for d in datasets:
            # filenames = tf.io.gfile.glob(f"{base_dir}/{training_dir}/train/*.tfrecord")
            filenames.extend(
                tf.io.gfile.glob(f"{str(self.data_dir)}/{d}/validation/*.tfrecord")
            )
        if self.second_data_dir is not None:
            second_filenames = tf.io.gfile.glob(
                f"{str(self.second_data_dir)}/validation/*.tfrecord"
            )
        logging.info("Loading Val")
        self.validation, _, _ = get_dataset(
            # dir,
            filenames,
            self.labels,
            batch_size=self.batch_size,
            image_size=self.input_shape,
            resample=False,
            excluded_labels=excluded_labels,
            mean_sub=self.mean_sub,
            filenames_2=second_filenames,
            embeddings=self.model_name == "embeddings"
            # preprocess_fn=self.preprocess_fn,
        )

        if test:
            logging.info("Loading test")
            filenames = []
            for d in datasets:
                # filenames = tf.io.gfile.glob(f"{base_dir}/{training_dir}/train/*.tfrecord")
                filenames.extend(
                    tf.io.gfile.glob(f"{str(self.data_dir)}/{d}/test/*.tfrecord")
                )
            if self.second_data_dir is not None:
                second_filenames = tf.io.gfile.glob(
                    f"{str(self.second_data_dir)}/test/*.tfrecord"
                )

            self.test, _, _ = get_dataset(
                # dir,
                filenames,
                self.labels,
                batch_size=self.batch_size,
                image_size=self.input_shape,
                resample=False,
                excluded_labels=excluded_labels,
                mean_sub=self.mean_sub,
                shuffle=False,
                filenames_2=second_filenames,
                embeddings=self.model_name == "embeddings"
                # preprocess_fn=self.preprocess_fn,
            )
        self.remapped = remapped
        for l in excluded_labels:
            self.labels.remove(l)
        self.training_data_meta = meta

    def get_base_model(self, input_shape, weights="imagenet"):
        pretrained_model = self.model_name
        if pretrained_model == "wr-resnet":
            model = wr_resnet.WRResNet(
                input_shape,
                len(self.labels),
            )
            return model, None
        # if pretrained_model == "wr-resnet":
        #     decay_step = lr_step_epoch * self.num_train_instance / self.batch_size
        #
        #     hp = resnet.HParams(
        #         batch_size=self.batch_size,
        #         num_classes=self.num_classes,
        #         num_residual_units=num_residual_units,
        #         k=k,
        #         weight_decay=l2_weight,
        #         initial_lr=self.learning_rate,
        #         decay_step=decay_step,
        #         lr_decay=lr_decay,
        #         momentum=momentum,
        #     )
        #     network = resnet.ResNet(hp, input_shape, self.labels)
        #     network.build_model()
        #     return network, None
        elif pretrained_model == "resnet":
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
    if pretrained_model == "wr-resnet":
        return None
    elif pretrained_model == "resnet":
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


@tf.function
def weighted_binary_cross(y_true, y_pred):
    # possibly help train for specific birds over generic
    # weights should already be in y_true i.e. [0.8,1]
    weight = y_true
    y_pred = tf.clip_by_value(
        y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()
    )
    y_true = tf.math.greater(y_true, 0)
    y_true = tf.cast(y_true, tf.float64)
    # keep logits as 1 and 0s
    term_0 = (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
    term_1 = y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())
    term_1 = term_1 * weight
    loss = tf.keras.backend.mean(term_0 + term_1, axis=1)
    return -loss


@tf.function
def sigmoid_binary_cross(y_true, y_pred):
    # equiv of optax.igmoid_binary_cross_entropy(logits, labels)

    log_p = -tf.math.log(1 + tf.math.exp(-y_pred))
    log_not_p = -tf.math.log(1 + tf.math.exp(y_pred))

    loss = -y_true * log_p - (1.0 - y_true) * log_not_p
    loss_m = tf.keras.backend.mean(loss, axis=1)
    return loss_m


def loss(multi_label=False, smoothing=0):
    if multi_label:
        # logging.info("Using focal binary cross")
        # return tf.keras.losses.BinaryFocalCrossentropy(
        #     gamma=2.0, from_logits=False, apply_class_balancing=True
        # )
        logging.info("Using binary")
        loss_fn = tf.keras.losses.BinaryCrossentropy(
            label_smoothing=smoothing,
        )
        # loss_fn = sigmoid_binary_cross
    else:
        logging.info("Using cross")
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=smoothing,
        )

    return loss_fn


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


def plot_to_image(figure):
    import io

    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def log_confusion_matrix(epoch, logs, model, dataset, writer, labels):
    # Use the model to predict the values from the validation dataset.
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(classes=np.arange(len(labels)))
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

    y_true = mlb.fit_transform(y_true)
    predicted_categories = mlb.fit_transform(predicted_categories)
    cms = multilabel_confusion_matrix(
        y_true, predicted_categories, labels=np.arange(len(labels))
    )

    for i, cm in enumerate(cms):
        print("saving cm for", labels[i])
        cm2 = np.empty((cm.shape))
        cm2[0] = np.flip(cm[1])
        cm2[1] = np.flip(cm[0])
        figure = plot_confusion_matrix(cm2, class_names=[labels[i], "not"])
        # logging.info("Saving confusion to %s", filename)
        # plt.savefig(f"{labels[i]}-{filename}", format="png")
        #
        # cm = confusion_matrix(
        #     true_categories, predicted_categories, labels=np.arange(len(model.labels))
        # )
        # Log the confusion matrix as an image summary.
        # figure = plot_confusion_matrix(cm, class_names=model.labels)
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with writer.as_default():
            tf.summary.image(f"Confusion Matrix {i}", cm_image, step=epoch)


def confusion(model, labels, dataset, filename="confusion.png"):
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(classes=np.arange(len(labels)))
    true_categories = [y for x, y in dataset]
    print(true_categories)
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
        wrong_labels = {}
        for ll in labels:
            wrong_labels[ll] = 0
        for y, p in zip(y_true, predicted_categories):
            if i in y:
                lbl_count += 1
                if i in p:
                    tp += 1
                else:
                    fp += 1
                    for lbl_p in p:
                        wrong_labels[labels[lbl_p]] += 1
            else:
                neg_c += 1
                if i in p:
                    fn += 1
                else:
                    tn += 1

        print("Have", lbl_count)
        print("Incorrects are", wrong_labels)
        if lbl_count == 0:
            continue
        print(
            "{}( {}%)\t{}( {}% )".format(
                tp, round(100 * tp / (tp + fp)), fp, round(100 * fp / (tp + fp))
            )
        )
        print(
            "{}( {}%)\t{}( {}% )".format(
                fn, round(100 * fn / (tn + fn)), tn, round(100 * tn / (tn + fn))
            )
        )
    y_true = mlb.fit_transform(y_true)
    predicted_categories = mlb.fit_transform(predicted_categories)
    cms = multilabel_confusion_matrix(
        y_true, predicted_categories, labels=np.arange(len(labels))
    )
    # Log the confusion matrix as an image summary.
    for i, cm in enumerate(cms):
        cm2 = np.empty((cm.shape))
        cm2[0] = np.flip(cm[1])
        cm2[1] = np.flip(cm[0])
        figure = plot_confusion_matrix(cm2, class_names=[labels[i], "not"])
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
    if args.confusion is not None:
        load_model = Path(args.name)
        logging.info("Loading %s with weights %s", load_model, "val_acc")
        model = tf.keras.models.load_model(
            str(load_model),
            compile=False,
        )

        # model.load_weights(load_model / "val_loss").expect_partial()

        meta_file = load_model / "metadata.txt"
        print("Meta", meta_file)
        with open(str(meta_file), "r") as f:
            meta_data = json.load(f)

        labels = meta_data.get("labels")
        print("model labels are", labels)
        # labels = meta_file.get("labels")
        model_name = meta_data.get("name")
        mean_sub = meta_data.get("mean_sub")

        preprocess = get_preprocess_fn(model_name)
        base_dir = Path(args.dataset_dir)
        meta_f = base_dir / "training-meta.json"
        dataset_meta = None
        with open(meta_f, "r") as f:
            dataset_meta = json.load(f)
        labels = dataset_meta.get("labels")
        if "bird" not in labels:
            labels.append("bird")
        if "noise" not in labels:
            labels.append("noise")
        if "other" not in labels:
            labels.append("other")
        labels.sort()
        excluded_labels = get_excluded_labels(labels)
        # self.labels = meta.get("labels", [])
        dataset, _, _ = get_dataset(
            tf.io.gfile.glob(f"{str(base_dir)}/test/*.tfrecord"),
            labels,
            image_size=DIMENSIONS,
            shuffle=False,
            resample=False,
            deterministic=True,
            reshuffle=False,
            batch_size=64,
            mean_sub=mean_sub,
            excluded_labels=excluded_labels,
            stop_on_empty=False,
        )
        for l in excluded_labels:
            labels.remove(l)
        # acc = tf.metrics.binary_accuracy
        acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        model.compile(
            optimizer=optimizer(lr=1),
            loss=loss(True),
            metrics=[
                acc,  #
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
            ],
        )
        # model.evaluate(dataset)

        if dataset is not None:
            # best_threshold(model, labels, dataset, args.confusion)
            # return
            confusion(model, labels, dataset, args.confusion)

    else:
        am = AudioModel(args.model_name, args.dataset_dir, args.second_dataset_dir)
        if args.cross:
            am.cross_fold_train(run_name=args.name)
        else:
            # args.multi = args.multi == 1
            am.train_model(
                run_name=args.name,
                epochs=args.epochs,
                weights=args.weights,
                multi_label=args.multi,
            )


def none_or_str(value):
    if value in ["null", "none", "None"]:
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Epochs to use")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/data/audio-data/training-data",
        help="Dataset directory to use",
    )
    parser.add_argument(
        "--second-dataset-dir",
        type=none_or_str,
        default="/data/audio-data/flickr-training-data",
        help="Secondary dataset directory to use",
    )
    parser.add_argument("--confusion", help="Save confusion matrix for model")
    parser.add_argument("-w", "--weights", help="Weights to use")
    parser.add_argument("--cross", action="count", help="Cross fold val")
    parser.add_argument("--multi", default=True, action="count", help="Multi label")
    parser.add_argument(
        "--model-name",
        default="badwinner2",
        help="Model to use badwinner, badwinner2, inc3",
    )

    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument("name", help="Run name")

    args = parser.parse_args()
    args.multi = args.multi > 0
    if args.dataset_dir is not None:
        args.dataset_dir = Path(args.dataset_dir)
    if args.second_dataset_dir is not None:
        args.second_dataset_dir = Path(args.second_dataset_dir)
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


# https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """

    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels

    return macro_cost


def best_threshold(model, labels, dataset, filename):
    from sklearn.metrics import roc_curve, auc

    # sklearn.metrics.auc(
    y_pred = model.predict(dataset)
    print(y_pred.shape)
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import RocCurveDisplay

    true_categories = [y for x, y in dataset]
    true_categories = tf.concat(true_categories, axis=0)

    true_categories = np.array(true_categories)
    label_binarizer = LabelBinarizer().fit(true_categories)
    y_onehot_test = label_binarizer.transform(true_categories)
    # y_onehot_test.shape  # (n_samples, n_classes)
    print(label_binarizer.classes_)
    for i, class_of_interest in enumerate(labels):
        # class_of_interest = "virginica"
        class_id = np.flatnonzero(label_binarizer.classes_ == i)[0]
        print("plt show for", class_of_interest)

        fpr, tpr, thresholds = roc_curve(
            y_onehot_test[:, class_id], y_pred[:, class_id]
        )
        # RocCurveDisplay.from_predictions(
        #     y_onehot_test[:, class_id],
        #     y_pred[:, class_id],
        #     name=f"{class_of_interest} vs the rest",
        #     color="darkorange",
        # )
        plt.plot(fpr, tpr, marker=".", label=class_of_interest)

        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)")
        plt.legend()
        plt.savefig(f"{labels[i]}-{filename}.png", format="png")
        plt.clf()
        print(tpr.shape, tpr.dtype, fpr.shape)
        # tpr = np.array(tpr)
        # fpr = np.array(fpr)
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)
        print("Best Threshold=%f, G-Mean=%.3f" % (thresholds[ix], gmeans[ix]))
        # for area, thresh in zip(areas, thresholds):
        #     # print("f", f, t)
        #     print("Thresholds are", thresh, area)


def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = (
        1 - soft_f1_class1
    )  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = (
        1 - soft_f1_class0
    )  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (
        cost_class1 + cost_class0
    )  # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost


def log_hist_weights(model, writer):
    def log_hist(epoch, logs):
        # predict images
        with writer.as_default():
            for tf_var in model.trainable_weights:
                tf.summary.histogram(tf_var.name, tf_var.numpy(), step=epoch)

    return log_hist


def get_linear_model(embedding_dim, num_classes):
    """Create a simple linear Keras model."""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=embedding_dim),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(num_classes, activation="sigmoid"),
        ]
    )
    return model


if __name__ == "__main__":
    main()
