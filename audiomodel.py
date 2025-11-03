# Build a segment dataset for training.
# Segment headers will be extracted from a track database and balanced
# according to class. Some filtering occurs at this stage as well, for example
# tracks with low confidence are excluded.

import argparse
import os

TRAINING = False

# https://www.tensorflow.org/guide/profiler#improve_device_performance
# dont know if does anything
if TRAINING:
    import ctypes

    _libcudart = ctypes.CDLL("libcudart.so")
    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128

    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    os.environ["TF_GPU_THREAD_COUNT"] = "1"

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

# hopefully the good gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# #
import tensorflow as tf

# import tensorflow_decision_forests as tfdf
from tensorflow.keras import layers
import ydf
from utils import get_label_to_ebird_map, get_ebird_id, get_ebird_ids_to_labels

MIXED_PRECISION = False

if MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
#
#
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from birdsconfig import (
    NOISE_LABELS,
    BIRD_TRAIN_LABELS,
    ALL_BIRDS,
    EXTRA_LABELS,
    OTHER_LABELS,
    HUMAN_LABELS,
)

from tfdataset import (
    get_dataset,
    DIMENSIONS,
    get_weighting,
    get_excluded_labels,
    EMBEDDING_SHAPE,
    set_specific_by_count,
    N_MELS,
    BREAK_FREQ,
)

# from tfdatasetembeddings import get_dataset, DIMENSIONS
import time
from pathlib import Path
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt

import badwinner
import badwinner2
from sklearn.model_selection import KFold
import math

from resnet import wr_resnet

DEBUG_PROFILE = False
#
# num_residual_units = 2
# momentum = 0.9
# l2_weight = 0.0001
# k = 2
# lr_step_epoch = 100
# lr_decay = 0.1
# tensorflow stealing my log handler
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
        root_logger.removeHandler(handler)


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
        # self.remapped = None
        self.input_shape = DIMENSIONS
        if model_name == "embeddings":
            self.input_shape = EMBEDDING_SHAPE
        elif model_name == "efficientnetb0":
            self.input_shape = (self.input_shape[0], self.input_shape[1], 3)
        elif model_name == "dual-badwinner2":
            self.input_shape = (96, 511, 1)
        elif model_name == "cnn-features":
            self.input_shape = [(68, 60), (136, 3)]
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

        labels = set(meta.get("labels", []))
        if self.second_data_dir is not None:
            logging.info("Loading second metadata %s", self.second_data_dir)
            file = self.second_data_dir / "training-meta.json"
            with open(file, "r") as f:
                second_meta = json.load(f)
            second_labels = set(second_meta.get("labels", []))
            labels.update(second_labels)

        self.labels = list(labels)
        if "bird" not in self.labels:
            self.labels.append("bird")
        if "noise" not in self.labels:
            self.labels.append("noise")
        self.labels.sort()
        self.training_data_meta = meta

    def load_weights(self, weights_file):
        logging.info("Loading %s", weights_file)
        self.model.load_weights(weights_file)

    def cross_fold_train(
        self, run_name="test", epochs=15, multi=True, use_generic_bird=True
    ):
        datasets = ["./training-data"]
        labels = set()
        filenames = []

        # filenames = tf.io.gfile.glob(f"{base_dir}/{training_dir}/train/*.tfrecord")
        filenames.extend(tf.io.gfile.glob(f"{self.data_dir}/train/*.tfrecord"))
        filenames.extend(tf.io.gfile.glob(f"{self.data_dir}/validation/*.tfrecord"))
        filenames.extend(tf.io.gfile.glob(f"{self.data_dir}/test/*.tfrecord"))

        file = f"{self.data_dir}/training-meta.json"
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
        set_specific_by_count(meta)

        excluded_labels = get_excluded_labels(self.labels)
        filenames = np.array(filenames)
        np.random.shuffle(filenames)
        test_percent = 0.2
        test_i = int(test_percent * len(filenames))
        print("Using this many test files ", test_i)
        self.test, _, _ = get_dataset(
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
            mean_sub=self.mean_sub,
            use_generic_bird=use_generic_bird,
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
            self.train, remapped, _, _ = get_dataset(
                # dir,
                filenames[train_index],
                og_labels,
                batch_size=self.batch_size,
                image_size=self.input_shape,
                augment=True,
                resample=False,
                excluded_labels=excluded_labels,
                mean_sub=self.mean_sub,
                use_generic_bird=use_generic_bird,
                # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
            )
            self.validation, remapped, _, _ = get_dataset(
                # dir,
                filenames[test_index],
                og_labels,
                batch_size=self.batch_size,
                image_size=self.input_shape,
                augment=False,
                resample=False,
                excluded_labels=excluded_labels,
                mean_sub=self.mean_sub,
                use_generic_bird=use_generic_bird,
                # preprocess_fn=tf.keras.applications.inception_v3.preprocess_input,
            )
            # self.load_datasets(self.data_dir, self.labels, self.species, self.input_shape)
            self.build_model(multi_label=multi)

            if fold == 1:
                self.model.summary()
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
                        cm_dir / "val_loss.weights.h5",
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=True,
                        mode="min",
                    )
                ],
                # callbacks=[
                #     tf.keras.callbacks.TensorBoard(
                #         self.log_dir, write_graph=True, write_images=True
                #     ),
                #     # *checkpoints,
                # ],  # log metricslast_stats
            )
            logging.info("Finished fold %s", fold)
            self.model.load_weights(str(cm_dir / "val_loss"))

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
                tp_percent = -0 if tp + fp == 0 else tp / (tp + fp)
                fp_percent = -0 if tp + fp == 0 else fp / (tp + fp)
                fn_percent = -0 if tn + fn == 0 else fn / (tn + fn)
                tn_percent = -0 if tn + fn == 0 else tn / (tn + fn)

                print(
                    "{}( {}%)\t{}( {}% )".format(
                        tp, round(100 * tp_percent), fp, round(100 * fp_percent)
                    )
                )
                print(
                    "{}( {}%)\t{}( {}% )".format(
                        fn, round(100 * fn_percent), tn, round(100 * tn_percent)
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

    global_epoch = None

    def train_model(
        self,
        run_name="test",
        epochs=100,
        weights=None,
        **args,
    ):
        global global_epoch
        global_epoch = tf.Variable(
            1, name="global_epoch", trainable=False, dtype=tf.int32
        )

        self.log_dir = self.log_dir / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        labels, excluded_labels, self.training_data_meta = init_labels(
            self.data_dir, **args
        )

        (
            self.train,
            self.validation,
            self.labels,
            remapped,
            extra_label_map,
        ) = load_datasets(
            labels,
            excluded_labels,
            self.data_dir,
            self.batch_size,
            **args,
        )

        from tfdataset import DIMENSIONS

        self.input_shape = DIMENSIONS
        if self.model_name == "embeddings":
            self.input_shape = EMBEDDING_SHAPE
        elif "efficientnet" in self.model_name:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 3)
        elif self.model_name == "dual-badwinner2":
            self.input_shape = (96, 511, 1)

        if args.get("n_mels") != 160:
            self.input_shape = (args.get("n_mels"), self.input_shape[1], 3)
        args["excluded_labels"] = excluded_labels
        args["remapped_labels"] = remapped
        args["extra_label_map"] = extra_label_map
        self.num_classes = len(self.labels)
        if args.get("rf_model") and args.get("cnn_model"):
            models = []
            inputs = []

            outputs = []

            rf = tf.keras.models.load_model(args.get("rf_model"))
            cnn = tf.keras.models.load_model(args.get("cnn_model"), compile=False)
            cnn.load_weights(Path(args.get("cnn_model")) / "val_binary_accuracy")
            cnn.trainable = False
            rf.trainable = False

            inputs = [
                tf.keras.Input(shape=(68 * 60 + 136 * 3), name="features"),
                cnn.input,
            ]
            output = tf.keras.layers.Concatenate()([rf(inputs[0]), cnn.outputs[0]])
            output = layers.Dense(len(self.labels))(output)
            output = tf.keras.activations.sigmoid(output)

            self.model = tf.keras.models.Model(inputs=inputs, outputs=output)
            self.model.summary()
            if args.get("multi_label"):
                acc = tf.metrics.binary_accuracy
            else:
                acc = tf.metrics.categorical_accuracy
            # loss_fn = loss(args.get("multi_label", True))
            loss_fn = WeightedCrossEntropy(self.labels)
            self.loss_fn = loss_fn.name
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

            if weights is not None:
                self.load_weights(weights)
        else:
            # can use for 2 gpus, but seems to be much slower on our setup
            # strategy = tf.distribute.MirroredStrategy()
            # with strategy.scope():
            weights_labels = None
            if weights is not None:
                weights_metadata = Path(weights).parent / "metadata.txt"
                if weights_metadata.exists():
                    with weights_metadata.open("r") as f:
                        weights_meta = json.load(f)
                    weights_labels = weights_meta.get("labels")
                    print("Weight labels are ", weights_labels)
            self.build_model(
                multi_label=args.get("multi_label", True),
                loss_fn=args.get("loss_fn", "keras"),
                weight_labels=weights_labels,
                weights=weights,
            )
            (self.checkpoint_folder / run_name).mkdir(parents=True, exist_ok=True)
            if self.model_name != "rf-features":
                self.model.save(self.checkpoint_folder / run_name / f"{run_name}.keras")
            self.save_metadata(run_name, None, None, **args)

        checkpoints = self.checkpoints(
            run_name, multi_label=args.get("multi_label", True)
        )
        class_weights = None
        if args.get("use_weighting", True):
            class_weights = get_weighting(self.train, self.labels)
            logging.info("Weights are %s", class_weights)
        history = None
        if self.model_name == "rf-features":
            ydf_ds = tf_to_ydf(self.train)

            self.model = self.model.train(ydf_ds)
            self.model.save(str(self.checkpoint_folder / run_name / "rf"))
            self.load_test_set(**args)
            self.train = None
            ydf_ds = tf_to_ydf(self.test)

            evaluation = self.model.evaluate(ydf_ds)
            print(evaluation)
            self.save_metadata(
                run_name + "/rf",
                history,
                test_results=(evaluation.accuracy, evaluation.loss),
                **args,
            )
            return
            # history = self.model.fit(self.train, validation_data=self.validation)
        else:
            policy = tf.keras.mixed_precision.global_policy()
            logging.info("Policy is %s", policy)
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
        self.load_test_set(**args)
        self.save(run_name, history=history, test_results=test_accuracy, **args)

    def save(self, run_name=None, history=None, test_results=None, **args):
        # create a save point
        if run_name is None:
            run_name = self.model_name
        self.model.save(self.checkpoint_folder / run_name / f"{run_name}.keras")
        self.save_metadata(run_name, history, test_results, **args)
        if self.test is not None:
            acc = (
                "val_binary_accuracy.weights.h5"
                if args.get("multi_label")
                else "val_categorical_accuracy.weights.h5"
            )
            weights = "val_loss.weights.h5"
            self.model.load_weights(
                os.path.join(self.checkpoint_folder, run_name, weights)
            )
            confusion_file = self.checkpoint_folder / run_name / "confusion-val_loss"
            if args.get("multi_label"):
                multi_confusion_single(
                    self.model,
                    self.labels,
                    self.test,
                    confusion_file,
                    model_name=self.model_name,
                )
            else:
                confusion(self.model, self.labels, self.test, confusion_file)

    def save_metadata(self, run_name, history, test_results, **args):
        #  save metadata
        if run_name is None:
            run_name = self.params.model_name
        model_stats = {}
        model_stats.update(self.training_data_meta)
        model_stats.update(args)
        model_stats["name"] = self.model_name
        model_stats["magv2"] = True
        model_stats["bird_thresh"] = 1
        model_stats["use_generic_bird"] = args.get("use_generic_bird", False)
        # model_stats["filter_freq"] = args.get("filter_freq", False)
        model_stats["ebird_labels"] = self.labels
        # model_stats["multi_label"] = args.get("multi_label", False)
        model_stats["mean_sub"] = self.mean_sub
        model_stats["loss_fn"] = self.loss_fn
        model_stats["bird_labels"] = BIRD_TRAIN_LABELS
        model_stats["noise_labels"] = NOISE_LABELS
        model_stats["extra_labels"] = EXTRA_LABELS
        model_stats["other_labels"] = OTHER_LABELS
        from tfdataset import FMIN, FMAX, NFFT

        model_stats["fmin"] = FMIN
        model_stats["fmax"] = FMAX

        if model_stats.get("n_mels") is None:
            model_stats["n_mels"] = N_MELS
        if model_stats.get("break_freq") is None:
            model_stats["break_freq"] = BREAK_FREQ
        model_stats["power"] = 2
        model_stats["n_fft"] = NFFT
        model_stats["lme"] = self.lme
        model_stats["db_scale"] = False
        # model_stats["hyperparams"] = self.params
        model_stats["training_date"] = str(time.time())
        model_stats["version"] = self.VERSION
        # if self.remapped is not None:
        # model_stats["remapped"] = self.remapped

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

    def build_model(
        self, multi_label=False, loss_fn="keras", weight_labels=None, weights=None
    ):
        if weights is not None and weight_labels is not None:
            # build model wight these labels then pop the dense layer and add a new one for our labels
            labels = weight_labels
        else:
            labels = self.labels

        num_labels = len(labels)

        activation = "softmax"
        if multi_label:
            activation = "sigmoid"
        if self.model_name == "merge":
            inputs = []

            bad_model = self.model = badwinner2.build_model(
                self.input_shape,
                None,
                num_labels,
                multi_label=multi_label,
                lme=self.lme,
            )
            inputs.append(bad_model.inputs[0])
            inputs.append(tf.keras.Input(shape=(68, 60), name="short_f"))
            inputs.append(tf.keras.Input(shape=(136, 3), name="mid_f"))

            short, mid = feature_cnn(inputs[1], inputs[2], num_labels)
            output = tf.keras.layers.Concatenate()([bad_model.output, short, mid])
            # output = tf.keras.layers.Dense(128, activation="relu")(output)

            # output = tf.keras.layers.Conv2D(
            #     num_labels,
            #     1,
            #     activation=tf.keras.layers.LeakyReLU(),
            #     kernel_initializer=tf.keras.initializers.Orthogonal(),
            # )(output)
            # # x = logmeanexp(x, sharpness=1, axis=2)
            # output = tf.keras.layers.GlobalAveragePooling2D()(x)

            # rather than dense could just weight each output with a learnable weight and then average
            #  i think this should learn it the same but allow for more complex patterns
            output = layers.Dense(num_labels)(output)
            output = tf.keras.layers.Activation(
                activation, dtype="float32", name="predictions"
            )(output)
            self.model = tf.keras.models.Model(inputs, outputs=output)
            self.model.summary()
        elif self.model_name == "dual-badwinner2":
            model = badwinner2.build_model(
                self.input_shape,
                None,
                num_labels,
                multi_label=multi_label,
                lme=self.lme,
                n_mels=96,
            )
            model_2 = badwinner2.build_model(
                self.input_shape,
                None,
                num_labels,
                multi_label=multi_label,
                lme=self.lme,
                input_name="input2",
                n_mels=96,
            )
            inputs = []
            model_2.input.name = "input2"
            inputs.append(model.input)
            inputs.append(model_2.input)
            output = tf.keras.layers.Concatenate()([model.output, model_2.output])

            #  i think this should learn it the same but allow for more complex patterns
            output = layers.Dense(num_labels)(output)
            output = tf.keras.layers.Activation(
                activation, dtype="float32", name="predictions"
            )(output)
            self.model = tf.keras.models.Model(inputs, outputs=output)
            self.model.summary()
            print("MODEL MADE")
        elif self.model_name == "badwinner2":
            logging.info("Building bad winner2")
            self.model = badwinner2.build_model(
                self.input_shape,
                None,
                num_labels,
                multi_label=multi_label,
                lme=self.lme,
            )
        elif self.model_name == "cnn-features":
            inputs = []
            inputs.append(tf.keras.Input(shape=(68, 60), name="short_f"))
            inputs.append(tf.keras.Input(shape=(136, 3), name="mid_f"))

            short, mid = feature_cnn(inputs[0], inputs[1], num_labels)
            output = tf.keras.layers.Concatenate()([short, mid])
            output = layers.Dense(num_labels)(output)
            output = tf.keras.layers.Activation(
                activation, dtype="float32", name="predictions"
            )(output)
            self.model = tf.keras.models.Model(inputs, outputs=output)
            self.model.summary()
        elif self.model_name == "rf-features":
            self.model = ydf.RandomForestLearner(label="y")

            # self.model = tfdf.keras.RandomForestModel()
            return
        elif self.model_name == "badwinner2-res":
            logging.info("Building bad winner2 res")
            self.model = badwinner2.build_model_res(
                self.input_shape, None, num_labels, multi_label=multi_label
            )

        elif self.model_name == "embeddings":
            self.model = get_linear_model(self.input_shape, len(labels))
        elif self.model_name == "wr-resnet":
            self.model = wr_resnet.WRResNet(
                self.input_shape,
                len(labels),
            )
        else:
            # norm_layer = tf.keras.layers.Normalization()
            # norm_layer.adapt(data=self.train.map(map_func=lambda spec, label: spec))
            input = tf.keras.Input(shape=self.input_shape, name="input")

            base_model, self.preprocess_fn = self.get_base_model(self.input_shape)
            # x = norm_layer(input)
            x = badwinner2.MagTransform()(input)

            x = base_model(x)
            # , training=True)

            if self.lme:
                x = badwinner2.LMELayer(axis=1, sharpness=5)(x)
                x = badwinner2.LMELayer(axis=2, sharpness=5)(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            activation = "softmax"
            if multi_label:
                activation = "sigmoid"
            logging.info("Using %s activation", activation)
            x = tf.keras.layers.Dense(num_labels, name="prediction")(x)

            birds = tf.keras.layers.Activation(
                activation, dtype="float32", name="predictions"
            )(x)

            outputs = [birds]
            self.model = tf.keras.models.Model(input, outputs=outputs)

        if multi_label:
            acc = tf.metrics.binary_accuracy
        else:
            acc = tf.metrics.categorical_accuracy
        if loss_fn == "WeightedCrossEntropy":
            logging.info("Using weighted cross entropy")
            loss_fn = WeightedCrossEntropy(labels)
        else:
            logging.info("Using cross entropy")
            loss_fn = loss(multi_label)

        self.loss_fn = loss_fn.name

        if weights is not None:
            logging.info("Loading weights %s", weights)
            self.model.load_weights(weights)
            if weight_labels is not None:
                logging.info(
                    "Changing final layer as number of labels differs from loaded weights"
                )
                self.model = tf.keras.Model(
                    inputs=self.model.input, outputs=self.model.layers[-3].output
                )
                num_labels = len(self.labels)
                self.model.summary()
                logging.info("Using %s activation", activation)
                x = tf.keras.layers.Dense(num_labels, name="prediction")(
                    self.model.output
                )

                birds = tf.keras.layers.Activation(
                    activation, dtype="float32", name="predictions"
                )(x)

                outputs = [birds]
                self.model = tf.keras.models.Model(input, outputs=outputs)

        self.model.compile(
            optimizer=optimizer(lr=self.learning_rate),
            loss=loss_fn,
            metrics=[
                acc,
                # precAtK( not working on gpu
                #     k=3,
                #     num_labels=len(self.labels),
                #     # , bird_i=self.labels.index("bird")
                # ),
                tf.keras.losses.BinaryFocalCrossentropy(),
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
                tf.keras.losses.Huber(),
            ],
        )
        self.model.summary()

    def checkpoints(self, run_name, multi_label=True):
        metrics = [
            "val_loss",
            "val_precision",
            "val_auc",
            "val_recall",
            "val_huber_loss",
            # "val_precK",
            "val_binary_focal_crossentropy",
        ]
        accuracy = "val_categorical_accuracy"
        if multi_label:
            accuracy = "val_binary_accuracy"
        metrics.append(accuracy)
        checks = []
        for m in metrics:
            m_dir = self.checkpoint_folder / run_name / f"{m}.weights.h5"
            if "loss" in m or "focal" in m:
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
            patience=10,
            monitor="val_loss",
            mode="min",
        )
        checks.append(earlyStopping)
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", verbose=1, mode="max"
        )
        checks.append(reduce_lr_callback)
        # file_writer_cm = tf.summary.create_file_writer(str(self.log_dir / "cm"))
        # cm_callback = tf.keras.callbacks.LambdaCallback(
        #     on_epoch_end=lambda epoch, logs: log_confusion_matrix(
        #         epoch, logs, self.model, self.validation, file_writer_cm, self.labels
        #     )
        # )
        # checks.append(cm_callback)
        weight_writer = tf.summary.create_file_writer(str(self.log_dir / "weights"))

        hist_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_hist_weights(self.model, weight_writer)
        )
        checks.append(hist_callback)

        model_checkpt = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_folder / run_name / f"chkpt.weights.h5",
            verbose=1,
            save_weights_only=True,
            save_freq="epoch",
        )
        checks.append(model_checkpt)

        if DEBUG_PROFILE:
            tboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir / run_name / "profiler",
                histogram_freq=1,
                profile_batch=(10, 30),
            )
            checks.append(tboard_callback)

        epoch_updater = EpochUpdater()
        checks.append(epoch_updater)
        return checks

    def load_test_set(self, **args):
        logging.info("Loading test")
        second_dir = None
        if self.second_data_dir is not None:
            second_dir = self.second_data_dir / "test"

        test_args = dict(args)
        test_args["shuffle"] = False
        self.test, _, _, _, _ = get_dataset(
            self.data_dir / "test",
            self.labels,
            batch_size=self.batch_size,
            image_size=self.input_shape,
            mean_sub=self.mean_sub,
            second_dir=second_dir,
            embeddings=self.model_name == "embeddings",
            deterministic=True,
            **test_args,
        )

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
        elif pretrained_model == "efficientnetv2b0":
            return (
                tf.keras.applications.EfficientNetV2B0(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                    include_preprocessing=False,
                ),
                None,
            )
        elif pretrained_model == "efficientnetv2b3":
            return (
                tf.keras.applications.EfficientNetV2B3(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                    include_preprocessing=False,
                ),
                None,
            )
        elif pretrained_model == "efficientnetv2bs":
            return (
                tf.keras.applications.EfficientNetV2S(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                    include_preprocessing=False,
                ),
                None,
            )
        elif pretrained_model == "efficientnetv2bm":
            return (
                tf.keras.applications.EfficientNetV2M(
                    weights=weights,
                    include_top=False,
                    input_shape=input_shape,
                    include_preprocessing=False,
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
    y_true = tf.cast(y_true, tf.float32)
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
        logging.info("Using binary loss")
        loss_fn = tf.keras.losses.BinaryCrossentropy(
            label_smoothing=smoothing,
        )
        # loss_fn = sigmoid_binary_cross
    else:
        logging.info("Using cross loss")
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
    dataset_data = [(x, y) for x, y in dataset]

    true_categories = [y for x, y in dataset_data]
    true_categories = tf.concat(true_categories, axis=0)
    y_true = []
    for y in true_categories:
        non_zero = tf.where(y).numpy()
        y_true.append(list(non_zero.flatten()))
    y_true = y_true

    true_categories = np.int64(tf.argmax(true_categories, axis=1))
    data = [x for x, y in dataset_data]
    data = tf.concat(data, axis=0)
    y_pred = model.predict(data)

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


def confusion(
    model,
    model_labels,
    dataset,
    filename="confusion.png",
    one_hot=True,
    model_name=None,
    other_models=None,
):
    ebird_labels = model_labels.copy()
    labels = []
    ebird_id_map = get_ebird_ids_to_labels()
    for ebird_id in ebird_labels:
        label = ebird_id_map.get(ebird_id, [ebird_id])
        labels.append(label[0])
    true_categories = [y[0] if isinstance(y, tuple) else y for x, y in dataset]
    true_categories = tf.concat(true_categories, axis=0)
    y_true = []
    if one_hot:
        y_true = np.int64(tf.argmax(true_categories, axis=1))
    else:
        y_true = np.array(true_categories)

    if model_name == "rf-features":
        dataset = tf_to_ydf(dataset)
    y_pred = model.predict(dataset)
    all_preds = None
    weighted_max = False
    weights = [0.6, 0.4]

    if other_models is not None:
        all_preds = [y_pred]

        for other in other_models:
            y_pred_2 = other.predict(dataset)
            # y_pred = np.where(y_pred[0] * weights[1] >= y_pred_2 * weights[1], y_pred, y_pred_2)
            # y_pred = np.maximum(y_pred_2,y_pred)

            all_preds.append(y_pred_2)
        all_preds = np.array(all_preds)
        logging.info("Taking weighted average all models %s", weights)
        if not weighted_max:
            # Calculate the weighted average
            y_pred = np.average(all_preds, axis=0, weights=weights)

            # y_pred =(y_pred + y_pred_2) / 2.0
            # logging.info("All preds is %s", all_preds.shape)

            # y_pred = np.maximum(all_preds)
            logging.info("y_pred preds is %s", y_pred.shape)
        # y_pred = np.int64(tf.argmax(y_pred, axis=1))

    predicted_categories = []
    if "None" not in labels:
        labels.append("None")

    if all_preds is not None and weighted_max:
        logging.info("Running weighted max")
        for row, pred in enumerate(all_preds[0]):
            max_i = np.argmax(pred)
            max_p = pred[max_i]
            max_p_weighted = max_p * weights[0]
            weights_i = 1
            for other_preds in all_preds[1:]:
                pred_2 = other_preds[row]
                max_2 = np.argmax(pred_2)
                max_2_p = pred_2[max_2] * weights[weights_i]
                if max_2_p >= max_p_weighted:
                    max_p_weighted = max_2_p
                    max_p = pred_2[max_2]
                    max_i = max_2
            if max_p > 0.7:
                predicted_categories.append(max_i)
            else:
                predicted_categories.append(len(labels) - 1)
    else:
        for pred in y_pred:
            max_i = np.argmax(pred)
            max_p = pred[max_i]
            if max_p > 0.7:
                predicted_categories.append(max_i)
            else:
                predicted_categories.append(len(labels) - 1)
    cm = confusion_matrix(y_true, predicted_categories, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(filename.with_suffix(".png"), format="png")
    np.save(str(filename.with_suffix(".npy")), cm)

    # plt.savefig(f"./confusions/{filename}", format="png")


def confusion_with_pre(
    model,
    labels,
    dataset,
    filename="confusion.png",
    one_hot=True,
    model_name=None,
    other_models=None,
    pre_model=None,
    pre_labels=None,
):
    labels = labels.copy()
    true_categories = [y[0] if isinstance(y, tuple) else y for x, y in dataset]
    true_categories = tf.concat(true_categories, axis=0)
    y_true = []
    if one_hot:
        y_true = np.int64(tf.argmax(true_categories, axis=1))
    else:
        y_true = np.array(true_categories)

    if model_name == "rf-features":
        dataset = tf_to_ydf(dataset)
    if pre_model:
        pre_model_pred = pre_model.predict(dataset)

        for l in pre_labels:
            if l not in labels:
                labels.append(l)
    y_pred = model.predict(dataset)
    all_preds = None
    weighted_max = False
    weights = [0.6, 0.4]

    if other_models is not None:
        all_preds = [y_pred]

        for other in other_models:
            y_pred_2 = other.predict(dataset)
            all_preds.append(y_pred_2)
        all_preds = np.array(all_preds)
        if not weighted_max:
            logging.info("Taking average all models")

            # Calculate the weighted average
            y_pred = np.mean(all_preds, axis=0)
            logging.info("y_pred preds is %s", y_pred.shape)

    predicted_categories = []

    if "None" not in labels:
        labels.append("None")

    morepork_i = labels.index("morepo2")
    if all_preds is not None and weighted_max:
        logging.info("Running weighted max")
        for row, pred in enumerate(all_preds[0]):
            max_i = np.argmax(pred)
            max_p = pred[max_i]
            max_p_weighted = max_p * weights[0]
            weights_i = 1
            for other_preds in all_preds[1:]:
                pred_2 = other_preds[row]
                max_2 = np.argmax(pred_2)
                max_2_p = pred_2[max_2] * weights[weights_i]
                if max_2_p >= max_p_weighted:
                    max_p_weighted = max_2_p
                    max_p = pred_2[max_2]
                    max_i = max_2
            if max_p > 0.7:
                predicted_categories.append(max_i)
            else:
                predicted_categories.append(len(labels) - 1)
    else:
        for i, pred in enumerate(y_pred):
            max_i = np.argmax(pred)
            max_p = pred[max_i]
            lbl = labels[max_i]
            if pre_model is not None and max_p <= 0.7:
                pre_pred = pre_model_pred[i]
                pre_max_i = np.argmax(pre_pred)
                pre_max_p = pre_pred[pre_max_i]

                pre_lbl = pre_labels[pre_max_i]
                pre_max_i = labels.index(pre_lbl)
                logging.info(
                    "Pre labl is %s- %s max is %s - %s",
                    pre_lbl,
                    pre_max_p,
                    labels[max_i],
                    max_p,
                )
                filter_moreporks = False
                if pre_max_p > 0.7:
                    max_i = pre_max_i
                    max_p = pre_max_p
                    # only use as fallback
                    continue
                    if pre_lbl == "noise":
                        if lbl not in ["insect", "tree-weta", "weta"] or max_p <= 0.7:
                            max_i = pre_max_i
                            max_p = pre_max_p
                    elif pre_lbl in ["morepo2", "human"]:
                        max_i = pre_max_i
                        max_p = pre_max_p
                    else:
                        filter_moreporks = True
                        if max_i == morepork_i or max_p <= 0.7:
                            max_i = pre_max_i
                            max_p = pre_max_p
                # elif pre_max_p > 0.5 and pre_lbl in ["noise", "human"]:
                #     if max_i == morepork_i:
                #         predicted_categories.append(len(labels) - 1)
                #         logging.info("Unknown")
                #         continue
            if max_p > 0.7:
                # logging.info(
                #     "Final result is %s true %s", labels[max_i], labels[y_true[i]]
                # )
                predicted_categories.append(max_i)
            else:
                # logging.info("Final unknown")
                predicted_categories.append(len(labels) - 1)
    cm = confusion_matrix(y_true, predicted_categories, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(filename.with_suffix(".png"), format="png")
    np.save(str(filename.with_suffix(".npy")), cm)

    # plt.savefig(f"./confusions/{filename}", format="png")


def multi_confusion_single(
    model,
    labels,
    dataset,
    filename="confusion.png",
    one_hot=True,
    prob_thresh=0.7,
    model_name=None,
):
    filename = Path(filename)
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(classes=np.arange(len(labels)))
    true_categories = [y[0] if isinstance(y, tuple) else y for x, y in dataset]
    true_categories = tf.concat(true_categories, axis=0)
    # y_true = []
    # if one_hot:
    #     for y in true_categories:
    #         non_zero = tf.where(y).numpy()
    #         y_true.append(list(non_zero.flatten()))
    # only usefull if not multi
    # true_categories = np.int64(tf.argmax(true_categories, axis=1))
    # else:
    # y_true = np.array(true_categories)
    if model_name == "rf-features":
        dataset = tf_to_ydf(dataset)

    y_pred = model.predict(dataset)
    if "nothing" not in labels:
        labels.append("nothing")
    none_p = []
    none_y = []
    flat_p = []
    flat_y = []
    bird_index = None
    if "bird" in labels:
        bird_index = labels.index("bird")
    for y, p in zip(true_categories, y_pred):
        # get predicted label that isn't bird and use this as overall tag
        # may not work when we have 2 tags for a recording but i dont think this is the case at the moment
        index = 0
        arg_sorted = np.argsort(p)
        best_label = arg_sorted[-1]
        if (
            bird_index is not None
            and best_label == bird_index
            and p[arg_sorted[-2]] != 0
        ):
            best_label = arg_sorted[-2]
        best_prob = p[best_label]

        best_labels = np.argwhere(p > prob_thresh).ravel()

        # get true label that isn't bird if available and use this as overall tag
        # arg_sorted = np.argsort(y)

        true_labels = np.argwhere(y == 1).ravel()
        # true_label = arg_sorted[-1]
        # if true_label == bird_index and y[arg_sorted[-2]] != 0:
        #     true_label = arg_sorted[-2]
        # print("Y true is", y)
        # print("Actual y is ", true_labels)
        # print("P is ", np.round(p * 100))
        # print("Actual p is ", best_labels)
        for y_l, p_l in zip(y, p):
            predicted = p_l >= prob_thresh
            if y_l == 0 and predicted:
                # if predicted but is not true

                # add this index as wrong for all true labels. where it hasn't got it right

                for true_label in true_labels:
                    # if true_label not in best_labels:
                    flat_y.append(true_label)
                    flat_p.append(index)
            elif y_l == 1 and predicted:
                flat_y.append(index)
                flat_p.append(index)
            elif y_l == 1 and not predicted:
                flat_y.append(index)
                flat_p.append(len(labels) - 1)  # None

                # if we have no prediction over a threshold put the best one over a reasonable threshold (0.3) into the None matrix
                if best_prob >= 0.3 and len(best_labels) == 0:
                    if best_label not in true_labels:
                        none_y.append(index)
                        none_p.append(best_label)

                # # is this index but we predicted something else add to missing conf
                # # if index != bird_index and best_prob > 0.5:
                # for best_label in best_labels:
                #     if best_label not in true_labels:
                #         none_y.append(index)
                #         none_p.append(best_label)

            index += 1

    flat_p = np.int64(flat_p)
    flat_y = np.int64(flat_y)

    none_p = np.int64(none_p)
    none_y = np.int64(none_y)

    cm = confusion_matrix(flat_y, flat_p, labels=np.arange(len(labels)))
    # confusion_path = Path(f"./confusions/{filename}")

    np.save(str(filename.with_suffix(".npy")), cm)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=labels)
    logging.info("Saving confusion to %s", filename.with_suffix(".png"))
    plt.savefig(filename.with_suffix(".png"), format="png")

    none_p = np.int64(none_p)
    none_y = np.int64(none_y)
    cm = confusion_matrix(none_y, none_p, labels=np.arange(len(labels)))
    none_path = filename.parent / f"{filename.stem}-none"
    logging.info("Saving confusion to %s", none_path.with_suffix(".png"))

    np.save(str(none_path.with_suffix(".npy")), cm)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(none_path.with_suffix(".png"), format="png")


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

    figure = plt.figure(figsize=(24, 24))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    counts = cm.copy()
    threshold = counts.max() / 2.0

    print("Threshold is", threshold, " for ", cm.max())
    # Normalize the confusion matrix.

    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cm = np.nan_to_num(cm)
    cm = np.uint8(np.round(cm * 100))

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if counts[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def ktest():

    labels = ["bird", "whistler", "bittern"]
    metric = precAtK(num_labels=3, bird_i=0)
    y_true = np.array([[1, 0, 0]])
    y_pred = np.array([[0.1, 0.1, 0]])
    metric.update_state(y_true, y_pred)
    print("Result is ", metric.result())


def load_datasets(labels, excluded_labels, data_dir, batch_size, **args):
    second_data_dir = None
    second_dir = None
    if args.get("second_dataset_dir") is not None:
        second_data_dir = args.get("second_dataset_dir")
        second_dir = second_data_dir / "train"

    human_dir = None
    human_dataset_dir = None
    if args.get("human_dataset_dir") is not None:
        human_dataset_dir = args.get("human_dataset_dir")
        human_dir = human_dataset_dir / "train"

    logging.info(
        "Loading datasets with %s and secondary dir %s human dir %s generic bird %s",
        data_dir,
        second_data_dir,
        human_dataset_dir,
        args.get("use_generic_bird"),
    )

    training_files_dir = data_dir / "train"
    model_name = args.get("model_name")

    global global_epoch
    train, remapped, epoch_size, new_labels, extra_label_map = get_dataset(
        training_files_dir,
        labels,
        global_epoch=global_epoch,
        batch_size=batch_size,
        augment=True,  # seems to perform worse
        excluded_labels=excluded_labels,
        second_dir=second_dir,
        embeddings=model_name == "embeddings",
        load_seperate_ds=True,
        cache=False,
        human_dir=human_dir,
        **args,
    )
    if args.get("only_load_train", False):
        return remapped, excluded_labels, extra_label_map

    if second_data_dir is not None:
        second_dir = second_data_dir / "validation"

    if human_dataset_dir is not None:
        human_dir = human_dataset_dir / "validation"
    logging.info("Loading Val")

    validation, _, _, _, _ = get_dataset(
        data_dir / "validation",
        labels,
        batch_size=batch_size,
        excluded_labels=excluded_labels,
        second_dir=second_dir,
        embeddings=model_name == "embeddings",
        human_dir=human_dir,
        **args,
    )

    return train, validation, new_labels, remapped, extra_label_map


def init_labels(data_dir, **args):
    labels = set()

    training_files_dir = data_dir / "train"
    second_data_dir = args.get("second_dataset_dir")
    file = f"{data_dir}/training-meta.json"
    with open(file, "r") as f:
        meta = json.load(f)

    labels.update(meta.get("labels", []))
    second_meta = None
    if second_data_dir is not None:
        logging.info("Loading second metadata %s", second_data_dir)
        file = second_data_dir / "training-meta.json"
        with open(file, "r") as f:
            second_meta = json.load(f)
        second_labels = set(second_meta.get("labels", []))
        labels.update(second_labels)

    if args.get("extra_datasets"):
        for extra_ds in args["extra_datasets"]:
            logging.info("Loading second metadata %s", extra_ds)
            file = Path(extra_ds) / "training-meta.json"
            with open(file, "r") as f:
                extra_meta = json.load(f)
            second_labels = set(extra_meta.get("labels", []))
            labels.update(second_labels)
    human_dataset_dir = None
    if args.get("human_dataset_dir") is not None:
        human_dataset_dir = args.get("human_dataset_dir")
        logging.info("Loading second metadata %s", human_dataset_dir)
        file = human_dataset_dir / "training-meta.json"
        with open(file, "r") as f:
            extra_meta = json.load(f)
        second_labels = set(extra_meta.get("labels", []))
        labels.update(second_labels)

    set_specific_by_count(meta)
    if second_meta is not None:
        set_specific_by_count(second_meta)

    labels = list(labels)

    ebird_map = get_label_to_ebird_map()
    ebird_labels = []
    for l in labels:
        ebird_labels.append(get_ebird_id(l, ebird_map))
    labels = list(set(ebird_labels))
    labels.sort()

    if args.get("use_generic_bird", False) and "bird" not in labels:
        labels.append("bird")

    # if "other" not in labels:
    # labels.append("other")
    labels.sort()
    logging.info("Loading train")

    excluded_labels = get_excluded_labels(labels)
    if not args.get("use_generic_bird", False):
        excluded_labels.append("bird")
    if args.get("only_features", False):
        from tfdataset import ANIMAL_LABELS, set_merge_labels

        if "animal" not in labels:
            labels.append("animal")
        labels.sort()

        merge_labels = {}
        excluded_labels = []
        for l in labels:
            if l == "bird":
                continue
            if l in ALL_BIRDS:
                print("Setting", l, " to bird")
                merge_labels[l] = "bird"
            elif l in ANIMAL_LABELS:
                merge_labels[l] = "animal"
            elif l == "insect":
                merge_labels[l] = "noise"
                # merge_labels[l] = "insect"
            elif l in NOISE_LABELS:
                merge_labels[l] = "noise"
        excluded_labels = ["false-positive"]

        set_merge_labels(merge_labels)
    elif args.get("morepork_model", False):
        from tfdataset import (
            ANIMAL_LABELS,
            set_merge_labels,
            HUMAN_LABELS,
            INSECT_LABELS,
        )

        # if "animal" not in labels:
        #     labels.append("animal")
        labels.sort()

        merge_labels = {}
        excluded_labels = []
        for l in labels:
            if l == "morepo2":
                continue
            elif l == "bird":
                continue
            if l in ALL_BIRDS:
                print("Setting", l, " to bird")
                merge_labels[l] = "bird"
            elif l in ANIMAL_LABELS:
                merge_labels[l] = "noise"
            elif l == "insect":
                merge_labels[l] = "noise"
                # merge_labels[l] = "insect"
            elif l in NOISE_LABELS:
                merge_labels[l] = "noise"
            elif l in HUMAN_LABELS:
                merge_labels[l] = "human"
            elif l in INSECT_LABELS:
                merge_labels[l] = "noise"
        excluded_labels = ["false-positive"]

        set_merge_labels(merge_labels)
    else:
        if "human" not in excluded_labels:
            excluded_labels.append("human")
        if "noise" not in excluded_labels:
            excluded_labels.append("noise")

    logging.info("labels are %s Excluding %s", labels, excluded_labels)
    return labels, excluded_labels, meta


from functools import partial

from multiprocessing import Pool


def evaluate_dir(model, model_meta, dir_name, filename, rec_ids):
    filename = Path("./confusions") / filename
    meta_data_f = dir_name.glob("**/*.txt")
    meta_data_f = list(meta_data_f)
    predicted_categories = []
    y_true = []
    labels = model_meta["labels"]

    include_labels = set(labels)

    pre_labels = ["bird", "human", "noise"]

    for pre_l in pre_labels:
        if pre_l not in labels:
            labels.append(pre_l)
    labels.append("None")

    remapped = model_meta["remapped_labels"]
    for k, v in remapped.items():
        if v >= 0:
            include_labels.add(k)

    include_labels.add("noise")
    include_labels.add("human")

    for l in NOISE_LABELS:
        include_labels.add(l)
        remapped[l] = labels.index("noise")
    for l in HUMAN_LABELS:
        include_labels.add(l)

        remapped[l] = labels.index("human")
    remapped["human"] = labels.index("human")

    for l in ALL_BIRDS:
        if l in labels:
            continue
        include_labels.add(l)

        remapped[l] = labels.index("bird")
    remapped["bird"] = labels.index("bird")

    include_labels = list(include_labels)
    include_labels.sort()
    print("Include labels is ", include_labels)
    if rec_ids is not None:
        filtered_meta = []
        for f in meta_data_f:
            file_name = f.stem
            try:
                first_hyphen = file_name.index("-")
            except:
                continue
            rec_id = file_name[:first_hyphen]
            rec_id = int(rec_id)
            if rec_id in rec_ids:
                filtered_meta.append(f)
        print("Runing on ", len(filtered_meta), " supplied recs ", len(rec_ids))

    else:
        filtered_meta = meta_data_f
    # meta_data_f = meta_data_f[:1]
    pre_fn = partial(preprocess_audio, labels=include_labels)
    total_count = len(filtered_meta)
    count = 0

    predicted_mean = []
    predicted_max = []
    predicted_counts = []
    confidences = []
    track_ids = []

    with Pool(processes=8) as pool:
        for result in pool.imap_unordered(pre_fn, filtered_meta, chunksize=8):
            if count % 100 == 0:
                logging.info("Done %s / %s", count, total_count)
            count += 1
            if result is None:
                continue
            file_name, tracks, all_samples = result
            try:
                if len(all_samples) == 0:
                    logging.info("No samples for %s", file_name)
                    continue

                filtered_tracks = []
                filtered_samples = []
                counts = []
                for track, samples in zip(tracks, all_samples):
                    if len(samples) == 0:
                        logging.info("No samples for track %s", track)
                        continue
                    filtered_tracks.append(track)
                    filtered_samples.extend(samples)
                    counts.append(len(samples))
                if len(filtered_samples) == 0:
                    logging.info("No samples for %s", file_name)
                    continue

                # all_samples = np.array(all_samples)
                # all_samples = np.concat(all_samples, axis=0)
                all_samples = np.array(filtered_samples)
                all_samples = np.repeat(all_samples, 3, -1)
                predictions = model.predict(all_samples)
                offset = 0
                for track, count in zip(tracks, counts):
                    track_preds = predictions[offset : offset + count]

                    track_pred = np.max(track_preds, axis=0)
                    max_i = np.argmax(track_pred)
                    max_p = track_pred[max_i]
                    if max_p > 0.7:
                        predicted_max.append(max_i)
                    else:
                        predicted_max.append(len(labels) - 1)

                    track_pred = np.mean(track_preds, axis=0)
                    confidences.append(track_pred)
                    track_ids.append(track.id)
                    max_i = np.argmax(track_pred)
                    max_p = track_pred[max_i]
                    if max_p > 0.7:
                        predicted_mean.append(max_i)
                    else:
                        predicted_mean.append(len(labels) - 1)

                    arg_max = np.argmax(track_preds, axis=1)
                    rows = np.arange(len(track_preds))
                    prob_max = track_preds[rows, arg_max]
                    over_thresh = prob_max > 0.7
                    args_over_thresh = arg_max[over_thresh]
                    if len(args_over_thresh) == 0:
                        predicted_counts.append(len(labels) - 1)
                    else:
                        counts = np.bincount(args_over_thresh)
                        max_i = np.argmax(counts)
                        max_c = counts[max_i]
                        max_counts = np.where(counts == max_c)
                        if len(max_counts) > 1:
                            # could use mean will just be unknown???
                            predicted_counts.append(predicted_mean[-1])
                        elif len(max_counts) == 1:
                            predicted_counts.append(max_i)

                    if track.tag in remapped:
                        lbl_i = remapped[track.tag]
                    else:
                        lbl_i = labels.index(track.tag)
                    y_true.append(lbl_i)
                    offset += count
            except:
                logging.error("Could not process %s", file_name, exc_info=True)

    predicted_mean = np.array(predicted_mean)
    predicted_max = np.array(predicted_max)
    predicted_counts = np.array(predicted_counts)
    confidences = np.array(confidences)
    track_ids = np.array(track_ids)

    npy_file = filename.parent / f"{filename.stem}-raw.npy"
    with npy_file.open("wb") as f:
        np.save(f, track_ids)
        np.save(f, y_true)
        np.save(f, predicted_mean)
        np.save(f, confidences)
        np.save(f, labels)

    print("Saving to ", filename)
    cm_file = filename.parent / f"{filename.stem}-mean"
    cm = confusion_matrix(y_true, predicted_mean, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(cm_file.with_suffix(".png"), format="png")
    np.save(str(cm_file.with_suffix(".npy")), cm)

    print("Saving to ", filename)
    cm_file = filename.parent / f"{filename.stem}-max"
    cm = confusion_matrix(y_true, predicted_max, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(cm_file.with_suffix(".png"), format="png")
    np.save(str(cm_file.with_suffix(".npy")), cm)

    print("Saving to ", filename)
    cm_file = filename.parent / f"{filename.stem}-counts"
    cm = confusion_matrix(y_true, predicted_counts, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(cm_file.with_suffix(".png"), format="png")
    np.save(str(cm_file.with_suffix(".npy")), cm)


from identifytracks import get_tracks_from_signals, signal_noise, get_end, Signal
from audiowriter import load_recording
from predict_utils import load_samples
from audiodataset import Recording


def preprocess_audio(metadata_f, labels=None):
    try:
        audio_f = metadata_f.with_suffix(".m4a")
        if not audio_f.exists():
            audio_f = metadata_f.with_suffix(".wav")
        if not audio_f.exists():
            audio_f = metadata_f.with_suffix(".mp3")
        if not audio_f.exists():
            audio_f = metadata_f.with_suffix(".flac")
        if not audio_f.exists():
            logging.info("Could not find audio file for %s", metadata_f)
            return None

        try:
            with metadata_f.open("r") as f:
                metadata = json.load(f)
        except:
            logging.info("Could not load metadata for %s", metadata_f, exc_info=True)
            return None
        rec = Recording(metadata, audio_f, None, False, True)
        frames, sr = load_recording(audio_f)
        end = get_end(frames, sr)
        frames = frames[: int(sr * end)]
        tracks = [track for track in rec.tracks if track.tag in labels]
        if len(tracks) == 0:
            return None
        samples = load_samples(frames, sr, tracks)
        for s, t in zip(samples, tracks):
            if len(s) == 0:
                logging.error(
                    "No samples for some track %s from %s",
                    t.id,
                    metadata_f,
                    exc_info=True,
                )
                return None
    except:
        logging.error("Could not load audio for %s", metadata_f, exc_info=True)
        return None
    return metadata_f, tracks, samples


def main():
    init_logging()
    args = parse_args()

    # ktest()
    # return

    if args.confusion is not None:
        model_path = Path(args.name)
        if model_path.is_dir():
            meta_file = model_path / "metadata.txt"
        else:
            meta_file = model_path.parent / "metadata.txt"
        print("Meta", meta_file)
        with open(str(meta_file), "r") as f:
            meta_data = json.load(f)
        model_name = meta_data.get("name")
        multi = meta_data.get("multi_label", True)
        labels = meta_data.get("labels")
        other_models = None
        if model_name == "rf-features":
            model = ydf.load_model(str(model_path.parent))
        else:
            if model_path.is_dir():
                model_path = model_path / f"{model_path.stem}.keras"
            logging.info("Loading %s with weights %s", model_path, "val_acc")
            model = tf.keras.models.load_model(
                str(model_path),
                compile=False,
            )

            model.summary()
            excluded_labels = meta_data.get("excluded_labels")
            remapped_labels = meta_data.get("remapped_labels")
            if args.model_2 is not None and len(args.model_2) > 0:
                other_models = []
                for other_model in args.model_2:
                    model_2_path = Path(other_model)
                    if model_2_path.is_dir():
                        model_2_path = model_2_path / f"{model_2_path.stem}.keras"
                    logging.info(
                        "Loading other model model %s with weights %s",
                        model_2_path,
                        "val_acc",
                    )
                    model_2 = tf.keras.models.load_model(
                        str(model_2_path),
                        compile=False,
                    )
                    other_models.append(model_2)
            pre_model = None
            pre_labels = None
            if args.pre_model is not None:
                pre_model_path = Path(args.pre_model)
                if pre_model_path.is_dir():
                    pre_model_path = pre_model_path / f"{pre_model_path.stem}.keras"

                pre_model = tf.keras.models.load_model(
                    str(pre_model_path),
                    compile=False,
                )
                pre_meta_f = pre_model_path.parent / "metadata.txt"

                with pre_meta_f.open("r") as f:
                    pre_meta = json.load(f)
                pre_labels = pre_meta["labels"]
                pre_remapped = pre_meta.get("remapped_labels")

                for l in pre_labels:
                    if l in excluded_labels:
                        excluded_labels.remove(l)
                    if l not in labels:
                        labels.append(l)
                for k, v in pre_remapped.items():
                    if v != -1:
                        if k in excluded_labels:
                            excluded_labels.remove(k)
                        if k in remapped_labels and remapped_labels[k] != -1:
                            continue
                        pre_lbl = pre_labels[v]
                        remapped_labels[k] = labels.index(pre_lbl)
        if args.evaluate_dir is not None:
            rec_ids = None
            if args.dataset_dir is not None:
                base_dir = Path(args.dataset_dir)
                meta_f = base_dir.parent / "training-meta.json"
                with meta_f.open("r") as f:
                    data_json = json.load(f)

                ds_name = base_dir.stem
                rec_ids = data_json["recs"][ds_name]

            if args.weights is not None:
                logging.info("Loading weights %s", args.weights)
                model.load_weights(args.weights)
            evaluate_dir(
                model, meta_data, Path(args.evaluate_dir), args.confusion, rec_ids
            )
            return

        base_dir = Path(args.dataset_dir)
        second_dir = None
        if args.second_dataset_dir is not None:
            second_dir = args.second_dataset_dir / "test"

        dataset, _, _, _, _ = get_dataset(
            base_dir / "test",
            labels,
            excluded_labels=excluded_labels,
            remapped_labels=remapped_labels,
            second_dir=second_dir,
            image_size=DIMENSIONS,
            shuffle=False,
            deterministic=True,
            batch_size=64,
            mean_sub=meta_data.get("mean_sub", False),
            extra_label_map=meta_data.get("extra_label_map"),
            stop_on_empty=False,
            one_hot=args.one_hot,
            use_generic_bird=meta_data.get("use_generic_bird", False),
            filter_freq=meta_data.get("filter_freq", False),
            only_features=meta_data.get("only_features", False),
            features=meta_data.get("features", False),
            multi_label=meta_data.get("multi_label", True),
            loss_fn=meta_data.get("loss_fn"),
            load_raw=False,
            # meta_data.get("load_raw"),
            fmin=meta_data.get("fmin"),
            fmax=meta_data.get("fmax"),
            break_freq=meta_data.get("break_freq"),
            n_fft=meta_data.get("n_fft"),
            model_name=model_name,
            cache=True,
        )
        # acc = tf.metrics.binary_accuracy
        if model_name != "rf-features":
            acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
            model.compile(
                optimizer=optimizer(lr=1.0),
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
            if meta_data.get("only_features"):
                weight_files = [None]
            else:
                weight_files = [
                    "final",
                    "val_loss.weights.h5",
                    # "val_precK.weights.h5",
                    (
                        "val_binary_accuracy.weights.h5"
                        if multi
                        else "val_categorical_accuracy.weights.h5"
                    ),
                ]
            weight_base_path = model_path.parent
            logging.info("Using %s weights", weight_base_path / weight_files[-1])
            # model.load_weights(weight_base_path / weight_files[-1])
            # best_threshold(model, labels, dataset, args.confusion)
            # return

            args.confusion = Path(args.confusion)

            for w in weight_files:
                file_prefix = "final"
                if w != "final":
                    logging.info("Using %s weights", weight_base_path / w)
                    model.load_weights(weight_base_path / w)
                    index = w.index(".weights")
                    file_prefix = w[:index]
                    if other_models is not None:
                        for model_dir, other_model in zip(args.model_2, other_models):
                            model_dir = Path(model_dir)
                            other_model.load_weights(model_dir / w)
                    if pre_model is not None:
                        pre_path = Path(args.pre_model)
                        pre_model.load_weights(pre_path / w)

                confusion_file = (
                    Path("./confusions")
                    / model_path.stem
                    / (args.confusion.parent / f"{args.confusion.stem}-{file_prefix}")
                )
                confusion_file.parent.mkdir(exist_ok=True, parents=True)
                if multi:
                    multi_confusion_single(
                        model,
                        labels,
                        dataset,
                        confusion_file,
                        one_hot=not meta_data.get("only_features"),
                        model_name=model_name,
                    )
                else:
                    confusion(
                        model,
                        labels,
                        dataset,
                        confusion_file,
                        one_hot=not meta_data.get("only_features"),
                        other_models=other_models,
                    )

                    # confusion_with_pre(
                    #     model,
                    #     labels,
                    #     dataset,
                    #     confusion_file,
                    #     one_hot=not meta_data.get("only_features"),
                    #     model_name=model_name,
                    #     other_models=other_models,
                    #     pre_model=pre_model,
                    #     pre_labels=pre_labels,
                    # )

    else:
        am = AudioModel(args.model_name, args.dataset_dir, args.second_dataset_dir)
        am.lme = args.lme

        if args.cross:
            am.cross_fold_train(run_name=args.name, use_generic_bird=args.use_bird)
        else:
            # args.multi = args.multi == 1
            am.train_model(
                run_name=args.name,
                **args.__dict__,
            )


def none_or_str(value):
    if value in ["null", "none", "None"]:
        return None
    return value


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Epochs to use")
    parser.add_argument(
        "--extra-datasets",
        default=[
            "/data/audio-data/ambientsounds/training-data/",
            "/data/audio-data/ESC-50-master/training-data/",
        ],
        help="Dataset directory to use",
    )

    parser.add_argument(
        "-d",
        "--dataset-dir",
        type=str,
        default=None,
        # "/data/audio-data/training-data",
        help="Dataset directory to use",
    )
    parser.add_argument(
        "--second-dataset-dir",
        type=none_or_str,
        default=None,
        help="Secondary dataset directory to use",
    )
    parser.add_argument(
        "--human-dataset-dir",
        type=none_or_str,
        default=None,
        help="Secondary dataset directory to use",
    )
    parser.add_argument("--confusion", help="Save confusion matrix for model")
    parser.add_argument("--rf_model", help="RF Model to use")
    parser.add_argument("--cnn_model", help="CNN Model to use with val_acc weights")

    parser.add_argument("-w", "--weights", help="Weights to use")
    parser.add_argument("--cross", action="count", help="Cross fold val")
    parser.add_argument(
        "--no-low-samples", action="count", help="Don't use over sampled samples"
    )

    parser.add_argument(
        "--lme", action="count", help="Use log mean expo instead of global avg"
    )
    parser.add_argument(
        "--filter-freq", default=False, action="count", help="Filter Freq"
    )
    parser.add_argument(
        "--filter-signal",
        default=False,
        action="count",
        help="Filter by signal percent > 0.0",
    )
    parser.add_argument(
        "--random-butter",
        default=0,
        type=float,
        help="Random butter a percentage between 0-1 of using butter",
    )
    parser.add_argument(
        "--use-weighting",
        default=False,
        action="count",
        help="Use weighting for classes",
    )
    parser.add_argument(
        "--only-features", default=False, action="count", help="Train on features"
    )
    parser.add_argument(
        "--features", default=False, action="count", help="Train on features"
    )
    parser.add_argument(
        "--use_bird_tags",
        default=False,
        action="count",
        help="Use tracks of generic bird tags ( without specific birds) in training",
    )

    parser.add_argument(
        "--one-hot", type=str2bool, default=True, help="One hot labeling "
    )
    parser.add_argument("--resample", default=False, action="count", help="Resample DS")
    parser.add_argument(
        "--morepork-model", default=False, action="store_true", help="Morepork model"
    )

    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle DS")
    parser.add_argument(
        "--multi-label", type=str2bool, default=False, help="Multi label"
    )
    parser.add_argument(
        "--use-generic-bird",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use bird as well as specific label",
    )

    parser.add_argument(
        "--model-name",
        default="efficientnetv2b3",
        help="Model to use badwinner, badwinner2, inc3",
    )

    parser.add_argument(
        "--load-raw",
        default=False,
        action="count",
        help="Load raw audio data rather than preprocessed spect",
    )

    parser.add_argument(
        "--loss_fn",
        default="keras",
        help="loss function to use either keras (Binary or Categorical) depending on multi label or custom loss WeightedCrossEntropy",
    )

    parser.add_argument("-c", "--config-file", help="Path to config file to use")
    parser.add_argument(
        "--model_2", nargs="+", help="Second model for mean model confusions"
    )
    parser.add_argument("--pre_model", help="Pre model for confusions")
    parser.add_argument("name", help="Run name")

    parser.add_argument(
        "--fmin",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--fmax",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--n_mels",
        default=160,
        type=int,
    )
    parser.add_argument(
        "--n_fft",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--break-freq",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--evaluate-dir",
        default=None,
    )

    args = parser.parse_args()
    args.extra_datasets = None
    # args.multi = args.multi > 0
    args.resample = args.resample > 0
    args.only_features = args.only_features > 0
    args.features = args.features > 0
    args.filter_freq = args.filter_freq > 0
    if args.dataset_dir is not None:
        args.dataset_dir = Path(args.dataset_dir)
    if args.second_dataset_dir is not None:
        args.second_dataset_dir = Path(args.second_dataset_dir)

    if args.human_dataset_dir is not None:
        args.human_dataset_dir = Path(args.human_dataset_dir)
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
        if isinstance(obj, Path):
            return str(obj)
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
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

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
    thresholds_best = []
    for i, class_of_interest in enumerate(labels):
        # class_of_interest = "virginica"
        class_id = np.flatnonzero(label_binarizer.classes_ == i)[0]
        print("plt show for", class_of_interest)

        precision, recall, thresholds = precision_recall_curve(
            y_onehot_test[:, class_id], y_pred[:, class_id]
        )
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)

        # fpr, tpr, thresholds = roc_curve(
        #     y_onehot_test[:, class_id], y_pred[:, class_id]
        # )
        # RocCurveDisplay.from_predictions(
        #     y_onehot_test[:, class_id],
        #     y_pred[:, class_id],
        #     name=f"{class_of_interest} vs the rest",
        #     color="darkorange",
        # )
        testy = y_onehot_test[:, class_id]
        no_skill = len(testy[testy == 1]) / len(testy)

        plt.plot(recall, precision, marker=".", label="Logistic")
        plt.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
        plt.axis("square")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Recall vs Precivsion - {labels[i]}")
        plt.legend()
        plt.scatter(recall[ix], precision[ix], marker="o", color="black", label="Best")

        plt.savefig(f"{labels[i]}-{filename}.png", format="png")
        plt.clf()
        print("Best Threshold=%f, F-Score=%.3f" % (thresholds[ix], fscore[ix]))
        thresholds_best.append(thresholds[ix])
        # for area, thresh in zip(areas, thresholds):
        #     # print("f", f, t)
        #     print("Thresholds are", thresh, area)
    thresholds = np.array(thresholds_best)
    logging.info(
        "ALl thresholds are %s mean %s median %s",
        thresholds,
        np.mean(thresholds),
        np.median(thresholds),
    )


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
                if tf_var.name == "a-power":
                    tf.summary.histogram(tf_var.name, tf_var.numpy(), step=epoch)
                # tf.summary.histogram(tf_var.name, tf_var.numpy(), step=epoch)

    return log_hist


def get_linear_model(embedding_dim, num_classes):
    """Create a simple linear Keras model."""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=embedding_dim),
            tf.keras.layers.Dense(num_classes, activation="sigmoid"),
        ]
    )
    return model


class WeightedCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, labels, name="bird_cross_entropy", **kwargs):
        super().__init__(name=name, **kwargs)
        bird_mask = np.zeros((len(labels)))
        if "bird" in labels:
            bird_mask[labels.index("bird")] = 1
        self.bird_mask = tf.constant(bird_mask, dtype=tf.float32)

        self.bird_loss = self.bird_mask
        self.normal_loss = tf.constant(tf.ones(len(labels), dtype=tf.float32))
        # print(
        #     "Bird weight cross entropy  labels",
        #     labels,
        #     " bird mask",
        #     self.bird_mask,
        #     " bird loss",
        #     self.bird_loss,
        #     " normal ",
        #     self.normal_loss,
        # )

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()
        )
        possible_labels = y_true[1]
        y_true = y_true[0]
        # birds = tf.math.reduce_all(tf.math.equal(y_true, self.bird_mask), axis=1)
        # birds = tf.expand_dims(birds, 1)

        # loss_matrix = tf.where(birds, possible_labels, self.normal_loss)

        term_0 = (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
        term_0 = term_0 * possible_labels
        # since a lot of our bird tags may have specific birds lets not penalize the model for
        # choosing a specific bird in this scenario, only calculate loss from not being bird

        term_1 = y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())
        loss = tf.keras.backend.mean(term_0 + term_1, axis=1)
        return [-loss, -loss]

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}


class precAtK(tf.keras.metrics.Metric):
    def __init__(
        self, name="precK", k=3, num_labels=None, bird_i=None, weighting=None, **kwargs
    ):
        super(precAtK, self).__init__(name=name, **kwargs)
        self.bird_mask = None
        if bird_i is not None:
            bird_mask = np.ones((num_labels))
            bird_mask[bird_i] = 0
            self.bird_mask = tf.constant(bird_mask, dtype=tf.bool)
        self.weighting = None
        if weighting is not None:
            self.weighting = tf.constant(weighting)
        self.num_labels = num_labels
        self.k = k
        self.k_percent = self.add_weight(
            name="k_percent", initializer="zeros", dtype=tf.float32
        )
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.bird_mask is not None:

            p_masked = tf.cast(
                tf.logical_and(tf.cast(y_pred, tf.bool), self.bird_mask),
                tf.int32,
            )
            y_masked = tf.cast(
                tf.logical_and(tf.cast(y_true, tf.bool), self.bird_mask),
                tf.int32,
            )

        else:
            p_masked = y_pred
            y_masked = y_true
        top_pred = tf.math.top_k(p_masked, k=3)
        top_true = tf.math.top_k(y_masked, k=3)
        mask = tf.math.greater(top_true.values, 0)
        # dont want to count 0 values in top_k
        top_true_indices = tf.ragged.boolean_mask(top_true.indices, mask)
        non_zero = tf.size(top_true_indices)
        top_true_indices = top_true_indices.to_tensor(default_value=-1)

        mask = tf.math.greater(top_pred.values, 0)
        # dont want to count 0 values in top_k
        top_pred_indices = tf.ragged.boolean_mask(top_pred.indices, mask)
        non_zero = tf.size(top_true_indices)
        top_pred_indices = top_pred_indices.to_tensor(default_value=-1)

        intersection = tf.sets.intersection(top_pred_indices, top_true_indices).values
        k_percent = tf.size(intersection)
        if self.weighting is not None:
            one_hot_intersection = tf.one_hot(intersection, self.num_labels)
            print(self.weighting.dtype, one_hot_intersection.dtype)

            k_percent = tf.math.reduce_sum(one_hot_intersection * self.weighting)
        self.total.assign_add(non_zero)
        self.k_percent.assign_add(tf.cast(k_percent, tf.float32))

    def reset_state(self):
        self.k_percent.assign(0)
        self.total.assign(0)

    def result(self):
        return self.k_percent / tf.math.maximum(tf.cast(self.total, tf.float32), 1)


def keras_model_memory_usage_in_bytes(model, batch_size):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory


def feature_cnn(short_features, mid_features, num_labels):
    short_features = tf.keras.layers.Dense(128, activation="relu")(short_features)
    short_features = tf.keras.layers.Dense(128, activation="relu")(short_features)
    short_features = tf.keras.layers.Dropout(0.1)(short_features)
    short_features = tf.keras.layers.GlobalAveragePooling1D()(short_features)

    short_features = tf.keras.layers.Dense(num_labels, activation="sigmoid")(
        short_features
    )

    mid_features = tf.keras.layers.Dense(128, activation="relu")(mid_features)
    mid_features = tf.keras.layers.Dense(128, activation="relu")(mid_features)
    mid_features = tf.keras.layers.Dropout(0.1)(mid_features)
    mid_features = tf.keras.layers.GlobalAveragePooling1D()(mid_features)

    mid_features = tf.keras.layers.Dense(num_labels, activation="sigmoid")(mid_features)

    return short_features, mid_features


def tf_to_ydf(dataset):
    ydf_ds = {"f1": [], "f2": [], "y": []}
    # not sure if you loose anything by flattening, i.e. time info
    for x, y in dataset.unbatch():
        short_f = x[0].numpy().ravel()
        long_f = x[1].numpy().ravel()
        ydf_ds["f1"].append(short_f)
        ydf_ds["f2"].append(long_f)
        ydf_ds["y"].append(tf.argmax(y).numpy())

    ydf_ds["f1"] = np.float32(ydf_ds["f1"])
    ydf_ds["f2"] = np.float32(ydf_ds["f2"])
    ydf_ds["y"] = np.int16(ydf_ds["y"])
    return ydf_ds


class EpochUpdater(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None, steps=50):
        global global_epoch
        # global_epoch = tf.Variable(1, name='global_epoch', trainable=False, dtype=tf.int32)
        global_epoch.assign(epoch + 1)


if __name__ == "__main__":
    main()
