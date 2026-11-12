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
    EVALUATED_LABELS,
)
from evaluate import (
    preprocess_audio,
    plot_confusion_matrix,
    evaluate_weakly_labelled_dir,
)

# from tfdatasetembeddings import get_dataset, DIMENSIONS
import time
from pathlib import Path
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt

import badwinner
import badwinner2

# registers the MelSpectrogram layer so models with it load
from mellayer import MelSpectrogram
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
        model_name="efficientnetv2b3",
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
        self.preprocess_fn = None
        # self.learning_rate = 0.01
        self.training_data_meta = None
        self.loss_fn = None
        # multi label models output logits, the sigmoid is only added when saving
        self.from_logits = False
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

            predictions = self.probability_model().predict(self.test)
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

            val_history = history.history["val_pr_auc"]
            best_val = np.amax(val_history)
            test_results_acc["val_pr_auc"] = best_val

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

        if args.get("train_batch_size"):
            self.batch_size = args["train_batch_size"]
        logging.info("Batch size %s", self.batch_size)
        self.log_dir = self.log_dir / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        labels, excluded_labels, self.training_data_meta = init_labels(
            self.data_dir, **args
        )
        weights_meta = None
        if weights is not None:
            weights_meta = load_weights_meta(weights)
        if weights_meta is not None and not args.get("reset_labels", False):
            # use the same labels and mappings as the weights so the trained
            # final layer is kept and each output still means the same label
            labels, excluded_labels = use_weights_labels(
                weights_meta, labels, excluded_labels, args
            )

        (
            self.train,
            self.validation,
            self.labels,
            remapped,
            extra_label_map,
            epoch_size,
        ) = load_datasets(
            labels,
            excluded_labels,
            self.data_dir,
            self.batch_size,
            **args,
        )
        if args.get("benchmark_pipeline"):
            benchmark_pipeline(self.train, args["benchmark_pipeline"])
            return

        from tfdataset import DIMENSIONS

        self.input_shape = DIMENSIONS
        if self.model_name == "embeddings":
            self.input_shape = EMBEDDING_SHAPE
        elif "efficientnet" in self.model_name:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 3)

        if args.get("n_mels") != 160:
            # inputs are [mels, time, channels]
            self.input_shape = (args.get("n_mels"), self.input_shape[1], 3)
        args["excluded_labels"] = excluded_labels
        args["remapped_labels"] = remapped
        args["extra_label_map"] = extra_label_map
        self.num_classes = len(self.labels)

        # can use for 2 gpus, but seems to be much slower on our setup
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        weights_labels = None
        if weights_meta is not None:
            weights_labels = weights_meta.get("ebird_labels")
            logging.info("Weight labels are %s", weights_labels)
            # same labels so keep the trained final layer
            if weights_labels == self.labels:
                weights_labels = None
        self.build_model(
            multi_label=args.get("multi_label", True),
            loss_fn=args.get("loss_fn", "keras"),
            weight_labels=weights_labels,
            weights=weights,
            pcen=args.get("pcen", False),
            epoch_size=epoch_size,
            epochs=epochs,
            lr=args.get("lr", 5e-4),
            mel_in_model=args.get("mel_in_model", False),
            spec_augment=not args.get("no_augment", False),
            steps_per_execution=args.get("steps_per_execution", 1),
        )
        (self.checkpoint_folder / run_name).mkdir(parents=True, exist_ok=True)
        if self.model_name != "rf-features":
            self.probability_model().save(
                self.checkpoint_folder / run_name / f"{run_name}.keras"
            )
        self.save_metadata(run_name, None, None, **args)

        checkpoints = self.checkpoints(
            run_name,
            multi_label=args.get("multi_label", True),
            patience=args.get("patience", 10),
            backup=args.get("backup", False),
        )
        class_weights = None
        # off by default, keras turns class_weight into one weight per sample
        # from the argmax label which doesnt suit multi label targets
        if args.get("use_weighting", False):
            class_weights = get_weighting(self.train, self.labels)
            logging.info("Weights are %s", class_weights)
        else:
            logging.info("Not using class weighting")
        history = None

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
        self.probability_model().save(
            self.checkpoint_folder / run_name / f"{run_name}.keras"
        )
        thresholds = None
        pr_weights = self.checkpoint_folder / run_name / "val_pr_auc.weights.h5"
        if (
            args.get("multi_label")
            and self.validation is not None
            and pr_weights.exists()
        ):
            # best per label thresholds on the validation set, with the pr auc
            # weights as these are best over all labels. self.validation is
            # used by fit so can't have the track ids
            logging.info("Finding best thresholds on validation set")
            self.model.load_weights(pr_weights)
            thresholds = best_threshold(
                self.probability_model(),
                self.labels,
                self.load_eval_set("validation", **args),
            )
        self.save_metadata(
            run_name, history, test_results, thresholds=thresholds, **args
        )
        if self.test is not None:
            weights = "val_pr_auc.weights.h5"
            self.model.load_weights(
                os.path.join(self.checkpoint_folder, run_name, weights)
            )
            confusion_file = self.checkpoint_folder / run_name / "confusion-val_loss"
            if args.get("multi_label"):
                multi_confusion_single(
                    self.probability_model(),
                    self.labels,
                    self.test,
                    confusion_file,
                    thresholds or {},
                    model_name=self.model_name,
                    split_secondary=True,
                )
            else:
                confusion(self.model, self.labels, self.test, confusion_file)

    def save_metadata(self, run_name, history, test_results, thresholds=None, **args):
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
        model_stats["evaluated_labels"] = [
            l for l in self.labels if l in EVALUATED_LABELS
        ]
        # model_stats["multi_label"] = args.get("multi_label", False)
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
        if thresholds is not None:
            model_stats["original_thresholds"] = thresholds
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
        self,
        multi_label=False,
        loss_fn="keras",
        weight_labels=None,
        weights=None,
        pcen=False,
        epoch_size=None,
        epochs=35,
        lr=5e-4,
        mel_in_model=False,
        spec_augment=False,
        steps_per_execution=1,
    ):
        if weights is not None and weight_labels is not None:
            # build model wight these labels then pop the dense layer and add a new one for our labels
            labels = weight_labels
        else:
            labels = self.labels

        num_labels = len(labels)

        # these models end in their own sigmoid
        self.from_logits = multi_label and self.model_name not in (
            "badwinner2-res",
            "embeddings",
            "wr-resnet",
        )
        activation = "softmax"
        if multi_label:
            # train on logits, probability_model adds the sigmoid for inference
            activation = "linear"
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
        elif self.model_name == "badwinner2":
            logging.info("Building bad winner2")
            self.model = badwinner2.build_model(
                self.input_shape,
                None,
                num_labels,
                multi_label=multi_label,
                lme=self.lme,
                logits=self.from_logits,
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
            base_model, self.preprocess_fn = self.get_base_model(self.input_shape)
            # base_model.summary()
            if mel_in_model:
                # raw audio in, the mel (and spec augment when training) is
                # made on the GPU by the model rather than in the dataset
                from tfdataset import (
                    SR,
                    NFFT,
                    HOP_LENGTH,
                    N_MELS,
                    FMIN,
                    FMAX,
                    BREAK_FREQ,
                )

                input = tf.keras.Input(shape=(SR * 3,), name="input")
                mel = MelSpectrogram(
                    sr=SR,
                    n_fft=NFFT,
                    hop_length=HOP_LENGTH,
                    n_mels=N_MELS,
                    fmin=FMIN,
                    fmax=FMAX,
                    break_freq=BREAK_FREQ,
                    power=not pcen,
                    channels=self.input_shape[-1],
                    spec_augment=spec_augment,
                    name="mel_spectrogram",
                )(input)
                logging.info(
                    "Making mels in the model, spec augment %s", spec_augment
                )
            else:
                input = tf.keras.Input(shape=self.input_shape, name="input")
                mel = input
            # x = norm_layer(input)
            if pcen:
                logging.info("Adding pcen layer")
                from tfpcen import PCEN

                # inputs are [batch, mels, time, channels]
                x = PCEN(time_axis=2, dtype="float32")(mel)
            else:
                # dataset gives power mels, MagTransform is pow(x, a) so needs
                # non negative inputs, batch norm then normalizes per channel
                # over batch, mels and time
                # inputs are [batch, mels, time, channels]
                logging.info("Adding mag transform and batch norm")
                x = badwinner2.MagTransform()(mel)
                x = tf.keras.layers.BatchNormalization(axis=-1)(x)

            x = base_model(x)
            # , training=True)

            if self.lme:
                x = badwinner2.LMELayer(axis=1, sharpness=5)(x)
                x = badwinner2.LMELayer(axis=2, sharpness=5)(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            # effnetv2b3 is 0.2
            # v2bm is 0.2
            x = tf.keras.layers.Dropout(0.5)(x)

            logging.info("Using %s activation", activation)
            x = tf.keras.layers.Dense(num_labels, name="prediction")(x)

            birds = tf.keras.layers.Activation(
                activation, dtype="float32", name="predictions"
            )(x)

            outputs = birds
            self.model = tf.keras.models.Model(input, outputs=outputs)

        logging.info(
            "Using  %s cross entropy from logits %s",
            "binary" if multi_label else "",
            self.from_logits,
        )
        loss_fn = loss(multi_label, from_logits=self.from_logits)

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

                outputs = birds
                self.model = tf.keras.models.Model(input, outputs=outputs)

        logging.info("Steps per execution %s", steps_per_execution)
        self.model.compile(
            optimizer=optimizer(
                epoch_size=epoch_size,
                batch_size=self.batch_size,
                total_epochs=epochs,
                lr=lr,
            ),
            # runs several batches per call to the GPU, less per step overhead
            steps_per_execution=steps_per_execution,
            loss=loss_fn,
            # given the per label sample weights, so soft labels, which may or
            # may not be in a sample, don't count. Binary accuracy and roc auc
            # aren't used as with many labels they are dominated by easy
            # negatives
            weighted_metrics=compile_metrics(
                multi_label, num_labels, self.from_logits
            ),
        )
        self.model.summary()

    def probability_model(self):
        # model outputting probabilities, for saving and predicting
        if not self.from_logits:
            return self.model
        # model.output is a list as the model is built with outputs=[birds],
        # passing the list would add a leading dim to the output
        probabilities = tf.keras.layers.Activation(
            "sigmoid", dtype="float32", name="probabilities"
        )(self.model.outputs[0])
        return tf.keras.models.Model(self.model.input, outputs=probabilities)

    def checkpoints(self, run_name, multi_label=True, patience=10, backup=False):
        """Best weights per metric, early stopping and, with backup, the full
        training state (weights, optimizer, epoch) every epoch so an
        interrupted run (e.g. a spot machine) resumes where it stopped when
        started again with the same command"""
        # val_ and the names of the compile_metrics
        metrics = ["val_loss", "val_pr_auc", "val_precision", "val_recall"]
        if not multi_label:
            metrics.append("val_categorical_accuracy")
        run_dir = self.checkpoint_folder / run_name
        best_file = run_dir / "backup-best.json"
        best = {}
        if backup and best_file.exists():
            # a resumed run must only replace best weights with better ones
            with best_file.open() as f:
                best = json.load(f)
            logging.info("Resuming best metric values %s", best)
        checks = []
        modes = {}
        for m in metrics:
            m_dir = run_dir / f"{m}.weights.h5"
            if "loss" in m or "focal" in m:
                mode = "min"
            else:
                mode = "max"
            modes[m] = mode
            m_check = tf.keras.callbacks.ModelCheckpoint(
                m_dir,
                monitor=m,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode=mode,
                initial_value_threshold=best.get(m),
            )
            checks.append(m_check)
        if backup:
            checks.append(BestMetrics(best_file, modes, best))
            # restores weights, optimizer state (Adam moments, lr schedule step)
            # and the epoch, and is removed once training finishes
            checks.append(tf.keras.callbacks.BackupAndRestore(run_dir / "backup"))
        earlyStopping = tf.keras.callbacks.EarlyStopping(
            patience=patience,
            monitor="val_pr_auc",
            mode="max",
        )
        checks.append(earlyStopping)
        # dont use with cosine
        # reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_loss", verbose=1, mode="min", patience=4
        # )
        # checks.append(reduce_lr_callback)
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
        self.test = self.load_eval_set("test", **args)

    def load_eval_set(self, split, **args):
        """Unshuffled split with all y loaded, as track ids are needed for the
        per track thresholds and confusion"""
        logging.info("Loading %s", split)
        second_dir = None
        if self.second_data_dir is not None:
            second_dir = self.second_data_dir / split

        eval_args = dict(args)
        eval_args["shuffle"] = False
        eval_args["load_all_y"] = True
        dataset, _, _, _, _ = get_dataset(
            self.data_dir / split,
            self.labels,
            batch_size=self.batch_size,
            image_size=self.input_shape,
            second_dir=second_dir,
            deterministic=True,
            **eval_args,
        )
        return dataset

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


def benchmark_pipeline(dataset, batches, warmup=20):
    """Times the training input pipeline alone, no model, and logs samples a
    second. The first warmup batches (filling the shuffle buffer etc) aren't
    timed"""
    import time

    samples = 0
    start = None
    for i, batch in enumerate(dataset.take(warmup + batches)):
        if i == warmup:
            start = time.time()
        if i >= warmup:
            samples += int(tf.shape(batch[0])[0])
    if start is None or samples == 0:
        logging.info("Dataset has fewer than %s batches to benchmark", warmup + 1)
        return
    took = time.time() - start
    logging.info(
        "Pipeline benchmark: %s samples in %.1fs, %.0f samples/s, %.0f ms a batch",
        samples,
        took,
        samples / took,
        1000 * took / batches,
    )


class BestMetrics(tf.keras.callbacks.Callback):
    """Saves the best value of each monitored metric every epoch, so a run
    resumed with BackupAndRestore keeps its best weights checkpoints. Removed
    when training finishes"""

    def __init__(self, filename, modes, best=None):
        super().__init__()
        self.filename = Path(filename)
        self.modes = modes
        self.best = dict(best or {})

    def on_epoch_end(self, epoch, logs=None):
        for m, mode in self.modes.items():
            value = (logs or {}).get(m)
            if value is None:
                continue
            value = float(value)
            old = self.best.get(m)
            if old is None or (value < old if mode == "min" else value > old):
                self.best[m] = value
        with self.filename.open("w") as f:
            json.dump(self.best, f, indent=4)

    def on_train_end(self, logs=None):
        self.filename.unlink(missing_ok=True)


def compile_metrics(multi_label, num_labels, from_logits):
    """Metrics the model is compiled with, all weighted by the per label sample
    weights. checkpoints saves the best weights for each"""
    metrics = [
        # per label pr auc averaged over labels so rare labels count as much
        # as common ones
        tf.keras.metrics.AUC(
            curve="PR",
            multi_label=True,
            num_labels=num_labels,
            from_logits=from_logits,
            name="pr_auc",
        ),
        # logits threshold of 0 is a probability of 0.5
        tf.keras.metrics.Recall(thresholds=0.0 if from_logits else None),
        tf.keras.metrics.Precision(thresholds=0.0 if from_logits else None),
    ]
    if not multi_label:
        metrics.append(tf.keras.metrics.CategoricalAccuracy())
    return metrics


def huber_loss(y_true, y_pred):
    # huber loss on probabilities for models outputting logits
    return tf.keras.losses.huber(y_true, tf.math.sigmoid(y_pred))


def loss(multi_label=False, smoothing=0, from_logits=False):
    if multi_label:
        # logging.info("Using focal binary cross")
        # return tf.keras.losses.BinaryFocalCrossentropy(
        #     gamma=2.0, from_logits=False, apply_class_balancing=True
        # )
        # logging.info("Using binary loss")
        # loss_fn = tf.keras.losses.BinaryCrossentropy(
        #     label_smoothing=smoothing,
        # )
        logging.info("Using MultiHotMixupBCE loss")

        loss_fn = MultiHotMixupBCE(from_logits=from_logits)
        # loss_fn = sigmoid_binary_cross
    else:
        logging.info("Using cross loss")
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=smoothing,
        )

    return loss_fn


def optimizer(epoch_size, batch_size=32, total_epochs=35, lr=5e-4):
    STEPS_PER_EPOCH = epoch_size // batch_size

    # Calculate exact step targets
    warmup_steps = STEPS_PER_EPOCH * 1  # 1 full epoch of warmup
    total_steps = STEPS_PER_EPOCH * total_epochs

    # 2. Define the Learning Rate Schedule with Native Warmup
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,  # warmup starts from 0
        decay_steps=total_steps - warmup_steps,  # decay starts after warmup
        alpha=1e-2,  # Drops down to lr * 1e-2 by the final epoch
        warmup_target=lr,  # peak learning rate
        warmup_steps=warmup_steps,  # Linearly ramp from 0.0 to lr over this many steps
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
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
    pre_model=None,
    pre_labels=None,
):
    ebird_labels = model_labels.copy()
    labels = []
    ebird_id_map = get_ebird_ids_to_labels()
    for ebird_id in ebird_labels:
        label = ebird_id_map.get(ebird_id, [ebird_id])
        labels.append(label[0])
    true_categories = [y for x, y in dataset]
    recs = []
    tracks = []
    starts = []
    y_true = []

    if isinstance(true_categories[0], tuple):
        for batch in true_categories:
            recs.extend(batch[3].numpy())
            tracks.extend(batch[4].numpy())
            starts.extend(batch[7].numpy())
            y_true.extend(batch[0])
        true_categories = y_true
        dataset = dataset.map(
            lambda x, y: (x, y[0]),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
    else:
        # true_categories = [y[0] if isinstance(y, tuple) else yfor x, y in dataset ]
        y_true = tf.concat(true_categories, axis=0)
    if one_hot:
        y_true = np.int64(tf.argmax(y_true, axis=1))
    else:
        y_true = np.array(y_true)
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
    if pre_model is not None:
        pre_pred = pre_model.predict(dataset)

    predicted_categories = []
    if "None" not in labels:
        labels.append("None")
    # save raw values for use later
    npy_file = filename.parent / f"{filename.stem}-raw.npy"
    with npy_file.open("wb") as f:
        np.save(f, np.array(model_labels))
        np.save(f, y_pred)
        np.save(f, y_true)
        np.save(f, np.array(tracks))
        np.save(f, np.array(recs))
        np.save(f, np.array(starts))

        if pre_model is not None:
            np.save(f, np.array(["premodel"]))
            np.save(f, pre_pred)
            np.save(f, np.array(pre_labels))

        if all_preds is not None and len(all_preds) > 1:
            np.save(f, np.array(["other_model"]))
            np.save(f, all_preds[1])

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
    np.savez(filename.with_suffix(".npz"), cm=cm, labels=np.array(labels))

    # plt.savefig(f"./confusions/{filename}", format="png")


def predict_dataset(model, dataset):
    """Returns (true labels, predictions) for every sample in dataset"""
    y_ds = dataset.map(
        lambda x, y: y[0] if isinstance(y, tuple) else y,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    true_categories = np.concatenate(list(y_ds.as_numpy_iterator()), axis=0)
    x_ds = dataset.map(
        lambda x, y: x,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    y_pred = model.predict(x_ds)
    if isinstance(y_pred, tf.RaggedTensor):
        # predict returns (batches, None, labels) when each batch output has an
        # extra leading dim, flatten back to one row per sample
        y_pred = y_pred.merge_dims(0, 1)
    y_pred = np.asarray(y_pred)
    assert (
        y_pred.shape == true_categories.shape
    ), f"Predictions {y_pred.shape} don't match labels {true_categories.shape}"
    return true_categories, y_pred


def group_by_track(dataset, predictions):
    """Mean prediction per track and label, from a dataset loaded with
    load_all_y where y[4] is the track id. A sample covering several tracks
    has them all in its track id e.g. "123 124" and is grouped with samples of
    the same tracks

    Each label is pooled over the samples that know it, as e.g. a secondary
    bird may only be in some of a track's samples:
    - if any samples have it as a true label the track is true for it, and its
      prediction is the mean over just those samples
    - otherwise the track is false for it, and its prediction is the mean over
      the samples it isn't soft in
    - if it is soft in every sample it is soft for the track

    Returns (true labels, mean predictions, soft labels, secondary labels)
    with one row per track. Secondary labels are true labels that are only
    ever a secondary tag in the track's samples. Soft and secondary labels are
    all zeros if the dataset has none (y[9] and y[10])
    """
    true_categories, y_pred = predictions
    track_ds = dataset.map(
        lambda x, y: y[4],
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    track_ids = np.concatenate(list(track_ds.as_numpy_iterator()), axis=0)
    assert len(track_ids) == len(
        y_pred
    ), f"Track ids {len(track_ids)} don't match predictions {len(y_pred)}"
    unique_tracks, track_index = np.unique(track_ids, return_inverse=True)
    num_tracks = len(unique_tracks)

    true = (true_categories > 0).astype(np.float64)

    def y_values(index):
        # zeros for datasets without y[index]
        spec = dataset.element_spec[1]
        if not isinstance(spec, tuple) or len(spec) <= index:
            return np.zeros_like(true)
        values_ds = dataset.map(
            lambda x, y: y[index],
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        values = np.concatenate(list(values_ds.as_numpy_iterator()), axis=0)
        return (values > 0).astype(np.float64)

    soft = y_values(9) * (1 - true)
    secondary = y_values(10) * true
    negative = (1 - true) * (1 - soft)

    def per_track(values):
        summed = np.zeros((num_tracks, y_pred.shape[1]), dtype=np.float64)
        np.add.at(summed, track_index, values)
        return summed

    pos_count = per_track(true)
    neg_count = per_track(negative)
    pos_pred = per_track(y_pred * true) / np.maximum(pos_count, 1)
    neg_pred = per_track(y_pred * negative) / np.maximum(neg_count, 1)
    # a label soft in every sample is pooled over all of them, it isn't
    # counted in the confusions but save_soft_predictions reports it
    all_pred = per_track(y_pred) / np.bincount(
        track_index, minlength=num_tracks
    )[:, np.newaxis]
    track_true = np.float32(pos_count > 0)
    track_soft = np.float32((pos_count == 0) & (neg_count == 0))
    # true in some samples, but never as a primary tag
    primary_count = per_track(true * (1 - secondary))
    track_secondary = np.float32((pos_count > 0) & (primary_count == 0))
    track_pred = np.where(
        pos_count > 0, pos_pred, np.where(neg_count > 0, neg_pred, all_pred)
    )
    if np.any(track_soft):
        logging.info(
            "%s tracks have soft labels", np.count_nonzero(track_soft.any(axis=1))
        )
    logging.info("Grouped %s samples into %s tracks", len(y_pred), num_tracks)
    return track_true, track_pred, track_soft, track_secondary


def multi_confusion_single(
    model,
    labels,
    dataset,
    filename,
    thresholds_per_label,
    one_hot=True,
    model_name=None,
    threshold=0.5,
    split_secondary=False,
):
    """Save multi label confusion matrices at threshold, at
    thresholds_per_label ({label: threshold}, missing labels use threshold)
    and at thresholds_per_label capped between 0.5 and 0.8, predicting
    dataset once.

    With split_secondary also saves filename-primary-* and
    filename-secondary-* confusions. primary leaves out labels that are only
    a secondary tag of a track (like soft labels, neither right nor wrong).
    secondary is only tracks with secondary labels, with those as the true
    labels and the track's primary labels left out

    Uses the mean prediction per track, so dataset must be loaded with
    load_all_y (y[4] is the track id)
    """
    filename = Path(filename)
    predictions = group_by_track(dataset, predict_dataset(model, dataset))
    label_thresholds = np.array(
        [
            thresholds_per_label.get(l, threshold)
            for l in labels[: predictions[1].shape[1]]
        ]
    )
    # no dots in the names as with_suffix would cut them off
    confusion_thresholds = {
        str(threshold).replace(".", "_"): threshold,
        "thresholds": label_thresholds,
        "thresholds-capped": np.clip(label_thresholds, 0.5, 0.8),
    }
    true_categories, y_pred, soft, secondary = predictions
    # (suffix, true labels, soft labels, tracks to use)
    variants = [("", true_categories, soft, None)]
    if split_secondary:
        primary_true = true_categories * (1 - secondary)
        variants.append(
            ("-primary", primary_true, np.maximum(soft, secondary), None)
        )
        has_secondary = secondary.any(axis=1)
        logging.info("%s tracks have secondary labels", np.count_nonzero(has_secondary))
        if np.any(has_secondary):
            variants.append(
                (
                    "-secondary",
                    secondary,
                    np.maximum(soft, primary_true),
                    has_secondary,
                )
            )
    for suffix, var_true, var_soft, rows in variants:
        var_pred = y_pred
        if rows is not None:
            var_true, var_pred, var_soft = var_true[rows], y_pred[rows], var_soft[rows]
        for name, prob_thresh in confusion_thresholds.items():
            save_multi_confusion(
                var_true,
                var_pred,
                labels,
                filename.parent / f"{filename.name}{suffix}-{name}",
                prob_thresh,
                soft=var_soft,
            )


def save_multi_confusion(
    true_categories, y_pred, labels, filename, prob_thresh, soft=None
):
    """Saves filename-combined, filename-matches and filename-none confusions.

    combined: each true label is correct if predicted, and every prediction
    that isn't a true label counts against every true label. A true label
    that isn't predicted, with no wrong predictions, counts as None.

    matches: each true label is correct if predicted, whatever else was
    predicted. Otherwise it counts against the highest wrong prediction, or
    None if there are no wrong predictions.

    If noise isn't a label, tracks with no true labels are noise and get a
    noise row (and column) before None. They are correct if nothing is
    predicted, otherwise combined counts every prediction against noise and
    matches the highest one. A true label that is missed with nothing
    predicted counts as noise rather than None. If noise is a label tracks
    with no true labels are counted in the None row

    prob_thresh is a single threshold or one per label

    soft (one row per track like true_categories) are labels that may or may
    not be in a track, predicting them is neither right nor wrong so they are
    left out. How often each was predicted is saved to filename-soft.csv
    """
    num_labels = y_pred.shape[1]
    # one threshold per label
    prob_thresh = np.broadcast_to(
        np.asarray(prob_thresh, dtype=np.float32), (num_labels,)
    )
    cm_labels = list(labels[:num_labels])
    no_true_count = np.sum(true_categories.sum(axis=1) == 0)
    # true count of each class
    totals_row = np.int64(true_categories.sum(axis=0))
    noise_index = None
    if "noise" not in cm_labels:
        noise_index = len(cm_labels)
        cm_labels.append("noise")
        totals_row = np.append(totals_row, no_true_count)
    cm_labels.append("None")
    none_index = len(cm_labels) - 1
    # predicting nothing is predicting noise when noise isn't a label
    nothing_index = none_index if noise_index is None else noise_index
    # with a noise row, nothing true goes there rather than None
    totals_row = np.append(totals_row, 0 if noise_index is not None else no_true_count)

    if soft is not None and np.any(soft):
        save_soft_predictions(soft, y_pred, labels[:num_labels], prob_thresh, filename)
        # below any threshold and argmax so never predicted
        y_pred = np.where(soft > 0, -1.0, y_pred)

    none_p = []
    none_y = []
    flat_p = []
    flat_y = []
    match_p = []
    match_y = []
    for y, p in zip(true_categories, y_pred):
        predicted = np.flatnonzero(p >= prob_thresh)
        true_labels = np.flatnonzero(y == 1)
        if len(true_labels) == 0:
            if noise_index is not None:
                if len(predicted) == 0:
                    flat_y.append(noise_index)
                    flat_p.append(noise_index)
                    match_y.append(noise_index)
                    match_p.append(noise_index)
                else:
                    for pred in predicted:
                        flat_y.append(noise_index)
                        flat_p.append(pred)
                    match_y.append(noise_index)
                    match_p.append(max(predicted, key=lambda pred: p[pred]))
            else:
                # nothing is true, record any predictions against "nothing"
                for pred in predicted:
                    flat_y.append(none_index)
                    flat_p.append(pred)
                    match_y.append(none_index)
                    match_p.append(pred)
            continue

        wrong_preds = [pred for pred in predicted if pred not in true_labels]
        for true_label in true_labels:
            if true_label in predicted:
                flat_y.append(true_label)
                flat_p.append(true_label)
                match_y.append(true_label)
                match_p.append(true_label)
            else:
                match_y.append(true_label)
                if len(wrong_preds) > 0:
                    match_p.append(max(wrong_preds, key=lambda pred: p[pred]))
                else:
                    match_p.append(nothing_index)
            # every wrong prediction counts against every true label
            for pred in wrong_preds:
                flat_y.append(true_label)
                flat_p.append(pred)
            if true_label not in predicted and len(wrong_preds) == 0:
                flat_y.append(true_label)
                flat_p.append(nothing_index)

                # if we have no prediction over a threshold put the best one over a reasonable threshold (0.3) into the None matrix
                best_label = np.argmax(p)
                if (
                    len(predicted) == 0
                    and p[best_label] >= 0.3
                    and best_label not in true_labels
                ):
                    none_y.append(true_label)
                    none_p.append(best_label)

    for name, cm_y, cm_p in (
        ("combined", flat_y, flat_p),
        ("matches", match_y, match_p),
        ("none", none_y, none_p),
    ):
        cm = confusion_matrix(
            np.int64(cm_y), np.int64(cm_p), labels=np.arange(len(cm_labels))
        )
        cm_path = filename.parent / f"{filename.name}-{name}"
        np.savez(
            cm_path.with_suffix(".npz"),
            cm=cm,
            labels=np.array(cm_labels),
            thresholds=prob_thresh,
        )
        figure = plot_confusion_matrix(cm, class_names=cm_labels, totals_row=totals_row)
        logging.info("Saving confusion to %s", cm_path.with_suffix(".png"))
        plt.savefig(cm_path.with_suffix(".png"), format="png")
        plt.close(figure)


def save_soft_predictions(soft, y_pred, labels, prob_thresh, filename):
    """For each label, the tracks it is soft on and how many of them predict
    it, a high fraction suggests the soft label is often really there"""
    rows = []
    for i, label in enumerate(labels):
        on = soft[:, i] > 0
        if not np.any(on):
            continue
        predicted = int(np.count_nonzero(y_pred[on, i] >= prob_thresh[i]))
        rows.append((label, int(on.sum()), predicted, predicted / on.sum()))
    rows.sort(key=lambda r: -r[1])
    out = filename.parent / f"{filename.name}-soft.csv"
    with open(out, "w") as f:
        f.write("label,soft_tracks,predicted,predicted_fraction\n")
        for label, count, predicted, fraction in rows:
            f.write(f"{label},{count},{predicted},{fraction:.3f}\n")
            logging.info(
                "Soft %s predicted on %s of %s tracks", label, predicted, count
            )
    logging.info("Saved soft label predictions to %s", out)


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
        augment=not args.get("no_augment", False),
        drop_remainder=True,  # keep batch shape fixed so XLA only compiles once
        excluded_labels=excluded_labels,
        second_dir=second_dir,
        load_seperate_ds=True,
        cache=False,
        human_dir=human_dir,
        **args,
    )
    if args.get("only_load_train", False):
        return remapped, excluded_labels, extra_label_map
    if args.get("benchmark_pipeline"):
        # only the training input is timed
        return train, None, new_labels, remapped, extra_label_map, epoch_size

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
        human_dir=human_dir,
        load_seperate_ds=True,
        cache=False,
        **args,
    )

    return train, validation, new_labels, remapped, extra_label_map, epoch_size


def load_weights_meta(weights):
    """metadata.txt saved next to weights, or None if there isn't one"""
    weights_metadata = Path(weights).parent / "metadata.txt"
    if not weights_metadata.exists():
        logging.warning(
            "No metadata for weights %s, using labels from the datasets, these "
            "must match the labels the weights were trained with",
            weights,
        )
        return None
    with weights_metadata.open("r") as f:
        return json.load(f)


WEIGHTS_LABEL_KEYS = ["ebird_labels", "excluded_labels", "remapped_labels"]


def use_weights_labels(weights_meta, labels, excluded_labels, args):
    """Labels, excluded labels, remapped labels, extra label map and generic
    bird from the weights metadata. remapped_labels and extra_label_map are set
    in args so get_dataset uses them instead of working them out again

    Returns labels and excluded labels, unchanged if the metadata is missing any
    of WEIGHTS_LABEL_KEYS
    """
    missing = [k for k in WEIGHTS_LABEL_KEYS if weights_meta.get(k) is None]
    if missing:
        logging.warning(
            "Weights metadata is missing %s, using labels from the datasets",
            missing,
        )
        return labels, excluded_labels

    weights_labels = weights_meta["ebird_labels"]
    # remapped labels has every dataset label the weights were trained with,
    # including excluded and merged ones
    new_labels = sorted(set(labels) - set(weights_meta["remapped_labels"]))
    if new_labels:
        logging.warning(
            "Datasets have labels %s which the weights weren't trained with, "
            "these are excluded",
            new_labels,
        )
    use_generic_bird = weights_meta.get("use_generic_bird", False)
    if args.get("use_generic_bird", False) != use_generic_bird:
        logging.warning(
            "Setting use generic bird to %s to match the weights", use_generic_bird
        )
    args["use_generic_bird"] = use_generic_bird
    args["remapped_labels"] = weights_meta["remapped_labels"]
    # an empty map rather than None so get_dataset uses the remapped labels
    args["extra_label_map"] = weights_meta.get("extra_label_map") or {}
    logging.info("Using labels, excluded and remapped labels from the weights")
    return list(weights_labels), list(weights_meta["excluded_labels"])


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
        pass
        # if "human" not in excluded_labels:
        #     excluded_labels.append("human")
        if "noise" not in excluded_labels:
            excluded_labels.append("noise")

    logging.info("labels are %s Excluding %s", labels, excluded_labels)
    return labels, excluded_labels, meta


from functools import partial

from multiprocessing import Pool


def evaluate_dir(model, model_meta, dir_name, filename, rec_ids):
    if model_meta.get("mel_in_model"):
        raise NotImplementedError("evaluate_dir makes mels, this model takes raw audio")
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
    all_pred_confidences = []
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
                    all_pred_confidences.append(track_preds)
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

    import pickle

    # save all confs for further analysis
    pkl_file = filename.parent / f"{filename.stem}-raw-confidences.pkl"
    with pkl_file.open("wb") as f:
        pickle.dump(all_pred_confidences, f)
    print("Saving to ", filename)
    cm_file = filename.parent / f"{filename.stem}-mean"
    cm = confusion_matrix(y_true, predicted_mean, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(cm_file.with_suffix(".png"), format="png")
    np.savez(cm_file.with_suffix(".npz"), cm=cm, labels=np.array(labels))

    print("Saving to ", filename)
    cm_file = filename.parent / f"{filename.stem}-max"
    cm = confusion_matrix(y_true, predicted_max, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(cm_file.with_suffix(".png"), format="png")
    np.savez(cm_file.with_suffix(".npz"), cm=cm, labels=np.array(labels))

    print("Saving to ", filename)
    cm_file = filename.parent / f"{filename.stem}-counts"
    cm = confusion_matrix(y_true, predicted_counts, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(cm_file.with_suffix(".png"), format="png")
    np.savez(cm_file.with_suffix(".npz"), cm=cm, labels=np.array(labels))


from identifytracks import get_tracks_from_signals, signal_noise, get_end, Signal
from audiowriter import load_recording
from predict_utils import load_samples
from audiodataset import Recording


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
        labels = meta_data.get("ebird_labels")
        other_models = None
        if model_name == "rf-features":
            model = ydf.load_model(str(model_path.parent))
        else:
            if model_path.is_dir():
                model_path = model_path / f"{model_path.stem}.keras"
            logging.info("Loading %s", model_path)
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
        if args.weak_dir:
            weights_file, _ = single_weight_file(args.weights, model_path.parent)
            logging.info("Loading weights %s", weights_file)
            model.load_weights(weights_file)
            evaluate_weakly_labelled_dir(
                model, meta_data, Path(args.evaluate_dir), args.confusion
            )
            return
        if args.evaluate_dir is not None:
            rec_ids = None
            if args.dataset_dir is not None:
                base_dir = Path(args.dataset_dir)
                meta_f = base_dir.parent / "training-split.json"
                if not meta_f.exists():
                    # datasets built before the split had its own file
                    meta_f = base_dir.parent / "training-meta.json"
                with meta_f.open("r") as f:
                    data_json = json.load(f)

                ds_name = base_dir.stem
                rec_ids = data_json["recs"][ds_name]

            weights_file, _ = single_weight_file(args.weights, model_path.parent)
            logging.info("Loading weights %s", weights_file)
            model.load_weights(weights_file)
            evaluate_dir(
                model, meta_data, Path(args.evaluate_dir), args.confusion, rec_ids
            )
            return

        # default to the dirs the model was trained with, training saves its
        # args in the metadata
        for key in ("dataset_dir", "second_dataset_dir", "noise_dirs"):
            if getattr(args, key) is None and meta_data.get(key) is not None:
                setattr(args, key, meta_data[key])
                logging.info("Using %s %s from the model metadata", key, meta_data[key])
        if args.dataset_dir is None:
            raise ValueError(
                f"No --dataset-dir given and none in {meta_file} to evaluate on"
            )
        base_dir = Path(args.dataset_dir)
        second_dir = None
        if args.second_dataset_dir is not None:
            args.second_dataset_dir = Path(args.second_dataset_dir)
            second_dir = args.second_dataset_dir / "test"

        dataset_args = dict(
            excluded_labels=excluded_labels,
            remapped_labels=remapped_labels,
            second_dir=second_dir,
            image_size=DIMENSIONS,
            shuffle=False,
            deterministic=True,
            batch_size=64,
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
            pcen=meta_data.get("pcen", False),
            fmin=meta_data.get("fmin"),
            fmax=meta_data.get("fmax"),
            break_freq=meta_data.get("break_freq"),
            n_fft=meta_data.get("n_fft"),
            model_name=model_name,
            cache=True,
            load_all_y=True,
            noise_dirs=args.noise_dirs,
            # models with a mel layer take raw audio
            mel_in_model=meta_data.get("mel_in_model", False),
        )
        dataset, _, _, _, _ = get_dataset(base_dir / "test", labels, **dataset_args)
        # (file prefix, dataset), with a second dataset also confusions of only
        # the main (cacophony) dataset's test files
        eval_datasets = [("", dataset)]
        if second_dir is not None:
            cacophony_dataset, _, _, _, _ = get_dataset(
                base_dir / "test", labels, **dict(dataset_args, second_dir=None)
            )
            eval_datasets.append(("cacophony-", cacophony_dataset))

        thresholds = meta_data.get("original_thresholds")
        if multi and thresholds is None and not meta_data.get("only_features"):
            # best per label thresholds on the validation set, saved so they
            # are only worked out once
            logging.info("Finding best thresholds on validation set")
            val_dataset, _, _, _, _ = get_dataset(
                base_dir / "validation",
                labels,
                **dict(
                    dataset_args,
                    second_dir=(
                        args.second_dataset_dir / "validation"
                        if args.second_dataset_dir is not None
                        else None
                    ),
                ),
            )
            model.load_weights(model_path.parent / "val_pr_auc.weights.h5")
            thresholds = best_threshold(model, labels, val_dataset)
            del val_dataset
            meta_data["original_thresholds"] = thresholds
            with open(str(meta_file), "w") as f:
                json.dump(meta_data, f, indent=4, cls=MetaJSONEncoder)
            logging.info("Saved thresholds to %s", meta_file)

        if dataset is not None:
            if meta_data.get("only_features"):
                # only_features models have no weight files
                weight_files = [(None, "model")]
            else:
                weight_files = confusion_weight_files(
                    args.weights, model_path.parent, multi=multi
                )

            args.confusion = Path(args.confusion)

            for w, file_prefix in weight_files:
                if w is not None:
                    logging.info("Using %s weights", w)
                    model.load_weights(w)
                    if other_models is not None:
                        for model_dir, other_model in zip(args.model_2, other_models):
                            model_dir = Path(model_dir)
                            other_model.load_weights(model_dir / w.name)
                    if pre_model is not None:
                        pre_path = Path(args.pre_model)
                        pre_model.load_weights(pre_path / w.name)

                confusion_file = (
                    Path("./confusions")
                    / model_path.stem
                    / (args.confusion.parent / f"{args.confusion.stem}-{file_prefix}")
                )
                confusion_file.parent.mkdir(exist_ok=True, parents=True)
                if multi:
                    for ds_prefix, eval_dataset in eval_datasets:
                        multi_confusion_single(
                            model,
                            labels,
                            eval_dataset,
                            confusion_file.parent
                            / f"{ds_prefix}{confusion_file.name}",
                            thresholds or {},
                            one_hot=not meta_data.get("only_features"),
                            model_name=model_name,
                            split_secondary=args.split_secondary,
                        )
                else:
                    confusion(
                        model,
                        labels,
                        dataset,
                        confusion_file,
                        one_hot=not meta_data.get("only_features"),
                        other_models=other_models,
                        pre_model=pre_model,
                        pre_labels=pre_labels,
                    )

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


# weights saved by training, "all" for the confusion uses those that exist
ALL_WEIGHTS = ["val_loss", "val_pr_auc", "val_binary_accuracy"]


def confusion_weight_files(weights, model_dir, multi=True):
    """[(weights file, name)] to evaluate. weights is None for val_pr_auc,
    "all" for every saved weights file, a path to a weights file, or a name in
    model_dir e.g. val_loss for model_dir/val_loss.weights.h5"""
    if weights != "all":
        return [single_weight_file(weights, model_dir)]
    names = list(ALL_WEIGHTS)
    if not multi:
        names[-1] = "val_categorical_accuracy"
    found = [
        (Path(model_dir) / f"{name}.weights.h5", name)
        for name in names
        if (Path(model_dir) / f"{name}.weights.h5").exists()
    ]
    if not found:
        raise FileNotFoundError(f"No weights {names} in {model_dir}")
    return found


def single_weight_file(weights, model_dir):
    """(weights file, name) for weights, see confusion_weight_files"""
    if weights is None:
        weights = "val_pr_auc"
    if weights == "all":
        raise ValueError("--weights all is only for the test set confusion")
    suffix = ".weights.h5"
    path = Path(weights)
    if not path.is_file():
        name = weights if weights.endswith(suffix) else f"{weights}{suffix}"
        path = Path(model_dir) / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Weights {weights} isn't a file and {path} doesn't exist"
            )
    name = path.name
    name = name[: -len(suffix)] if name.endswith(suffix) else path.stem
    return path, name


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
    parser.add_argument("--epochs", type=int, default=35, help="Epochs to use")
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Peak learning rate after warmup"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Epochs without val_pr_auc improving before stopping early",
    )
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
        "--noise-dirs",
        nargs="+",
        default=None,
        help="Noise record dirs to mix into training audio, each containing a"
        "train dir e.g. training-data/noise from build.py --noise-dataset",
    )
    parser.add_argument(
        "--extra-dirs",
        dest="extra_datasets",
        nargs="+",
        default=None,
        help="Extra record dirs added to the main dataset, each containing a "
        "training-meta.json and train/validation dirs e.g. training-data/animal "
        "from otherdata.py --csv",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Train without augmentations (roll, mixup, noise mixing, spec augment)",
    )
    parser.add_argument(
        "--bird-mix-prob",
        type=float,
        default=0.25,
        help="Probability of mixing each training clip with another clip",
    )
    parser.add_argument(
        "--mixup-mode",
        choices=["lam", "union"],
        default="lam",
        help="lam blends two clips and their labels by a random lambda, union "
        "adds the second clip at an audible level (--mixup-db-range) and labels "
        "every bird in either clip 1",
    )
    parser.add_argument(
        "--mixup-db-range",
        nargs=2,
        type=float,
        default=[-12.0, 0.0],
        help="With --mixup-mode union, the range of levels in dB to add the "
        "second clip at relative to the first, by rms",
    )
    parser.add_argument(
        "--noise-mix-fraction",
        type=float,
        default=0.8,
        help="Fraction of the --noise-dirs records used for mixing, the rest "
        "are added to the training data as noise samples",
    )
    parser.add_argument(
        "--noise-train-fraction",
        type=float,
        default=0.05,
        help="Noise only samples to add to training from the --noise-dirs records "
        "not used for mixing, as a fraction of the number of samples",
    )
    parser.add_argument(
        "--noise-eval-fraction",
        type=float,
        default=0.1,
        help="Noise only samples to add to validation and test from --noise-dirs, "
        "as a fraction of the number of samples",
    )
    parser.add_argument(
        "--human-dataset-dir",
        type=none_or_str,
        default=None,
        help="Secondary dataset directory to use",
    )
    parser.add_argument("--confusion", help="Save confusion matrix for model")
    parser.add_argument(
        "--split-secondary",
        action="store_true",
        help="With --confusion also save confusions of primary labels only and "
        "of secondary labels only",
    )

    parser.add_argument(
        "-w",
        "--weights",
        help="Weights to use. Training starts from this weights file. With "
        "--confusion a weights file or a name in the model dir e.g. val_loss for "
        "val_loss.weights.h5, all for every saved weights file, default val_pr_auc",
    )
    parser.add_argument(
        "--reset-labels",
        action="store_true",
        help="With --weights, use labels from the datasets rather than the labels "
        "in the weights metadata, replacing the final layer if they differ",
    )
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
        help="Use weighting for classes, turns off the soft label weights as "
        "keras can't use both",
    )
    parser.add_argument(
        "--drop-untrained",
        action="store_true",
        help="Drop samples of bird species that aren't trained on instead of "
        "using them as negatives",
    )
    parser.add_argument(
        "--no-soft-weights",
        action="store_true",
        help="Train on soft labels as negatives instead of leaving them out of "
        "the loss and pr auc",
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
        "--cache-noise",
        action="store_true",
        help="Keep the decoded noise clips for mixing in memory (about 0.6 MB "
        "a clip, e.g. 7.4 GB for 12,867), much less CPU work each epoch",
    )
    parser.add_argument(
        "--steps-per-execution",
        type=int,
        default=1,
        help="Batches run per call to the GPU, more lowers the per step "
        "overhead e.g. 8-16. Training metrics are only logged every this many "
        "batches",
    )
    parser.add_argument(
        "--benchmark-pipeline",
        type=int,
        default=None,
        metavar="BATCHES",
        help="Time this many training batches of the input pipeline alone (no "
        "model), log samples a second and exit",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Save the full training state every epoch and resume from it when "
        "the same command is run again, e.g. on spot machines",
    )
    parser.add_argument(
        "--batch-size",
        # batch_size is passed by name alongside **args
        dest="train_batch_size",
        type=int,
        default=32,
        help="Training batch size, the learning rate schedule's steps follow it "
        "but --lr isn't scaled, e.g. try --lr 7e-4 to 1e-3 for 64",
    )
    parser.add_argument(
        "--mel-before-model",
        action="store_true",
        help="Make the mel spectrogram (and spec augment) in the input pipeline "
        "on the CPU, the model then takes mels. By default the model makes them "
        "on the GPU from raw audio",
    )
    parser.add_argument(
        "--pcen",
        action="store_true",
        help="Use a trainable PCEN layer on magnitude mels instead of normalized db mels",
    )
    parser.add_argument(
        "--multi-label", type=str2bool, default=True, help="Multi label"
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
        help="loss function to use, keras (Binary or Categorical) depending on multi label",
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

    parser.add_argument("--weak-dir", default=False, action="store_true")

    args = parser.parse_args()
    # saved in the model metadata, models without it take mels
    args.mel_in_model = not args.mel_before_model
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


def best_threshold(model, labels, dataset, filename=None):
    """Per label threshold that maximises F1 on the mean prediction per track,
    as {label: threshold}, to match the per track confusion. dataset must be
    loaded with load_all_y (y[4] is the track id)

    Labels with no true tracks in dataset get 0.5. If filename is given a
    precision recall plot is saved for each label
    """
    from sklearn.metrics import precision_recall_curve

    true_categories, y_pred, soft, _ = group_by_track(
        dataset, predict_dataset(model, dataset)
    )
    thresholds_best = {}
    for i, label in enumerate(labels[: y_pred.shape[1]]):
        # tracks the label is soft on are neither positives nor negatives
        known = soft[:, i] == 0
        y_true = true_categories[known, i] > 0
        if not np.any(y_true):
            logging.info("No samples of %s, using threshold 0.5", label)
            thresholds_best[label] = 0.5
            continue
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred[known, i])
        # last precision / recall point has no threshold
        precision = precision[:-1]
        recall = recall[:-1]
        fscore = np.nan_to_num((2 * precision * recall) / (precision + recall))
        ix = np.argmax(fscore)
        thresholds_best[label] = float(thresholds[ix])
        logging.info(
            "%s best threshold %.3f F-score %.3f precision %.3f recall %.3f",
            label,
            thresholds[ix],
            fscore[ix],
            precision[ix],
            recall[ix],
        )
        if filename is not None:
            no_skill = np.mean(y_true)
            plt.plot(recall, precision, marker=".", label="Model")
            plt.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
            plt.scatter(
                recall[ix], precision[ix], marker="o", color="black", label="Best"
            )
            plt.axis("square")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Recall vs Precision - {label}")
            plt.legend()
            plt.savefig(f"{filename}-{label}.png", format="png")
            plt.clf()
    thresholds = np.array(list(thresholds_best.values()))
    logging.info(
        "All thresholds mean %s median %s", np.mean(thresholds), np.median(thresholds)
    )
    return thresholds_best


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


class MultiHotMixupBCE(tf.keras.losses.Loss):
    def __init__(
        self,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name="multi_hot_mixup_bce",
        from_logits=False,
    ):
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # 1. Use native TF to get standard cross entropy per class slot
        # Ensure from_logits matches your model setup (True is highly recommended)
        # ops version is element wise [batch, classes], losses version averages
        # over classes giving [batch]
        bce = tf.keras.ops.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )

        # return bce
        # extra focal loss otherwise just return bce

        # 2. Calculate probabilities to build the focal weighting factor
        probs = tf.math.sigmoid(y_pred) if self.from_logits else y_pred

        # 3. Compute BirdNET-style focal weight: (1 - p_t)^gamma
        # if y_true is near 1, weight is (1 - prob). If y_true is near 0, weight is prob.
        focal_weight = y_true * tf.math.pow(1.0 - probs, 2.0) + (
            1.0 - y_true
        ) * tf.math.pow(probs, 2.0)

        # 4. Scale the standard BCE loss by the focal weights. Left per label
        # [batch, classes] so keras can apply per label sample weights (soft
        # labels weigh 0), keras's sum over batch size reduction then averages
        # over every label of every sample, the same as averaging over labels
        # then the batch
        return bce * focal_weight


if __name__ == "__main__":
    main()
