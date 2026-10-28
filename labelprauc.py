# Print per label pr auc for checkpoint weights on the validation (or test) set
# e.g. python3 labelprauc.py train/checkpoints/spectv4 -d /data2/audio-training/audio-training/training-data/
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from tfdataset import get_dataset

# imported so their custom layers are registered for load_model
import badwinner2
import tfpcen


def init_logging():
    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Checkpoint run dir e.g. train/checkpoints/spectv4")
    parser.add_argument("-d", "--dataset-dir", required=True, help="Dataset directory")
    parser.add_argument(
        "-w",
        "--weights",
        nargs="+",
        default=["val_pr_auc.weights.h5"],
        help="Weights files in the model dir",
    )
    parser.add_argument(
        "--split", default="validation", help="Dataset split to use e.g. validation, test"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for precision recall"
    )
    return parser.parse_args()


def load_split(meta, labels, dataset_dir, split):
    # same label mapping and audio processing as training
    dataset, _, _, _, _ = get_dataset(
        dataset_dir / split,
        labels,
        batch_size=64,
        excluded_labels=meta.get("excluded_labels"),
        remapped_labels=meta.get("remapped_labels"),
        extra_label_map=meta.get("extra_label_map"),
        extra_datasets=meta.get("extra_datasets"),
        noise_dirs=meta.get("noise_dirs"),
        noise_eval_fraction=meta.get("noise_eval_fraction", 0.1),
        use_generic_bird=meta.get("use_generic_bird", False),
        multi_label=meta.get("multi_label", True),
        pcen=meta.get("pcen", False),
        embeddings=meta.get("name") == "embeddings",
        model_name=meta.get("name"),
        load_seperate_ds=True,
        shuffle=False,
        deterministic=True,
        cache=False,
    )
    return dataset


def predict(model, dataset):
    y_true = []
    y_pred = []
    for x, y in dataset:
        if isinstance(y, (tuple, list)):
            y = y[0]
        y_true.append(y.numpy())
        y_pred.append(model.predict_on_batch(x))
    return np.concatenate(y_true).astype(np.float32), np.concatenate(y_pred)


def label_stats(y_true, y_pred, labels, threshold):
    # same metric as val_pr_auc in training, but one per label
    stats = []
    for i, label in enumerate(labels):
        pr_auc = tf.keras.metrics.AUC(curve="PR")
        pr_auc.update_state(y_true[:, i], y_pred[:, i])
        positives = y_true[:, i] > 0
        predicted = y_pred[:, i] >= threshold
        tp = np.sum(positives & predicted)
        precision = tp / max(np.sum(predicted), 1)
        recall = tp / max(np.sum(positives), 1)
        stats.append(
            (label, int(np.sum(positives)), float(pr_auc.result()), precision, recall)
        )
    return stats


def main():
    init_logging()
    args = parse_args()
    model_dir = Path(args.model_dir)
    with (model_dir / "metadata.txt").open("r") as f:
        meta = json.load(f)
    labels = meta["ebird_labels"]

    # probability model so outputs are sigmoid not logits
    model = tf.keras.models.load_model(
        str(model_dir / f"{model_dir.name}.keras"), compile=False
    )
    dataset = load_split(meta, labels, Path(args.dataset_dir), args.split)

    for weights in args.weights:
        logging.info("Loading weights %s", model_dir / weights)
        model.load_weights(model_dir / weights)
        y_true, y_pred = predict(model, dataset)
        stats = label_stats(y_true, y_pred, labels, args.threshold)

        print(f"\n{model_dir.name} {weights} on {args.split} ({len(y_true)} samples)")
        print(
            f"{'label':<10} {'positives':>9} {'pr_auc':>7} "
            f"{'precision':>9} {'recall':>7}  @{args.threshold}"
        )
        for label, count, pr_auc, precision, recall in sorted(
            stats, key=lambda s: s[2]
        ):
            print(
                f"{label:<10} {count:>9} {pr_auc:>7.3f} {precision:>9.3f} {recall:>7.3f}"
            )
        print(f"{'mean':<10} {'':>9} {np.mean([s[2] for s in stats]):>7.3f}")

        # samples with nothing labelled e.g. noise, any prediction is a false positive
        empty = np.sum(y_true, axis=1) == 0
        if np.any(empty):
            fp = np.any(y_pred[empty] >= args.threshold, axis=1)
            print(
                f"\nEmpty samples {np.sum(empty)} with a prediction >= "
                f"{args.threshold}: {np.sum(fp)} ({np.mean(fp):.3f})"
            )
            fp_counts = np.sum(y_pred[empty] >= args.threshold, axis=0)
            for i in np.argsort(-fp_counts):
                if fp_counts[i] == 0:
                    break
                print(f"  {labels[i]:<10} {fp_counts[i]}")


if __name__ == "__main__":
    main()
