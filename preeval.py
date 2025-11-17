import json
import numpy as np
from pathlib import Path
from audiomodel import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import logging
import sys

import argparse

import pickle


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rms_file",
        default=None,
    )

    parser.add_argument(
        "--bird_file",
        default=None,
    )
    parser.add_argument(
        "--birdrms_file",
        default=None,
    )
    parser.add_argument(
        "out_file",
        default=None,
    )

    args = parser.parse_args()
    return args


def main():
    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )
    args = parse_args()
    if args.birdrms_file is not None:
        birdrms_file = Path(args.birdrms_file)
        rms_conf = None
        print("Loading bird model results", birdrms_file)
        with birdrms_file.open("rb") as f:
            bird_labels = np.load(f)
            bird_conf = np.load(f)
            bird_true = np.load(f)
            tracks = np.load(f)
            recs = np.load(f)

            starts = np.load(f)
            print("Tracks are", tracks)
            print("Recs are ", recs)

            try:
                model_type = np.load(f)
                print("Extra model", model_type)
                rms_conf = np.load(f)
                rms_labels = np.load(f)

            except:
                pass
        bird_tracks = np.arange(len(bird_conf))
        if rms_conf is not None:
            rms_tracks = bird_tracks
            rms_true = bird_true
            rms_pred = np.argmax(rms_conf, axis=1)
        print(bird_conf.shape)
        bird_pred = np.argmax(bird_conf, axis=1)
        bird_all_conf = bird_conf
        # print("BIrd pre is ",bird_pred.shape,bird_tracks.shape)
        # 1/0

    else:
        rms_stats = Path(args.rms_file)
        bird_stats = Path(args.bird_file)

        print("Loading bird model results", bird_stats)
        with bird_stats.open("rb") as f:
            bird_tracks = np.load(f)
            bird_true = np.load(f)
            bird_pred = np.load(f)
            bird_conf = np.load(f)
            bird_labels = np.load(f)

        bird_raw_conf = bird_stats.parent / f"{bird_stats.stem}-confidences.pkl"
        with bird_raw_conf.open("rb") as f:
            bird_all_conf = pickle.load(f)
        print(bird_labels)

        print("Loading pre model results", rms_stats)
        with rms_stats.open("rb") as f:
            rms_tracks = np.load(f)
            rms_true = np.load(f)
            rms_pred = np.load(f)
            rms_conf = np.load(f)
            rms_labels = np.load(f)
    out_file = Path(args.out_file)

    # best_threshold(
    #     rms_labels,
    #     rms_true,
    #     rms_pred,
    #     rms_conf,
    #     out_file.parent / f"{out_file.stem}-rms-threshold",
    # )
    # return
    data_per_track = {}
    # for track, y_t, y_p, y_c in zip(rms_tracks, rms_true, rms_pred, rms_conf):
    #     data_per_track[track] = (y_t, y_p, y_c)

    bird_data_per_track = {}
    for track, y_t, y_p, y_c, all_preds in zip(
        bird_tracks, bird_true, bird_pred, bird_conf, bird_all_conf
    ):
        # meaned = np.mean(all_preds, axis=0)
        # # print("Meaned ",meaned.shape,y_c.shape,meaned,y_c)
        # assert np.all(np.isclose(meaned, y_c))
        bird_data_per_track[track] = (y_t, y_p, y_c, np.array(all_preds))

    # best_threshold(
    #     bird_labels,
    #     bird_true,
    #     bird_pred,
    #     bird_conf,
    #     out_file.parent / f"{out_file.stem}-threshold",
    # )
    y_true = []
    preds = []
    threshold = 0.7
    labels = list(bird_labels)
    # labels.append("bird")

    # labels.append("human")
    # labels.append("noise")
    # labels.append("None")

    thresholds = [
        0.8,
        90.4,
        0.0,
        0.0,
        62.1,
        0.0,
        87.7,
        1.1,
        30.7,
        0.0,
        0.0,
        0.0,
        30.5,
        0.0,
        93.6,
        70.2,
        2.0,
        30.9,
        77.7,
        0.0,
        8.6,
        72.4,
        3.0,
        89.3,
        55.0,
        0.0,
        75.7,
        1.3,
        0.0,
        14.5,
        87.8,
        19.6,
        0.0,
        37.5,
        0.0,
        0.0,
        89.7,
        35.3,
        0.0,
        3.8,
        24.2,
        0.4,
        0.0,
        0.2,
        0.0,
        0.1,
        22.5,
        83.0,
        2.2,
        32.7,
        96.8,
        0.0,
        49.6,
        0.0,
        0.0,
        99.9,
        29.6,
        0.0,
        18.8,
        0.0,
        0.0,
        0.0,
        30.8,
        8.6,
        0.0,
        0.0,
        0.0,
    ]
    thresholds = np.array(thresholds)
    thresholds = thresholds / 100
    thresholds[thresholds < 0.5] = 0.5
    thresholds[thresholds > 0.9] = 0.9
    # print(thresholds)

    pre_thresh = np.array([0.0, 61.3, 16.2, 92.2, 72.7, 0.0])
    pre_thresh = pre_thresh / 100
    pre_thresh[pre_thresh < 0.5] = 0.5
    pre_thresh[pre_thresh > 0.9] = 0.9
    # noise_i = labels.index("noise")
    # human_i = labels.index("human")
    labels.append("None")

    morepork_i = labels.index("morepo2")
    for track_id, data in bird_data_per_track.items():
        y_t, y_p, y_c, all_preds = data
        # if y_p == 0:
        #     print("Bittern")
        #     1/0
        if track_id not in data_per_track:
            # print("Track id missing", track_id)
            rms_c = None
        else:
            rms_t, rms_p, rms_c = data_per_track[track_id]

        y_true.append(y_t)
        y_p = np.argmax(y_c)
        threshold = thresholds[y_p]
        threshold = max(0.5, threshold)
        threshold = min(0.9, threshold)
        threshold = 0.7
        # print(labels[y_p], "Threshold is ", threshold)
        is_morepork = y_p == morepork_i
        # if is_morepork and

        if rms_c is not None:
            pre_p = np.argmax(rms_c)
            pre_t = pre_thresh[pre_p]
            if rms_c[pre_p] >= pre_t:
                rms_l = rms_labels[pre_p]
                pre_p = labels.index(rms_l)
                pre_p = pre_p
            else:
                pre_p = None
        else:
            pre_p = None

        if y_c[y_p] < threshold:
            y_p = len(labels) - 1
            # if
            is_bird = False
            bird_c = 0
            other_c = 0
            # for p in all_preds:
            p = np.max(all_preds, axis=0)

            p_max = np.argmax(p)
            lbl = labels[p_max]
            # print("Checking ",lbl,p[p_max])
            # if lbl in ["noise", "human", "insect"]:
            #     thresh = 0.7
            # else:
            #     thresh = thresholds[p_max]
            #     thresh = 0.5
            # if p[p_max] >= thresh:
            #     if lbl in ["noise", "human", "insect"]:
            #         other_c += 1
            #     else:
            #         bird_c += 1
            # is_bird = bird_c > 0 and other_c == 0
            # if is_bird:
            #     print(
            #         "have bird ",
            #         bird_c,
            #         " other c",
            #         other_c,
            #         " for ",
            #         labels[y_t],
            #         is_bird,
            #     )
            # is_bird=False
            if pre_p is not None:
                y_p = pre_p
        elif (
            is_morepork and pre_p is not None and (pre_p == noise_i or pre_p == human_i)
        ):
            print(
                "Pre doesnt get morepork so using pre label ",
                bird_labels[pre_p],
                " true is ",
                labels[y_t],
            )
            y_p = pre_p
        preds.append(y_p)
        # break

    for track_id, data in data_per_track.items():
        y_t, y_p, y_c = data
        if track_id in bird_data_per_track:
            continue
        rms_l = rms_labels[y_t]
        # if rms_l == "human":
        # 1/0
        y_t = labels.index(rms_l)
        y_true.append(y_t)
        y_p = np.argmax(y_c)
        threshold = pre_thresh[y_p]
        rms_l = rms_labels[y_p]
        pre_p = labels.index(rms_l)

        if y_c[y_p] < threshold:
            pre_p = len(labels) - 1

        preds.append(pre_p)
        # break

    preds = np.array(preds)
    y_true = np.array(y_true)
    cm = confusion_matrix(y_true, preds, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    filename = out_file.parent / f"{out_file.stem}-combined"
    plt.savefig(filename.with_suffix(".png"), format="png")
    np.save(str(filename.with_suffix(".npy")), cm)
    return

    y_true = []
    preds = []
    threshold = 0.7
    labels = list(bird_labels)
    labels.append("bird")
    labels.append("noise")
    labels.append("None")
    for track_id, data in bird_data_per_track.items():
        y_t, y_p, y_c = data
        y_true.append(y_t)
        y_p = np.argmax(y_c)

        if y_c[y_p] < threshold:
            y_p = len(labels) - 1
        preds.append(y_p)
        # break

    preds = np.array(preds)
    y_true = np.array(y_true)
    cm = confusion_matrix(y_true, preds, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    filename = out_file.parent / f"{out_file.stem}-birds"
    plt.savefig(filename.with_suffix(".png"), format="png")
    np.save(str(filename.with_suffix(".npy")), cm)

    labels = list(rms_labels)
    labels.append("None")

    y_true = []
    preds = []
    for track_id, data in data_per_track.items():
        y_t, y_p, y_c = data
        y_true.append(y_t)
        y_p = np.argmax(y_c)

        if y_c[y_p] < threshold:
            y_p = len(labels) - 1
        preds.append(y_p)
        # break

    preds = np.array(preds)
    y_true = np.array(y_true)
    cm = confusion_matrix(y_true, preds, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    filename = out_file.parent / f"{out_file.stem}-noise"
    plt.savefig(filename.with_suffix(".png"), format="png")
    np.save(str(filename.with_suffix(".npy")), cm)


def best_threshold(labels, y_true, y_pred, confidences, filename):
    from sklearn.metrics import precision_recall_curve, RocCurveDisplay

    from sklearn.preprocessing import LabelBinarizer

    print("Y_true is ", y_true.shape)
    print("Y_pred is ", y_pred.shape)
    print("Confidences ", confidences.shape)

    label_binarizer = LabelBinarizer().fit(y_true)
    y_onehot_test = label_binarizer.transform(y_true)
    thresholds_best = []
    for i, class_of_interest in enumerate(labels):
        print("Class ", class_of_interest)
        lbl_mask = y_true == i
        if len(y_true[lbl_mask]) == 0:
            thresholds_best.append(0)
            continue
        binary_true = np.uint8(lbl_mask)

        if len(confidences.shape) == 1:
            # just best lbl confidence
            lbl_pred = confidences.copy()
            lbl_pred[~lbl_mask] = 0
        else:
            lbl_pred = confidences[:, i]
            # print("CHooisng all of this labl", lbl_pred)
        print("plt show for", class_of_interest)

        precision, recall, thresholds = precision_recall_curve(binary_true, lbl_pred)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)

        scatters = []
        for t_i, th in enumerate(thresholds):
            if th >= 0.6 and len(scatters) == 0:
                scatters.append((t_i, th))
            if th >= 0.7 and len(scatters) == 1:
                scatters.append((t_i, th))
            if th >= 0.8 and len(scatters) == 2:
                scatters.append((t_i, th))
                break
        no_skill = len(binary_true[lbl_mask]) / len(binary_true)

        plt.plot(recall, precision, marker=".", label="Logistic")
        plt.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
        plt.axis("square")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Recall vs Precision - {labels[i]}")
        plt.legend()
        plt.scatter(recall[ix], precision[ix], marker="o", color="black", label="Best")

        colours = ["red", "yellow", "green"]
        for point, colour in zip(scatters, colours):
            plt.scatter(
                recall[point[0]],
                precision[point[0]],
                marker="o",
                color=colour,
                label=f"TX {point[1]}",
            )
            print("plotted ", point, " with colour ", colour)
        label_f = filename.parent / f"{filename.stem}-{labels[i]}.png"
        plt.savefig(label_f, format="png")
        plt.clf()
        print("Best Threshold=%f, F-Score=%.3f" % (thresholds[ix], fscore[ix]))
        thresholds_best.append(thresholds[ix])

    thresholds = np.array(thresholds_best)
    logging.info(
        "ALl thresholds are %s mean %s median %s",
        np.round(100 * thresholds, 1),
        np.mean(thresholds),
        np.median(thresholds),
    )


if __name__ == "__main__":
    main()
