import matplotlib.pyplot as plt
from identifytracks import get_tracks_from_signals, signal_noise, get_end, Signal


from functools import partial

from multiprocessing import Pool
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import itertools
import logging
import numpy as np
from birdsconfig import (
    NOISE_LABELS,
    BIRD_TRAIN_LABELS,
    ALL_BIRDS,
    EXTRA_LABELS,
    OTHER_LABELS,
    HUMAN_LABELS,
)
from pathlib import Path


def evaluate_weakly_labelled_dir(model, model_meta, dir_name, filename):
    filename = Path("./confusions") / filename

    predicted_categories = []
    y_true = []
    labels = model_meta["ebird_labels"]

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

    audio_files = []
    for sub_dir in dir_name.iterdir():
        if sub_dir.is_file():
            print("Ignoring ", sub_dir)
        if sub_dir.name !="kokako3":
            continue
        files = [sub_f for sub_f in sub_dir.iterdir() if sub_f.is_file()]
        audio_files.extend(files)
    print("Audio_files are ", audio_files)
    # audio_files = audio_files[1:2]
    # meta_data_f = meta_data_f[:1]
    pre_fn = partial(preprocess_weakly_lbl_audio, labels=include_labels)
    total_count = len(audio_files)
    count = 0

    predicted_mean = []
    predicted_counts = []
    confidences = []
    track_ids = []
    all_pred_confidences = []
    file_names = []
    with Pool(processes=1) as pool:
        for result in pool.imap_unordered(pre_fn, audio_files, chunksize=8):
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
                    file_names.append(filename)

                    if len(samples) == 0:
                        logging.info("No samples for track %s", track)
                        continue
                    filtered_tracks.append(track)
                    filtered_samples.extend(samples)
                    counts.append(len(samples))
                if len(filtered_samples) == 0:
                    logging.info("No samples for %s", file_name)
                    continue

                all_samples = np.array(filtered_samples)
                all_samples = np.repeat(all_samples, 3, -1)
                predictions = model.predict(all_samples)
                offset = 0
                threshold = 0.7
                logging.info("FOr %s", file_name)
                for track, count in zip(tracks, counts):
                    track_preds = predictions[offset : offset + count]
                    logging.info("Track preds is %s",track_preds.shape)
                    for i, p in enumerate(track_preds):

                        max_i = np.argmax(p)
                        max_p = p[max_i]
                        if max_p >= threshold:
                            logging.info("At %s have %s with %s",i, labels[max_i],round(100*max_p))
                    track_pred = np.mean(track_preds, axis=0)
                    confidences.append(track_pred)
                    track_ids.append(track.id)
                    max_i = np.argmax(track_pred)
                    max_p = track_pred[max_i]
                    logging.info("Mean max arg is %s with %s", labels[max_i], round(100*max_p))

                    if max_p > 0.7:
                        predicted_mean.append(max_i)
                    else:
                        predicted_mean.append(len(labels) - 1)
                    all_pred_confidences.append(track_preds)

                    # count of each label
                    arg_max = np.argmax(track_preds, axis=1)
                    rows = np.arange(len(track_preds))
                    prob_max = track_preds[rows, arg_max]
                    over_thresh = prob_max >= threshold
                    args_over_thresh = arg_max[over_thresh]
                    if len(args_over_thresh) == 0:
                        predicted_counts.append(len(labels) - 1)
                    else:
                        counts = np.bincount(args_over_thresh)
                        for i,c in enumerate(counts):
                            if c > 0:
                                logging.info("%s: %s times ",labels[i], c)
                        max_i = np.argmax(counts)
                        max_c = counts[max_i]
                        predicted_counts.append(max_i)

                    if track.tag in remapped:
                        lbl_i = remapped[track.tag]
                    else:
                        lbl_i = labels.index(track.tag)
                    y_true.append(lbl_i)
                    offset += count
            except:
                logging.error("Could not process %s", file_name, exc_info=True)

    predicted_counts = np.array(predicted_counts)
    confidences = np.array(confidences)
    track_ids = np.array(track_ids)
    file_names = np.array(file_names)
    npy_file = filename.parent / f"{filename.stem}-raw.npy"
    with npy_file.open("wb") as f:
        np.save(f, file_names)
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
    np.save(str(cm_file.with_suffix(".npy")), cm)

    print("Saving to ", filename)
    cm_file = filename.parent / f"{filename.stem}-counts"
    cm = confusion_matrix(y_true, predicted_counts, labels=np.arange(len(labels)))
    figure = plot_confusion_matrix(cm, class_names=labels)
    plt.savefig(cm_file.with_suffix(".png"), format="png")
    np.save(str(cm_file.with_suffix(".npy")), cm)


from audiowriter import load_recording
from predict_utils import load_samples
from audiodataset import Recording


def preprocess_weakly_lbl_audio(audio_f, labels=None):
    try:

        frames, sr = load_recording(audio_f)
        end = get_end(frames, sr)
        frames = frames[: int(sr * end)]

        ebird_id = audio_f.parent.name
        meta = {}
        meta["file"] = str(audio_f)
        tag = {"automatic": False, "what": ebird_id, "ebird_id": ebird_id}
        track_meta = {
            "id": f"{0}_1",
            "start": 0,
            "end": end,
            "tags": [tag],
        }
        meta["tracks"] = [track_meta]
        print(meta)
        rec = Recording(meta, audio_f, None, False)

        tracks = [track for track in rec.tracks if track.tag in labels]
        if len(tracks) == 0:
            return None
        samples = load_samples(frames, sr, tracks)
        for s, t in zip(samples, tracks):
            if len(s) == 0:
                logging.error(
                    "No samples for some track %s from %s",
                    t.id,
                    audio_f,
                    exc_info=True,
                )
                return None
    except:
        logging.error("Could not load audio for %s", audio_f, exc_info=True)
        return None
    return audio_f, tracks, samples


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
