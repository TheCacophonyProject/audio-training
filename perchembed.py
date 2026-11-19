import argparse
import csv
import json
import logging
import math
import warnings
from collections import Counter
from pathlib import Path

import librosa
import numpy as np

from birdnetcompare import (
    is_bird_track,
    load_taxonomy,
    primary_tags,
    rms_window,
    secondary_tags,
)
from birdsconfig import RELABEL_MAP
from utils import ID_PREFIX_DIRS, get_ebird_id, get_label_to_ebird_map, prefixed_id

AUDIO_EXTS = [".flac", ".wav", ".m4a", ".mp3", ".ogg"]
# perch v2 takes 5s of 32kHz audio
PERCH_SR = 32000
PERCH_LENGTH = 5
FIELDNAMES = [
    "index",
    "file",
    "recording_id",
    "track_id",
    "manual_tags",
    "secondary_tags",
    "rec_datetime",
    "track_start",
    "track_end",
    "rms_start",
    "rms_end",
    "perch_start",
    "perch_end",
]


def shard_paths(out_dir, recursive=False):
    """Completed shards in order. Names are zero padded but sort on the number
    anyway, a shard only counts once its csv exists as that is written last.
    Recursive also finds shards in sub directories, ordered by directory."""
    csvs = Path(out_dir).rglob("*.csv") if recursive else Path(out_dir).glob("*.csv")
    shards = [
        p.with_suffix(".npy")
        for p in csvs
        if p.stem.isdigit() and p.with_suffix(".npy").exists()
    ]
    return sorted(shards, key=lambda p: (str(p.parent), int(p.stem)))


def load_shard_rows(npy):
    with open(npy.with_suffix(".csv"), newline="") as f:
        return list(csv.DictReader(f))


def load_embeddings(out_dir, recursive=False):
    """All embeddings as one (N, 1536) array with a csv row per embedding, in
    shard order so row i of the csv is embedding i. To go through more than fits
    in memory, loop over shard_paths and np.load each shard instead. Each row
    gets the directory its shard came from."""
    embeddings = []
    rows = []
    for npy in shard_paths(out_dir, recursive):
        embeddings.append(np.load(npy))
        for row in load_shard_rows(npy):
            row["shard_dir"] = str(npy.parent)
            rows.append(row)
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.concatenate(embeddings), rows


def prefer_dir(embeddings, rows, prefer):
    """Drops rows of tracks that are also in the shards of the prefer directory,
    given as a path or just the directory name, so each track comes from one
    directory only"""
    prefer = Path(prefer)

    def preferred(row):
        shard_dir = Path(row["shard_dir"])
        return shard_dir.name == str(prefer) or shard_dir.resolve() == prefer.resolve()

    in_preferred = {(r["file"], r["track_id"]) for r in rows if preferred(r)}
    keep = [
        i
        for i, r in enumerate(rows)
        if preferred(r) or (r["file"], r["track_id"]) not in in_preferred
    ]
    logging.info(
        "%s tracks in %s, dropped %s embeddings of them from other directories",
        len(in_preferred),
        prefer,
        len(rows) - len(keep),
    )
    return embeddings[keep], [rows[i] for i in keep]


def save_shard(out_dir, number, embeddings, rows):
    # write to temp names then rename so a crash never leaves a partial shard
    npy = Path(out_dir) / f"{number:05d}.npy"
    tmp_npy = npy.with_name(npy.stem + ".tmp.npy")
    np.save(tmp_npy, np.stack(embeddings))
    tmp_npy.replace(npy)
    tmp_csv = npy.with_name(npy.stem + ".tmp.csv")
    with open(tmp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp_csv.replace(npy.with_suffix(".csv"))
    logging.info("Saved shard %s with %s embeddings", npy, len(rows))


def recording_id(txt, meta):
    """The txt's id, prefixed the same as audiodataset's Recording.id for
    Train_001 and Train_002 so build.py can match rows to recordings"""
    rec_id = prefixed_id(txt, meta.get("id"))
    return None if rec_id is None else str(rec_id)


def rec_datetime(meta):
    return meta.get("rec_datetime") or meta.get("recordingDateTime") or ""


def find_audio(txt, meta):
    for ext in AUDIO_EXTS:
        audio = txt.with_suffix(ext)
        if audio.exists():
            return audio
    original = meta.get("file")
    if original and Path(original).exists():
        return Path(original)
    return None


def perch_window(window, track, duration):
    # centre a perch length window on the best rms window, kept inside the
    # recording where possible
    if window is None:
        centre = (track.get("start", 0) + track.get("end", 0)) / 2
    else:
        centre = (window[0] + window[1]) / 2
    start = centre - PERCH_LENGTH / 2
    start = max(0, min(start, duration - PERCH_LENGTH))
    return start, start + PERCH_LENGTH


def track_windows(track, duration):
    """Perch length windows covering the whole track, the last one ends at the
    track end so it overlaps the one before rather than running past the track"""
    start = track.get("start", 0)
    end = min(track.get("end", start), duration)
    if end - start <= PERCH_LENGTH:
        return [perch_window(None, track, duration)]
    # rounded so float error doesn't add a window e.g. 10.0000001s
    count = math.ceil(round((end - start) / PERCH_LENGTH, 3))
    starts = [start + i * PERCH_LENGTH for i in range(count - 1)]
    starts.append(end - PERCH_LENGTH)
    return [(s, s + PERCH_LENGTH) for s in starts]


def embed(model, y, start):
    first = int(start * PERCH_SR)
    chunk = y[first : first + PERCH_LENGTH * PERCH_SR]
    # recordings shorter than 5s are padded at the end
    chunk = np.pad(chunk, (0, PERCH_LENGTH * PERCH_SR - len(chunk)))
    embeddings = np.asarray(model.embed(chunk).embeddings, dtype=np.float32)
    # one 5s window gives a single frame, average anything else down to 1536
    return embeddings.reshape(-1, embeddings.shape[-1]).mean(axis=0)


def analyse(args):
    """Flags tracks whose label doesn't fit their embedding. Uses a multi label
    probe, a logistic regression per label, with out of fold probabilities,
    folds grouped by recording so tracks from the same recording never help
    predict each other, and the labels of each track's nearest neighbours from
    other recordings."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neighbors import NearestNeighbors

    print("Loading from ", args.out)
    embeddings, rows = load_embeddings(args.out, args.recursive)
    if not rows:
        logging.info("No embeddings in %s", args.out)
        return
    if args.prefer:
        embeddings, rows = prefer_dir(embeddings, rows, args.prefer)
    x = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    # compare as ebird ids so tags and probe classes from datasets with
    # different names match, tags without one like "bird" are kept as is.
    # Then merged as in training e.g. wind is noise, parakeets are parake
    ebird_map = get_label_to_ebird_map()

    def to_label(tag):
        ebird_id = get_ebird_id(tag, ebird_map)
        return RELABEL_MAP.get(ebird_id, ebird_id)

    def parse_tags(value):
        return {to_label(t) for t in value.split(";") if t}

    tags = [parse_tags(r["manual_tags"]) for r in rows]
    # secondary tags are treated the same as primary tags, in tier1 the rarer
    # of a clip's labels is made primary so secondary doesn't mean quieter.
    # Shards made before they were saved have none
    secondary = [parse_tags(r.get("secondary_tags", "")) for r in rows]
    if not args.include_bird:
        # generic bird says nothing about the species, drop the tag and tracks
        # left with no other tag so they aren't scored or used as neighbours
        tags = [t - {"bird"} for t in tags]
        secondary = [t - {"bird"} for t in secondary]
        keep = [i for i, (t, s) in enumerate(zip(tags, secondary)) if t or s]
        logging.info(
            "Skipping %s windows only tagged bird, use --include-bird to keep them",
            len(rows) - len(keep),
        )
        rows = [rows[i] for i in keep]
        tags = [tags[i] for i in keep]
        secondary = [secondary[i] for i in keep]
        x = x[keep]
    groups = np.array([r["recording_id"] or r["file"] for r in rows])
    # the track's own tags, primary and secondary
    heard = [t | s for t, s in zip(tags, secondary)]
    # everything in a window, the track's own tags plus those of entries of
    # the same recording overlapping it. The probe trains on these and the
    # checks below accept any of them
    overlap_tags = window_tags(rows, heard, groups)
    logging.info(
        "%s of %s windows overlap an entry with other tags",
        sum(len(w) > len(t) for w, t in zip(overlap_tags, heard)),
        len(rows),
    )

    probs = None
    classes = []
    class_index = {}
    if args.load_probe:
        # score with a probe trained on other embeddings e.g. the rms windows,
        # so no out of fold training here
        saved = joblib.load(args.load_probe)
        if "thresholds" not in saved:
            raise ValueError(
                f"{args.load_probe} is a single label probe, retrain it with --save-probe"
            )
        classes = list(saved["classes"])
        # tags are relabelled before training so the classes should already be,
        # if not RELABEL_MAP has changed since the probe was saved
        if [to_label(c) for c in classes] != classes:
            raise ValueError(
                f"{args.load_probe} classes don't match the current RELABEL_MAP, retrain it"
            )
        class_index = {l: i for i, l in enumerate(classes)}
        thresholds = np.asarray(saved["thresholds"], dtype=np.float32)
        probs = saved["model"].predict_proba(x).astype(np.float32)
        logging.info(
            "Scoring %s windows with probe %s of %s labels",
            len(rows),
            args.load_probe,
            len(classes),
        )
    else:
        # labels heard in enough windows, rarer labels are still scored by knn
        counts = Counter(t for w in overlap_tags for t in w)
        classes = sorted(l for l, c in counts.items() if c >= args.min_label_count)
        class_index = {l: i for i, l in enumerate(classes)}
        if len(classes) >= 2:
            y = multi_hot(overlap_tags, class_index)
            logging.info(
                "Probe trained on %s windows, %s with more than one label, of %s "
                "labels heard in at least %s windows",
                len(rows),
                int((y.sum(axis=1) > 1).sum()),
                len(classes),
                args.min_label_count,
            )

            def probe():
                # an independent logistic regression per label, balanced
                # between its positive and negative windows
                return OneVsRestClassifier(
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                    n_jobs=-1,
                )

            # multi hot labels can't be stratified, use each window's rarest
            # label so rare labels are spread over the folds
            rarest = np.array(
                [
                    min((counts[t], class_index[t]) for t in w if t in class_index)[1]
                    if w & class_index.keys()
                    else len(classes)
                    for w in overlap_tags
                ]
            )
            probs = np.zeros(y.shape, dtype=np.float32)
            folds = min(args.folds, len(set(groups)))
            splitter = StratifiedGroupKFold(
                n_splits=folds, shuffle=True, random_state=0
            )
            with warnings.catch_warnings():
                # a fold can miss a rare label, it's then predicted as never
                warnings.filterwarnings("ignore", message="Label .* is present in all")
                warnings.filterwarnings("ignore", message="Label not .* is present")
                for fold_train, fold_test in splitter.split(x, rarest, groups):
                    model = probe().fit(x[fold_train], y[fold_train])
                    probs[fold_test] = model.predict_proba(x[fold_test])
            thresholds = best_thresholds(y, probs)
            if args.save_probe:
                model = probe().fit(x, y)
                joblib.dump(
                    {"model": model, "classes": classes, "thresholds": thresholds},
                    args.save_probe,
                )
                logging.info(
                    "Saved probe trained on all %s windows to %s",
                    len(rows),
                    args.save_probe,
                )
        else:
            logging.info("Need at least 2 labels with enough windows for the probe")
            classes, class_index = [], {}
    if probs is not None:
        log_probe_metrics(multi_hot(overlap_tags, class_index), probs, thresholds, classes)

    # neighbours from the same recording sound alike so are skipped
    neighbours = NearestNeighbors(n_neighbors=min(len(rows), args.knn * 4 + 1)).fit(x)
    _, nearest = neighbours.kneighbors(x)

    results = []
    for i, row in enumerate(rows):
        result = dict(row)
        result["label"] = ";".join(sorted(tags[i]))
        # relabelled the same as label, secondary_tags is as in the txt
        result["secondary_label"] = ";".join(sorted(secondary[i]))
        result["window_label"] = ";".join(sorted(overlap_tags[i]))

        own_prob = None
        doubtful = None
        result.update(
            probe_own_prob="", probe_doubtful="", probe_labels="", probe_missing=""
        )
        if probs is not None:
            # labels are scored independently so each of the track's own tags,
            # primary and secondary, is checked, the least likely one is its
            # probe score
            own = [class_index[t] for t in heard[i] if t in class_index]
            if own:
                own_prob = float(probs[i, own].min())
                result["probe_own_prob"] = round(own_prob, 3)
                doubtful = [
                    classes[c]
                    for c in own
                    if probs[i, c]
                    < (
                        thresholds[c]
                        if args.probe_threshold is None
                        else args.probe_threshold
                    )
                ]
                result["probe_doubtful"] = ";".join(sorted(doubtful))
            predicted = np.flatnonzero(probs[i] >= thresholds)
            predicted = predicted[np.argsort(-probs[i, predicted])]
            result["probe_labels"] = ";".join(
                f"{classes[c]}:{probs[i, c]:.2f}" for c in predicted
            )
            # predicted but not tagged on anything in the window, possibly a
            # missing tag
            result["probe_missing"] = ";".join(
                classes[c] for c in predicted if classes[c] not in overlap_tags[i]
            )

        others = [n for n in nearest[i] if groups[n] != groups[i]][: args.knn]
        agreement = None
        result.update(knn_agreement="", knn_label="", knn_label_fraction="")
        if others:
            agreement = sum(1 for n in others if heard[n] & overlap_tags[i]) / len(
                others
            )
            neighbour_tags = Counter(t for n in others for t in heard[n])
            knn_label, knn_count = neighbour_tags.most_common(1)[0]
            result["knn_agreement"] = round(agreement, 3)
            result["knn_label"] = knn_label
            result["knn_label_fraction"] = round(knn_count / len(others), 3)

        scores = [s for s in (own_prob, agreement) if s is not None]
        result["doubt"] = round(1 - sum(scores) / len(scores), 3) if scores else ""
        # both have to disagree, either alone is too noisy
        checks = []
        if doubtful is not None:
            checks.append(bool(doubtful))
        if agreement is not None:
            checks.append(agreement < args.knn_threshold)
        result["flagged"] = bool(checks) and all(checks)
        results.append(result)

    results.sort(key=lambda r: -1 if r["doubt"] == "" else -r["doubt"])
    with open(args.flags_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logging.info(
        "Wrote %s tracks, %s flagged, to %s",
        len(results),
        sum(r["flagged"] for r in results),
        args.flags_out,
    )

    logging.info(
        "Per label (tracks / flagged / mean own probe prob / mean knn agreement):"
    )
    per_label = {}
    for r in results:
        per_label.setdefault(r["label"], []).append(r)
    for label, label_rows in sorted(per_label.items(), key=lambda kv: -len(kv[1])):

        def mean(key):
            values = [r[key] for r in label_rows if r[key] != ""]
            return f"{np.mean(values):.2f}" if values else "-"

        logging.info(
            "  %s: %s / %s / %s / %s",
            label,
            len(label_rows),
            sum(r["flagged"] for r in label_rows),
            mean("probe_own_prob"),
            mean("knn_agreement"),
        )


def multi_hot(tag_sets, class_index):
    y = np.zeros((len(tag_sets), len(class_index)), dtype=np.int8)
    for i, tag_set in enumerate(tag_sets):
        for tag in tag_set:
            if tag in class_index:
                y[i, class_index[tag]] = 1
    return y


def best_thresholds(y, probs):
    """Per label the probability threshold with the best F1, from out of fold
    probabilities. Rare labels usually need a lower one than common labels"""
    from sklearn.metrics import precision_recall_curve

    thresholds = np.full(y.shape[1], 0.5, dtype=np.float32)
    for c in range(y.shape[1]):
        if not y[:, c].any():
            continue
        precision, recall, cuts = precision_recall_curve(y[:, c], probs[:, c])
        # the last precision and recall have no threshold
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
            precision[:-1] + recall[:-1], 1e-10
        )
        thresholds[c] = cuts[np.argmax(f1)]
    return thresholds


def log_probe_metrics(y, probs, thresholds, classes):
    """Average precision and F1 at each label's threshold, worst labels first"""
    from sklearn.metrics import average_precision_score, f1_score

    present = np.flatnonzero(y.any(axis=0))
    if not len(present):
        logging.info("No windows tagged with the probe's labels to score it on")
        return
    predicted = probs >= thresholds
    scores = {
        c: (
            average_precision_score(y[:, c], probs[:, c]),
            f1_score(y[:, c], predicted[:, c]),
        )
        for c in present
    }
    logging.info(
        "Probe mean average precision %.3f, mean F1 %.3f over %s labels",
        np.mean([ap for ap, _ in scores.values()]),
        np.mean([f1 for _, f1 in scores.values()]),
        len(present),
    )
    logging.info("Per label (average precision / F1 / threshold / windows):")
    for c, (ap, f1) in sorted(scores.items(), key=lambda kv: kv[1][0]):
        logging.info(
            "  %s: %.3f / %.3f / %.2f / %s",
            classes[c],
            ap,
            f1,
            thresholds[c],
            int(y[:, c].sum()),
        )


def window_tags(rows, tags, groups):
    """Each row's tags plus the tags of the other rows of the same recording
    overlapping its perch window e.g. a blackbird calling behind a fantail. A
    row with a best rms window overlaps when that window's centre is inside the
    perch window, full track rows have none so overlap when their perch windows
    do. tags should hold primary and secondary tags so all of both rows' tags
    are combined"""
    by_recording = {}
    for i, group in enumerate(groups):
        by_recording.setdefault(group, []).append(i)

    result = [None] * len(rows)
    for members in by_recording.values():
        members = np.array(members)
        starts = np.array([float(rows[i]["perch_start"]) for i in members])
        ends = np.array([float(rows[i]["perch_end"]) for i in members])
        has_rms = np.array(
            [rows[i].get("rms_start", "") != "" and rows[i].get("rms_end", "") != "" for i in members]
        )
        centres = np.array(
            [
                (float(rows[i]["rms_start"]) + float(rows[i]["rms_end"])) / 2
                if rms
                else np.nan
                for i, rms in zip(members, has_rms)
            ]
        )
        for i, start, end in zip(members, starts, ends):
            overlaps = np.where(
                has_rms,
                (centres >= start) & (centres <= end),
                (starts < end) & (ends > start),
            )
            window = set(tags[i])
            for other in members[overlaps]:
                window |= tags[other]
            result[i] = window
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Perch v2 embeddings of the best rms window of each manually tagged track"
    )
    parser.add_argument(
        "dir",
        nargs="?",
        help="Directory to search for metadata .txt files, not needed with --analyse",
    )
    parser.add_argument(
        "--out",
        default="perch-embeddings",
        help="Directory for shards, 00000.npy embeddings with 00000.csv rows etc. "
        "Re-running skips tracks already in it",
    )
    parser.add_argument(
        "--shard-size", type=int, default=10000, help="Embeddings per shard"
    )
    parser.add_argument(
        "--taxonomy",
        default=str(Path(__file__).parent / "eBird_taxonomy_v2024.csv"),
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=3,
        help="Length of the best rms window, the perch window is centred on it",
    )
    parser.add_argument(
        "--full-track",
        action="store_true",
        help="Embed the whole track in back to back perch length windows instead "
        "of one window on the best rms window, use a separate --out",
    )
    parser.add_argument(
        "--analyse",
        action="store_true",
        help="Flag doubtful labels from the saved embeddings instead of embedding",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="With --analyse also load shards from sub directories of --out",
    )
    parser.add_argument(
        "--prefer",
        help="With --recursive, a directory (path or name) whose embeddings are "
        "used for tracks that are in more than one directory",
    )
    parser.add_argument(
        "--save-probe",
        help="With --analyse save the probe, trained on all the training tracks, here",
    )
    parser.add_argument(
        "--load-probe",
        help="With --analyse score with this saved probe instead of training one",
    )
    parser.add_argument("--flags-out", default="perch-flags.csv")
    parser.add_argument(
        "--min-label-count",
        type=int,
        default=10,
        help="Labels heard in fewer windows aren't trained on by the probe",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--include-bird",
        action="store_true",
        help="With --analyse also analyse generic bird tags, by default they are "
        "dropped and tracks only tagged bird are skipped",
    )
    parser.add_argument("--knn", type=int, default=10, help="Neighbours to compare")
    parser.add_argument(
        "--probe-threshold",
        type=float,
        help="Flag when the out of fold probability of one of the track's tags "
        "is below this, by default each label's own threshold with the best F1",
    )
    parser.add_argument(
        "--knn-threshold",
        type=float,
        default=0.3,
        help="and the fraction of neighbours sharing its label is below this",
    )
    args = parser.parse_args()
    if args.analyse:
        analyse(args)
        return
    if args.dir is None:
        parser.error("dir is needed to make embeddings")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # resume, skip tracks already embedded and carry on numbering
    done = set()
    shards = shard_paths(out_dir)
    for npy in shards:
        done.update((r["file"], r["track_id"]) for r in load_shard_rows(npy))
    next_shard = int(shards[-1].stem) + 1 if shards else 0
    total = len(done)
    if done:
        logging.info("Resuming, %s tracks already in %s shards", total, len(shards))

    taxonomy = load_taxonomy(args.taxonomy)
    logging.info("Loading perch v2")
    from perch_hoplite.zoo import model_configs

    model = model_configs.load_model_by_name("perch_v2")

    embeddings = []
    rows = []
    files = 0
    try:
        for txt in sorted(Path(args.dir).rglob("*.txt")):
            try:
                with open(txt) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(meta, dict):
                continue
            tracks = [
                track
                for track in meta.get("Tracks", [])
                if any(t.get("automatic") is False for t in track.get("tags", []))
                and (str(txt), str(track.get("id"))) not in done
            ]
            if not tracks:
                continue
            audio = find_audio(txt, meta)
            if audio is None:
                logging.info("No audio for %s", txt)
                continue
            try:
                y, _ = librosa.load(audio, sr=PERCH_SR, mono=True)
            except Exception:
                logging.exception("Could not load %s", audio)
                continue
            duration = len(y) / PERCH_SR

            for track in tracks:
                # primary and secondary tags are saved separately
                manual = primary_tags(track)
                secondary = secondary_tags(track)
                if args.full_track:
                    window = None
                    perch_windows = track_windows(track, duration)
                else:
                    # bird tracks use bird_rms, others noise_rms, same as audiodataset
                    bird_track = is_bird_track(manual, taxonomy)
                    window = rms_window(track, meta, bird_track, args.segment_length)
                    perch_windows = [perch_window(window, track, duration)]
                for start, end in perch_windows:
                    embeddings.append(embed(model, y, start))
                    rows.append(
                        {
                            "index": total,
                            "file": str(txt),
                            "recording_id": recording_id(txt, meta),
                            "track_id": track.get("id"),
                            "manual_tags": ";".join(t["what"] for t in manual),
                            "secondary_tags": ";".join(t["what"] for t in secondary),
                            "rec_datetime": rec_datetime(meta),
                            "track_start": track.get("start"),
                            "track_end": track.get("end"),
                            "rms_start": "" if window is None else round(window[0], 2),
                            "rms_end": "" if window is None else round(window[1], 2),
                            "perch_start": round(start, 2),
                            "perch_end": round(end, 2),
                        }
                    )
                    total += 1
                # checked per track so a track's windows are never split across
                # shards, resuming skips whole tracks
                if len(rows) >= args.shard_size:
                    save_shard(out_dir, next_shard, embeddings, rows)
                    next_shard += 1
                    embeddings, rows = [], []
            files += 1
            if files % 100 == 0:
                logging.info("Embedded %s tracks from %s files", total, files)
    finally:
        # also save on ctrl-c or an error so the run can be resumed
        if rows:
            save_shard(out_dir, next_shard, embeddings, rows)
    logging.info("%s embeddings in %s", total, out_dir)


if __name__ == "__main__":
    # m4a falls back to audioread which warns on every file
    warnings.filterwarnings("ignore", message="PySoundFile failed")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
