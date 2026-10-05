import argparse
import csv
import json
import logging
import warnings
from collections import Counter
from pathlib import Path

import librosa
import numpy as np

from birdnetcompare import GENERIC_BIRD_TAGS, load_taxonomy, rms_window, tag_taxon

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
    "track_start",
    "track_end",
    "rms_start",
    "rms_end",
    "perch_start",
    "perch_end",
]


def shard_paths(out_dir):
    """Completed shards in order. Names are zero padded but sort on the number
    anyway, a shard only counts once its csv exists as that is written last."""
    shards = [
        p.with_suffix(".npy")
        for p in Path(out_dir).glob("*.csv")
        if p.stem.isdigit() and p.with_suffix(".npy").exists()
    ]
    return sorted(shards, key=lambda p: int(p.stem))


def load_shard_rows(npy):
    with open(npy.with_suffix(".csv"), newline="") as f:
        return list(csv.DictReader(f))


def load_embeddings(out_dir):
    """All embeddings as one (N, 1536) array with a csv row per embedding, in
    shard order so row i of the csv is embedding i. To go through more than fits
    in memory, loop over shard_paths and np.load each shard instead."""
    embeddings = []
    rows = []
    for npy in shard_paths(out_dir):
        embeddings.append(np.load(npy))
        rows.extend(load_shard_rows(npy))
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.concatenate(embeddings), rows


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


def embed(model, y, start):
    first = int(start * PERCH_SR)
    chunk = y[first : first + PERCH_LENGTH * PERCH_SR]
    # recordings shorter than 5s are padded at the end
    chunk = np.pad(chunk, (0, PERCH_LENGTH * PERCH_SR - len(chunk)))
    embeddings = np.asarray(model.embed(chunk).embeddings, dtype=np.float32)
    # one 5s window gives a single frame, average anything else down to 1536
    return embeddings.reshape(-1, embeddings.shape[-1]).mean(axis=0)


def analyse(args):
    """Flags tracks whose label doesn't fit their embedding. Uses a logistic
    regression probe with out of fold probabilities, folds grouped by recording
    so tracks from the same recording never help predict each other, and the
    labels of each track's nearest neighbours from other recordings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.neighbors import NearestNeighbors

    print("Loading from ", args.out)
    embeddings, rows = load_embeddings(args.out)
    if not rows:
        logging.info("No embeddings in %s", args.out)
        return
    x = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    tags = [set(r["manual_tags"].split(";")) for r in rows]
    groups = np.array([r["recording_id"] or r["file"] for r in rows])

    # train the probe on single tag tracks of labels with enough examples,
    # multi tag tracks and rare labels are still scored
    counts = Counter(next(iter(t)) for t in tags if len(t) == 1)

    classes = sorted(l for l, c in counts.items() if c >= args.min_label_count)
    class_index = {l: i for i, l in enumerate(classes)}
    train = np.array(
        [i for i, t in enumerate(tags) if len(t) == 1 and next(iter(t)) in class_index]
    )
    logging.info(
        "%s tracks, probe trained on %s tracks of %s labels with at least %s tracks",
        len(rows),
        len(train),
        len(classes),
        args.min_label_count,
    )

    probs = None
    if len(classes) >= 2:
        y = np.array([class_index[next(iter(tags[i]))] for i in train])
        probs = np.zeros((len(rows), len(classes)), dtype=np.float32)

        def probe():
            return LogisticRegression(max_iter=1000, class_weight="balanced")

        folds = min(args.folds, len(set(groups[train])))
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
        for fold_train, fold_test in splitter.split(x[train], y, groups[train]):
            model = probe().fit(x[train][fold_train], y[fold_train])
            # a fold can miss a rare class, map its columns back
            probs[train[fold_test][:, None], model.classes_] = model.predict_proba(
                x[train][fold_test]
            )
        others = np.setdiff1d(np.arange(len(rows)), train)
        if len(others):
            model = probe().fit(x[train], y)
            probs[others[:, None], model.classes_] = model.predict_proba(x[others])
        accuracy = np.mean(probs[train].argmax(axis=1) == y)
        logging.info("Out of fold probe accuracy %.3f", accuracy)
    else:
        logging.info("Need at least 2 labels with enough tracks for the probe")

    # neighbours from the same recording sound alike so are skipped
    neighbours = NearestNeighbors(n_neighbors=min(len(rows), args.knn * 4 + 1)).fit(x)
    _, nearest = neighbours.kneighbors(x)

    results = []
    for i, row in enumerate(rows):
        result = dict(row)
        result["label"] = ";".join(sorted(tags[i]))

        own_prob = None
        result.update(probe_own_prob="", probe_label="", probe_prob="")
        if probs is not None:
            known = [class_index[t] for t in tags[i] if t in class_index]
            if known:
                own_prob = float(probs[i, known].max())
                result["probe_own_prob"] = round(own_prob, 3)
            top = int(probs[i].argmax())
            result["probe_label"] = classes[top]
            result["probe_prob"] = round(float(probs[i, top]), 3)

        others = [n for n in nearest[i] if groups[n] != groups[i]][: args.knn]
        agreement = None
        result.update(knn_agreement="", knn_label="", knn_label_fraction="")
        if others:
            agreement = sum(1 for n in others if tags[n] & tags[i]) / len(others)
            neighbour_tags = Counter(t for n in others for t in tags[n])
            knn_label, knn_count = neighbour_tags.most_common(1)[0]
            result["knn_agreement"] = round(agreement, 3)
            result["knn_label"] = knn_label
            result["knn_label_fraction"] = round(knn_count / len(others), 3)

        scores = [s for s in (own_prob, agreement) if s is not None]
        result["doubt"] = round(1 - sum(scores) / len(scores), 3) if scores else ""
        # both have to disagree, either alone is too noisy
        result["flagged"] = bool(scores) and all(
            s < t
            for s, t in (
                (own_prob, args.probe_threshold),
                (agreement, args.knn_threshold),
            )
            if s is not None
        )
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
        "--analyse",
        action="store_true",
        help="Flag doubtful labels from the saved embeddings instead of embedding",
    )
    parser.add_argument("--flags-out", default="perch-flags.csv")
    parser.add_argument(
        "--min-label-count",
        type=int,
        default=10,
        help="Labels with fewer single tag tracks aren't trained on by the probe",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--knn", type=int, default=10, help="Neighbours to compare")
    parser.add_argument(
        "--probe-threshold",
        type=float,
        default=0.2,
        help="Flag when the out of fold probability of the track's label is below this",
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
                for track in meta.get("tracks", [])
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
                manual = [
                    t for t in track.get("tags", []) if t.get("automatic") is False
                ]
                # bird tracks use bird_rms, others noise_rms, same as audiodataset
                bird_track = any(
                    tag_taxon(t, taxonomy) is not None or t["what"] in GENERIC_BIRD_TAGS
                    for t in manual
                )
                window = rms_window(track, meta, bird_track, args.segment_length)
                start, end = perch_window(window, track, duration)
                embeddings.append(embed(model, y, start))
                rows.append(
                    {
                        "index": total,
                        "file": str(txt),
                        "recording_id": meta.get("id"),
                        "track_id": track.get("id"),
                        "manual_tags": ";".join(t["what"] for t in manual),
                        "track_start": track.get("start"),
                        "track_end": track.get("end"),
                        "rms_start": "" if window is None else round(window[0], 2),
                        "rms_end": "" if window is None else round(window[1], 2),
                        "perch_start": round(start, 2),
                        "perch_end": round(end, 2),
                    }
                )
                total += 1
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
