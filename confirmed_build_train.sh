#!/usr/bin/env bash
# From an embed_analyse.sh run of a model, writes each tier1 track's confirmed
# windows into its metadata, builds tier1 (with them) and cacophony, then
# trains.
#   1 best_windows.py          each tag's best probe / knn window, from
#                              tier1-probe-<model name>.csv
#   2 compare_best_windows.py  against each track's best rms window (needs the
#                              tier1 metadata)
#   3 write_best_windows.py    confirmed-tracks, primary_starts,
#                              secondary_starts, probe_starts and knn_starts
#                              into the tier1 metadata, every label unless
#                              FILTER_LABELS=1 (only the model's labels, others'
#                              primaries keep their rms window)
#   4 build.py tier1           --best-segment --confirmed-tracks
#   5 build.py cacophony       as build_datasets.sh
#   6 train.sh
#
#   ./confirmed_build_train.sh <model .keras> [train name]
# e.g.
#   ./confirmed_build_train.sh \
#       train/checkpoints/spectv12AutoNoiseClean/spectv12AutoNoiseClean.keras
# trains spectv12AutoNoiseCleanConfirmed by default.
#
# Paths can be overridden with environment variables as in build_datasets.sh,
# BUILD_ARGS go to both builds, WRITE_ARGS to write_best_windows.py e.g.
# WRITE_ARGS="--exempt tomtit1,comred". WRITE_DRY_RUN=1 stops after counting
# what step 3 would write. Each task writes its output to its own .txt file in
# LOG_DIR. Back up the tier1 metadata before running, step 3 edits it in place
# (re-running replaces what it wrote).
set -euo pipefail

cd "$(dirname "$0")"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <model .keras> [train name]" >&2
    exit 1
fi
MODEL=$1
NAME=$(basename "$MODEL" .keras)
TRAIN_NAME=${2:-${NAME}Confirmed}
MODEL_META=$(dirname "$MODEL")/metadata.txt
FILTER_LABELS=${FILTER_LABELS:-}
if [[ -n "$FILTER_LABELS" && ! -f "$MODEL_META" ]]; then
    echo "No $MODEL_META to read the model's labels from" >&2
    exit 1
fi

# audio and metadata .txt files
TIER1_DIR=${TIER1_DIR:-/data2/audio-training/tier1}
CACOPHONY_DIR=${CACOPHONY_DIR:-/data/audio-data/audio-rec}
# perch flags from build_datasets.sh, not remade here
TIER1_FLAGS=${TIER1_FLAGS:-/data2/audio-training/perchembeddings/perch-flags.csv}
CACOPHONY_FLAGS=${CACOPHONY_FLAGS:-/data2/audio-training/perchembeddings-cacophony/perch-flags.csv}
# build.py writes training-data/ in these
TIER1_OUT=${TIER1_OUT:-/data2/audio-training/audio-training/tier1}
CACOPHONY_OUT=${CACOPHONY_OUT:-/data2/audio-training/audio-training}
BUILD_ARGS=${BUILD_ARGS:-}
WRITE_ARGS=${WRITE_ARGS:-}
WRITE_DRY_RUN=${WRITE_DRY_RUN:-}
TRAIN_WEIGHTS=${TRAIN_WEIGHTS:-train/checkpoints/spectv11NoiseLbael/val_pr_auc.weights.h5}

# from embed_analyse.sh
PROBE_CSV=tier1-probe-$NAME.csv
BEST_CSV=tier1-best-windows-$NAME.csv
COMPARE_CSV=tier1-best-windows-vs-rms-$NAME.csv
for f in "$PROBE_CSV" "$TIER1_FLAGS" "$CACOPHONY_FLAGS" "$TRAIN_WEIGHTS"; do
    if [[ ! -f "$f" ]]; then
        echo "No $f" >&2
        exit 1
    fi
done

LOG_DIR=${LOG_DIR:-logs/confirmed-build-train-$(date '+%Y%m%d-%H%M%S')}
mkdir -p "$LOG_DIR"
echo "Logs in $LOG_DIR"

run() {
    # run a task with its output in $LOG_DIR/<name>.txt, stop if it fails
    local name=$1 description=$2
    shift 2
    local log=$LOG_DIR/$name.txt
    echo "==== $(date '+%F %T') $description, log $log"
    if ! "$@" &> "$log"; then
        echo "FAILED $name, last lines of $log:"
        tail -n 20 "$log"
        exit 1
    fi
}

LABEL_ARGS=()
if [[ -n "$FILTER_LABELS" ]]; then
    # only the model's labels get confirmed windows
    LABELS=$LOG_DIR/labels.txt
    python3 -c '
import json, sys
print("\n".join(json.load(open(sys.argv[1]))["labels"]))
' "$MODEL_META" > "$LABELS"
    echo "$(wc -l < "$LABELS") labels from $MODEL_META in $LABELS"
    LABEL_ARGS=(--labels "$LABELS")
fi

run best-windows "Best windows of $PROBE_CSV into $BEST_CSV" \
    python3 best_windows.py "$PROBE_CSV" --out "$BEST_CSV"
run compare "Comparing $BEST_CSV with the best rms windows into $COMPARE_CSV" \
    python3 compare_best_windows.py "$BEST_CSV" --out "$COMPARE_CSV"
# shellcheck disable=SC2086 # WRITE_ARGS is split into arguments on purpose
run write-dry-run "Counting what would be written to $TIER1_DIR" \
    python3 write_best_windows.py "$COMPARE_CSV" "${LABEL_ARGS[@]}" --dry-run \
    $WRITE_ARGS
if [[ -n "$WRITE_DRY_RUN" ]]; then
    echo "==== WRITE_DRY_RUN set, stopping, counts in $LOG_DIR/write-dry-run.txt"
    exit 0
fi
# shellcheck disable=SC2086
run write "Writing confirmed tracks into $TIER1_DIR" \
    python3 write_best_windows.py "$COMPARE_CSV" "${LABEL_ARGS[@]}" $WRITE_ARGS

# shellcheck disable=SC2086 # BUILD_ARGS is split into arguments on purpose
run tier1-build "Building tier1 with confirmed tracks into $TIER1_OUT" \
    python3 build.py -d "$TIER1_DIR" --best-segment --confirmed-tracks \
    --perch-flags "$TIER1_FLAGS" $BUILD_ARGS "$TIER1_OUT"
# noise only samples go to training-data/noise/<split>, train's --noise-dirs
# uses it
# shellcheck disable=SC2086
run cacophony-build "Building cacophony into $CACOPHONY_OUT" \
    python3 build.py -d "$CACOPHONY_DIR" --perch-flags "$CACOPHONY_FLAGS" \
    $BUILD_ARGS "$CACOPHONY_OUT"

TRAIN_LOG=logs/$TRAIN_NAME.txt
echo "==== $(date '+%F %T') Training $TRAIN_NAME, log $TRAIN_LOG"
./train.sh -w "$TRAIN_WEIGHTS" --lr 1e-4 \
    --epochs 25 --patience 25 "$TRAIN_NAME" \
    --bird-mix-prob 0.5 --mixup-mode union-fixed --mixup-db-range -12 -6 \
    --noise-bank --noise-share --backup --add-noise-label --auto-noise-fraction 0.3 \
    --no-mix-labels kokako3 morepo2 \
    --drop-tracks "$PWD/tier1-drop-b-tomtit-morepork.txt" \
    --noise-dirs \
        /data2/audio-training/audio-training/training-data/noise \
        /data2/audio-training/audio-training/training-data/noise/auto-noise \
        /data/audio-data/other-datasets/noise-dataset/ambient-noise/ \
        /data/audio-data/other-datasets/noise-dataset/esc-50-noise/ \
    &> "$TRAIN_LOG"

echo "==== $(date '+%F %T') Done, logs in $LOG_DIR and $TRAIN_LOG"
