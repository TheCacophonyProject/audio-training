#!/usr/bin/env bash
# Perch embeddings and flags for tier1 and the cacophony data, then builds
# both datasets with build.py using them.
#   tier1      perch on each track's best rms window, built with --best-segment
#   cacophony  perch on whole tracks (--full-track), built without --best-segment
# both split noise only samples into training-data/noise (build.py's default)
# Paths can be overridden with environment variables e.g.
#   TIER1_DIR=/other/tier1 ./build_datasets.sh
# Extra build.py arguments go in BUILD_ARGS e.g. BUILD_ARGS="--filter-dry-run"
# Each task writes its output to its own .txt file in LOG_DIR
set -euo pipefail

cd "$(dirname "$0")"

# audio and metadata .txt files
TIER1_DIR=${TIER1_DIR:-/data2/audio-training/tier1}
CACOPHONY_DIR=${CACOPHONY_DIR:-/data/audio-data/audio-rec}
# perch shards (00000.npy / 00000.csv ...), flags csv and probe, each dataset
# in its own folder. tier1 is best rms window embeddings, cacophony full track
TIER1_PERCH=${TIER1_PERCH:-/data2/audio-training/perchembeddings}
CACOPHONY_PERCH=${CACOPHONY_PERCH:-/data2/audio-training/perchembeddings-cacophony}
# build.py writes training-data/ in these
TIER1_OUT=${TIER1_OUT:-/data2/audio-training/audio-training/tier1}
CACOPHONY_OUT=${CACOPHONY_OUT:-/data2/audio-training/audio-training}
BUILD_ARGS=${BUILD_ARGS:-}

TIER1_FLAGS=$TIER1_PERCH/perch-flags.csv
CACOPHONY_FLAGS=$CACOPHONY_PERCH/perch-flags.csv
# each task's output goes to its own .txt file in here
LOG_DIR=${LOG_DIR:-logs/build-datasets-$(date '+%Y%m%d-%H%M%S')}
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

perch() {
    # perch shards and flags for one dataset, extra arguments go to the embedding
    local name=$1 dir=$2 out=$3 flags=$4
    shift 4
    mkdir -p "$out"
    # shards from before the id changes get their recording and track ids,
    # secondary tags and rec_datetime updated, so resuming skips their tracks
    run "$name-update-csvs" "Updating existing shards in $out" \
        python3 perchupdatecsvs.py "$out"
    run "$name-embed" "Embedding $dir into $out $*" \
        python3 perchembed.py "$dir" --out "$out" "$@"
    run "$name-analyse" "Analysing $out into $flags" \
        python3 perchembed.py --analyse --out "$out" --flags-out "$flags" \
        --save-probe "$out/probe.joblib"
}

perch tier1 "$TIER1_DIR" "$TIER1_PERCH" "$TIER1_FLAGS"
# shellcheck disable=SC2086 # BUILD_ARGS is split into arguments on purpose
run tier1-build "Building tier1 into $TIER1_OUT" \
    python3 build.py -d "$TIER1_DIR" --best-segment --perch-flags "$TIER1_FLAGS" \
    $BUILD_ARGS "$TIER1_OUT"

perch cacophony "$CACOPHONY_DIR" "$CACOPHONY_PERCH" "$CACOPHONY_FLAGS" --full-track
# noise only samples go to training-data/noise/<split> for mixing in as
# background (build.py's default), train.sh's --noise-dirs uses it
# shellcheck disable=SC2086
run cacophony-build "Building cacophony into $CACOPHONY_OUT" \
    python3 build.py -d "$CACOPHONY_DIR" --perch-flags "$CACOPHONY_FLAGS" \
    $BUILD_ARGS "$CACOPHONY_OUT"

echo "==== $(date '+%F %T') Done, logs in $LOG_DIR"
