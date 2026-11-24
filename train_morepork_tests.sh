#!/usr/bin/env bash
# Tests of whether weak tier1 morepork clips cause the morepork false
# positives, the v11 noise label settings with:
#   A: tier1 moreporks birdnet doesn't call a morepork / owl dropped
#   B: A plus owls with no rms or 300Hz-1.2kHz signal dropped
#   C: B and moreporks never bird mixed
# Drop lists are from scripts/tier1_morepork_check.py. Any arguments are added
# to every run e.g. ./train_morepork_tests.sh --epochs 20
# Each run's output is also saved to logs/<run name>.txt
set -euo pipefail

cd "$(dirname "$0")"

EPOCHS=10
COMMON=(
    --epochs "$EPOCHS" --patience "$EPOCHS"
    --bird-mix-prob 0.5 --mixup-mode union-fixed --mixup-db-range -12 -6
    --noise-bank --noise-share --backup --add-noise-label
)

mkdir -p logs

# run name last, the output also goes to logs/<run name>.txt
run() {
    local name="${@: -1}"
    echo "Training $name, log logs/$name.txt"
    ./train.sh "$@" 2>&1 | tee "logs/$name.txt"
}

for f in tier1-morepork-drop-a.txt tier1-morepork-drop-b.txt; do
    if [[ ! -f "$f" ]]; then
        echo "Missing $f" >&2
        exit 1
    fi
done

run "${COMMON[@]}" --no-mix-labels kokako3 \
    --drop-tracks "$PWD/tier1-morepork-drop-a.txt" "$@" spectv12DropA

run "${COMMON[@]}" --no-mix-labels kokako3 \
    --drop-tracks "$PWD/tier1-morepork-drop-b.txt" "$@" spectv12DropB

run "${COMMON[@]}" --no-mix-labels kokako3 morepo2 \
    --drop-tracks "$PWD/tier1-morepork-drop-b.txt" "$@" spectv12DropBNoMix
