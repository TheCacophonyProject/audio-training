#!/usr/bin/env bash
# Train with the default datasets, any arguments are added to the end e.g.
# ./train.sh -w <path>/spectv7/val_pr_auc.weights.h5 --lr 1e-4 --epochs 10 spectv7Fine
set -euo pipefail

cd "$(dirname "$0")"

exec python3 audiomodel.py \
    --second-dataset-dir /data2/audio-training/audio-training/tier1/training-data/ \
    -d /data2/audio-training/audio-training/training-data/ \
    --noise-dirs \
        /data2/audio-training/audio-training/training-data/noise \
        /data/audio-data/other-datasets/noise-dataset/ambient-noise/ \
        /data/audio-data/other-datasets/noise-dataset/cacophony/noise/ \
        /data/audio-data/other-datasets/noise-dataset/esc-50-noise/ \
    --extra-dirs \
        /data/audio-data/other-datasets/animal-dataset/animal-sounds/ \
        /data/audio-data/other-datasets/animal-dataset/esc-50-animal/ \
    "$@"
