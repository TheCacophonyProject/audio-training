#!/usr/bin/env bash
# Copy every .txt metadata file under a remote dir (and its sub dirs) into one
# flat local dir, over ssh with tar so only the wanted files are sent.
#   scripts/fetch_meta.sh [--audio] user@host /remote/dir /local/dir
# --audio also copies each recording's audio (.m4a .flac .wav .mp3), which
# build_auto_noise.py needs beside the metadata.
# Files with the same name in different sub dirs overwrite each other.
set -euo pipefail

audio=0
if [[ "${1:-}" == "--audio" ]]; then
    audio=1
    shift
fi
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 [--audio] user@host /remote/dir /local/dir" >&2
    exit 1
fi
host=$1
src=$2
dest=$3

if [[ $audio -eq 1 ]]; then
    match="\\( -name '*.txt' -o -name '*.m4a' -o -name '*.flac' -o -name '*.wav' -o -name '*.mp3' \\)"
else
    match="-name '*.txt'"
fi

mkdir -p "$dest"
ssh "$host" "cd $(printf %q "$src") && find . -type f $match -print0 | tar --null -T - -cf -" \
    | tar -xf - -C "$dest" --transform='s|.*/||'
echo "Copied $(find "$dest" -maxdepth 1 -name '*.txt' | wc -l) metadata files to $dest"
