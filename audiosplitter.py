import os
import soundfile as sf
import requests
from pathlib import Path
import argparse
import librosa
import math
from datetime import timedelta
import datetime
import json


def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def load_recording(file, resample=48000):
    frames, sr = librosa.load(str(file), sr=resample)
    if resample is not None and resample != sr:
        print("resampling", sr, " too", resample)
        frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
        sr = resample
    return frames, sr


args = parse_args()
output_dir = Path("./rifleman/split")

file_path = Path("./rifleman")
target_sr = None
filenames = list(file_path.glob("*.mp3"))
filenames.sort()
# name_template = "2023 01 {} 12 {} 00"
for _, f in enumerate(filenames):
    # print(f.stem)
    # break
    date = os.path.getmtime(f)
    date = datetime.datetime.fromtimestamp(date)
    frames, sr = load_recording(f, target_sr)
    splits = (len(frames) / sr) / 60
    splits = int(math.ceil(splits))
    duration = len(frames) / sr
    last_s = duration - (splits - 1) * 60
    if last_s < 20:
        splits -= 1
    if splits <= 1:
        print("no split for", f)
        continue
    print("For duration", duration, " have x splits", splits, " last split is", last_s)
    # break
    # print(f, "for ", len(frames), " at ", sr, " have ", splits, " 1 minute recs")
    split_frames = 60 * sr
    for i in range(splits):
        if i == splits - 1:
            split_f = frames[split_frames * i :]
        else:
            split_f = frames[split_frames * i : split_frames * i + split_frames]
        print("split ", i, " have ", len(split_f), len(split_f) / sr)
        # continue
        output_f = output_dir / f"{f.stem}-{i}.mp3"
        meta_file = f.with_suffix(".txt")
        with open(str(meta_file), "r") as t:
            # add in some metadata stats
            meta = json.load(t)

        sf.write(f"{output_f}", split_f, sr)
        print("writing to  ", f"{output_f}")
        meta["duration"] = len(split_f) / sr
        meta_out = output_f.with_suffix(".txt")
        with open(meta_out, "w") as meta_f:
            json.dump(meta, meta_f, indent=4)
    # break
# print("got results", results)
