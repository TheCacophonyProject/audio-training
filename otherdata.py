import librosa
import csv
import logging
import sys
from pathlib import Path
from audiodataset import Track, Recording, AudioDataset, RELABEL, AudioSample, Config
from build import split_randomly, validate_datasets
import psutil
import random
from audiomentations import AddBackgroundNoise, PolarityInversion, Compose
import soundfile as sf
import random
import audioread.ffdec  # Use ffmpeg decoder
import numpy as np
from multiprocessing import Pool
from identifytracks import signal_noise, get_tracks_from_signals, get_end, Signal
import argparse
import matplotlib.pyplot as plt


def load_recording(file, resample=48000):
    try:
        # librosa.load(file) giving strange results
        aro = audioread.ffdec.FFmpegAudioFile(file)
        frames, sr = librosa.load(aro, sr=None)
        aro.close()
        if resample is not None and resample != sr:
            frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
            sr = resample
        return frames, sr
    except:
        logging.error("Could not load %s", file, exc_info=True)
        # for some reason the original exception causes docker to hang
        raise Exception(f"Could not load {file}")


# csv_files = ["./ff10/ff1010bird_metadata.csv"]
csv_files = [
    "/data/audio-data/warblrb10k_public/warblrb10k_public_metadata.csv",
    "/data//audio-data/ff1010bird/ff1010bird_metadata.csv",
]


out_dir = Path("./other-data")
from audiowriter import create_tf_records
import json


chime_labels = {
    "c": "human",
    "m": "human",
    "f": "human",
    "v": "video-game",
    "p": "noise",
    "b": "noise",
    "o": "other",
    "S": "silence",
    "U": "unknown",
}
NOISE_LABELS = ["wind", "vehicle", "dog", "rain", "static", "noise", "cat"]

NOISE_PATH = []
BIRD_PATH = []
signals = Path("./signal-data/train")
bad_signals = signals.parent / "bad-train"
bad_signals.mkdir(parents=True, exist_ok=True)

wavs = list(signals.glob("*.wav"))
for w in wavs:
    if "bird" in w.stem:
        # frames, sr = load_recording(w)
        # if frames is None:
        #    w.rename(bad_signals/w.name)
        #    continue
        # if len(frames) / sr < 4:
        #    logging.info("skipping %s", w)
        #    continue
        BIRD_PATH.append(w)
    else:
        for noise in NOISE_LABELS:
            if noise in w.stem:
                # frames, sr = load_recording(w)
                # if frames is None:
                #    w.rename(bad_signals/w.name)

                #   continue
                # if len(frames) / sr < 4:
                #    logging.info("skipping %s", w)
                #    continue
                NOISE_PATH.append(w)
                break

# BIRD_LABELS = ["bird"]
# NOISE_LABELS = []
# NOISE_PATH = NOISE_PATH[:2]
# BIRD_PATH = BIRD_PATH[:2]


def process_noise():
    noisy_p = Path(
        "/data/audio-data/Flickr-Audio-Caption-Corpus/flickr_audio/noisy-wavs"
    )
    # noisy_p = Path("./flickr/noisy-wavs")
    if noisy_p.is_dir():
        logging.info("Clearing dir %s", noisy_p)
        for child in noisy_p.glob("*"):
            if child.is_file():
                child.unlink()
    noisy_p.mkdir(parents=True, exist_ok=True)

    p = Path("/data/audio-data/Flickr-Audio-Caption-Corpus/flickr_audio/wavs")
    # p = Path("./flickr/wavs")

    wav_files = list(p.glob("*.wav"))
    random.shuffle(wav_files)
    num_noisy = len(wav_files) // 2
    # wav_files = wav_files[:num_noisy]
    logging.info("adding noise to %s", len(wav_files))
    count = 0
    with Pool(processes=8, initializer=worker_init) as pool:
        [0 for x in pool.imap_unordered(mix_noise, wav_files, chunksize=8)]
    # pool.wait()
    logging.info("Finished adding noise for %s", len(wav_files))


add_noise = None
add_bird = None
count = 0


def worker_init():
    global add_noise
    global add_bird
    global count
    random.shuffle(NOISE_PATH)
    add_noise = AddBackgroundNoise(
        sounds_path=NOISE_PATH,
        min_snr_in_db=3.0,
        max_snr_in_db=30.0,
        noise_transform=PolarityInversion(),
        p=1,
    )
    random.shuffle(BIRD_PATH)

    add_bird = AddBackgroundNoise(
        sounds_path=BIRD_PATH,
        min_snr_in_db=3.0,
        max_snr_in_db=30.0,
        noise_transform=PolarityInversion(),
        p=1,
    )


def mix_noise(w):
    global add_noise
    global add_bird
    noisy_p = Path(
        "/data/audio-data/Flickr-Audio-Caption-Corpus/flickr_audio/noisy-wavs"
    )
    # noisy_p = Path("./flickr/noisy-wavs")
    label = ""
    frames, sr = load_recording(w)
    if frames is None:
        return
    rand_f = np.random.rand()
    if rand_f > 0.5:
        frames = add_bird(frames, 48000)
        label = "bird"
    else:
        frames = add_noise(frames, 48000)
        label = "noise"
    name = noisy_p / f"{label}-{w.stem}.wav"
    sf.write(str(name), frames, 48000)
    global count
    count += 1
    if count % 50 == 0:
        logging.info("Saved %s", count)


# more aggressive track merging
# since we are only looking for one bird in these clips try reduce to one concurrent track
def merge_again(tracks):
    post_filter = []
    tracks_sorted = sorted(tracks, key=lambda track: track.start)
    current_track = None
    for t in tracks_sorted:
        if current_track is None:
            current_track = t
            post_filter.append(current_track)

            continue
        overlap = current_track.time_overlap(t)
        percent_overlap = overlap / t.length
        percent_overlap_2 = overlap / current_track.length

        f_overlap = current_track.mel_freq_overlap(t)
        f_percent_overlap = f_overlap / t.mel_freq_range

        if percent_overlap_2 > 0.5:
            post_filter = post_filter[: len(post_filter) - 1]
            post_filter.append(t)
            current_track = t
        elif percent_overlap > 0.5 or (percent_overlap > 0 and f_percent_overlap > 0.5):
            if f_percent_overlap > 0.5:
                current_track.end = max(current_track.end, t.end)
        else:
            # encountered another big track shall we just make this current track and allow some small overlap???
            current_track = t
            post_filter.append(current_track)

        # if  percent_overlap>0.5 or percent_overlap_2> 0.5:
        #     print("Current track ", current_track, " over laps ", t, " with ", overlap)
        #     print("Track 1 over lap is ",percent_overlap, " track 2 overlap is ",percent_overlap_2)
        # el
        if overlap <= 0:
            current_track = t
            post_filter.append(current_track)
    return post_filter


# def generate_tracks(wav_file):
#     min_freq = lbl_meta.get("min_freq")
#     max_freq = lbl_meta.get("max_freq")

#     # wav_file = args[0]
#     # clip_id = args[1]
#     meta_f = wav_file.with_suffix(".txt")
#     metadata = {}
#     if meta_f.exists():
#         with meta_f.open("r") as f:
#             metadata = json.load(f)

#     if "Tracks" in metadata:
#         return
#     frames, sr = load_recording(wav_file, None)

#     length = get_end(frames, sr)
#     signals = signal_noise(frames[: int(sr * length)], sr, 281)
#     signals = [
#         s
#         for s in signals
#         if (min_freq is None or s.freq_start > min_freq)
#         and (max_freq is None or s.freq_start < max_freq)
#     ]

#     tracks = get_tracks_from_signals(signals, length)
#     tracks = merge_again(tracks)
#     tracks_meta = []
#     for i, t in enumerate(tracks):

#         track_meta = t.get_meta()
#         track_meta["id"] = f"{wav_file.name}-{i}"
#         tag = {"automatic": False, "what": wav_file.parent.name}
#         track_meta["tags"] = [tag]
#         tracks_meta.append(track_meta)
#     metadata["Tracks"] = tracks_meta
#     metadata["id"] = metadata["additionalMetadata"]["xeno-id"]
#     metadata["duration"] = length
#     # metadata["location"] =
#     # could get some metadata  from xeno canto

#     with meta_f.open("w") as f:
#         json.dump(metadata, f, indent=4)


lbl_meta = None


def xeno_init(metadata):
    global lbl_meta
    lbl_meta = metadata


def weakly_lbled_data(base_dir):
    logging.info("Weakly labeled xeno data %s", base_dir)
    config = Config()
    dataset = AudioDataset("Xeno", config)
    base_dir = Path(base_dir)
    child_dirs = [f for f in base_dir.iterdir() if f.is_dir()]
    for lbl_dir in child_dirs:
        logging.info("Loading from %s", lbl_dir.name)

        meta_f = lbl_dir / f"{lbl_dir.name}.txt"
        metadata = {}
        if meta_f.exists():
            with meta_f.open("r") as f:
                metadata = json.load(f)
        wav_files = list(lbl_dir.glob("*.wav"))
        wav_files.extend(list(lbl_dir.glob("*.mp3")))

        with Pool(processes=8, initializer=xeno_init, initargs=(metadata,)) as pool:
            [0 for x in pool.imap_unordered(generate_tracks, wav_files, chunksize=8)]
    # FIRST_SECONDS = 5

    dataset.load_meta(base_dir)
    dataset.print_counts()

    dataset.samples = []
    for k, r in dataset.recs.items():

        # acceptable_tracks = [t.id for t in r.tracks if t.start < FIRST_SECONDS]
        # could filter some tracks by small freq bands
        r.samples = [s for s in r.samples if s.length > 2]
        if len(r.samples) == 0:
            print("No sample is first seconds for ", r.filename)
        dataset.samples.extend(r.samples)
    dataset.print_counts()

    train, validation, _ = split_randomly(dataset, no_test=True)
    validation.name = "test"

    all_labels = set()
    for d in [train, validation]:
        logging.info("")
        logging.info("%s Dataset", d.name)
        d.print_sample_counts()

        all_labels.update(d.labels)

    all_labels = list(all_labels)
    record_dir = base_dir / "xeno-training-data/"
    print("saving to", record_dir)

    dataset_counts = {}
    for dataset in [train, validation]:
        dir = record_dir / dataset.name
        print("saving to ", dir)
        create_tf_records(dataset, dir, all_labels, num_shards=100)
        dataset_counts[dataset.name] = dataset.get_counts()
        # dataset.saveto_numpy(os.path.join(base_dir))
    # dont need dataset anymore just need some meta
    meta_filename = record_dir / "training-meta.json"
    meta_data = {
        "labels": list(all_labels),
        "type": "audio",
        "counts": dataset_counts,
        "by_label": False,
        "relabbled": RELABEL,
    }
    meta_data.update(config.__dict__)
    with open(meta_filename, "w") as f:
        json.dump(meta_data, f, indent=4)


def flickr_data():
    config = Config()
    dataset = AudioDataset("Flickr", config)
    p = Path("/data/audio-data/Flickr-Audio-Caption-Corpus/flickr_audio/wavs")
    # p = Path("./flickr/wavs")

    wav_files = list(p.glob("*.wav"))
    noisy_p = Path(
        "/data/audio-data/Flickr-Audio-Caption-Corpus/flickr_audio/noisy-wavs"
    )
    # noisy_p = Path("./flickr/noisy-wavs")

    # noisy_wav_files.extend(list(p.glob("*.wav")))
    random.shuffle(wav_files)

    for rec_name in wav_files:
        rand_f = np.random.rand()
        added = False
        labels = ["human"]
        # if rand_f > 0.7:
        #     if rand_f > 0.85:
        #         noisy_name = noisy_p / f"bird-{rec_name.name}"
        #         if noisy_name.exists():
        #             add_rec(
        #                 dataset,
        #                 noisy_name,
        #                 ["human"],
        #                 config,
        #                 mixed_label="bird",
        #             )
        #             # logging.info("Adding %s from  %s", noisy_name, rec_name)
        #             added = True
        #             labels.append("bird")
        #     else:
        #         noisy_name = noisy_p / f"noise-{rec_name.name}"
        #         print("looking for %s", noisy_name)
        #         if noisy_name.exists():
        #             add_rec(dataset, noisy_name, ["human"], config, mixed_label="noise")
        #             # logging.info("Adding %s from  %s", noisy_name, rec_name)
        #             added = True
        #             labels.append("noise")
        if not added:
            add_rec(dataset, rec_name, labels, config)
        if len(dataset.recs) > len(wav_files) / 3:
            break

    logging.info("Loaded samples mem %s", psutil.virtual_memory()[2])
    dataset.print_counts()
    # return
    datasets = split_randomly(dataset, no_test=False)
    # for d in datasets:
    #     for r in d.recs:
    #         name = r.filename
    #         noisy_name = noisy_p / f"bird-{name.name}"
    #         if noisy_name.exists():
    #             add_rec(d, noisy_name, ["human", "bird"], config)
    #             logging.info("Adding %s %s %s", noisy_name, " from ", name)
    #         noisy_name = noisy_p / f"noise-{name.name}"
    #
    #         if noisy_name.exists():
    #             add_rec(dataset, noisy_name, ["human", "noise"], config)
    #             print("Adding %s %s %s", noisy_name, " from ", name)
    logging.info("Split samples mem %s", psutil.virtual_memory()[2])

    all_labels = set()
    for d in datasets:
        logging.info("%s Dataset", d.name)
        d.print_sample_counts()

        all_labels.update(d.labels)
    # return
    all_labels = list(all_labels)
    all_labels.sort()
    for d in datasets:
        d.labels = all_labels
    base_dir = Path("/data/audio-data/")
    record_dir = base_dir / "flickr-training-data/"
    print("saving to", record_dir)
    logging.info("Saving pre samples mem %s", psutil.virtual_memory()[2])

    dataset_counts = {}
    for dataset in datasets:
        dir = record_dir / dataset.name
        print("saving to ", dir)
        create_tf_records(dataset, dir, datasets[0].labels, num_shards=100)
        dataset_counts[dataset.name] = dataset.get_counts()
        # dataset.saveto_numpy(os.path.join(base_dir))
    # dont need dataset anymore just need some meta
    meta_filename = f"{base_dir}/flickr-training-data/training-meta.json"
    meta_data = {
        "labels": datasets[0].labels,
        "type": "audio",
        "counts": dataset_counts,
        "by_label": False,
        "relabbled": RELABEL,
    }
    meta_data.update(config.__dict__)
    with open(meta_filename, "w") as f:
        json.dump(meta_data, f, indent=4)


def add_rec(dataset, rec_name, labels, config, mixed_label=None):
    id = None
    id, id_2, speaker = rec_name.stem.split("_")
    id = f"{id}-{id_2}-{speaker}"
    r = Recording({"id": id, "tracks": []}, rec_name, config)
    tags = []
    for l in labels:
        tag = {"automatic": False, "what": l}
        tags.append(tag)
        r.human_tags.add(l)
    # try:
    #     y, sr = librosa.load(rec_name)
    #     end = librosa.get_duration(y=y, sr=sr)
    #     y = None
    #     sr = None
    # except:
    #     continue
    t = Track({"id": id, "start": 0, "end": None, "tags": tags}, rec_name, r.id, r)
    t.mixed_label = mixed_label
    # r.load_samples()
    r.tracks.append(t)
    sample = AudioSample(
        r, r.human_tags, 0, None, [t.id], 1, None, mixed_label=mixed_label
    )
    r.samples = [sample]
    dataset.add_recording(r)
    dataset.samples.extend(r.samples)
    for l in labels:
        dataset.labels.add(l)


#  Child speech
# m Adult male speech
# f Adult female speech
# v Video game/TV
# p Percussive sounds, e.g. crash, bang, knock, footsteps
# b Broadband noise, e.g. household appliances
# o Other identifiable sounds
# S Silence / background noise only
# U Flag chunk (unidentifiable sounds, not sure how to label)
# }
def chime_data():
    dataset = AudioDataset("Chime")
    p = Path("./chime")
    csv_files = list(p.glob("**/*.csv"))
    for file in csv_files:
        with open(file, newline="") as f:
            dreader = csv.reader(f, delimiter=",", quotechar="|")
            i = -1
            label = None
            id = None
            for row in dreader:
                key = row[0]
                if key == "majorityvote":
                    label = row[1]
                    # break
                elif key == "segmentname":
                    id = row[1]
                    # .48kHz.wav
            rec_name = file.parent / f"{file.stem}.48kHz.wav"
            id = file.stem
            r = Recording({"id": id, "tracks": []}, rec_name)
            tags = []
            for code in label:
                tags.append({"automatic": False, "what": chime_labels[code]})
            try:
                y, sr = librosa.load(rec_name)
                end = librosa.get_duration(y=y, sr=sr)
                y = None
                sr = None
            except:
                continue
            t = Track(
                {"id": id, "start": 0, "end": end, "tags": tags}, rec_name, r.id, r
            )
            r.load_samples()
            r.human_tags.add(chime_labels[code])
            r.tracks.append(t)
            dataset.add_recording(r)
            dataset.samples.extend(r.samples)
    logging.info("Loaded samples mem %s", psutil.virtual_memory()[2])
    dataset.print_counts()
    # return
    datasets = split_randomly(dataset, no_test=True)
    logging.info("Split samples mem %s", psutil.virtual_memory()[2])

    all_labels = set()
    for d in datasets:
        logging.info("%s Dataset", d.name)
        d.print_sample_counts()

        all_labels.update(d.labels)
    all_labels = list(all_labels)
    all_labels.sort()
    for d in datasets:
        d.labels = all_labels
    base_dir = Path(".")
    record_dir = base_dir / "chime-training-data/"
    print("saving to", record_dir)
    logging.info("Saving pre samples mem %s", psutil.virtual_memory()[2])

    dataset_counts = {}
    for dataset in datasets:
        dir = record_dir / dataset.name
        print("saving to ", dir)
        create_tf_records(dataset, dir, datasets[0].labels, num_shards=100)
        dataset_counts[dataset.name] = dataset.get_counts()
        # dataset.saveto_numpy(os.path.join(base_dir))
    # dont need dataset anymore just need some meta
    meta_filename = f"{base_dir}/chime-training-data/training-meta.json"
    meta_data = {
        "labels": datasets[0].labels,
        "type": "audio",
        "counts": dataset_counts,
        "by_label": False,
        "relabbled": RELABEL,
    }

    with open(meta_filename, "w") as f:
        json.dump(meta_data, f, indent=4)

    logging.info("Loaded samples mem %s", psutil.virtual_memory()[2])
    dataset.print_counts()
    # return
    datasets = split_randomly(dataset, no_test=False)
    # for d in datasets:
    #     for r in d.recs:
    #         name = r.filename
    #         noisy_name = noisy_p / f"bird-{name.name}"
    #         if noisy_name.exists():
    #             add_rec(d, noisy_name, ["human", "bird"], config)
    #             logging.info("Adding %s %s %s", noisy_name, " from ", name)
    #         noisy_name = noisy_p / f"noise-{name.name}"
    #
    #         if noisy_name.exists():
    #             add_rec(dataset, noisy_name, ["human", "noise"], config)
    #             print("Adding %s %s %s", noisy_name, " from ", name)
    logging.info("Split samples mem %s", psutil.virtual_memory()[2])

    all_labels = set()
    for d in datasets:
        logging.info("%s Dataset", d.name)
        d.print_sample_counts()

        all_labels.update(d.labels)
    # return
    all_labels = list(all_labels)
    all_labels.sort()
    for d in datasets:
        d.labels = all_labels
    base_dir = Path("/data/audio-data/")
    record_dir = base_dir / "flickr-training-data/"
    print("saving to", record_dir)
    logging.info("Saving pre samples mem %s", psutil.virtual_memory()[2])

    dataset_counts = {}
    for dataset in datasets:
        dir = record_dir / dataset.name
        print("saving to ", dir)
        create_tf_records(dataset, dir, datasets[0].labels, num_shards=100)
        dataset_counts[dataset.name] = dataset.get_counts()
        # dataset.saveto_numpy(os.path.join(base_dir))
    # dont need dataset anymore just need some meta
    meta_filename = f"{base_dir}/flickr-training-data/training-meta.json"
    meta_data = {
        "labels": datasets[0].labels,
        "type": "audio",
        "counts": dataset_counts,
        "by_label": False,
        "relabbled": RELABEL,
    }
    meta_data.update(config.__dict__)
    with open(meta_filename, "w") as f:
        json.dump(meta_data, f, indent=4)


def tier1_data(base_dir, split_file=None):
    print("Loading tier1")
    test_labels = [
        "bellbird",
        "bird",
        "fantail",
        "morepork",
        "noise",
        "human",
        "grey warbler",
        "insect",
        "kiwi",
        "magpie",
        "tui",
        "house sparrow",
        "blackbird",
        "sparrow",
        "song thrush",
        "whistler",
        "rooster",
        "silvereye",
        "norfolk silvereye",
        "australian magpie",
        "new zealand fantail",
        "banded dotterel",
        "australasian bittern",
    ]

    ebird_map = {}
    first = True
    with open("eBird_taxonomy_v2024.csv") as f:
        for line in f:
            if first:
                first = False
                continue
            split_l = line.split(",")
            # for i,split in enumerate(split_l):
            # print(i,split)
            ebird_map[split_l[2]] = (split_l[4].lower(), split_l[8].lower())

    with open("classes.csv", newline="") as csvfile:
        dreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        i = -1
        for row in dreader:
            i += 1
            if i == 0:
                continue
            # ebird = (common, extra)
            ebird_map[row[2]] = (row[1].lower(), row[4].lower())
    config = Config()
    plot_signal = False
    signal_scale = 100
    dataset = AudioDataset("Tier1", config)
    folders = ["Train_001", "Train_002"]
    counts = {}
    label_percents = {}
    ignore_long_tracks = False
    for folder in folders:
        filtered_stats = {}
        # ignore_long_tracks = folder == "Train_002"
        dataset_dir = base_dir / folder
        metadata = dataset_dir / "001_metadata.csv"
        if not metadata.exists():
            logging.warning("No metadata at %s", metadata)
            continue
        logging.info("Loading from %s", dataset_dir)
        with open(metadata, newline="") as csvfile:
            dreader = csv.reader(csvfile, delimiter=",", quotechar='"')
            i = -1
            for row in dreader:
                i += 1
                if i == 0:
                    continue

                if len(row) == 6:
                    id, filename, label, other_labels, start, end = row
                else:
                    id = f"{folder}-{i}"
                    filename, label, other_labels, start, end = row

                start = float(start)
                end = float(end)
                length = end - start

                # if label != "dobplo1":
                # continue
                primary_label = ebird_map.get(label)

                # print("Dot mapped too",primary_label)
                if primary_label is None:
                    print("No Mapping for ", label)
                    continue

                if primary_label[0] in test_labels:
                    label = primary_label[0]
                elif primary_label[1] in test_labels:
                    label = primary_label[1]
                elif "kiwi" in primary_label[0]:
                    label = "kiwi"
                else:
                    label = primary_label[0].replace(" ", "-")
                if label not in counts:
                    counts[label] = 0
                counts[label] += 1
                # continue
                # if length > 5 and ignore_long_tracks:
                #     if label not in filtered_stats:
                #         filtered_stats[label] = 0
                #     filtered_stats[label] += 1
                #     logging.info("Track length %s so Ignoring %s", length, filename)
                #     continue
                # if length > 3:
                # # or "kiwi" in label or "more" in label:
                #     continue
                audio_file = dataset_dir / "train_audio" / filename

                if not audio_file.exists():
                    continue
                metadata_file = audio_file.with_suffix(".txt")
                if metadata_file.exists():
                    with metadata_file.open("r") as f:
                        # add in some metadata stats
                        meta = json.load(f)
                else:
                    meta = {}
                meta["id"] = id
                meta["tracks"] = []

                r = Recording(
                    meta,
                    audio_file,
                    dataset.config,
                    load_samples=False,
                )
                assert r.id not in dataset.recs
                track_length = end - start
                t_start = 0
                t_end = min(track_length, 5)

                if label != "banded dotterel" and track_length >= 4:
                    # just choose 1 track in centre
                    t_start = 1
                    t_end = 4
                if "best_track" not in meta:
                    print("No best track", audio_file)
                    continue
                meta["best_track"]["id"] = id

                track_meta = meta["best_track"]
                track_meta["tags"][0]["what"] = label
                meta_length = track_meta["end"] - track_meta["start"]
                if meta_length > length:
                    track_meta["end"] = end
                    # print("Adjusted end of ", filename, track_meta)
                    # return
                t = Track(
                    track_meta,
                    r.filename,
                    r.id,
                    r,
                )
                r.tracks = [t]
                r.signal_percent()

                if plot_signal:
                    if label not in label_percents:

                        label_percents[label] = [0] * (signal_scale + 1)
                    signal_percent = round(t.signal_percent * signal_scale)
                    label_percents[label][signal_percent] += 1
                r.human_tags.add(label)
                r.load_samples(
                    dataset.config.segment_length, dataset.config.segment_stride
                )
                # dataset
                dataset.add_recording(r)
        print("FIltereds are ", filtered_stats)
        keys = list(counts.keys())
        keys.sort()
        for k in keys:
            print(f"{k}, {counts[k]}")
        tootal = list(counts.values())
        print("total is ", np.sum(tootal))
        counts = {}
        if plot_signal:
            save_dir = dataset_dir / "signal-graphs"
            save_dir.mkdir(parents=True, exist_ok=True)
            for label, values in label_percents.items():
                plt.clf()
                plt.plot(np.arange(signal_scale + 1), values, marker="o", linestyle="-")

                # Add labels and title
                plt.xlabel("Signal percent")
                plt.ylabel("Tracks")
                plt.title(f"{label}")
                plt.legend()
                plt.savefig(str(save_dir / f"{label}.png"))
            label_percents = {}
    if plot_signal:
        return
    logging.info("Loaded tier 1 data")
    dataset.print_sample_counts()
    if split_file is not None:
        logging.info("Splitting by %s", split_file)
        with open(split_file, "r") as t:
            # add in some metadata stats
            split_meta = json.load(t)
        split_by_ds = split_meta["recs"]
        datasets = []
        for name in ["train", "validation", "test"]:
            split_clips = split_by_ds[name]
            ds = AudioDataset(name, dataset.config)
            datasets.append(ds)
            logging.info("Loading %s using ids from split # %s", name, len(split_clips))
            for clip_id in split_clips:
                if clip_id in dataset.recs:
                    rec = dataset.recs[clip_id]
                    ds.add_recording(rec)
                else:
                    logging.error("Missing clip id %s", clip_id)
    else:
        datasets = split_randomly(dataset)
    save_data(datasets, base_dir, dataset.config)


def plot_signal(dataset, out_dir):
    label_percents = {}
    signal_scale = 10
    for rec in dataset.recs.values():
        for t in rec.tracks:
            for label in t.human_tags:
                if label not in label_percents:

                    label_percents[label] = [0] * (signal_scale + 1)
                signal_percent = round(t.signal_percent * signal_scale)
                label_percents[label][signal_percent] += 1
    save_dir = out_dir / "signal-graphs"
    save_dir.mkdir(parents=True, exist_ok=True)
    for label, values in label_percents.items():
        plt.clf()
        plt.plot(np.arange(signal_scale + 1), values, marker="o", linestyle="-")

        # Add labels and title
        plt.xlabel("Signal percent")
        plt.ylabel("Tracks")
        plt.title(f"{label}")
        plt.savefig(str(save_dir / f"{label}.png"))


def save_data(datasets, base_dir, config):
    all_labels = set()
    for d in datasets:
        logging.info("")
        logging.info("%s Dataset", d.name)
        d.print_sample_counts()

        all_labels.update(d.labels)
    all_labels = list(all_labels)
    all_labels.sort()
    for d in datasets:
        d.labels = all_labels
        print("setting all labels", all_labels)
    validate_datasets(datasets)
    record_dir = base_dir / "training-data/"
    print("saving to", record_dir)
    # return
    dataset_counts = {}
    dataset_recs = {}
    for dataset in datasets:
        dir = record_dir / dataset.name
        r_counts = dataset.get_rec_counts()
        for k, v in r_counts.items():
            r_counts[k] = len(v)
        dataset_recs[dataset.name] = list(dataset.recs.keys())
        dataset_counts[dataset.name] = {
            "rec_counts": r_counts,
            "sample_counts": dataset.get_counts(),
        }
        create_tf_records(dataset, dir, datasets[0].labels, num_shards=100)

        # dataset.saveto_numpy(os.path.join(base_dir))
    # dont need dataset anymore just need some meta
    meta_filename = f"{base_dir}/training-data/training-meta.json"
    meta_data = {
        # "segment_length": SEGMENT_LENGTH,
        # "segment_stride": SEGMENT_STRIDE,
        # "hop_length": HOP_LENGTH,
        # "n_mels": N_MELS,
        # "fmin": FMIN,
        # "fmax": FMAX,
        # "break_freq": BREAK_FREQ,
        # "htk": HTK,
        "labels": datasets[0].labels,
        "type": "audio",
        "counts": dataset_counts,
        "recs": dataset_recs,
        "by_label": False,
        "relabbled": RELABEL,
    }
    meta_data.update(config.__dict__)
    with open(meta_filename, "w") as f:
        json.dump(meta_data, f, indent=4)


from identifytracks import (
    signal_noise as track_signals,
)


def signal_noise(file, hop_length=281):
    frames, sr = load_recording(file)
    end = get_end(frames, sr)
    # end = 5
    # frames = frames[int(20*sr): int(sr * 40)]

    frames = frames[: int(sr * end)]
    # frames = frames[: sr * 120]
    # n_fft = sr // 10
    n_fft = 4096
    # spectogram = librosa.stft(frames, n_fft=n_fft, hop_length=hop_length)
    # plot_spec(spectogram)
    signals, spectogram = track_signals(
        frames, sr, hop_length=hop_length, n_fft=n_fft, min_width=0, min_height=0
    )
    noise = []
    return signals, noise, spectogram, frames, end


def add_signal_meta(dir):

    test_labels = [
        "bellbird",
        "bird",
        "fantail",
        "morepork",
        "noise",
        "human",
        "grey warbler",
        "insect",
        "kiwi",
        "magpie",
        "tui",
        "house sparrow",
        "blackbird",
        "sparrow",
        "song thrush",
        "whistler",
        "rooster",
        "silvereye",
        "norfolk silvereye",
        "australian magpie",
        "new zealand fantail",
        "banded dotterel",
        "australasian bittern",
    ]
    tier1_data = True
    if tier1_data:
        ebird_labels = set()
        with open("eBird_taxonomy_v2024.csv") as f:
            for line in f:
                split_l = line.split(",")
                if (
                    split_l[1].lower() in test_labels
                    or split_l[9].lower() in test_labels
                    or "kiwi" in split_l[1]
                ):
                    ebird_labels.add(split_l[2])
        # 1/0
        # ebird_map = {}
        with open("classes.csv", newline="") as csvfile:
            dreader = csv.reader(csvfile, delimiter=",", quotechar="|")
            i = -1
            for row in dreader:
                i += 1
                if i == 0:
                    continue
                # ebird = (common, extra)
                if (
                    row[1].lower() in test_labels
                    or row[4].lower() in test_labels
                    or "kiwi" in row[1].lower()
                ):
                    ebird_labels.add(row[2])
        print("Only running on ", ebird_labels)
        meta_files = []
        folders = ["Train_001", "Train_002"]
        for folder in folders:
            audio_dir = dir / folder / "train_audio"
            for lbl in ebird_labels:
                lbl_dir = audio_dir / lbl
                meta_files.extend(lbl_dir.glob("**/*.flac"))
    else:
        meta_files = dir.glob("**/*.txt")

    with Pool(processes=8) as pool:
        [0 for x in pool.imap_unordered(process_signal, meta_files, chunksize=8)]


def process_signal(metadata_file):
    try:
        metadata_file = metadata_file.with_suffix(".txt")
        if metadata_file.exists():
            with metadata_file.open("r") as f:
                # add in some metadata stats
                meta = json.load(f)
        else:
            meta = {}
        if meta.get("signal", None) is not None:
            print("Zeroing existing signal")
            meta["signal"] = None
            # print("Already have signal data")
            # return
        file = metadata_file.with_suffix(".m4a")
        if not file.exists():
            file = metadata_file.with_suffix(".wav")
        if not file.exists():
            file = metadata_file.with_suffix(".mp3")
        if not file.exists():
            file = metadata_file.with_suffix(".flac")
        if not file.exists():
            logging.info("Not recording for %s", metadata_file)
            return

        logging.info("Calcing %s", file)
        signals, noise, _, _, end = signal_noise(file)

        signals = [s.to_array(decimals=2) for s in signals]
        meta["file"] = str(file)
        meta["signal"] = signals
        meta["noise"] = noise
        meta["rec_end"] = end
        with metadata_file.open("w") as f:
            json.dump(
                meta,
                f,
                indent=4,
            )
        logging.info("Updated %s", metadata_file)
    except:
        logging.error("Error processing %s", metadata_file, exc_info=True)
    return


def generate_tracks_master(dir):
    tier1_data = True
    filename_to_meta = {}
    meta_files = dir.glob("**/*.txt")

    if tier1_data:
        meta_files = []
        folders = ["Train_001", "Train_002"]
        for folder in folders:
            dataset_dir = dir / folder
            meta_files.extend(dataset_dir.glob("**/*.txt"))
            metadata = dataset_dir / "001_metadata.csv"
            with open(metadata, newline="") as csvfile:
                dreader = csv.reader(csvfile, delimiter=",", quotechar='"')
                i = -1

                for row in dreader:
                    i += 1
                    if i == 0:
                        continue
                    if len(row) == 6:
                        id, filename, label, other_labels, start, end = row
                    else:
                        id = f"{folder}-{i}"
                        filename, label, other_labels, start, end = row
                    start = float(start)
                    end = float(end)
                    length = end - start
                    filename_to_meta[filename] = {"start": start, "end": end}
        pool_data = []
        for f in meta_files:
            flac_file = Path(f.parent.name) / f.stem
            flac_file = flac_file.with_suffix(".flac")
            csv_meta = filename_to_meta.get(str(flac_file))
            if csv_meta is None:
                print("Skipping", f)
                continue
            pool_data.append((f, csv_meta))
    else:
        pool_data = [meta_files]
    with Pool(processes=8) as pool:
        [0 for x in pool.imap_unordered(generate_tracks, pool_data, chunksize=8)]


def generate_tracks(metadata):
    if isinstance(metadata, tuple):
        file = metadata[0]
        csv_meta = metadata[1]
    else:
        file = metadata
        csv_meta = None

    min_height = 105.46875
    min_width = 0.15981875

    meta_f = file.with_suffix(".txt")
    metadata = {}
    if meta_f.exists():
        if meta_f.is_dir():
            logging.error("Is dir %s ", meta_f)
            return
        with meta_f.open("r") as f:
            metadata = json.load(f)
    else:
        logging.error("No metadata found for %s", file)
        return
    end = metadata.get("rec_end", None)
    if csv_meta:
        # take which ever ends first
        c_end = csv_meta["end"] - csv_meta["start"]
        end = min(c_end, end)

    if "signal" not in metadata:
        logging.error("No Signals metadata found for %s", file)
        return
    meta_sig = metadata.get("signal")
    signals = []
    sig_end = None
    for s in meta_sig:
        if (s[1] - s[0]) < min_width or (s[3] - s[2]) < min_width:
            continue
        signals.append(Signal(s[0], s[1], s[2], s[3], 0))
        if end is None:
            if sig_end is None or s[1] > sig_end:
                sig_end = s[1]
    if end is None:
        end = sig_end + 3
        logging.info("Using last signal + 3 as end %s", end)
    tracks = get_tracks_from_signals(signals, end=end, filter_short=False)

    length_per_segment = []
    best_segment = (0, 0, 0)
    length_score = None
    starts = int(end) - 3 + 1
    starts = max(starts, 1)
    step = 0.5
    starts = np.arange(starts, step=step)
    # possibly could align starts with the signals instead of 0
    for start in starts:
        s_end = start + 3
        signal_length = signal_length_for_segment(tracks, start, s_end)

        if len(length_per_segment) > 0:
            # always adding for 1 before
            length_score = length_per_segment[-1]
            if len(length_per_segment) == 1:
                length_score += signal_length
            else:
                # 0.5 of prev and post score
                length_score += (signal_length + length_per_segment[-2]) / 2

            if best_segment is None or best_segment[2] < length_score:
                best_segment = (start - step, signal_length, length_score)
        else:
            best_segment = (start, signal_length, signal_length)
            # print(length_score)
        #    print(f"Signal length at {start}-{s_end} is {signal_length}")

        length_per_segment.append(signal_length)

    # note also skipping last seg above
    # do_last_seg = False
    # #end of audio normally is worse so probably  not worth using
    # if do_last_seg:
    #     last_seg = end - 3
    #     if last_seg > 0 and last_seg not in starts:
    #         signal_length = signal_length_for_segment(tracks, last_seg,last_seg+3)

    #         prev = max(0,last_seg - 3)
    #         prev_length = signal_length_for_segment(tracks, prev,prev+3)
    #         length_score = signal_length + prev_length
    #         if best_segment is None or best_segment[2] < length_score:
    #             best_segment = (last_seg, signal_length, length_score)

    #         length_per_segment.append(signal_length)

    best_track = {
        "score": best_segment[2],
        "signal_length": best_segment[1],
        "start": best_segment[0],
        "end": best_segment[0] + 3,
        "tags": [{"automatic": False, "what": file.parent.name}],
    }
    if best_segment[0] != metadata["best_track"]["start"]:
        print("Differing starts", file, best_track, metadata["best_track"])
        1 / 0
    # print("Best signal is ", best_segment)
    return
    metadata["best_track"] = best_track

    with open(meta_f, "w") as f:
        json.dump(metadata, f, indent=4)


def signal_length_for_segment(tracks, s_start, s_end):
    signal_length = 0
    for s in tracks:
        if s.start < s_start and s.end < s_end:
            continue

        if s.start > s_end:
            break
        signal_length += min(s.end, s_end) - max(s_start, s.start)
    return signal_length


def main():
    init_logging()
    args = parse_args()
    if args.tracks:
        logging.info("Adding best track estimates")
        generate_tracks_master(args.dir)
        return
    elif args.signal:
        print("Adding signal data to ", args.dir)
        add_signal_meta(args.dir)
        return
    else:
        print("Doing tier 1 data")
        tier1_data(args.dir, args.split_file)
    return
    # weakly_lbled_data(args.dir)
    return
    # process_noise()
    # return
    flickr_data()
    return
    chime_data()
    # return
    dataset = AudioDataset("Other")
    # dataset.print_counts()

    # return
    labels = ["other", "bird"]
    for csv_file in csv_files:
        print("loading", csv_file)
        csv_file = Path(csv_file)
        with open(csv_file, newline="") as csvfile:
            dreader = csv.reader(csvfile, delimiter=",", quotechar="|")
            i = -1
            for row in dreader:
                i += 1
                if i == 0:
                    continue
                rec_name = csv_file.parent / "wav" / row[0]
                rec_name = rec_name.with_suffix(".wav")
                r = Recording({"id": row[0], "tracks": []}, rec_name)
                try:
                    y, sr = librosa.load(rec_name)
                    end = librosa.get_duration(y=y, sr=sr)
                    y = None
                    sr = None
                except:
                    continue
                what = labels[int(row[1])]
                t = Track(
                    {
                        "id": row[0],
                        "start": 0,
                        "end": end,
                        "tags": [{"automatic": False, "what": what}],
                    },
                    csv_file,
                    r.id,
                    r,
                )

                r.tracks = [t]
                r.human_tags.add(what)
                r.load_samples()
                # dataset
                dataset.add_recording(r)
                dataset.samples.extend(r.samples)

    print("counts are")
    dataset.print_counts()
    datasets = split_randomly(dataset, no_test=True)
    all_labels = set()
    for d in datasets:
        logging.info("%s Dataset", d.name)
        d.print_sample_counts()

        all_labels.update(d.labels)
    all_labels = list(all_labels)
    all_labels.sort()
    for d in datasets:
        d.labels = all_labels
    base_dir = Path(".")
    record_dir = base_dir / "other-training-data/"
    print("saving to", record_dir)
    dataset_counts = {}
    for dataset in datasets:
        dir = record_dir / dataset.name
        print("saving to ", dir)
        create_tf_records(dataset, dir, datasets[0].labels, num_shards=100)
        dataset_counts[dataset.name] = dataset.get_counts()
        # dataset.saveto_numpy(os.path.join(base_dir))
    # dont need dataset anymore just need some meta
    meta_filename = f"{base_dir}/other-training-data/training-meta.json"
    meta_data = {
        "labels": datasets[0].labels,
        "type": "audio",
        "counts": dataset_counts,
        "by_label": False,
        "relabbled": RELABEL,
    }

    with open(meta_filename, "w") as f:
        json.dump(meta_data, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="Dir to load")
    parser.add_argument(
        "-s", "--signal", action="store_true", help="Add signal data to dir"
    )
    parser.add_argument(
        "-t", "--tracks", action="store_true", help="Add best track data"
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Split the dataset using clip ids specified in this file",
    )
    args = parser.parse_args()
    args.dir = Path(args.dir)
    return args


def init_logging():
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """

    fmt = "%(asctime)s %(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


if __name__ == "__main__":
    main()
