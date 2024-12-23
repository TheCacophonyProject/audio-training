import librosa
import csv
import logging
import sys
from pathlib import Path
from audiodataset import Track, Recording, AudioDataset, RELABEL, AudioSample, Config
from build import split_randomly
import psutil
import random
from audiomentations import AddBackgroundNoise, PolarityInversion, Compose
import soundfile as sf
import random
import audioread.ffdec  # Use ffmpeg decoder
import numpy as np
from multiprocessing import Pool
from identifytracks import signal_noise, get_tracks_from_signals, get_end
import argparse


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


def generate_tracks(args):
    min_freq = lbl_meta.get("min_freq")
    max_freq = lbl_meta.get("max_freq")

    wav_file = args[0]
    clip_id = args[1]
    meta_f = wav_file.with_suffix(".txt")
    metadata = {}
    if meta_f.exists():
        with meta_f.open("r") as f:
            metadata = json.load(f)

    if "Tracks" in metadata:
        return
    frames, sr = load_recording(wav_file, None)

    length = get_end(frames, sr)
    signals = signal_noise(frames[: int(sr * length)], sr, 281)
    signals = [
        s
        for s in signals
        if (min_freq is None or s.freq_start > min_freq)
        and (max_freq is None or s.freq_start < max_freq)
    ]

    tracks = get_tracks_from_signals(signals, length)
    tracks_meta = []
    for i, t in enumerate(tracks):

        track_meta = t.get_meta()
        track_meta["id"] = f"{wav_file.name}-{i}"
        tag = {"automatic": False, "what": wav_file.parent.name}
        track_meta["tags"] = [tag]
        tracks_meta.append(track_meta)
    metadata["Tracks"] = tracks_meta
    metadata["id"] = int(clip_id)
    metadata["duration"] = length
    # metadata["location"] =
    # could get some metadata  from xeno canto

    with meta_f.open("w") as f:
        json.dump(metadata, f, indent=4)


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

        clip_ids = np.arange(len(wav_files))
        with Pool(processes=8, initializer=xeno_init, initargs=(metadata,)) as pool:
            [
                0
                for x in pool.imap_unordered(
                    generate_tracks, zip(wav_files, clip_ids), chunksize=8
                )
            ]
    # FIRST_SECONDS = 5

    dataset.load_meta(base_dir)
    dataset.print_counts()

    dataset.samples = []
    for k,r in dataset.recs.items():
        # acceptable_tracks = [t.id for t in r.tracks if t.start < FIRST_SECONDS]
        # could filter some tracks by small freq bands
        r.samples = [s for s in r.samples if s.track_ids[0] in r.tracks and s.length> 2]
        if len(r.samples)==0:
            print("No sample is first seconds for ", r.filename)
        dataset.samples.extend(r.samples)
    dataset.print_counts()


    train,validation,_ = split_randomly(dataset, no_test=True)
    validation.name = "test"

    all_labels = set()
    for d in [train,validation]:
        logging.info("")
        logging.info("%s Dataset", d.name)
        d.print_sample_counts()

        all_labels.update(d.labels)

    all_labels = list(all_labels)
    record_dir = base_dir / "xeno-training-data/"
    print("saving to", record_dir)

    dataset_counts = {}
    for dataset in [train,validation]:
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


def main():
    init_logging()
    args = parse_args()
    weakly_lbled_data(args.dir)
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
