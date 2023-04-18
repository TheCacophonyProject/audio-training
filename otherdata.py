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


def load_recording(file, resample=48000):
    aro = None
    try:
        aro = audioread.ffdec.FFmpegAudioFile(file)
        frames, sr = librosa.load(aro)
        aro.close()
        if resample is not None and resample != sr:
            frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
            sr = resample
    except:
        try:
            aro.close()
        except:
            pass
        logging.error("Could not load %s",file, exc_info=True)
        return None,None
    return frames, sr


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
bad_signals = signals.parent/"bad-train"
bad_signals.mkdir(parents=True, exist_ok=True)

wavs = list(signals.glob("*.wav"))
for w in wavs:

    if "bird" in w.stem:
        #frames, sr = load_recording(w)
       # if frames is None:
        #    w.rename(bad_signals/w.name)
        #    continue
        #if len(frames) / sr < 4:
        #    logging.info("skipping %s", w)
        #    continue
        BIRD_PATH.append(w)
    else:
        for noise in NOISE_LABELS:
            if noise in w.stem:
                #frames, sr = load_recording(w)
                #if frames is None:
                #    w.rename(bad_signals/w.name)

                 #   continue
                #if len(frames) / sr < 4:
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


def flickr_data():
    config = Config()
    dataset = AudioDataset("Flickr", config)
    p = Path("./flickr/wavs")
    # p = Path("/data/audio-data/Flickr-Audio-Caption-Corpus/flickr_audio/wavs")

    wav_files = list(p.glob("*.wav"))
    p = Path("./flickr/noisy-wavs")
    wav_files.extend(list(p.glob("*.wav")))
    random.shuffle(wav_files)

    for rec_name in wav_files:
        label = "human"
        id = None
        id, id_2, speaker = rec_name.stem.split("_")
        id = f"{id}-{id_2}-{speaker}"

        r = Recording({"id": id, "tracks": []}, rec_name, config)
        tags = [{"automatic": False, "what": label}]
        # try:
        #     y, sr = librosa.load(rec_name)
        #     end = librosa.get_duration(y=y, sr=sr)
        #     y = None
        #     sr = None
        # except:
        #     continue
        t = Track({"id": id, "start": 0, "end": None, "tags": tags}, rec_name, r.id, r)
        # r.load_samples()
        r.human_tags.add(label)
        r.tracks.append(t)
        sample = AudioSample(r, r.human_tags, 0, None, [t.id], 1)
        r.samples = [sample]
        dataset.add_recording(r)
        dataset.samples.extend(r.samples)
        dataset.labels.add(label)
        if len(dataset.recs) > len(wav_files) / 3:
            break

    logging.info("Loaded samples mem %s", psutil.virtual_memory()[2])
    dataset.print_counts()
    # return
    datasets = split_randomly(dataset, no_test=False)
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
    base_dir = Path(".")
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
    process_noise()
    return
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
    args = parser.parse_args()
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
