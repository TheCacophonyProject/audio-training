import csv
import logging
import sys
from pathlib import Path
from audiodataset import Track,Recording,AudioDataset,RELABEL
from build import split_randomly
# csv_files = ["./ff10/ff1010bird_metadata.csv"]
csv_files = ["/home/cp/cacophony/audio-data/warblrb10k_public/warblrb10k_public_metadata.csv","/home/cp/cacophony/audio-data/ff1010bird/ff1010bird_metadata.csv"]
out_dir = Path("./other-data")
from audiowriter import create_tf_records
import json


chime_labels ={
"c":"human",
"m":"human",
"f":"human",
"v":"video-game",
"p":"noise",
"b": "noise",
"o":"other",
"S":"silence",
"U":"unknown",
}
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
    csv_files = list(p.glob('**/*.csv'))
    for file in csv_files:
        with open(file, newline='') as f:
            dreader = csv.reader(f, delimiter=',', quotechar='|')
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
            r = Recording({"id":id,"tracks":[]},rec_name)
            print(rec_name)
            tags = []
            for code in label:
                tags.append({"automatic":False,"what":chime_labels[code]})
            t = Track({"id":id,"start":None,"end":None,"tags":tags},rec_name, r.id, r)
            r.human_tags.add(chime_labels[code])
            r.tracks.append(t)

            dataset.add_recording(r)

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
    record_dir =base_dir / "chime-training-data/"
    print("saving to", record_dir)
    dataset_counts = {}
    for dataset in datasets:
        dir =record_dir / dataset.name
        print("saving to ", dir)
        create_tf_records(
            dataset, dir, datasets[0].labels, num_shards=100, by_label=False
        )
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
    chime_data()
    dataset = AudioDataset("Other")
    # dataset.print_counts()

    # return
    labels = ["other","bird"]
    for csv_file in csv_files:
        print("loading",csv_file)
        csv_file = Path(csv_file)
        with open(csv_file, newline='') as csvfile:


            dreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            i = -1
            for row in dreader:
                i+=1
                if i == 0:
                    continue
                rec_name = csv_file.parent / "wav"/row[0]
                rec_name = rec_name.with_suffix(".wav")
                r = Recording({"id":row[0],"tracks":[]},rec_name)
                what = labels[int(row[1])]
                t = Track({"id":row[0],"start":None,"end":None,"tags":[{"automatic":False,"what":what}]},csv_file, r.id, r)

                r.tracks =[t]
                r.human_tags.add(what)
                dataset.add_recording(r)
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
    record_dir =base_dir / "other-training-data/"
    print("saving to", record_dir)
    dataset_counts = {}
    for dataset in datasets:
        dir =record_dir / dataset.name
        print("saving to ", dir)
        create_tf_records(
            dataset, dir, datasets[0].labels, num_shards=100, by_label=False
        )
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

    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


if __name__ == "__main__":
    main()