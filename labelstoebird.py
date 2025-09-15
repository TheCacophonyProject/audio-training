import sys
import json
from pathlib import Path
import csv
from utils import get_ebird_map, get_ebird_id


def main():
    metadata_f = Path(sys.argv[1])
    with metadata_f.open("r") as f:
        metadata = json.load(f)

    ebird_map = get_ebird_map()
    ebird_ids = []
    print("Migrating old meta to new meta")
    for l in metadata["labels"]:
        l = l.replace(" ", "-")
        if l not in ebird_map:
            ebird_ids.append(l)
            print(f"{l} is missing from ebird map")
        else:
            ebird_ids.append(ebird_map[l])
            # continue
            # print(f"{l} = {ebird_map[l]}")
    counts = metadata["counts"]

    #
    # for dataset in ["train", "validation", "test"]:
    #     training = counts[dataset]["sample_counts"]
    #     new_training = {}
    #     for k, v in training.items():
    #         ebird_label = get_ebird_id(k, ebird_map)
    #         new_training[ebird_label] = v
    #     counts[dataset]["sample_counts"] = new_training
    #     training = counts[dataset]["rec_counts"]
    #     new_training = {}
    #     for k, v in training.items():
    #         ebird_label = get_ebird_id(k, ebird_map)
    #         new_training[ebird_label] = v
    #     counts[dataset]["rec_counts"] = new_training

    metadata["ebird_ids"] = ebird_ids
    with metadata_f.open("w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()
