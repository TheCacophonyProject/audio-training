import argparse
from pathlib import Path
import tensorflow as tf
import shutil
import csv
import json
from utils import get_ebird_ids_to_labels
from badwinner2 import MagTransform

# Script used to prepare model for deployment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", help="Weights to freeze in model")
    parser.add_argument("model", help="Model keras file")
    parser.add_argument("out_dir", help="Path to save to")

    args = parser.parse_args()
    args.model = Path(args.model)
    args.out_dir = Path(args.out_dir)
    return args


# convert model labels to the exact text used on the api server
# and add ebird id list
def format_metadata(metadata):
    lbl_paths = Path("label_paths.json")
    with lbl_paths.open("r") as f:
        lbl_metadata = json.load(f)

    hyphenated_lbls = {}
    for lbl in lbl_metadata.keys():
        hyphenated_lbls[lbl.replace(" ", "-")] = lbl
    # metadata["ebird_labels"] = metadata["labels"].copy()
    ebird_labels = metadata["ebird_labels"]
    ebird_map = get_ebird_ids_to_labels()
    text_labels = []
    for ebird_id in ebird_labels:
        e_text_labels = ebird_map.get(ebird_id, [ebird_id])
        match = None
        for text_label in e_text_labels:
            if text_label in hyphenated_lbls:
                match = hyphenated_lbls[text_label]
                # print(f"Found match ebird {ebird_id} textual: {text_label}  {match}")
                break
        if match is None:
            print("Could not find api name for ", ebird_id, e_text_labels)
            match = ebird_id
            # raise Exception(f"Could not find api name for {ebird_id}  {text_labels}")
        text_labels.append(match)
    print(f"Converted ebird labels {ebird_labels} to api text labels is {text_labels}")
    metadata["labels"] = text_labels

    print("Findig all ebird ids under lbls")
    remapped_lbls = metadata["remapped_labels"]
    lbl_to_ebirds = {}
    for k, v in remapped_lbls.items():
        if v == -1:
            continue
        if k not in ebird_map:
            continue
        ebird_id = ebird_labels[v]
        if ebird_id not in lbl_to_ebirds:
            lbl_to_ebirds[ebird_id] = []
        lbl_to_ebirds[ebird_id].append(k)

    ebird_ids = []
    for lbl in ebird_labels:
        lbl_ebird_ids = set()
        if lbl in ebird_map:
            lbl_ebird_ids.add(lbl)
        if lbl in lbl_to_ebirds:
            lbl_ebird_ids.update(lbl_to_ebirds[lbl])
        ebird_ids.append(list(lbl_ebird_ids))
    metadata["ebird_ids"] = ebird_ids
    return metadata


def main():
    args = parse_args()
    print(f"Loading {args.model}")
    model = tf.keras.models.load_model(
        str(args.model),
        compile=False,
    )
    model.trainable = False
    if args.weights is not None:
        print(f"With weights {args.weights}")
        model.load_weights(str(args.weights))
    model.summary()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_model = args.out_dir / "audioModel.keras"
    print("Saving to ", out_model)
    model.save(out_model)
    shutil.copyfile(args.model.parent / "metadata.txt", args.out_dir / "metadata.txt")

    # save ebird ids
    metadata_file = args.out_dir / "metadata.txt"
    print("Fomatting ", metadata_file)

    with metadata_file.open("r") as f:
        meta = json.load(f)
    meta = format_metadata(meta)

    with open(args.out_dir / "metadata.txt", "w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    main()
