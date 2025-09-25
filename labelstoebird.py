import sys
import json
from pathlib import Path
import csv
from utils import get_label_to_ebird_map, get_ebird_id, get_ebird_ids_to_labels


def labels_to_ebird_links(metadata):
    remapped = metadata["remapped_labels"]
    labels = metadata["labels"]
    ebird_id_mappings = {}

    for l in labels:
        ebird_id_mappings[l] = {l}
    for original_id, v in remapped.items():
        if v == -1:
            continue
        ebird_id = labels[v]
        ebird_id_mappings[ebird_id].add(original_id)

    print(ebird_id_mappings)

    ids_to_lbls = get_ebird_ids_to_labels()

    for ebird_id, merged_ids in ebird_id_mappings.items():
        translated = ids_to_lbls.get(ebird_id, [ebird_id])[0]

        if len(merged_ids) > 1:
            print(translated)
        for og_id in merged_ids:
            translated = ids_to_lbls.get(og_id, [og_id])[0]
            print(
                f"-,{translated},https://ebird.org/species/{og_id}?siteLanguage=en_NZ"
            )


def test_labels():
    text_labels = [
        "bellbird",
        "fantail",
        "morepork",
        "grey warbler",
        "kiwi",
        "magpie",
        "tui",
        "house sparrow",
        "blackbird",
        "sparrow",
        "song thrush",
        "silvereye",
        "norfolk silvereye",
        "new zealand fantail",
    ]

    ebird_map = get_label_to_ebird_map()
    ebird_ids = set()
    for text_label in text_labels:
        ebird_id = get_ebird_id(text_label, ebird_map)
        print(f"{text_label} ebirdid: {ebird_id}")
        ebird_ids.add(ebird_id)
    e_ids = list(ebird_ids)
    e_ids.sort()
    print("Ids are", e_ids)


def main():
    test_labels()
    return
    ebird_map = get_label_to_ebird_map()

    metadata_f = Path(sys.argv[1])
    with metadata_f.open("r") as f:
        metadata = json.load(f)
    labels_to_ebird_links(metadata)
    ebird_ids = []
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

    from birdsconfig import BIRD_TRAIN_LABELS

    RELABEL = {}
    RELABEL["mohoua novaeseelandiae"] = "brown-creeper"
    RELABEL["new zealand fantail"] = "fantail"
    RELABEL["shining bronze-cuckoo"] = "shining-cuckoo"
    RELABEL["long-tailed koel"] = "long-tailed-cuckoo"
    RELABEL["masked lapwing"] = "spur-winged-plover"
    RELABEL["sacred kingfisher (new zealand)"] = "new-zealand-kingfisher"
    RELABEL["norfolk island gerygone"] = "norfolk gerygone"
    RELABEL["kelp gull"] = "southern-black-backed-gull"
    RELABEL["common myna"] = "indian-myna"
    RELABEL["baillon's crake"] = "marsh-crake"
    RELABEL["north island brown kiwi"] = "kiwi"
    RELABEL["great spotted kiwi"] = "kiwi"
    RELABEL["norfolk morepork"] = "morepork"
    RELABEL["golden whistler"] = "whistler"
    RELABEL["norfolk golden whistler"] = "whistler"
    RELABEL["golden-backed whistler"] = "whistler"
    for k, v in RELABEL.items():
        k = get_ebird_id(k, ebird_map)
        v = get_ebird_id(v, ebird_map)
        print(f'"{k}":"{v}",')


if __name__ == "__main__":
    main()
