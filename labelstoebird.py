import sys
import json
from pathlib import Path
import csv
from utils import get_label_to_ebird_map, get_ebird_id, get_ebird_ids_to_labels

# def debug_labels():
#     ebird_map = get_ebird_map()
#     existing_ids = []
#     for lbl in existing_labels:
#         ebird_id = get_ebird_id(lbl, ebird_map)
#         if ebird_id in existing_ids:
#             print("Added ", ebird_id, " twice ", lbl)
#         existing_ids.append(ebird_id)
#     global train_labels
#     existing_ids = set(existing_ids)
#     train_labels = set(train_labels)

#     diff = existing_ids - train_labels
#     ids_to_lbls = get_ebird_ids_to_labels()

#     for e_id in diff:
#         print(e_id, " Missing in new but in old ", ids_to_lbls.get(e_id, e_id))

#     diff = train_labels - existing_ids
#     for e_id in diff:
#         print(e_id, " Missing in old but in new", ids_to_lbls.get(e_id, e_id))

#     print("Diff in old and not in new ", existing_ids - train_labels)
#     print("Diff in new and not in old", train_labels - existing_ids)
#     1 / 0

#     text_labels = []
#     for l in train_labels:
#         lbls = ids_to_lbls.get(l, [f"{l} not matched"])
#         text_labels.append(lbls[0])
#     text_labels.sort()
#     print(text_labels)


def main():
    ebird_map = get_label_to_ebird_map()

    metadata_f = Path(sys.argv[1])
    with metadata_f.open("r") as f:
        metadata = json.load(f)

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
