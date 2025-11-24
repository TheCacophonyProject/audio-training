import matplotlib.pyplot as plt
import sys
import json
from pathlib import Path
import csv
from utils import get_label_to_ebird_map, get_ebird_id, get_ebird_ids_to_labels
import numpy as np

# bunch of debug and helper functions
new_labels = [
    "ausbit1",
    "ausmag2",
    "auspip3",
    "baicra4",
    "blknod",
    "blkswa",
    "calqua",
    "cangoo",
    "comcha",
    "commyn",
    "comred",
    "duck",
    "dunnoc1",
    "eurbla",
    "eurgre1",
    "eursta",
    "fernbi1",
    "frog",
    "gryger1",
    "gull",
    "houspa",
    "insect",
    "kea1",
    "kiwi",
    "kokako3",
    "lotkoe1",
    "maslap1",
    "morepo2",
    "nezbel1",
    "nezfal1",
    "nezfan1",
    "nezkak1",
    "nezpig2",
    "nezrob3",
    "noiger1",
    "oyster1",
    "pacrob2",
    "parake",
    "parshe1",
    "pipipi1",
    "purswa6",
    "rebdot1",
    "rettro",
    "riflem1",
    "rooster",
    "rosella",
    "sackin3",
    "sbweye1",
    "shbcuc1",
    "silver3",
    "skylar",
    "sonthr1",
    "sooter1",
    "tomtit1",
    "tui1",
    "weka1",
    "weta",
    "whiteh1",
    "whiter",
    "y01193",
    "yellow2",
    "yellow3",
]


current_labels = [
    "ausmag2",
    "auspip3",
    "baicra4",
    "blknod",
    "blkswa",
    "brncre",
    "calqua",
    "cangoo",
    "comcha",
    "commyn",
    "criros2",
    "dunnoc1",
    "eurbla",
    "eurgre1",
    "eursta",
    "fernbi1",
    "frog",
    "gryger1",
    "houspa",
    "insect",
    "kea1",
    "kelgul",
    "kiwi",
    "lotkoe1",
    "maslap1",
    "morepo2",
    "nezbel1",
    "nezfal1",
    "nezfan1",
    "nezkak1",
    "nezpig2",
    "nezrob2",
    "nezrob3",
    "noiger1",
    "oyster1",
    "pacrob2",
    "parake",
    "parshe1",
    "purswa6",
    "redjun",
    "redpol1",
    "rettro",
    "riflem1",
    "sackin1",
    "sbweye1",
    "shbcuc1",
    "silver3",
    "skylar",
    "sonthr1",
    "sooter1",
    "tomtit1",
    "tui1",
    "weka1",
    "weta",
    "whiteh1",
    "whiter",
    "x00458",
    "y01193",
    "yefpar3",
    "yellow2",
    "yellow3",
]


def compare_labels():
    to_labels = get_ebird_ids_to_labels()
    global new_labels
    current = set(current_labels)
    new_labels = set(new_labels)
    added = new_labels - current
    for add in added:
        print("Added", to_labels.get(add, [add])[0])
    print("Have added ", new_labels - current)

    added = current - new_labels
    for add in added:
        print("Removed ", to_labels.get(add, [add])[0])

    print("Have removed ", current - new_labels)


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
        ebird_id = get_ebilabels_to_ebird_links
        rd_id(text_label, ebird_map)
        print(f"{text_label} ebirdid: {ebird_id}")
        ebird_ids.add(ebird_id)
    e_ids = list(ebird_ids)
    e_ids.sort()
    print("Ids are", e_ids)


# convert model labels to the exact text used on the api server
def labels_to_api(metadata_f):
    with metadata_f.open("r") as f:
        metadata = json.load(f)

    lbl_paths = Path("label_paths.json")
    with lbl_paths.open("r") as f:
        lbl_metadata = json.load(f)

    hyphenated_lbls = {}
    for lbl in lbl_metadata.keys():
        hyphenated_lbls[lbl.replace(" ", "-")] = lbl

    ebird_labels = metadata["labels"]
    ebird_map = get_ebird_ids_to_labels()
    text_labels = []
    for ebird_id in ebird_labels:
        e_text_labels = ebird_map.get(ebird_id, [ebird_id])
        match = None
        for text_label in e_text_labels:
            if text_label in hyphenated_lbls:
                match = hyphenated_lbls[text_label]
                print(f"Found match ebird {ebird_id} textual: {text_label}  {match}")
                break
        if match is None:
            print("Could not find api name for ", ebird_id, text_labels)
            raise Exception(f"Could not find api name for {ebird_id}  {text_labels}")
        text_labels.append(match)
    print("Text labels is ", text_labels)
    metadata["labels"] = text_labels
    metadata["ebird_labels"] = ebird_labels

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
        print(lbl_ebird_ids)
    metadata["ebird_ids"] = ebird_ids
    with metadata_f.open("w") as f:
        json.dump(metadata, f, indent=4)


# convert model labels to the exact text used on the api server
def labels_to_api_old(metadata_f):
    with metadata_f.open("r") as f:
        metadata = json.load(f)

    lbl_paths = Path("label_paths.json")
    with lbl_paths.open("r") as f:
        lbl_metadata = json.load(f)

    hyphenated_lbls = {}
    for lbl in lbl_metadata.keys():
        hyphenated_lbls[lbl.replace(" ", "-")] = lbl
    hyphenated_lbls["indian-myna"] = "common myna"
    hyphenated_lbls["mohua"] = "yellowhead"
    hyphenated_lbls["new-zealand-fernbird"] = "fernbird"
    hyphenated_lbls["new-zealand-kaka"] = "kaka"
    hyphenated_lbls["new-zealand-kingfisher"] = "sacred kingfisher"

    text_labels = metadata["labels"]
    new_labels = []
    for lbl in text_labels:
        match = None
        hyphened = lbl.replace(" ", "-")
        if hyphened in hyphenated_lbls:
            match = hyphenated_lbls[hyphened]
        if match is None:
            print("Could not find lbl", lbl)
            1 / 0
        if match != lbl:
            print(f"{lbl} changed to {match}")
        new_labels.append(match)
    metadata["original_labels"] = text_labels

    metadata["labels"] = new_labels
    with metadata_f.open("w") as f:
        json.dump(metadata, f, indent=4)


def show_remapped(meta):
    ebird_map = get_ebird_ids_to_labels()

    labels = meta["ebird_labels"]
    remapped = meta["remapped_labels"]
    for k, v in remapped.items():
        if v != -1:
            k2 = ebird_map.get(k, [k])
            # ()
            print(
                f"{k} {k2} is remapped to {labels[v]}-{ebird_map.get(labels[v],labels[v])}"
            )


def text_labels(meta):
    labels = meta["ebird_labels"]
    ebird_map = get_ebird_ids_to_labels()
    for l in labels:
        print(l, ",", ebird_map.get(l, [l]))

    for k, v in meta["remapped_labels"].items():
        if v != -1:
            print(k, ",", ebird_map.get(k, [k]))


def graph_counts_vs_accuracy(confusion_file, metadata_f, train_counts):
    confusion_data = np.load(confusion_file)
    with metadata_f.open("r") as f:
        metadata = json.load(f)

    labels = metadata["ebird_labels"]
    accuracies = []
    counts = []
    print(
        "Labels are ",
        labels,
    )
    total_train_counts = np.sum(list(train_counts.values()))
    total_train_counts /= len(labels) / 8
    ebird_map = get_ebird_ids_to_labels()

    graph_data = {}
    for i, row in enumerate(confusion_data[:-1]):
        correct = row[i]
        total = np.sum(row)
        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy)
        label = labels[i]

        train_count = train_counts.get(label, 0)
        if accuracy < 0.01:
            print(
                "low acc for ",
                ebird_map.get(label, [label])[0],
                "train counts",
                train_count,
            )
        train_count_percent = train_count / total_train_counts
        # print("Total train acc for ",label, round(train_acc*100))
        counts.append(train_count_percent)
        # train_counts.get(label,0)/total_train_counts)
        graph_data[label] = {
            "acc": accuracy,
            "train_count": train_count_percent,
            "lbl": ebird_map.get(label, [label])[0],
        }

    sorted_by_acc = sorted(graph_data.values(), key=lambda item: item["train_count"])

    counts = [item["train_count"] for item in sorted_by_acc]
    accuracies = [item["acc"] for item in sorted_by_acc]
    text_labels = [item["lbl"] for item in sorted_by_acc]

    fig, ax = plt.subplots()
    x = np.arange(len(text_labels))
    rects1 = ax.bar(x, accuracies, 0.25, label="Accuracy", color="g")
    rects2 = ax.bar(x + 0.25, counts, 0.25, label="Accuracy", color="b")

    ax.set_xticklabels(text_labels)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelrotation=90)

    plt.title("Accuracy vs data count")

    plt.tight_layout()
    plt.show()


def main():
    # compare_labels()
    # # test_labels()
    # return
    metadata_f = Path(sys.argv[2])
    confusion_f = Path(sys.argv[1])

    graph_counts_vs_accuracy(confusion_f, metadata_f, training_counts)
    return
    # labels_to_api_old(metadata_f)
    # return
    ebird_map = get_label_to_ebird_map()

    with metadata_f.open("r") as f:
        metadata = json.load(f)
    text_labels(metadata)
    return
    show_remapped(metadata)

    # labels_to_ebird_links(metadata)
    return
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


training_counts = {
    "ausbit1": 187.0,
    "ausmag2": 214.0,
    "auspip3": 127.0,
    "baicra4": 21.0,
    "blknod": 1083.0,
    "blkswa": 25.0,
    "calqua": 160.0,
    "cangoo": 111.0,
    "comcha": 7104.0,
    "commyn": 79.0,
    "comred": 643.0,
    "duck": 74.0,
    "dunnoc1": 459.0,
    "eurbla": 12717.0,
    "eurgre1": 317.0,
    "eursta": 144.0,
    "fernbi1": 30.0,
    "frog": 305.0,
    "gryger1": 16248.0,
    "gull": 132.0,
    "houspa": 268.0,
    "insect": 4393.0,
    "kea1": 2032.0,
    "kiwi": 2842.0,
    "kokako3": 392.0,
    "lotkoe1": 5389.0,
    "maslap1": 134.0,
    "morepo2": 3644.0,
    "nezbel1": 31838.0,
    "nezfal1": 180.0,
    "nezfan1": 10818.0,
    "nezkak1": 8599.0,
    "nezpig2": 108.0,
    "nezrob3": 3612.0,
    "noiger1": 1790.0,
    "oyster1": 54.0,
    "pacrob2": 634.0,
    "parake": 2378.0,
    "parshe1": 83.0,
    "pipipi1": 720.0,
    "purswa6": 713.0,
    "rebdot1": 303.0,
    "rettro": 92.0,
    "riflem1": 6379.0,
    "rooster": 1249.0,
    "rosella": 743.0,
    "sackin3": 758.0,
    "sbweye1": 931.0,
    "shbcuc1": 636.0,
    "silver3": 24948.0,
    "skylar": 382.0,
    "sonthr1": 2323.0,
    "sooter1": 384.0,
    "tomtit1": 25918.0,
    "tui1": 8886.0,
    "weka1": 2298.0,
    "weta": 3367.0,
    "whiteh1": 656.0,
    "whiter": 1528.0,
    "y01193": 10382.0,
    "yellow2": 450.0,
    "yellow3": 74.0,
}
if __name__ == "__main__":
    main()
