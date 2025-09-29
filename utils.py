import csv


def get_all_birds():
    allbirds = set()
    first = True
    with open("eBird_taxonomy_v2024.csv", newline="") as csvfile:
        dreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        _ = next(dreader)
        for row in dreader:
            allbirds.add(row[2])
    allbirds = list(allbirds)
    allbirds.sort()
    return allbirds


def get_label_to_ebird_map():
    ebird_map = {}
    first = True
    with open("classes.csv", newline="") as csvfile:
        dreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        _ = next(dreader)
        for row in dreader:
            # ebird = (common, extra)
            ebird_map[row[1].lower().replace(" ", "-")] = row[2]
            ebird_map[row[4].lower().replace(" ", "-")] = row[2]

    with open("eBird_taxonomy_v2024.csv") as f:
        dreader = csv.reader(f, delimiter=",", quotechar='"')
        _ = next(dreader)
        for split_l in dreader:
            # for i,split in enumerate(split_l):
            # print(i,split)
            # ebird_map[split_l[2]] = (split_l[4].lower(), split_l[8].lower())
            ebird_map[split_l[4].lower().replace(" ", "-")] = split_l[2]
            ebird_map[split_l[8].lower().replace(" ", "-")] = split_l[2]
    ebird_map["norfolk-silvereye"] = "silver3"
    ebird_map["norfolk-gerygone"] = "noiger1"
    ebird_map["whistler"] = "y01193"
    ebird_map["common-starling"] = "eursta"
    ebird_map["turkey"] = "wiltur"
    ebird_map["rooster"] = "redjun"
    ebird_map["crow"] = "rook1"
    ebird_map["norfolk-parrot"] = "noipar1"
    ebird_map["pigeon"] = "rocpig"
    ebird_map["new zealand dotterel"] = "dobplo1"

    return ebird_map


def get_ebird_id(label, ebird_map):
    return ebird_map.get(label.lower().replace(" ", "-"), label)


def get_ebird_ids_to_labels():
    ebird_map = get_label_to_ebird_map()
    labels_to_ebird = {}
    for lbl, ebird_id in ebird_map.items():
        if ebird_id in labels_to_ebird:
            labels_to_ebird[ebird_id].append(lbl)
        else:
            labels_to_ebird[ebird_id] = [lbl]

    return labels_to_ebird
