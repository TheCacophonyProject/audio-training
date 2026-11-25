import csv
from datetime import datetime
from pathlib import Path

# ids restart in each of these datasets so are prefixed with its name
ID_PREFIX_DIRS = ["Train_001", "Train_002"]


# tracks in these datasets are mostly centred on the call, so the best rms
# segment prefers centred windows, see birdnetcompare.best_rms
CENTRED_TRACK_DIRS = ["Train_001", "Train_002"]
CENTRE_BONUS = 0.1


def centre_bonus(path):
    return CENTRE_BONUS if any(d in str(path) for d in CENTRED_TRACK_DIRS) else 0


# recordings in these groups are split by their own id and day, not location
# e.g. uploads from other datasets where the location fields don't mean a site
SPLIT_BY_RECORDING_GROUPS = {667: "ebird", 664: "tier", 351: "xeno"}


def split_key(path, metadata):
    """The key recordings are split into train, validation and test by, so
    recordings sharing it stay together e.g. all recordings from a site and
    device. The first of site and device, site, stationId, location lat and
    lng, deviceId or the day of the recording that the metadata has, and the
    recording's own id only when it has none of them. Train_001 and Train_002
    keys start with the dataset so they can't clash with other datasets.
    Recordings in SPLIT_BY_RECORDING_GROUPS always use their id and day"""
    dataset = next((d for d in ID_PREFIX_DIRS if d in str(path)), None)

    def key(*parts):
        return "-".join(([dataset] if dataset else []) + list(parts))

    rec_id = metadata.get("id")
    if rec_id is None:
        rec_id = Path(path).stem
    when = metadata.get("rec_datetime") or metadata.get("recordingDateTime")
    try:
        group = int(metadata.get("groupId"))
    except (TypeError, ValueError):
        group = None
    if group in SPLIT_BY_RECORDING_GROUPS:
        if when:
            return key(f"id:{rec_id}", f"day:{recording_day(when)}")
        return key(f"id:{rec_id}")

    site = metadata.get("site")
    device = metadata.get("device")
    if site and device:
        return key(f"site:{site}", f"device:{device}")
    if site:
        return key(f"site:{site}")
    if metadata.get("stationId") is not None:
        return key(f"station:{metadata['stationId']}")
    location = metadata.get("location")
    if isinstance(location, list):
        location = location[0] if location else None
    if isinstance(location, dict):
        lat, lng = location.get("lat"), location.get("lng")
        if lat is not None and lng is not None:
            return key(f"location:{lat},{lng}")
    if metadata.get("deviceId") is not None:
        return key(f"device:{metadata['deviceId']}")
    if when:
        return key(f"day:{recording_day(when)}")
    return key(f"id:{rec_id}")


def recording_day(when):
    # the date of an iso datetime e.g. 2012-03-21T12:43:16 or 2022-10-09T01:31:07.000Z
    try:
        return (
            datetime.fromisoformat(str(when).replace("Z", "+00:00")).date().isoformat()
        )
    except ValueError:
        return str(when)[:10]


def prefixed_id(path, rec_id):
    """A recording's id, prefixed with Train_001- or Train_002- when its file
    is in one of those datasets so ids are unique across them. Other ids are
    returned unchanged"""
    if rec_id is None:
        return None
    for prefix in ID_PREFIX_DIRS:
        if prefix in str(path):
            rec_id = str(rec_id)
            # some ids already have it e.g. /data2/.../Train_001-846
            if f"{prefix}-" in rec_id:
                return rec_id[rec_id.index(f"{prefix}-") :]
            return f"{prefix}-{rec_id}"
    return rec_id


def get_all_birds():
    allbirds = set()
    first = True
    with open("eBird_taxonomy_v2024.csv", newline="") as csvfile:
        dreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        _ = next(dreader)
        for row in dreader:
            allbirds.add(row[2])
    with open("classes.csv", newline="") as csvfile:
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
    ebird_map["chicken"] = "redjun1"
    ebird_map["crow"] = "rook1"
    ebird_map["norfolk-parrot"] = "noipar1"
    ebird_map["pigeon"] = "rocpig"
    ebird_map["brown-creeper"] = "pipipi1"
    ebird_map["new-zealand-dotterel"] = "rebdot1"
    ebird_map["sacred-kingfisher"] = "sackin3"
    ebird_map["rock-wren"] = "soiwre1"
    ebird_map["redpoll"] = "comred"
    ebird_map["norfolk-golden-whistler"] = "y01193"
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
