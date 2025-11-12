import argparse
from pathlib import Path
import json
import logging
from audiodataset import (
    AudioDataset,
    RELABEL,
    Track,
    AudioSample,
    Config,
    LOW_SAMPLES_LABELS as dataset_low_samples,
)


import sys


def init_logging():
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """

    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", help="Directory of hdf5 files")
    parser.add_argument("out_dir", help="Directory of hdf5 files")

    args = parser.parse_args()
    args.out_dir = Path(args.out_dir)

    args.data_dir = Path(args.data_dir)
    return args


def main():
    init_logging()
    args = parse_args()
    config = Config(**vars(args))
    dataset = AudioDataset("all", config)
    dataset.load_meta(args.data_dir)
    location_id_map = {}
    global_loc_uid = 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for rec_id, rec in dataset.recs.items():
        json_meta = {}
        json_meta["id"] = rec_id
        json_meta["device"] = rec.device_id
        if rec.rec_date is not None:
            json_meta["datetime"] = rec.rec_date.isoformat()

        if rec.location is not None and len(rec.location) == 2:
            lng, lat = rec.location
            location_uid = f"{rec.device_id}#{lng}#{lat}"

            # make fuzzy
            lng = round(lng, 1)
            lat = round(lat, 1)
            json_meta["gps_location"] = [lng, lat]
        else:
            location_uid = f"{rec.device_id}#0#0"

        if location_uid in location_id_map:
            json_meta["location"] = location_id_map[location_uid]
        else:
            location_id_map[location_uid] = global_loc_uid
            json_meta["location"] = global_loc_uid
            global_loc_uid += 1

        json_meta["labels"] = list(rec.human_tags)
        tracks = []
        for t in rec.tracks:
            track = {"start": t.start, "end": t.end, "id": t.id}
            track["min_freq"] = t.min_freq if t.min_freq is not None else -1
            track["max_freq"] = t.max_freq if t.max_freq is not None else -1
            track["human_tags"] = list(t.human_tags)
            tracks.append(track)
        json_meta["tracks"] = tracks
        out_file = args.out_dir / f"{rec_id}_metdata.json"
        with out_file.open("w") as f:
            json.dump(json_meta, f, indent=4)


if __name__ == "__main__":
    main()
