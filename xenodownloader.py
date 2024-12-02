import requests
from pathlib import Path
import datetime
import json
import argparse


def download_file(url, local_filename):
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return local_filename


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bird", help="Bird data to download")
    parser.add_argument("dir", help="Directory to download into")

    args = parser.parse_args()
    args.dir = Path(args.dir)
    return args


base_url = "https://www.xeno-canto.org/api/2/recordings"
args = parse_args()
bird = args.bird
dl_path = Path(args.dir)
url = f"{base_url}?query={bird}"
print("getting", url)
r = requests.get(url)
r.raise_for_status()
results = r.json()
for r in results.get("recordings"):
    dl = r.get("file")
    date = r.get("date")
    time = r.get("time")
    try:
        date_time = datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    except:
        print("Error with", r.get("id"))
        if date[-2:] == "00":
            date = f"{date[:-2]}01"
        date_time = datetime.datetime.strptime(f"{date}", "%Y-%m-%d")

    filename = r.get("file-name")
    filename = dl_path / filename
    meta_file = filename.with_suffix(".txt")
    print("Saving meta", meta_file, meta_file.exists())
    if not meta_file.exists():
        if r.get("lat") is None or  r.get("lng") is None:
            location = {}
        else:
            location = {"lat": float(r.get("lat")), "lng": float(r.get("lng"))},
        meta_data = {
            "recordingDateTime": date_time.isoformat(),
            "location": location,
            "additionalMetadata": {
                "source": "xeno",
                "url": r.get("url"),
                "xeno-id": r.get("id"),
            },
        }
        duration = r.get("length")
        index = duration.index(":")
        duration = duration[index + 1 :]
        meta_data["duration"] = int(duration)
        with open(meta_file, "w") as f:
            json.dump(meta_data, f, indent=4)
        print("saving meta", meta_data)
    if filename.exists():
        print("already exists", filename)
    else:
        download_file(dl, filename)

# print("got results", results)
