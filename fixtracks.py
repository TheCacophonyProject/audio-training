import argparse
import json
import logging
from pathlib import Path


def fix_tracks(meta):
    if "tracks" not in meta:
        return None
    if "Tracks" in meta:
        del meta["tracks"]
        return "removed tracks"
    # rename in place to keep key order
    return_meta = {("Tracks" if k == "tracks" else k): v for k, v in meta.items()}
    meta.clear()
    meta.update(return_meta)
    return "renamed tracks to Tracks"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory to search for .txt/.json metadata files")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report changes without writing"
    )
    args = parser.parse_args()

    counts = {}
    for path in Path(args.dir).rglob("*"):
        if path.suffix not in (".txt", ".json") or not path.is_file():
            continue
        try:
            with open(path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        change = fix_tracks(meta)
        if change is None:
            continue
        counts[change] = counts.get(change, 0) + 1
        logging.info("%s: %s", path, change)
        if not args.dry_run:
            with open(path, "w") as f:
                json.dump(meta, f, indent=4)
    logging.info("%s%s", "Dry run " if args.dry_run else "", counts or "no changes")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
