import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

# how close birdnet's guess is to the manual tag, best first
MATCH = "match"
RELATED = "related"
DIFFERENT = "different"
NO_BIRD = "no_bird"
NON_BIRD_TAG = "non_bird_tag"
STATUS_ORDER = [MATCH, RELATED, DIFFERENT, NO_BIRD]

# report sections, in output order
NO_TAGS_SECTION = "Tracks with no birdnet tags"
DIFFERING_SECTION = "Tracks with differing tags"
# birdnet found no bird in the track but there is a clear signal in the best
# rms window, so the manual tag is probably right and birdnet missed it
MISSED_SECTION = "Tracks with no birdnet tags but a clear signal"
MATCH_SECTION = "Tracks with matches"
NON_BIRD_SECTION = "Tracks with non bird tags"
SECTIONS = [
    NO_TAGS_SECTION,
    DIFFERING_SECTION,
    MISSED_SECTION,
    MATCH_SECTION,
    NON_BIRD_SECTION,
]

# signals starting below this many Hz are usually wind or handling noise, the
# default matches the lower edge of bird_rms in otherdata.py
SIGNAL_MIN_FREQ = 500
# species that call lower than that, by ebird code
SPECIES_SIGNAL_MIN_FREQ = {
    "ausbit1": 100,  # bittern
    "morepo2": 300,  # morepork
}

# manual tags that just mean "some bird"
GENERIC_BIRD_TAGS = {"bird", "unidentified", "other"}
HUMAN_SCIENTIFIC = "Homo Sapiens"


def load_taxonomy(path):
    by_code = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            by_code[row["SPECIES_CODE"]] = {
                "code": row["SPECIES_CODE"],
                "category": row["CATEGORY"],
                "scientific": row["SCI_NAME"],
                "common": row["PRIMARY_COM_NAME"],
                "family": row["FAMILY"],
                "report_as": row["REPORT_AS"],
            }
    by_scientific = {}
    by_common = {}
    by_genus = {}
    for taxon in by_code.values():
        by_scientific.setdefault(taxon["scientific"].lower(), taxon)
        by_common.setdefault(label_key(taxon["common"]), taxon)
        if taxon["category"] == "species":
            by_genus.setdefault(genus(taxon).lower(), taxon)
    return by_code, by_scientific, by_common, by_genus


def label_key(name):
    return name.lower().replace(" ", "-").replace("'", "")


def species_taxon(taxon, by_code):
    # subspecies / forms report as their parent species
    while taxon["category"] != "species" and taxon["report_as"] in by_code:
        taxon = by_code[taxon["report_as"]]
    return taxon


def tag_taxon(tag, taxonomy):
    by_code, _, by_common, _ = taxonomy
    taxon = by_code.get(tag.get("ebird_id")) or by_common.get(label_key(tag["what"]))
    return None if taxon is None else species_taxon(taxon, by_code)


def detection_taxon(detection, taxonomy):
    # birdnet 3.0 includes frogs, insects, mammals etc so only names in the
    # ebird taxonomy count as birds
    by_code, by_scientific, by_common, by_genus = taxonomy
    scientific = detection["scientific_name"]
    taxon = by_scientific.get(scientific.lower()) or by_common.get(
        label_key(detection["common_name"])
    )
    if taxon is not None:
        return species_taxon(taxon, by_code)
    # birdnet uses a few species names newer or older than this taxonomy, if the
    # genus is a bird genus treat it as its own bird species in that family
    relative = by_genus.get(scientific.split(" ")[0].lower())
    if relative is None:
        return None
    return {
        "code": scientific,
        "category": "species",
        "scientific": scientific,
        "common": detection["common_name"],
        "family": relative["family"],
        "report_as": "",
    }


def genus(taxon):
    return taxon["scientific"].split(" ")[0]


def same_species_renamed(a, b):
    # genus changes between taxonomy versions e.g. Charadrius -> Anarhynchus bicinctus
    epithet_a = a["scientific"].split(" ")[-1]
    epithet_b = b["scientific"].split(" ")[-1]
    return epithet_a == epithet_b and a["family"] == b["family"]


def compare(tag_taxa, generic_bird, bird_detections):
    if not bird_detections:
        return NO_BIRD, None
    if generic_bird and not tag_taxa:
        return MATCH, bird_detections[0]
    best = (DIFFERENT, None)
    for detection, taxon in bird_detections:
        for tag in tag_taxa:
            if taxon["code"] == tag["code"] or same_species_renamed(taxon, tag):
                return MATCH, (detection, taxon)
            if genus(taxon) == genus(tag) or taxon["family"] == tag["family"]:
                if best[0] != RELATED:
                    best = (RELATED, (detection, taxon))
    return best


def best_rms(rms, segment_length=3, sr=48000, hop_length=281):
    # copied from audiodataset.py, importing that pulls in tensorflow
    window_size = sr * segment_length / hop_length
    window_size = int(window_size)
    first_window = np.sum(rms[:window_size])
    rolling_sum = first_window
    max_index = (0, first_window)
    for i in range(1, len(rms) - window_size):
        rolling_sum = rolling_sum - rms[i - 1] + rms[i + window_size]
        if rolling_sum > max_index[1]:
            max_index = (i, rolling_sum)
    return max_index


def rms_window(track, meta, bird_track, segment_length):
    # loudest segment_length window of the track, rms starts at the track start
    rms = track.get("bird_rms" if bird_track else "noise_rms")
    if not rms:
        return None
    rms_hop = meta.get("rms_hop_length", 281)
    # rms is calculated on audio resampled to 48k
    rms_sr = meta.get("rms_sr", 48000)
    frame_length = rms_hop / rms_sr
    best_offset, _ = best_rms(np.array(rms), segment_length, rms_sr, rms_hop)
    start = track.get("start", 0) + best_offset * frame_length
    end = min(start + segment_length, track.get("end", start))
    return start, end


def signal_cutoff(tag_taxa, generic_bird):
    if not tag_taxa and not generic_bird:
        # noise, human etc can be at any frequency
        return 0
    cutoffs = [
        SPECIES_SIGNAL_MIN_FREQ.get(t["code"], SIGNAL_MIN_FREQ) for t in tag_taxa
    ]
    if generic_bird:
        cutoffs.append(SIGNAL_MIN_FREQ)
    return min(cutoffs)


def track_signals(meta, start, end, min_freq):
    """Signals from identifytracks.signal_noise overlapping the track and
    starting at or above min_freq, strongest first. Each is
    [start, end, min_freq, max_freq] plus strength (mean db above background)
    from signal_version 1.1."""
    signals = meta.get("signal")
    if signals is None:
        return None
    found = [s for s in signals if s[0] < end and s[1] > start and s[2] >= min_freq]

    def strength(s):
        if len(s) > 4:
            return s[4]
        # older signals have no strength so use the box size
        return (s[1] - s[0]) * (s[3] - s[2])

    return sorted(found, key=lambda s: -strength(s))


def row_section(row):
    if row["status"] == MATCH:
        return MATCH_SECTION
    if row["status"] == NON_BIRD_TAG:
        return NON_BIRD_SECTION
    if row["status"] == NO_BIRD:
        if row["rms_clear_signal"]:
            return MISSED_SECTION
        return NO_TAGS_SECTION
    return DIFFERING_SECTION


def clear_signal(signals, window, min_strength):
    """Strongest signal overlapping the best rms window, signals without a
    strength (signal_version 1.0) count if they overlap at all."""
    if not signals or window is None:
        return None
    for s in signals:
        if s[0] < window[1] and s[1] > window[0]:
            if len(s) <= 4 or s[4] >= min_strength:
                return s
    return None


def format_signals(signals):
    formatted = []
    for s in signals:
        text = f"{s[0]}-{s[1]}s {s[2]:.0f}-{s[3]:.0f}Hz"
        if len(s) > 4:
            text += f" {s[4]:.1f}dB"
        formatted.append(text)
    return "; ".join(formatted)


def overlapping(detections, start, end, tolerance):
    found = [
        d
        for d in detections
        if d["start"] < end + tolerance and d["end"] > start - tolerance
    ]
    # keep each species once at its highest confidence
    best_per_species = {}
    for d in sorted(found, key=lambda d: -d["confidence"]):
        best_per_species.setdefault(d["scientific_name"], d)
    return list(best_per_species.values())


def evaluate(found, tag_taxa, generic_bird, non_bird, taxonomy):
    bird_detections = []
    for d in found:
        taxon = detection_taxon(d, taxonomy)
        if taxon is not None:
            bird_detections.append((d, taxon))

    if tag_taxa or generic_bird:
        status, best = compare(tag_taxa, generic_bird, bird_detections)
    else:
        status, best = NON_BIRD_TAG, None
        human = [d for d in found if d["scientific_name"] == HUMAN_SCIENTIFIC]
        if "human" in non_bird and human:
            status, best = MATCH, (human[0], None)
    if best is None and bird_detections:
        best = bird_detections[0]
    return status, None if best is None else best[0], bird_detections


def format_detections(detections):
    return "; ".join(
        f"{d['common_name']}:{d['confidence']:.2f} ({d['start']}-{d['end']}s)"
        for d in detections
    )


def track_rows(txt, meta, taxonomy, args):
    detections = [d for d in meta["birdnet"] if d["confidence"] >= args.min_conf]
    tracks = [
        track
        for track in meta.get("tracks", [])
        if any(t.get("automatic") is False for t in track.get("tags", []))
    ]
    if not tracks:
        return

    for track in tracks:
        manual = [t for t in track.get("tags", []) if t.get("automatic") is False]
        start = track.get("start", 0)
        end = track.get("end", start)

        tag_taxa = []
        generic_bird = False
        non_bird = []
        for tag in manual:
            taxon = tag_taxon(tag, taxonomy)
            if taxon is not None:
                tag_taxa.append(taxon)
            elif tag["what"] in GENERIC_BIRD_TAGS:
                generic_bird = True
            else:
                non_bird.append(tag["what"])

        found = overlapping(detections, start, end, args.tolerance)
        status, best, bird_detections = evaluate(
            found, tag_taxa, generic_bird, non_bird, taxonomy
        )

        # same comparison using only the loudest window of the track
        window = rms_window(
            track, meta, bool(tag_taxa or generic_bird), args.segment_length
        )
        rms_status, rms_best = "", None
        if window is not None:
            rms_found = overlapping(detections, *window, args.tolerance)
            rms_status, rms_best, _ = evaluate(
                rms_found, tag_taxa, generic_bird, non_bird, taxonomy
            )

        cutoff = signal_cutoff(tag_taxa, generic_bird)
        signals = track_signals(meta, start, end, cutoff)
        strongest = signals[0] if signals else None
        rms_signal = clear_signal(signals, window, args.clear_signal_db)

        yield {
            "file": txt.name,
            "recording_id": meta.get("id"),
            "track_id": track.get("id"),
            "start": start,
            "end": end,
            "rms_start": "" if window is None else round(window[0], 2),
            "rms_end": "" if window is None else round(window[1], 2),
            "manual_tags": ";".join(t["what"] for t in manual),
            "manual_scientific": ";".join(t["scientific"] for t in tag_taxa),
            "status": status,
            "birdnet_tag": "" if best is None else best["common_name"],
            "birdnet_confidence": "" if best is None else best["confidence"],
            "birdnet_start": "" if best is None else best["start"],
            "birdnet_end": "" if best is None else best["end"],
            "rms_status": rms_status,
            "rms_birdnet_tag": "" if rms_best is None else rms_best["common_name"],
            "rms_birdnet_confidence": (
                "" if rms_best is None else rms_best["confidence"]
            ),
            "rms_birdnet_start": "" if rms_best is None else rms_best["start"],
            "rms_birdnet_end": "" if rms_best is None else rms_best["end"],
            "signal_cutoff": cutoff,
            "signal_count": "" if signals is None else len(signals),
            "signal_start": "" if strongest is None else strongest[0],
            "signal_end": "" if strongest is None else strongest[1],
            "signal_min_freq": "" if strongest is None else strongest[2],
            "signal_max_freq": "" if strongest is None else strongest[3],
            "signal_strength": (
                strongest[4] if strongest is not None and len(strongest) > 4 else ""
            ),
            "signals": "" if signals is None else format_signals(signals),
            "rms_clear_signal": (
                "" if rms_signal is None else format_signals([rms_signal])
            ),
            "birdnet_birds": format_detections([d for d, _ in bird_detections]),
            "birdnet_other": format_detections(
                [d for d in found if detection_taxon(d, taxonomy) is None]
            ),
            "birdnet_model": meta.get("birdnet_model"),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Compare manual track tags with birdnet detections"
    )
    parser.add_argument("dir", help="Directory to search for metadata .txt files")
    parser.add_argument("--out", default="birdnet-compare.csv")
    parser.add_argument(
        "--taxonomy",
        default=str(Path(__file__).parent / "eBird_taxonomy_v2024.csv"),
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.1,
        help="Ignore birdnet detections below this confidence",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=3,
        help="Length of the best rms window, birdnet uses 3s segments",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Seconds of slack when overlapping detections with tracks",
    )
    parser.add_argument(
        "--clear-signal-db",
        type=float,
        default=12,
        help="Minimum signal strength (db above background) to count as a clear "
        "signal, signals have to be at least ~9.5db to be found at all",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    rows = []
    skipped = 0
    for txt in sorted(Path(args.dir).rglob("*.txt")):
        try:
            with open(txt) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(meta, dict) or "birdnet" not in meta:
            skipped += 1
            continue
        rows.extend(track_rows(txt, meta, taxonomy, args))

    if not rows:
        logging.info("No tracks with manual tags and birdnet results found")
        return

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        for title in SECTIONS:
            section = [r for r in rows if row_section(r) == title]
            if not section and title == NON_BIRD_SECTION:
                continue
            section.sort(key=lambda r: (r["manual_tags"], r["file"], r["start"]))
            writer.writerow([f"{title} ({len(section)})"])
            writer.writerow(fieldnames)
            writer.writerows([r[k] for k in fieldnames] for r in section)
            writer.writerow([])

    counts = Counter(r["status"] for r in rows)
    rms_counts = Counter(r["rms_status"] for r in rows)
    logging.info("Wrote %s tracks to %s", len(rows), args.out)
    if skipped:
        logging.info("Skipped %s metadata files without birdnet results", skipped)
    for status in STATUS_ORDER + [NON_BIRD_TAG]:
        logging.info(
            "%s: %s (best rms window %s)",
            status,
            counts.get(status, 0),
            rms_counts.get(status, 0),
        )

    # tracks birdnet heard nothing in but which have a signal are the likely
    # birdnet misses, no signal either suggests an empty or wrong track
    no_bird = [r for r in rows if r["status"] == NO_BIRD and r["signal_count"] != ""]
    if no_bird:
        with_signal = sum(1 for r in no_bird if r["signal_count"] > 0)
        logging.info(
            "no_bird tracks with a signal: %s, without: %s",
            with_signal,
            len(no_bird) - with_signal,
        )
    missing_signal = sum(1 for r in rows if r["signal_count"] == "")
    if missing_signal:
        logging.info("No signal metadata for %s tracks", missing_signal)

    # per manual tag breakdown, worst performing tags are the interesting ones
    per_tag = {}
    for r in rows:
        for tag in r["manual_tags"].split(";"):
            per_tag.setdefault(tag, Counter())[r["status"]] += 1
    logging.info("Per tag (match / related / different / no_bird / total):")
    for tag, c in sorted(per_tag.items(), key=lambda kv: -sum(kv[1].values())):
        logging.info(
            "  %s: %s / %s / %s / %s / %s",
            tag,
            c[MATCH],
            c[RELATED],
            c[DIFFERENT],
            c[NO_BIRD],
            sum(c.values()),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
