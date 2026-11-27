"""Proposes tighter tracks from a perchembed --analyse flags csv. For each
recording and label, keeps the windows whose probe probability for the track's
own label is >= --min-prob, clipped to the track they came from, and merges
overlapping kept windows into segments. Overlapping tracks of the same label in
a recording are merged so a call isn't counted twice.

Mostly a filter: the probe's scores are skewed towards 1 (median own score
0.994), so for common labels nearly every window passes and only tracks the
probe can't hear at all lose windows. For labels it has seen little of, e.g.
tomtit, gaps score near 0 and calls 0.3-0.9, so long loose tracks are cut down
to their calls.

  python3 scripts/tighten_tracks.py cacophony-flags-<model>.csv tightened.csv
Tracks are kept whole when they have fewer than --min-windows windows (default
2), when their label isn't scored by the probe (too rare), or when they carry a
--keep-labels label.

--keep-labels defaults to comred and nezpig2, which have so few tracks (5 and
3) that the probe scores nearly all of them 0, so filtering would delete them.
To choose these labels with data rather than by hand: per label, the share of
its own windows scoring below --min-prob. Where that's over about 50% the probe
can't hear the label rather than the tracks being wrong, so the label should be
kept whole or reviewed by ear, as for tomtit (52% below, reviewed) and nezkak1
(44% below, see nezkak1-review.csv).
Writes a row per segment, source "original" for kept whole, "tightened" for
cut down, and a row with empty start/end for recordings where nothing was kept,
so they can be checked or dropped."""

import argparse
import logging

import pandas as pd


def merge(intervals):
    """Merges overlapping (start, end) intervals"""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("flags", help="perchembed --analyse flags csv")
    parser.add_argument("out", help="csv of proposed segments")
    parser.add_argument(
        "--labels", nargs="+", help="Only these labels, default all"
    )
    parser.add_argument(
        "--keep-labels", nargs="+", default=["comred", "nezpig2"],
        help="Tracks with any of these labels are kept whole",
    )
    parser.add_argument("--min-prob", type=float, default=0.4)
    parser.add_argument(
        "--min-windows", type=int, default=2,
        help="Only tighten tracks with at least this many windows, shorter are kept whole",
    )
    args = parser.parse_args()

    d = pd.read_csv(args.flags, low_memory=False)
    d = d[d.label.notna()]
    tags = d.label.str.split(";").apply(set)
    if args.labels:
        d = d[tags.apply(lambda t: bool(t & set(args.labels)))]
        tags = tags[d.index]
    keep_whole = tags.apply(lambda t: bool(t & set(args.keep_labels)))
    d = d.assign(keep_whole=keep_whole | d.probe_own_prob.isna())
    rows = []
    for (rec, label), g in d.groupby(["recording_id", "label"]):
        tracks = g.drop_duplicates("track_id")
        base = dict(
            recording_id=rec,
            label=label,
            file=g.file.iloc[0],
            track_ids=";".join(str(t) for t in tracks.track_id),
            tagged_s=round(
                sum(e - s for s, e in merge(zip(tracks.track_start, tracks.track_end))),
                2,
            ),
        )
        # a track of fewer than --min-windows windows is short enough already,
        # and keep_whole ones can't be judged by the probe, both are kept whole.
        # Longer ones keep only their windows that pass
        intervals = []
        for _, t in g.groupby("track_id"):
            if len(t) < args.min_windows or t.keep_whole.iloc[0]:
                intervals.append((t.track_start.iloc[0], t.track_end.iloc[0], "original", None))
                continue
            for _, w in t[t.probe_own_prob >= args.min_prob].iterrows():
                intervals.append(
                    (max(w.window_start, w.track_start), min(w.window_end, w.track_end),
                     "tightened", w.probe_own_prob)
                )
        segments = merge((s, e) for s, e, _, _ in intervals)
        if not segments:
            rows.append(
                dict(base, start="", end="", source="discarded",
                     max_prob=round(g.probe_own_prob.max(), 3))
            )
        for s, e in segments:
            inside = [i for i in intervals if i[0] < e and i[1] > s]
            probs = [p for *_, p in inside if p is not None]
            rows.append(
                dict(base, start=round(s, 2), end=round(e, 2),
                     source=";".join(sorted({i[2] for i in inside})),
                     max_prob=round(max(probs), 3) if probs else "")
            )
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    found = out[out.start != ""]
    logging.info(
        "%s recording labels, %s discarded, %.0fs tagged -> %.0fs kept",
        len(out.drop_duplicates(["recording_id", "label"])),
        (out.source == "discarded").sum(),
        out.drop_duplicates(["recording_id", "label"]).tagged_s.sum(),
        (found.end.astype(float) - found.start.astype(float)).sum(),
    )
    for label, g in out.groupby("label"):
        recs = g.drop_duplicates("recording_id")
        found = g[g.start != ""]
        if len(recs) < 20:
            continue
        logging.info(
            "  %s: %s recordings, %s discarded (no window >= %s), %s segments, %.0fs tagged -> %.0fs kept",
            label,
            len(recs),
            recs.recording_id.nunique() - found.recording_id.nunique(),
            args.min_prob,
            len(found),
            recs.tagged_s.sum(),
            (found.end.astype(float) - found.start.astype(float)).sum(),
        )
    logging.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
