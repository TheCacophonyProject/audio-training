"""Tests for loading recordings, merging clips of one recording (e.g. tier1) and
making their samples, with and without best_segment.

python3 -m unittest discover tests
"""

import json
import logging
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audiodataset as ad  # noqa: E402
import build  # noqa: E402

SEG = 3


def fake_tighten(self, metadata, segment_length, tighten, filter_rms):
    # tighten_track needs rms data, the best window is the metadata's
    # best_start (test only) or the track start
    self.best_start = metadata.get("best_start", self.start)
    self.best_end = min(self.best_start + segment_length, self.end)


def track(track_id, label, start, end, offset=None, best_start=None, secondary=()):
    t = {
        "id": track_id,
        "start": start,
        "end": end,
        "tags": [{"automatic": False, "what": label}]
        + [{"automatic": False, "what": s, "secondary": True} for s in secondary],
    }
    if offset is not None:
        t["offset_start"] = offset
        t["offset_end"] = offset + end
    if best_start is not None:
        t["best_start"] = best_start
    return t


def clip_meta(rec_id, tracks, duration=5.0, birdnet=None):
    meta = {"id": rec_id, "duration": duration, "signal": [], "Tracks": tracks}
    if birdnet is not None:
        meta["birdnet"] = birdnet
    return meta


def write_dir(directory, files):
    """files {name: metadata}, written with an empty audio file next to each"""
    for name, meta in files.items():
        (directory / f"{name}.txt").write_text(json.dumps(meta))
        (directory / f"{name}.flac").touch()


def load(directory, best_segment, seed=0):
    np.random.seed(seed)
    config = ad.Config(best_segment=best_segment, dont_filter_rms=True)
    dataset = ad.AudioDataset("test", config)
    dataset.load_meta(directory)
    return dataset


def old_load(directory, best_segment, seed=0):
    """load_meta as it was, each file's samples made as it loads, before merging"""
    np.random.seed(seed)
    config = ad.Config(best_segment=best_segment, dont_filter_rms=True)
    dataset = ad.AudioDataset("test", config)
    for f in sorted(Path(directory).glob("**/*.txt")):
        dataset.add_recording(
            ad.Recording(ad.load_metadata(f), f.with_suffix(".flac"), config)
        )
    return dataset


def sample_keys(dataset):
    return [
        (
            s.rec_id,
            tuple(s.track_ids),
            round(float(s.start), 6),
            round(float(s.end), 6),
            tuple(s.tags),
            tuple(s.secondary_tags),
        )
        for s in dataset.samples
    ]


def own_track(sample):
    return sample.track_ids[-1]


class LoadTest(unittest.TestCase):
    """Recordings that aren't merged come out the same as before samples were
    made after loading"""

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        rng = np.random.RandomState(3)
        labels = ["morepork", "tui", "bellbird", "kiwi", "rain"]
        files = {}
        for i in range(60):
            duration = float(rng.uniform(20, 60))
            tracks = []
            for j in range(rng.randint(1, 5)):
                start = float(rng.uniform(0, duration - 2))
                end = float(min(duration, start + rng.uniform(1, 30)))
                best = float(rng.uniform(start, max(start, end - SEG)))
                tracks.append(
                    track(
                        i * 10 + j,
                        labels[rng.randint(len(labels))],
                        start,
                        end,
                        best_start=best,
                    )
                )
            files[f"{i:03d}"] = clip_meta(i, tracks, duration)
        write_dir(cls.dir, files)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        logging.disable(logging.NOTSET)

    def check_same(self, best_segment):
        with mock.patch.object(ad.Track, "tighten_track", fake_tighten):
            new = load(self.dir, best_segment)
            old = old_load(self.dir, best_segment)
        self.assertEqual(len(new.recs), 60)
        self.assertGreater(len(new.samples), 0)
        self.assertEqual(sample_keys(new), sample_keys(old))
        # the dataset's samples are its recordings' samples
        self.assertEqual(
            len(new.samples), sum(len(r.samples) for r in new.recs.values())
        )
        return new

    def test_without_best_segment(self):
        self.check_same(False)

    def test_best_segment(self):
        dataset = self.check_same(True)
        tracks = {t.id: t for r in dataset.recs.values() for t in r.tracks}
        for s in dataset.samples:
            # each sample is its track's best window, a window shorter than a
            # segment is expanded to a full segment
            t = tracks[own_track(s)]
            self.assertLessEqual(s.start, t.best_start + 1e-5)
            self.assertGreaterEqual(s.end, t.best_end - 1e-5)
            self.assertAlmostEqual(s.end - s.start, SEG, places=5)

    def test_best_segment_one_sample_per_track(self):
        with mock.patch.object(ad.Track, "tighten_track", fake_tighten):
            dataset = load(self.dir, True)
        per_track = {}
        for s in dataset.samples:
            per_track[own_track(s)] = per_track.get(own_track(s), 0) + 1
        self.assertTrue(all(n == 1 for n in per_track.values()))


class MergedTest(unittest.TestCase):
    """Clips of one recording are merged before samples are made, overlaps use
    offset_start to put each clip on the recording's timeline"""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()
        logging.disable(logging.NOTSET)

    def load(self, files, best_segment):
        write_dir(self.dir, files)
        with mock.patch.object(ad.Track, "tighten_track", fake_tighten):
            return load(self.dir, best_segment)

    def three_clips(self, kiwi_best=0.0):
        # morepork clip at 100-105s, kiwi 102-107s overlapping it, tui 300-305s
        return {
            "R_001": clip_meta("R", [track(1, "morepork", 0, 5.0, offset=100.0)]),
            "R_002": clip_meta(
                "R", [track(2, "kiwi", 0, 5.0, offset=102.0, best_start=kiwi_best)]
            ),
            "R_003": clip_meta("R", [track(3, "tui", 0, 5.0, offset=300.0)]),
        }

    def labels_by_track(self, dataset):
        out = {}
        for s in dataset.samples:
            out.setdefault(own_track(s), []).append(set(s.tags))
        return out

    def test_clips_merge_into_one_recording(self):
        for best_segment in (False, True):
            with self.subTest(best_segment=best_segment):
                dataset = self.load(self.three_clips(), best_segment)
                self.assertEqual(list(dataset.recs), ["R"])
                r = dataset.recs["R"]
                self.assertEqual(
                    [(t.id, t.filename.name) for t in r.tracks],
                    [(1, "R_001.flac"), (2, "R_002.flac"), (3, "R_003.flac")],
                )
                # each track's clip is its own recording, for its file/metadata
                clips = build.clip_tracks_of(r)
                self.assertEqual(
                    [(c.filename.name, [t.id for t in ts]) for c, ts in clips.items()],
                    [("R_001.flac", [1]), ("R_002.flac", [2]), ("R_003.flac", [3])],
                )
                self.assertEqual(len(dataset.samples), len(r.samples))

    def test_rms_filtered_track_doesnt_label_overlaps(self):
        def filtered_tighten(self, metadata, segment_length, tighten, filter_rms):
            fake_tighten(self, metadata, segment_length, tighten, filter_rms)
            self.rms_filtered = metadata.get("rms_filtered", False)

        files = self.three_clips()
        files["R_002"]["Tracks"][0]["rms_filtered"] = True
        write_dir(self.dir, files)
        kiwi = ad.get_ebird_id("kiwi", ad.labels_to_ebird_map)
        for best_segment in (False, True):
            with self.subTest(best_segment=best_segment):
                with mock.patch.object(ad.Track, "tighten_track", filtered_tighten):
                    dataset = load(self.dir, best_segment)
                labels = self.labels_by_track(dataset)
                # the filtered kiwi track makes no samples and isn't added to
                # the morepork samples it overlaps
                self.assertNotIn(2, labels)
                self.assertTrue(labels[1])
                self.assertTrue(all(kiwi not in tags for tags in labels[1]))
                self.assertTrue(all(2 not in s.track_ids for s in dataset.samples))

    def test_overlapping_clips_share_labels(self):
        dataset = self.load(self.three_clips(), False)
        labels = self.labels_by_track(dataset)
        kiwi = ad.get_ebird_id("kiwi", ad.labels_to_ebird_map)
        self.assertTrue(any(kiwi in tags for tags in labels[1]))
        self.assertTrue(all(len(tags) == 1 for tags in labels[3]))

    def test_best_segment_overlap_uses_other_best_window(self):
        # kiwi's best window 102-105s overlaps morepork's 100-103s
        dataset = self.load(self.three_clips(kiwi_best=0.0), True)
        self.assertEqual([len(t) for t in self.labels_by_track(dataset)[1]], [2])
        self.tmp.cleanup()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # kiwi's best window 104-107s, the clips overlap but the windows don't
        dataset = self.load(self.three_clips(kiwi_best=2.0), True)
        labels = self.labels_by_track(dataset)
        self.assertEqual([len(t) for t in labels[1]], [1])
        self.assertEqual([len(t) for t in labels[2]], [1])

    def test_no_offsets_different_files_dont_overlap(self):
        files = {
            "R_001": clip_meta("R", [track(1, "morepork", 0, 5.0)]),
            "R_002": clip_meta("R", [track(2, "kiwi", 0, 5.0)]),
        }
        dataset = self.load(files, False)
        for s in dataset.samples:
            self.assertEqual(len(s.track_ids), 1)
            self.assertEqual(len(s.tags), 1)

    def test_same_label_overlapping_clips_deduplicated(self):
        files = {
            "R_001": clip_meta("R", [track(1, "morepork", 0, 5.0, offset=100.0)]),
            "R_002": clip_meta("R", [track(2, "morepork", 0, 5.0, offset=101.0)]),
        }
        dataset = self.load(files, False)
        # the second clip's samples hold the same audio as the first's, which
        # are made first
        self.assertEqual({own_track(s) for s in dataset.samples}, {1})

    def test_same_label_no_offsets_not_deduplicated(self):
        files = {
            "R_001": clip_meta("R", [track(1, "morepork", 0, 5.0)]),
            "R_002": clip_meta("R", [track(2, "morepork", 0, 5.0)]),
        }
        dataset = self.load(files, False)
        self.assertEqual({own_track(s) for s in dataset.samples}, {1, 2})

    def test_short_sample_clamped_to_own_clip(self):
        # a 2s track in a 2s clip, merged with a 5s clip first
        files = {
            "R_001": clip_meta("R", [track(1, "morepork", 0, 5.0, offset=100.0)], 5.0),
            "R_002": clip_meta(
                "R", [track(2, "kiwi", 0, 2.0, offset=300.0)], duration=2.0
            ),
        }
        dataset = self.load(files, False)
        for s in dataset.samples:
            if own_track(s) == 2:
                self.assertLessEqual(s.end, 2.0)


class DropRemovedOverlapsTest(unittest.TestCase):
    """An overlapping track is dropped, with its tags, from a kept sample if
    its overlap section in the sample is rejected"""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()
        logging.disable(logging.NOTSET)

    def load(self, files, best_segment):
        write_dir(self.dir, files)
        with mock.patch.object(ad.Track, "tighten_track", fake_tighten):
            dataset = load(self.dir, best_segment)
        return dataset, {str(t.id): t for t in dataset.recs["R"].tracks}

    def two_clips(self, birdnet=None):
        return {
            "R_001": clip_meta(
                "R", [track(1, "morepork", 0, 5.0, offset=100.0)], birdnet=birdnet
            ),
            "R_002": clip_meta(
                "R", [track(2, "kiwi", 0, 5.0, offset=102.0)], birdnet=birdnet
            ),
        }

    def reject(self, sample, track_id):
        start, end, _ = sample.overlaps[track_id]
        return {build.overlap_key(track_id, start, end)}

    def test_overlaps_stored_in_other_tracks_file_times(self):
        dataset, _ = self.load(self.two_clips(), True)
        morepork = [s for s in dataset.samples if own_track(s) == 1][0]
        # morepork's best window 100-103s, kiwi's 102-105s, so 0-1s of kiwi's file
        start, end, min_overlap = morepork.overlaps[2]
        self.assertAlmostEqual(start, 0.0, places=5)
        self.assertAlmostEqual(end, 1.0, places=5)
        self.assertEqual(min_overlap, 0.5)

    def test_rejected_section_dropped(self):
        dataset, tracks = self.load(self.two_clips(), True)
        morepork = [s for s in dataset.samples if own_track(s) == 1][0]
        self.assertEqual(len(morepork.tags), 2)
        lost = build.drop_removed_overlaps(morepork, self.reject(morepork, 2), tracks)
        self.assertEqual(lost, {ad.get_ebird_id("kiwi", ad.labels_to_ebird_map)})
        self.assertEqual(morepork.track_ids, [1])
        self.assertEqual(morepork.overlaps, {})
        self.assertEqual(morepork.tags, sorted(tracks["1"].human_tags))

    def test_other_section_kept(self):
        dataset, tracks = self.load(self.two_clips(), True)
        morepork = [s for s in dataset.samples if own_track(s) == 1][0]
        lost = build.drop_removed_overlaps(
            morepork, {build.overlap_key(2, 3.0, 5.0)}, tracks
        )
        self.assertEqual(lost, set())
        self.assertIn(2, morepork.track_ids)

    def test_label_kept_if_own_track_has_it(self):
        # one file, two overlapping morepork tracks, the second with a secondary
        files = {
            "R_001": clip_meta(
                "R",
                [
                    track(1, "morepork", 0, 10.0),
                    track(2, "morepork", 1.0, 10.0, secondary=["tui"]),
                ],
                duration=10.0,
            )
        }
        dataset, tracks = self.load(files, True)
        sample = [s for s in dataset.samples if own_track(s) == 1 and 2 in s.track_ids][
            0
        ]
        tui = ad.get_ebird_id("tui", ad.labels_to_ebird_map)
        self.assertIn(tui, sample.secondary_tags)
        lost = build.drop_removed_overlaps(sample, self.reject(sample, 2), tracks)
        self.assertEqual(lost, set())
        self.assertNotIn(2, sample.track_ids)
        self.assertEqual(sample.tags, sorted(tracks["1"].human_tags))
        # the removed track's secondary goes too
        self.assertNotIn(tui, sample.secondary_tags)

    def filter(self, dataset, reject_track, best_segment, perch_flags=None, seen=None):
        """filter_bird_windows with birdnet judging stubbed, the windows of
        reject_track are rejected, returns the windows judged per track. seen
        gets the perch rows each window was judged with"""
        judged = {}

        def fake_track_results(filename, meta, taxonomy, args, window_rows=None):
            for t in meta["Tracks"]:
                track_id = str(t["id"])
                if args.windows is not None:
                    windows = args.windows.get(track_id, [])
                elif track_id in args.segments:
                    windows = [args.segments[track_id]]
                else:
                    windows = [(t["start"], t["end"])]
                for start, end in windows:
                    judged.setdefault(track_id, []).append((start, end))
                    window_rows.append(
                        {"track_id": t["id"], "window_start": start, "window_end": end}
                    )
            return iter(())

        def fake_outcome(window, perch_rows, args, tags=()):
            if seen is not None:
                seen.setdefault(str(window["track_id"]), []).extend(perch_rows)
            if str(window["track_id"]) == str(reject_track):
                return "rejected", True
            return "kept", False

        args = Namespace(
            perch_flags=perch_flags,
            birdnet_min_conf=0.5,
            clear_signal_db=6,
            best_segment=best_segment,
            filter_dry_run=False,
        )
        with mock.patch("birdnetcompare.track_results", fake_track_results), mock.patch(
            "birdnetcompare.load_taxonomy", lambda path: None
        ), mock.patch.object(build, "window_outcome", fake_outcome):
            build.filter_bird_windows(dataset, args)
        return judged

    def test_filter_judges_overlap_sections_in_own_clip(self):
        for best_segment in (False, True):
            with self.subTest(best_segment=best_segment):
                self.tmp.cleanup()
                self.tmp = tempfile.TemporaryDirectory()
                self.dir = Path(self.tmp.name)
                dataset, tracks = self.load(self.two_clips(birdnet=[]), best_segment)
                morepork = [s for s in dataset.samples if own_track(s) == 1]
                sections = {
                    tuple(round(v, 3) for v in s.overlaps[2][:2])
                    for s in morepork
                    if 2 in s.overlaps
                }
                self.assertTrue(sections)
                # morepork's windows kept, kiwi's rejected
                judged = self.filter(dataset, 2, best_segment)
                # each of morepork's overlap sections was judged for kiwi
                self.assertTrue(sections <= set(judged["2"]))
                kiwi = ad.get_ebird_id("kiwi", ad.labels_to_ebird_map)
                for s in dataset.samples:
                    self.assertEqual(own_track(s), 1)
                    self.assertNotIn(kiwi, s.tags)
                    self.assertEqual(s.track_ids, [1])

    def test_filter_keeps_tags_of_passed_sections(self):
        dataset, tracks = self.load(self.two_clips(birdnet=[]), False)
        before = {s.id: list(s.tags) for s in dataset.samples if own_track(s) == 2}
        # morepork rejected, kiwi kept, so kiwi samples lose morepork
        self.filter(dataset, 1, False)
        morepork = ad.get_ebird_id("morepork", ad.labels_to_ebird_map)
        kiwi_samples = [s for s in dataset.samples if own_track(s) == 2]
        self.assertTrue(kiwi_samples)
        self.assertTrue(any(morepork in tags for tags in before.values()))
        for s in kiwi_samples:
            self.assertNotIn(morepork, s.tags)

    def test_perch_matched_by_recording_and_track(self):
        import csv

        dataset, _ = self.load(self.two_clips(birdnet=[]), False)
        perch_file = self.dir / "perch.csv"
        with open(perch_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, ["recording_id", "track_id", "perch_start", "perch_end", "which"]
            )
            writer.writeheader()
            # track 2 of this recording, and a track 2 of another recording
            writer.writerow(
                {
                    "recording_id": "R",
                    "track_id": "2",
                    "perch_start": 0,
                    "perch_end": 5,
                    "which": "mine",
                }
            )
            writer.writerow(
                {
                    "recording_id": "OTHER",
                    "track_id": "2",
                    "perch_start": 0,
                    "perch_end": 5,
                    "which": "other",
                }
            )
        seen = {}
        self.filter(dataset, None, False, perch_flags=perch_file, seen=seen)
        self.assertTrue(seen["2"])
        self.assertEqual({row["which"] for row in seen["2"]}, {"mine"})
        # track 1 has no perch rows of its own
        self.assertEqual(seen.get("1"), [])

    def test_filter_without_birdnet_keeps_overlaps(self):
        dataset, tracks = self.load(self.two_clips(), False)
        before = [(s.id, list(s.tags)) for s in dataset.samples]
        self.filter(dataset, 2, False)
        self.assertEqual([(s.id, list(s.tags)) for s in dataset.samples], before)


class SecondaryPerClipTest(unittest.TestCase):
    """check_secondary_tags uses the birdnet results of each sample's own clip"""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()
        logging.disable(logging.NOTSET)

    def test_secondary_checked_against_own_clip(self):
        tui_detection = {
            "start": 0.0,
            "end": 5.0,
            "confidence": 0.9,
            "scientific_name": "Prosthemadera novaeseelandiae",
            "common_name": "Tui",
        }
        # both clips list tui as secondary, only the second clip's birdnet
        # found it, the first clip loads first so is the merged rec's metadata
        files = {
            "R_001": clip_meta(
                "R",
                [track(1, "morepork", 0, 5.0, offset=100.0, secondary=["tui"])],
                birdnet=[],
            ),
            "R_002": clip_meta(
                "R",
                [track(2, "kiwi", 0, 5.0, offset=300.0, secondary=["tui"])],
                birdnet=[tui_detection],
            ),
        }
        write_dir(self.dir, files)
        with mock.patch.object(ad.Track, "tighten_track", fake_tighten):
            dataset = load(self.dir, False)
        args = Namespace(birdnet_min_conf=0.5, secondary_min_overlap=1.0)
        build.check_secondary_tags(dataset, args)
        tui = ad.get_ebird_id("tui", ad.labels_to_ebird_map)
        for s in dataset.samples:
            if own_track(s) == 1:
                self.assertEqual((s.secondary_tags, s.soft_tags), ([], [tui]))
            else:
                self.assertEqual((s.secondary_tags, s.soft_tags), ([tui], []))


if __name__ == "__main__":
    unittest.main()


class GroupByTrackTest(unittest.TestCase):
    """Evaluation groups samples by recording and track id, track ids are only
    unique within a recording (tier1 clip numbers)"""

    def test_same_track_id_in_two_recordings(self):
        import tensorflow as tf

        import audiomodel

        rec = np.array([b"A", b"A", b"B", b"A"])
        trk = np.array([b"16", b"16", b"16", b"17"])
        lab = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], np.float32)
        zeros = np.zeros_like(lab)
        y = (lab, zeros, np.zeros(4, np.float32), rec, trk) + (zeros,) * 6
        ds = tf.data.Dataset.from_tensor_slices(
            (np.zeros((4, 1), np.float32), y)
        ).batch(2)
        pred = np.array([[0.9, 0.1], [0.7, 0.1], [0.2, 0.8], [0.1, 0.6]], np.float32)
        true, track_pred, _, _ = audiomodel.group_by_track(ds, (lab, pred))
        # A|16, A|17, B|16
        self.assertEqual(len(true), 3)
        rows = sorted(zip(true.tolist(), np.round(track_pred, 3).tolist()))
        self.assertEqual(
            rows,
            [
                ([0.0, 1.0], [0.1, 0.6]),
                ([0.0, 1.0], [0.2, 0.8]),
                ([1.0, 0.0], [0.8, 0.1]),
            ],
        )
