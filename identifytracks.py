import numpy as np
import librosa
from custommel import mel_spec
import math
import cv2

MAX_FRQUENCY = 48000 / 2
SIGNAL_WIDTH = 0.25
TOP_FREQ = 48000 / 2


def get_nfft(sr):
    base2 = round(math.log2(sr // 10))
    n_fft = int(math.pow(2, base2))
    return n_fft
    # n_fft = sr // 10
    nfft = sr / 10


def get_end(frames, sr):
    hop_length = 281
    n_fft = get_nfft(sr)
    spectogram = np.abs(librosa.stft(frames, n_fft=n_fft, hop_length=hop_length))
    mel = mel_spec(
        spectogram,
        sr,
        n_fft,
        hop_length,
        120,
        50,
        11000,
        1750,
        power=1,
    )
    start = 0
    chunk_length = sr // hop_length
    # this is roughtly a third of our spectogram used for classification
    end = start + chunk_length
    file_length = len(frames) / sr
    while end < mel.shape[1]:
        data = mel[:, start:end]
        if np.amax(data) == np.amin(data):
            # end of data
            return start * hop_length // sr
        start = end
        end = start + chunk_length
    return file_length


def signal_noise(
    frames, sr, hop_length=281, n_fft=4096, min_width=None, min_height=None
):
    # frames = frames[:sr]
    # n_fft = 4096
    # frames = frames[: sr * 3]
    spectogram = np.abs(librosa.stft(frames, n_fft=n_fft, hop_length=hop_length))
    og_spec = spectogram.copy()
    a_max = np.amax(spectogram)
    spectogram = spectogram / a_max
    row_medians = np.median(spectogram, axis=1)
    column_medians = np.median(spectogram, axis=0)
    rows, columns = spectogram.shape

    column_medians = column_medians[np.newaxis, :]
    row_medians = row_medians[:, np.newaxis]
    row_medians = np.repeat(row_medians, columns, axis=1)
    column_medians = np.repeat(column_medians, rows, axis=0)
    kernel = np.ones((4, 4), np.uint8)
    signal = (spectogram > 3 * column_medians) & (spectogram > 3 * row_medians)

    signal = signal.astype(np.uint8)
    signal = cv2.morphologyEx(signal, cv2.MORPH_OPEN, kernel)

    width = SIGNAL_WIDTH * sr / hop_length
    width = int(width)
    freq_range = 100
    height = 0
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    for i, f in enumerate(freqs):
        if f > freq_range:
            height = i + 1
            break

    signal = cv2.dilate(signal, np.ones((height, width), np.uint8))
    signal = cv2.erode(signal, np.ones((height // 10, width), np.uint8))

    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(signal)
    stats = stats[1:]
    stats = sorted(stats, key=lambda stat: stat[0])
    if min_height is None:
        min_height = height - height // 10
    if min_width is None:
        min_width = 0.65 * width

    print("Min height", min_height, " min width", min_width)
    stats = [s for s in stats if s[2] > min_width and s[3] > min_height]

    i = 0
    # indicator_vector = np.uint8(indicator_vector)
    s_start = -1
    signals = []

    bins = len(freqs)
    for s in stats:
        max_freq = min(len(freqs) - 1, s[1] + s[3])
        freq_range = (freqs[s[1]], freqs[max_freq])
        start = s[0] * 281 / sr
        end = (s[0] + s[2]) * 281 / sr
        signals.append(Signal(start, end, freq_range[0], freq_range[1], s[4]))

    return signals, og_spec


def segment_overlap(first, second):
    return (
        (first[1] - first[0])
        + (second[1] - second[0])
        - (max(first[1], second[1]) - min(first[0], second[0]))
    )


def mel_freq(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


# an attempt at getting frequency based tracks
# try and merge signals that are close together in time and frequency


def merge_signals(signals):
    unique_signals = []
    to_delete = []
    something_merged = False
    i = 0

    signals = sorted(signals, key=lambda s: s.mel_freq_end, reverse=True)
    signals = sorted(signals, key=lambda s: s.start)

    for s in signals:
        if s in to_delete:
            continue
        merged = False
        for u_i, u in enumerate(signals):
            if u in to_delete:
                continue
            if u == s:
                continue
            in_freq = u.mel_freq_end < 1500 and s.mel_freq_end < 1500
            in_freq = in_freq or u.mel_freq_end > 1500 and s.mel_freq_end > 1500
            # ensure both are either below 1500 or abov
            if not in_freq:
                continue
            overlap = s.time_overlap(u)
            if s.mel_freq_start > 1000 and u.mel_freq_start > 1000:
                freq_overlap = 0.1
                freq_overlap_time = 0.5
            else:
                freq_overlap = 0.5
                freq_overlap_time = 0.75
            if s.start > u.end:
                time_diff = s.start - u.end
            else:
                time_diff = u.start - s.end
            mel_overlap = s.mel_freq_overlap(u)
            if overlap > u.length * 0.75 and mel_overlap > -20:
                s.merge(u)
                merged = True

                break
            elif overlap > 0 and mel_overlap > u.mel_freq_range * freq_overlap_time:
                # time overlaps at all with more freq overlap
                s.merge(u)
                merged = True

                break

            elif mel_overlap > u.mel_freq_range * freq_overlap_time and time_diff <= 2:
                if u.mel_freq_end > s.mel_freq_range:
                    range_overlap = s.mel_freq_range / u.mel_freq_range
                else:
                    range_overlap = u.mel_freq_range / s.mel_freq_range
                if range_overlap < 0.75:
                    continue
                # freq range similar
                s.merge(u)
                merged = True

                break

        if merged:
            something_merged = True
            to_delete.append(u)

    for s in to_delete:
        signals.remove(s)

    return signals, something_merged


def get_tracks_from_signals(signals, end):
    # probably a much more efficient way of doing this
    # just keep merging until there are no more valid merges
    merged = True
    min_mel_range = 50

    while merged:
        signals, merged = merge_signals(signals)

    to_delete = []
    min_length = 0.35
    min_track_length = 0.7
    for s in signals:
        if s in to_delete:
            continue
        if s.length < min_length:
            to_delete.append(s)
            continue
        # s.enlarge(1.4, min_track_length=min_track_length)

        # s.end = min(end, s.end)
        for s2 in signals:
            if s2 in to_delete:
                continue
            if s == s2:
                continue

            # continue
            overlap = s.time_overlap(s2)
            mel_overlap = s.freq_overlap(s2)
            min_length = min(s.length, s2.length)

            # print(
            #     "TIme overlap between ",
            #     s,
            #     " and ",
            #     s2,
            #     " is ",
            #     overlap / min_length,
            #     mel_overlap,
            # )
            if overlap > 0.7 * min_length and abs(mel_overlap) < 2200:
                s.merge(s2)
                to_delete.append(s2)
                continue
            # engulfed = overlap >= 0.9 * s2.length
            # f_overlap = s.mel_freq_overlap(s2)
            # range = s2.mel_freq_range
            # range *= 0.7
            # if f_overlap > range and engulfed:
            #     to_delete.append(s2)

    for s in to_delete:
        signals.remove(s)
    for s in signals:
        s.enlarge(1.4, min_track_length=min_track_length)
        s.end = min(end, s.end)
    to_delete = []
    for s in signals:
        if s.mel_freq_range < min_mel_range:
            to_delete.append(s)
    for s in to_delete:
        signals.remove(s)
    return signals


SIGNAL_ID = 0


class Signal:
    def __init__(self, start, end, freq_start, freq_end, mass):
        global SIGNAL_ID
        self.id = SIGNAL_ID
        SIGNAL_ID += 1
        self.start = start
        self.end = end
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.mass = mass
        self.mel_freq_start = mel_freq(freq_start)
        self.mel_freq_end = mel_freq(freq_end)
        self.predictions = []
        self.track_id = None
        # self.model = None
        # self.labels = None
        # self.confidences = None
        # self.raw_tag = None
        # self.raw_confidence = None

    def to_features(self):
        return np.float32(
            [
                self.start,
                self.end,
                self.freq_start,
                self.freq_end,
                self.mel_freq_start,
                self.mel_freq_end,
            ]
        )

    def to_array(self, decimals=1):
        a = [self.start, self.end, self.freq_start, self.freq_end]
        if decimals is not None:
            a = list(
                np.round(
                    np.array(a),
                    decimals,
                )
            )
        return a

    def copy(self):
        return Signal(self.start, self.end, self.freq_start, self.freq_end)

    def time_overlap(self, other):
        return segment_overlap(
            (self.start, self.end),
            (other.start, other.end),
        )

    def mel_freq_overlap(self, other):
        return segment_overlap(
            (self.mel_freq_start, self.mel_freq_end),
            (other.mel_freq_start, other.mel_freq_end),
        )

    def freq_overlap(self, other):
        return segment_overlap(
            (self.freq_start, self.freq_end),
            (other.freq_start, other.freq_end),
        )

    @property
    def mel_freq_range(self):
        return self.mel_freq_end - self.mel_freq_start

    @property
    def freq_range(self):
        return self.freq_end - self.freq_start

    @property
    def length(self):
        return self.end - self.start

    def enlarge(self, scale, min_track_length):
        new_length = self.length * scale
        if new_length < min_track_length:
            new_length = min_track_length
        extension = (new_length - self.length) / 2
        self.start = self.start - extension
        self.end = self.end + extension
        self.start = max(self.start, 0)

        # also enlarge freq
        new_length = (self.freq_end - self.freq_start) * scale
        extension = (new_length - (self.freq_end - self.freq_start)) / 2
        self.freq_start = self.freq_start - extension
        self.freq_end = int(self.freq_end + extension)
        self.freq_start = int(max(self.freq_start, 0))

        self.mel_freq_start = mel_freq(self.freq_start)
        self.mel_freq_end = mel_freq(self.freq_end)

    def merge(self, other):
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)
        self.freq_start = min(self.freq_start, other.freq_start)
        self.freq_end = max(self.freq_end, other.freq_end)
        self.mel_freq_start = mel_freq(self.freq_start)
        self.mel_freq_end = mel_freq(self.freq_end)
        self.mass += other.mass

    def __str__(self):
        return f"Signal: {self.start}-{self.end} f: {self.freq_start}-{self.freq_end} mass {self.mass}"

    def get_meta(self):
        meta = {}
        meta["id"] = self.id
        meta["start"] = self.start
        meta["end"] = self.end
        meta["freq_start"] = self.freq_start
        meta["freq_end"] = self.freq_end
        meta["positions"] = [
            {
                "y": self.freq_start / TOP_FREQ,
                "height": (self.freq_end - self.freq_start) / TOP_FREQ,
            }
        ]
        meta["predictions"] = [r.get_meta() for r in self.predictions]
        if self.track_id is not None:
            meta["track_id"] = self.track_id
        return meta
