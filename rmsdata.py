"""Per track rms data, used by otherdata.py and addtracks.py. Kept apart from
otherdata.py as importing that has side effects e.g. making directories"""

import logging

import librosa
import numpy as np

from audiodataset import Track

# 1.2 adds track_rms over each tracks own frequency range
RMS_VERSION = 2.0
# only calculate track_rms for tracks with a frequency range up to this many Hz
MAX_TRACK_RMS_RANGE = 10000


def add_rms_data_to_tracks(y, sr, tracks):

    n_fft = 4096
    hop_length = 281
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # just for bittern as the calls are in 0-500Hz
    min_noise_max_freq = 100
    min_noise_bin = 0

    noise_max_freq = 500
    lower_noise_bin = 0
    morepork_max_freq = 1200
    bittern_max_freq = 500
    upper_max_freq = 3000
    morepork_upper_bin = 0
    upper_noise_bin = 0
    bittern_upper_bin = 0
    max_bin = 0
    for i, f in enumerate(freqs):
        if f < min_noise_max_freq:
            min_noise_bin = i
        if f < noise_max_freq:
            lower_noise_bin = i
        if f < morepork_max_freq:
            morepork_upper_bin = i + 1
        if f < bittern_max_freq:
            bittern_upper_bin = i + 1
        if f > upper_max_freq:
            upper_noise_bin = i
            break
    # print("Bins", lower_noise_bin, morepork_upper_bin,upper_noise_bin)

    for t in tracks:

        track = Track(t, None, 0, None, tighten=False, filter_rms=False)

        # start = t["start"]
        # end = t["end"]
        track_frames = y[int(sr * track.start) : int(sr * track.end)]
        stft = librosa.stft(track_frames, n_fft=n_fft, hop_length=hop_length)

        # bird_tag = False
        # for tag in track.human_tags:
        #     if tag in SPECIFIC_BIRD_LABELS or tag in GENERIC_BIRD_LABELS:
        #         bird_tag = True

        noise_stft = stft.copy()
        noise_stft[lower_noise_bin + 1 :, :] = 0
        S, phase = librosa.magphase(noise_stft)
        noise_rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)

        upper_stft = stft.copy()
        upper_stft[:upper_noise_bin, :] = 0
        S, phase = librosa.magphase(upper_stft)
        upper_rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)
        t["upper_noise_bin"] = upper_noise_bin
        lower_bin = lower_noise_bin
        upper_bin = None
        logging.info("Human tags %s", track.human_tags)
        if "ausbit1" in track.human_tags:
            upper_bin = bittern_upper_bin
            lower_bin = min_noise_bin
            logging.info("Using bittern lower %s upper %s", lower_bin, upper_bin)
            # choose different bins depending
        if "morepo2" in track.human_tags:
            upper_bin = morepork_upper_bin
            # choose different bins depending
        t["lower_nose_bin"] = lower_bin + 1

        # rms of just the tracks frequency range, unless the range is so wide it
        # is not much different to bird_rms, or a non noise track goes into the
        # noise band which suggests the box frequencies weren't set properly
        if (
            track.min_freq is not None
            and track.max_freq is not None
            and track.max_freq - track.min_freq <= MAX_TRACK_RMS_RANGE
            and (track.noise_track or track.min_freq >= noise_max_freq)
        ):
            track_lower_bin = int(np.searchsorted(freqs, track.min_freq))
            track_upper_bin = int(np.searchsorted(freqs, track.max_freq, side="right"))
            track_upper_bin = max(track_upper_bin, track_lower_bin + 1)
            track_stft = np.zeros_like(stft)
            track_stft[track_lower_bin:track_upper_bin, :] = stft[
                track_lower_bin:track_upper_bin, :
            ]
            S, phase = librosa.magphase(track_stft)
            track_rms = librosa.feature.rms(
                S=S, frame_length=n_fft, hop_length=hop_length
            )
            t["track_rms_bin"] = [track_lower_bin, track_upper_bin]
            t["track_rms"] = track_rms[0].tolist()

        stft[:lower_bin, :] = 0
        if upper_bin is not None:
            stft[upper_bin:, :] = 0
            t["bird_rms_bin"] = [lower_bin + 1, upper_bin]
        else:
            t["bird_rms_bin"] = [lower_bin + 1]

        S, phase = librosa.magphase(stft)
        bird_rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)
        # noise or human
        t["upper_rms"] = upper_rms[0].tolist()
        t["noise_rms"] = noise_rms[0].tolist()
        t["bird_rms"] = bird_rms[0].tolist()
