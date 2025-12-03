from custommel import mel_spec

import logging
import librosa
import numpy as np
import tensorflow as tf


def load_samples(
    frames,
    sr,
    tracks,
    segment_length=3,
    stride=1,
    hop_length=281,
    mean_sub=False,
    use_mfcc=False,
    mel_break=1000,
    htk=True,
    n_mels=160,
    fmin=100,
    fmax=11000,
    channels=1,
    power=2,
    db_scale=False,
    filter_freqs=False,
    filter_below=None,
    normalize=True,
    n_fft=4096,
    pad_short_tracks=False,
):
    logging.info(
        "Loading samples with length %s stride %s hop length %s and mean_sub %s mfcc %s break %s htk %s n mels %s fmin %s fmax %s filtering freqs %s filter below %s n_fft %s pad short tracks %s",
        segment_length,
        stride,
        hop_length,
        mean_sub,
        use_mfcc,
        mel_break,
        htk,
        n_mels,
        fmin,
        fmax,
        filter_freqs,
        filter_below,
        n_fft,
        pad_short_tracks,
    )
    mels = []
    i = 0
    # hop_length = 640  # feature frame rate of 75

    sample_size = int(sr * segment_length)
    jumps_per_stride = int(sr * stride)
    length = len(frames) / sr
    end = segment_length
    mel_samples = []
    
    for t in tracks:
        track_data = []
        if (
            t.freq_start is not None
            and t.freq_end is not None
            and (t.freq_start > fmax or t.freq_end < fmin)
        ):
            mel_samples.append(track_data)
            # no need to id these tracks
            continue
        start = 0
        end = start + segment_length

        sr_end = int(t.end * sr)
        sr_start = int(sr * t.start)

        if pad_short_tracks:
            end = min(end, t.length)
            track_frames = frames[sr_start:sr_end]
        else:

            missing = sample_size - (sr_end - sr_start)
            if missing > 0:
                offset = np.random.randint(0, missing)
                offset = missing //2
                sr_start = sr_start - offset

                if sr_start <= 0:
                    sr_start = 0
                    sr_end = sr_start + sample_size
                    sr_end = min(sr_end, len(frames))
                else:
                    end_offset = sr_end + missing - offset
                    if end_offset > len(frames):
                        end_offset = len(frames)
                        sr_start = end_offset - sample_size
                        sr_start = max(sr_start, 0)
                    sr_end = end_offset
                if len(frames) >= sample_size:
                    assert sr_end - sr_start == sample_size

            track_frames = frames[sr_start:sr_end]
            print("Track is ",sr_start/sr, sr_end/sr)
        sr_start = 0
        sr_end = min(sr_end, sample_size)
        if filter_freqs:
            track_frames = butter_bandpass_filter(
                track_frames, t.freq_start, t.freq_end, sr
            )
        elif filter_below and t.freq_end < filter_below:
            logging.info(
                "Filter freq below %s %s %s", filter_below, t.freq_start, t.freq_end
            )
            track_frames = butter_bandpass_filter(
                track_frames, t.freq_start, t.freq_end, sr
            )
        while True:
            data = track_frames[sr_start:sr_end]
            if len(data) != sample_size:
                extra_frames = sample_size - len(data)
                offset = np.random.randint(0, extra_frames)
                data = np.pad(data, (offset, extra_frames - offset))
            if normalize:
                data = normalize_data(data)
            spect = get_spect(
                data,
                sr,
                hop_length,
                mean_sub,
                use_mfcc,
                mel_break,
                htk,
                n_mels,
                fmin,
                fmax,
                n_fft,
                power,
                db_scale,
                channels,
                # pass_freqs=[t.freq_start, t.freq_end],
            )

            track_data.append(spect)
            start = start + stride
            end = start + segment_length
            sr_start = int(start * sr)
            sr_end = min(int(end * sr), sr_start + sample_size)
            # always take 1 sample
            if end > t.length:
                break

        mel_samples.append(track_data)
    return mel_samples


def normalize_data(x):
    min_v = np.min(x, -1, keepdims=True)
    x = x - min_v
    max_v = np.max(x, -1, keepdims=True)
    x = x / max_v + 0.000001
    x = x - 0.5
    x = x * 2
    return x


def get_spect(
    data,
    sr,
    hop_length,
    mean_sub,
    use_mfcc,
    mel_break,
    htk,
    n_mels,
    fmin,
    fmax,
    n_fft,
    power,
    db_scale,
    channels=1,
    pass_freqs=None,
):
    if not htk:
        mel = librosa.feature.melspectrogram(
            y=data,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            n_mels=n_mels,
        )
    else:
        # if pass_freqs is not None:
        #     data = butter_bandpass_filter(data, pass_freqs[0], pass_freqs[1], sr)

        spectogram = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))
        # bins = 1 + n_fft / 2
        # max_f = sr / 2
        # gap = max_f / bins
        # if low_pass is not None:
        #     min_bin = low_pass // gap
        #     spectogram[: int(min_bin)] = 0
        #
        # if high_pass is not None:
        #     max_bin = high_pass // gap
        #     spectogram[int(max_bin) :] = 0
        mel = mel_spec(
            spectogram,
            sr,
            n_fft,
            hop_length,
            n_mels,
            100 if fmin is None else fmin,
            11000 if fmin is None else fmax,
            mel_break,
            power=power,
        )
    if db_scale:
        mel = librosa.power_to_db(mel, ref=np.max)
    mel = tf.expand_dims(mel, axis=2)

    if use_mfcc:
        mfcc = librosa.feature.mfcc(
            y=data,
            sr=sr,
            hop_length=hop_length,
            htk=True,
            fmin=50,
            fmax=11000,
            n_mels=80,
        )
        mfcc = tf.image.resize_with_pad(mfcc, *mel.shape)
        mel = tf.concat((mel, mfcc), axis=0)
    # end = start + sample_size
    if mean_sub:
        mel_m = tf.reduce_mean(mel, axis=1)
        mel_m = tf.expand_dims(mel_m, axis=1)
        mel = mel - mel_m
    if channels > 1:
        mel = tf.repeat(mel, channels, axis=2)
    return mel


from scipy.signal import butter, sosfilt, sosfreqz, freqs


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    btype = "lowpass"
    freqs = []
    if lowcut > 0:
        btype = "bandpass"
        low = lowcut / nyq
        freqs.append(low)
    high = highcut / nyq
    freqs.append(high)
    sos = butter(order, freqs, analog=False, btype=btype, output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = sosfilt(sos, data)
    return filtered
