import librosa
import numpy as np

# replicating code from librosa but changing break freq from 700 -> 1750
def hz_to_mel(frequencies, break_freq):
    frequencies = np.array(frequencies)
    return 2595.0 * np.log10(1.0 + frequencies / break_freq)


def mel_frequencies(n_mels, fmin, fmax, break_freq):
    min_mel = hz_to_mel(fmin, break_freq)
    max_mel = hz_to_mel(fmax, break_freq)
    mels = np.linspace(min_mel, max_mel, n_mels)

    return break_freq * (10.0 ** (mels / 2595.0) - 1.0)


def mel_f(sr, n_mels, fmin, fmax, n_fft, break_freq):

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)

    # Center freqs of each FFT bin
    fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin, fmax, break_freq)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # slaney
    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        print(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels."
        )

    return weights


def mel_spec(stft, sr, n_fft, hop_length, n_mels, fmin, fmax, break_freq=1750):
    # fft_windows = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft) ** 2
    mels = mel_f(sr, n_mels, fmin, fmax, n_fft, break_freq)
    return mels.dot(magnitude)
