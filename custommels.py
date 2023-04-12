import librosa
import numpy as np
break_freq = 1750
# replicating code from librosa but changing break freq from 700 -> 1750
def hz_to_mel(frequencies):
    frequencies = np.array(frequencies)
    return 2595.0 * np.log10(1.0 + frequencies / break_freq)

def mel_frequencies(n_mels,fmin,fmax):
    min_mel =hz_to_mel(fmin)
    max_mel= hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels)

    return break_freq* (10.0 ** (mels / 2595.0) - 1.0)


def mel_f(sr, n_mels,fmin,fmax,n_fft):

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)

    # Center freqs of each FFT bin
    fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)

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
            "reducing n_mels."        )

    return weights

def mel_spec(stft,sr,n_fft,hop_length,n_mels,fmin,fmax):
    # fft_windows = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)**2
    mels = mel_f(sr,n_mels,fmin,fmax,n_fft)
    return mels.dot(magnitude)

# import matplotlib
# import matplotlib.pyplot as plt
# import librosa.display
#
# matplotlib.use("TkAgg")
# def plot_mel(mel, i=0):
#     plt.figure(figsize=(10, 10))
#
#     ax = plt.subplot(1, 1, 1)
#     img = librosa.display.specshow(
#         mel, x_axis="time", y_axis="mel", sr=48000, fmax=11000,fmin=50, ax=ax,hop_length=281
#     )
#     plt.show()
#     # plt.savefig(f"mel-power-{i}.png", format="png")
#     plt.clf()
#     plt.close()
#
# def main():
#     frames, sr = librosa.load("./signal-data/human-1.wav",sr = 48000)
#     frames = frames[:sr*3]
#     n_fft = 4800
#     hop_length = 281
#     n_mels = 120
#
#     S = librosa.feature.melspectrogram(frames, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,htk=True,fmin=50,fmax=11000)
#     # plot_mel(S)
#
#     mels = mel_spec(frames, sr,n_fft, hop_length,n_mels,50,11000)
#     # plot_mel(mels)
#     print(mels)
#     print(S)
#     assert (mels == S).all()
#
# if __name__ == "__main__":
#     main()
