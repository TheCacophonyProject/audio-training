import matplotlib.pyplot as plt
import matplotlib
import librosa

matplotlib.use("TkAgg")
import numpy as np


def plot_mfcc(mfcc):
    plt.figure(figsize=(10, 10))

    img = librosa.display.specshow(mfcc, x_axis="time", ax=ax)
    ax = plt.subplot(1, 1, 1)
    plt.show()
    # plt.savefig(f"mel-power-{i}.png", format="png")
    plt.clf()
    plt.close()


def plot_mel(mel, i=0):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    img = librosa.display.specshow(
        mel,
        x_axis="time",
        y_axis="mel",
        sr=48000,
        fmax=11000,
        fmin=50,
        ax=ax,
        hop_length=201,
    )
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    plt.show()
    # plt.savefig(f"mel-power-{i}.png", format="png")
    plt.clf()
    plt.close()
