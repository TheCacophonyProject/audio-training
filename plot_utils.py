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


import matplotlib.patches as patches


def plot_mel_signals(mel, signals, signals2=[], i=0):
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
        hop_length=281,
    )
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    for s in signals:
        start_x = s[0]
        end_x = s[1]
        # start_x = int(start_x)
        # end_x = int(end_x)
        rect = patches.Rectangle(
            (start_x, 0),
            end_x - start_x,
            1000,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        # print("Added rect", start_x, end_x)
        # break
    for s in signals2:
        start_x = s[0]
        end_x = s[1]
        # start_x = int(start_x)
        # end_x = int(end_x)
        rect = patches.Rectangle(
            (start_x, 1000),
            end_x - start_x,
            2000,
            linewidth=1,
            edgecolor="g",
            facecolor="none",
        )
        ax.add_patch(rect)
        # print("Added rect", start_x, end_x)
        # break
    plt.show()
    plt.savefig(f"mel-signal-{i}.png", format="png")
    plt.clf()
    plt.close()


def plot_spec(mel, filename="spec"):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    img = librosa.display.specshow(
        mel,
        x_axis="time",
        y_axis="log",
        sr=48000,
        fmax=11000,
        fmin=50,
        ax=ax,
        hop_length=281,
    )
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    plt.show()
    # plt.savefig(f"{filename}.png", format="png")
    plt.clf()
    plt.close()


def plot_mel(mel, filename="mel"):
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
        hop_length=281,
    )
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    # plt.show()
    plt.savefig(f"{filename}.png", format="png")
    plt.clf()
    plt.close()
