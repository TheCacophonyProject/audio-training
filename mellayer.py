"""Keras layer turning raw audio into the mel spectrogram the models are trained
on, so it runs on the GPU as part of the model rather than in the tf.data input
pipeline (which always runs on the CPU). Matches tfdataset.raw_to_mel.

Models saved with it need this module imported before tf.keras.models.load_model
so the layer is registered."""

import tensorflow as tf

from custommel import mel_f


@tf.keras.utils.register_keras_serializable(package="cacophony")
class MelSpectrogram(tf.keras.layers.Layer):
    """[batch, samples] raw audio to [batch, n_mels, frames, channels] mels.

    power True gives power mels for MagTransform, False magnitude mels for
    PCEN. With spec_augment, SpecAugment masking is applied while training
    only. Always runs in float32, FFTs don't suit lower precision"""

    def __init__(
        self,
        sr=48000,
        n_fft=4096,
        hop_length=273,
        n_mels=160,
        fmin=100,
        fmax=11000,
        break_freq=1000,
        power=True,
        channels=3,
        spec_augment=False,
        **kwargs,
    ):
        kwargs["dtype"] = "float32"
        super().__init__(**kwargs)
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.break_freq = break_freq
        self.power = power
        self.channels = channels
        self.spec_augment = spec_augment
        # [mels, fft bins], a constant rather than a weight so weights files
        # of models without this layer still load
        self.mel_weights = tf.constant(
            mel_f(sr, n_mels, fmin, fmax, n_fft, break_freq), dtype=tf.float32
        )

    def call(self, inputs, training=None):
        raw = tf.cast(inputs, tf.float32)
        # [batch, frames, fft bins]
        stft = tf.signal.stft(
            raw,
            self.n_fft,
            self.hop_length,
            fft_length=self.n_fft,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )
        spec = tf.math.abs(stft)
        if self.power:
            spec = tf.math.square(spec)
        # [batch, mels, frames] as raw_to_mel gives
        mel = tf.einsum("mf,btf->bmt", self.mel_weights, spec)
        mel = tf.repeat(tf.expand_dims(mel, axis=-1), self.channels, axis=-1)
        if training and self.spec_augment:
            # only needed when training, so inference doesn't need tfdataset
            from tfdataset import tf_batch_spec_augment

            mel = tf_batch_spec_augment(mel)
        return mel

    def compute_output_shape(self, input_shape):
        frames = None
        if input_shape[-1] is not None:
            frames = 1 + (input_shape[-1] - self.n_fft) // self.hop_length
        return (input_shape[0], self.n_mels, frames, self.channels)

    def get_config(self):
        config = super().get_config()
        config.update(
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            break_freq=self.break_freq,
            power=self.power,
            channels=self.channels,
            spec_augment=self.spec_augment,
        )
        return config
