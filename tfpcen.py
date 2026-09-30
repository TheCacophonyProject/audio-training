import tensorflow as tf


# https://github.com/google-research/leaf-audio/blob/7ead2f9fe65da14c693c566fe8259ccaaf14129d/leaf_audio/postprocessing.py#L27


@tf.keras.utils.register_keras_serializable(
    package="MyLayers", name="ExponentialMovingAverage"
)
class ExponentialMovingAverage(tf.keras.layers.Layer):
    """Computes of an exponential moving average of an sequential input."""

    def __init__(
        self, coeff_init, trainable=False, kernel_len=128, time_axis=2, **kwargs
    ):
        """Initializes the ExponentialMovingAverage.

        Args:
          coeff_init: the value of the initial coeff.
          trainable: whether the smoothing should be trained or not.
          kernel_len: unused, kept so older configs still load.
          time_axis: axis of the [batch, x, y, channels] input that is time,
            1 for [batch, time, mels, channels], 2 for [batch, mels, time, channels].
        """
        super().__init__(name="EMA", **kwargs)
        if time_axis not in (1, 2):
            raise ValueError(f"time_axis must be 1 or 2, got {time_axis}")
        self._coeff_init = coeff_init
        self._trainable = trainable
        self.kernel_len = kernel_len
        self.time_axis = time_axis

        self._weights = self.add_weight(
            name="smooth",
            shape=[1],
            initializer=tf.keras.initializers.Constant(self._coeff_init),
            trainable=self._trainable,
        )

    def call(self, inputs: tf.Tensor):
        """Inputs is of shape [batch, x, y, channels], smoothed over time_axis.

        Equivalent to s_t = w * x_t + (1 - w) * s_{t-1} with s_{-1} = x_0, but
        computed as a single matmul with a [time, time] lower triangular decay
        matrix. The depthwise conv version was very slow on GPU because the
        filter gradient for a 1x128 depthwise kernel has no fast cuDNN path.
        kernel_len is no longer used, the average is exact over all frames.
        """
        w = tf.clip_by_value(self._weights, clip_value_min=0.0, clip_value_max=1.0 - 1e-6)
        w = tf.cast(w, inputs.dtype)
        T = inputs.shape[self.time_axis]
        if T is None:
            T = tf.shape(inputs)[self.time_axis]
        t = tf.range(T, dtype=inputs.dtype)
        # diff[k, t] = t - k, how far input frame k is behind output frame t
        diff = t[None, :] - t[:, None]
        mask = tf.cast(diff >= 0, inputs.dtype)
        m = w * (1.0 - w) ** tf.maximum(diff, 0.0) * mask
        # initial state s_{-1} = x_0 adds (1 - w) ** (t + 1) * x_0
        init = (1.0 - w) ** (t + 1.0)
        m = m + tf.pad(init[None, :], [[0, T - 1], [0, 0]])
        if self.time_axis == 1:
            return tf.einsum("bkmc,kt->btmc", inputs, m)
        return tf.einsum("bmkc,kt->bmtc", inputs, m)

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="MyLayers", name="PCEN")
class PCEN(tf.keras.layers.Layer):
    def __init__(self, time_axis=2, **kwargs):
        """time_axis is the input axis that is time, 1 for [batch, time, mels, channels]
        or 2 for [batch, mels, time, channels]. Defaults to 2 so models saved before
        this option existed load with the same behaviour.
        """
        super(PCEN, self).__init__(**kwargs)
        self.time_axis = time_axis

        self.gain = self.add_weight(
            initializer=tf.keras.initializers.Constant(value=0.98),
            name="gain",
            dtype="float32",
            shape=[1],
            trainable=True,
        )
        self.bias = self.add_weight(
            initializer=tf.keras.initializers.Constant(value=2.0),
            name="bias",
            dtype="float32",
            shape=[1],
            trainable=True,
        )
        self.root = self.add_weight(
            initializer=tf.keras.initializers.Constant(value=2.0),
            name="root",
            dtype="float32",
            shape=[1],
            trainable=True,
        )

        self.eps = 1e-6

        self.ema = ExponentialMovingAverage(
            coeff_init=0.04,
            trainable=True,
            time_axis=time_axis,
            dtype=self.dtype_policy,
        )

    def get_config(self):
        config = super().get_config()
        config["time_axis"] = self.time_axis
        return config

    def call(self, inputs):
        gain = tf.math.minimum(self.gain, 1.0)
        root = tf.math.maximum(self.root, 1.0)
        ema_smoother = self.ema(inputs)
        one_over_root = 1.0 / root
        output = (
            inputs / (self.eps + ema_smoother) ** gain + self.bias
        ) ** one_over_root - self.bias**one_over_root

        output = normalize_minmax(output)
        # output = tf.expand_dims(output, axis=-1)
        # output = tf.repeat(output, 3, 3)
        return output


# normalize between -1 and 1
# not a tf.function, a nested function call made XLA training steps ~50x slower
def normalize_minmax(data):
    max_v = tf.reduce_max(data, axis=(1, 2, 3), keepdims=True)
    min_v = tf.reduce_min(data, axis=(1, 2, 3), keepdims=True)
    return 2 * ((data - min_v) / (max_v - min_v + 1e-6)) - 1
