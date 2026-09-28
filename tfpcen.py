import tensorflow as tf


# https://github.com/google-research/leaf-audio/blob/7ead2f9fe65da14c693c566fe8259ccaaf14129d/leaf_audio/postprocessing.py#L27


@tf.keras.utils.register_keras_serializable(
    package="MyLayers", name="ExponentialMovingAverage"
)
class ExponentialMovingAverage(tf.keras.layers.Layer):
    """Computes of an exponential moving average of an sequential input."""

    def __init__(self, coeff_init, trainable=False, kernel_len=128, **kwargs):
        """Initializes the ExponentialMovingAverage.

        Args:
          coeff_init: the value of the initial coeff.
          trainable: whether the smoothing should be trained or not.
          kernel_len: number of time frames the EMA is truncated to. The
            ignored tail weight is (1 - coeff) ** kernel_len.
        """
        super().__init__(name="EMA", **kwargs)
        self._coeff_init = coeff_init
        self._trainable = trainable
        self.kernel_len = kernel_len

        self._weights = self.add_weight(
            name="smooth",
            shape=[1],
            initializer=tf.keras.initializers.Constant(self._coeff_init),
            trainable=self._trainable,
        )

    def call(self, inputs: tf.Tensor):
        """Inputs is of shape [batch, mels, time, channels], smoothed over time.

        Equivalent to s_t = w * x_t + (1 - w) * s_{t-1} with s_{-1} = x_0, but
        computed as a causal convolution with kernel w * (1 - w) ** n instead
        of tf.scan, which has to keep every step around for backprop.
        """
        w = tf.clip_by_value(self._weights, clip_value_min=0.0, clip_value_max=1.0)
        w = tf.cast(w, inputs.dtype)
        n = tf.range(self.kernel_len, dtype=inputs.dtype)
        kernel = w * (1.0 - w) ** n
        # conv2d correlates, so reverse to put the newest frame last
        kernel = tf.reverse(kernel, [0])
        channels = tf.shape(inputs)[-1]
        kernel = tf.tile(kernel[None, :, None, None], [1, 1, channels, 1])

        # left pad with the first frame, same as using it as the initial state
        first = inputs[:, :, :1, :]
        padded = tf.concat(
            [tf.repeat(first, self.kernel_len - 1, axis=2), inputs], axis=2
        )
        return tf.nn.depthwise_conv2d(
            padded, kernel, strides=[1, 1, 1, 1], padding="VALID"
        )


import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="MyLayers", name="PCEN")
class PCEN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PCEN, self).__init__(**kwargs)

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
            dtype=self.dtype_policy,
        )

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


# normalize between 0 and 1
@tf.function
def normalize_minmax(data):
    max_v = tf.reduce_max(data, axis=(1,2,3),keepdims=True)
    min_v = tf.reduce_min(data, axis=(1,2,3),keepdims=True)
    return 2 * ((data - min_v) / (max_v - min_v)) - 1
