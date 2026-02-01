import tensorflow as tf




# https://github.com/google-research/leaf-audio/blob/7ead2f9fe65da14c693c566fe8259ccaaf14129d/leaf_audio/postprocessing.py#L27

@tf.keras.utils.register_keras_serializable(package="MyLayers", name="ExponentialMovingAverage")
class ExponentialMovingAverage(tf.keras.layers.Layer):
  """Computes of an exponential moving average of an sequential input."""

  def __init__(
      self,
      coeff_init,
      trainable = False):
    """Initializes the ExponentialMovingAverage.

    Args:
      coeff_init: the value of the initial coeff.
      per_channel: whether the smoothing should be different per channel.
      trainable: whether the smoothing should be trained or not.
    """
    super().__init__(name='EMA')
    self._coeff_init = coeff_init
    self._trainable = trainable

    self._weights = self.add_weight(
        name='smooth',
        shape=[1],
        initializer=tf.keras.initializers.Constant(self._coeff_init),
        trainable=self._trainable)

  def call(self, inputs: tf.Tensor, initial_state: tf.Tensor):
    """Inputs is of shape [batch, seq_length, num_filters]."""
    w = tf.clip_by_value(self._weights, clip_value_min=0.0, clip_value_max=1.0)
    result = tf.scan(lambda a, x: w * x + (1.0 - w) * a,
                     tf.transpose(inputs, (1, 0, 2)),
                     initializer=initial_state)
    return tf.transpose(result, (1, 0, 2))
  
import tensorflow as tf
@tf.keras.utils.register_keras_serializable(package="MyLayers", name="MagTransform")
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
        )

        self.a = self.add_weight(
            initializer=tf.keras.initializers.Constant(value=-1.0),
            name="a-power",
            dtype="float32",
            shape=[1],
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(
                min_value=-2.0, max_value=1.0, rate=1.0, axis=-1
            ),
        )

    def call(self, inputs):
        gain = tf.math.minimum(self.gain, 1.0)
        root = tf.math.maximum(self.root, 1.0)
        ema_smoother = self.ema(inputs, initial_state=tf.gather(inputs, 0, axis=1))
        one_over_root = 1. / root
        output = ((inputs / (self.eps + ema_smoother)**gain + self.bias)
                **one_over_root - self.bias**one_over_root)
        


        return normalize_minmax(output)
    



# normalize between 0 and 1
@tf.function
def normalize_minmax(data):

    max_v = tf.reduce_max(data)
    min_v = tf.reduce_min(data)
    return 2 * ((data - min_v) / (max_v - min_v)) - 1