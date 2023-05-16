# Input 1×1000×80
# Conv(3×3) 16×998×78
# leaky relu
# ( after ll conv)
# Pool(3×3) 16×332×26
# Conv(3×3) 16×330×24
# Pool(3×3) 16×110×8

# Conv(3×1) 16×108×8
# Pool(3×1) 16×36×8
# Conv(3×1) 16×34×8
# Pool(3×1) 16×11×8
# Dense 256
# leaky relu (after all dense)
# Dense 32
# Dense 1
#
#
#  leaky rectifier
# nonlinearity max(x, x/100)
import sys
import argparse
import logging
import tensorflow as tf


# Worth looking into lme pooling as proposed in  https://github.com/f0k/birdclef2018/blob/master/experiments/model.py
# Research/2018-birdclef.pdf


class MagTransform(tf.keras.layers.Layer):
    def __init__(self):
        super(MagTransform, self).__init__()
        self.a = self.add_weight(
            initializer=tf.keras.initializers.Constant(value=0.0),
            name="a-power",
            dtype="float32",
            shape=(),
            trainable=True,
        )

    def call(self, inputs):
        c = tf.math.pow(inputs, tf.math.sigmoid(self.a))
        return c

#         arch = 'conv:32x3x3
# conv:32x3x3
# pool:3x3
# #
# # conv:32x3x3
# # conv:32x3x3

# conv:64x3x-Shift
# # pool:3xShift

# # Conv:Fullx9x-1
# Conv:Fullx1x1'

def build_model(input_shape, norm_layer, num_labels, multi_label=False):
    input = tf.keras.Input(shape=(*input_shape, 1), name="input")
    # x = norm_layer(input)
    # if multi_label:
    filters = 256
    # y = x σ(a) , where σ(a) = 1/ (1 + exp(−a))

    x = MagTransform()(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU())(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU())(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D((3, 3))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU())(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU())(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(128, (17, 3), activation=tf.keras.layers.LeakyReLU())(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D((5,3))(x)
    x = tf.keras.layers.Dropout(0.5)(x)


# probably dont need to be as big
    x = tf.keras.layers.Conv2D(1024, (1,9),activation=tf.keras.layers.LeakyReLU(),kernel_initializer= tf.keras.initializers.Orthogonal())(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(1024, 1,activation=tf.keras.layers.LeakyReLU(),kernel_initializer= tf.keras.initializers.Orthogonal())(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(num_labels, 1,activation=tf.keras.layers.LeakyReLU(),kernel_initializer= tf.keras.initializers.Orthogonal())(
        x
    )
    x = logmeanexp(x,sharpness=1,axis = 2)
    x = tf.keras.activations.sigmoid(x)

    model = tf.keras.models.Model(input, outputs=x)
    return model

def logmeanexp(x, axis=None, keepdims=False, sharpness=5):
     return tf.math.reduce_logsumexp(x*sharpness,axis = axis)/sharpness

def main():
    init_logging()
    args = parse_args()
    model = build_model((80, 480), None, 2)
    model.summary()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confusion", help="Save confusion matrix for model")
    parser.add_argument("-w", "--weights", help="Weights to use")

    parser.add_argument("-c", "--config-file", help="Path to config file to use")

    args = parser.parse_args()
    return args


def init_logging():
    """Set up logging for use by various classifier pipeline scripts.

    Logs will go to stderr.
    """

    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )


if __name__ == "__main__":
    main()
