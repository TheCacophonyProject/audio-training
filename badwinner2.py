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

# import tensorflow_probability as tfp

# Worth looking into lme pooling as proposed in  https://github.com/f0k/birdclef2018/blob/master/experiments/model.py
# Research/2018-birdclef.pdf


class MagTransform(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MagTransform, self).__init__(**kwargs)
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


def res_block(X, filters, stage, block, stride=1):
    # defining name basis
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve Filters

    # Save the input value. You'll need this later to add back to the main path.

    X_shortcut = X
    # First component of main path
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        name=conv_name_base + "2a",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        name=conv_name_base + "2b",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        name=conv_name_base + "2c",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2d")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        name=conv_name_base + "2d",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)

    X_shortcut = tf.keras.layers.Conv2D(X.shape[-1], strides=stride, kernel_size=1)(
        X_shortcut
    )
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation("relu")(X)
    return X


def build_model_res(
    input_shape, norm_layer, num_labels, multi_label=False, add_dense=True
):
    input = tf.keras.Input(shape=input_shape, name="input")
    # x = norm_layer(input)
    # if multi_label:
    filters = 256
    # y = x σ(a) , where σ(a) = 1/ (1 + exp(−a))

    x = MagTransform()(input)
    # x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(64, (3, 3))
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # RESNET
    x = res_block(x, 64, 1, "b")
    x = tf.keras.layers.MaxPool2D((3, 3))(x)
    x = res_block(x, 128, 2, "b")
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(128, (14, 3))(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (22, 3))(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    # probably dont need to be as big
    x = tf.keras.layers.Conv2D(
        1024,
        (1, 9),
        kernel_initializer=tf.keras.initializers.Orthogonal(),
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(
        1024,
        1,
        kernel_initializer=tf.keras.initializers.Orthogonal(),
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    if add_dense:
        x = tf.keras.layers.Conv2D(
            num_labels,
            1,
            kernel_initializer=tf.keras.initializers.Orthogonal(),
        )(x)
        x = tf.keras.layers.LeakyReLU()(x)

        # x = logmeanexp(x, sharpness=1, axis=2)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.activations.sigmoid(x)

    model = tf.keras.models.Model(input, outputs=x)
    return model


def build_model(
    input_shape,
    norm_layer,
    num_labels,
    multi_label=False,
    lme=False,
    add_dense=True,
    big_condense=True,
):
    input = tf.keras.Input(shape=input_shape, name="input")
    # x = norm_layer(input)
    # if multi_label:
    filters = 256
    # y = x σ(a) , where σ(a) = 1/ (1 + exp(−a))

    x = MagTransform()(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3))(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(64, (3, 3))(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D((3, 3))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Original is based of 80 mels, which only needs one conv 17x 3
    # we can either have a bigger conv or 2 large ones

    # At this point we have 48 mel bands remaining if we started with 160
    # Squish the information into smaller features essentially combining mel bands
    if big_condense:
        x = tf.keras.layers.Conv2D(128, (44, 3))(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tf.keras.layers.Conv2D(128, (28, 3))(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, (17, 3))(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Squish again so that we have 5 condense mel bands

    # Pool the mel bands so that we have a shape of (1,X) essentially brining all the mel bands
    # into a set of features per time range
    x = tf.keras.layers.MaxPool2D((5, 3))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(
        1024,
        (1, 9),
        kernel_initializer=tf.keras.initializers.Orthogonal(),
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(
        1024,
        1,
        kernel_initializer=tf.keras.initializers.Orthogonal(),
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    if add_dense:
        x = tf.keras.layers.Conv2D(
            num_labels,
            1,
            kernel_initializer=tf.keras.initializers.Orthogonal(),
        )(x)
        x = tf.keras.layers.LeakyReLU()(x)
        # Since we have quite specific track information, LME might not be so usefull, as this is more
        #  like an inbetween max and average, higher the sharpness the more like max it becomes
        # haven't found any benefit using LME
        if lme:
            x = logmeanexp(x, axis=1, sharpness=5, keepdims=False)
            x = logmeanexp(x, axis=2, sharpness=5, keepdims=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.activations.sigmoid(x)

    model = tf.keras.models.Model(input, outputs=x)
    return model


def logmeanexp(x, axis=None, keepdims=False, sharpness=5):
    return (
        tfp.math.reduce_logmeanexp(x * sharpness, axis=axis, keepdims=keepdims)
        / sharpness
    )


#
# def logmeanexp_2(x, axis=None, keepdims=False, sharpness=5):
#     return (
#         tf.math.log(tf.math.reduce_mean(tf.math.exp(x * sharpness), axis=axis))
#         / sharpness
#     )
#     # return (
#     #     tfp.math.reduce_logmeanexp(x * sharpness, axis=axis, keepdims=True) / sharpness
#     # )


# higher ther sharpness closer to max it becomes
# in 45 samples if we have 7 stronge predictions of 0.9 this will equate to 0.8 in this label
# 7 being roughly 1/2 second of audio


def logmeanexp(x, axis=None, keepdims=False, sharpness=5):
    return (
        tfp.math.reduce_logmeanexp(x * sharpness, axis=axis, keepdims=True) / sharpness
    )


def main():
    init_logging()
    args = parse_args()
    model = build_model(
        (160, 513, 1), None, 21, multi_label=True, lme=False, big_condense=False
    )
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=CustomBinaryCrossEntropy,
        metrics=tf.keras.metrics.AUC(),
    )


def CustomBinaryCrossEntropy(y_true, y_pred):
    y_pred = tf.keras.backend.clip(y_pred, K.epsilon(), 1 - tf.keras.backend.epsilon())
    term_0 = (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
    term_1 = y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())
    return -tf.keras.backend.mean(term_0 + term_1, axis=0)


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
