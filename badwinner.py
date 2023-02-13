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


def build_model(input_shape, norm_layer, num_labels, multi_label=False):
    input = tf.keras.Input(shape=(*input_shape, 1), name="input")
    # x = norm_layer(input)
    filters = 16
    if multi_label:
        filters = 32
    x = tf.keras.layers.BatchNormalization()(input)
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation=tf.keras.layers.LeakyReLU())(
        x
    )
    x = tf.keras.layers.MaxPool2D((3, 3))(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation=tf.keras.layers.LeakyReLU())(
        x
    )
    x = tf.keras.layers.MaxPool2D((3, 3))(x)

    x = tf.keras.layers.Conv2D(filters, (1, 3), activation=tf.keras.layers.LeakyReLU())(
        x
    )
    x = tf.keras.layers.MaxPool2D((1, 3))(x)
    #
    # x = tf.keras.layers.Conv2D(filters, (1, 3), activation=tf.keras.layers.LeakyReLU())(x)
    # x = tf.keras.layers.MaxPool2D((1, 3))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    if num_labels > 1:
        dense = [1024, 512]
    else:
        dense = [256, 32]
    for d in dense:
        x = tf.keras.layers.Dense(d, activation=tf.keras.layers.LeakyReLU())(x)
        x = tf.keras.layers.Dropout(0.5)(x)
    # x = tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU())(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    activation = "softmax"
    if multi_label:
        activation = "sigmoid"
    logging.info("Using %s activation", activation)
    x = tf.keras.layers.Dense(num_labels, activation=activation)(x)
    # x = tf.keras.layers.Dense(2, activation="softmax")(x)
    # x = tf.keras.activations.sigmoid(x)

    model = tf.keras.models.Model(input, outputs=x)
    return model


def main():
    init_logging()
    args = parse_args()
    build_model((80, 226), None, 2)


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
