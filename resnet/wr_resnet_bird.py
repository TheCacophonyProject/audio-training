import tensorflow as tf
import tensorflow_probability as tfp


# based of birdnet
# https://arxiv.org/pdf/1605.07146.pdf
def WRResNet(input_shape=(120, 512, 1), classes=6, depth=22, k=4):
    filters = [16, 16 * k, 32 * k, 64 * k]

    FILTERS = [8, 16, 32, 64, 128]
    FILTERS = FILTERS * k
    FILTERS[0] = 8
    print("Filters are", filters)
    KERNEL_SIZES = [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3)]

    # Define the input as a tensor with shape input_shape
    X_input = tf.keras.Input(input_shape)
    n = int((depth - 4) / 6)

    for stage, f in enumerate(filters):
        if stage == 0:
            X = tf.keras.layers.Conv2D(
                f,
                KERNEL_SIZES[stage],
                padding="same",
                name=f"conv1_{stage+1}",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
            )(X_input)
            X = tf.keras.layers.BatchNormalization(axis=3)(X)
            X = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(X)
        else:
            X = wr_block(
                X,
                3,
                f,
                kernel_size=KERNEL_SIZES[stage],
                stage=stage,
                block="b",
                stride=2,
                depth=n,
            )
    X = tf.keras.layers.BatchNormalization(axis=3, name="final_bn")(X)
    X = tf.keras.layers.Activation("relu")(X)
    #
    # X = tf.keras.layers.AveragePooling2D(pool_size=8)(X)
    # classification branch
    X = tf.keras.layers.Conv2D(
        FILTERS[-1],
        (4, 10),
        padding="same",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Dropout(rate=0.1)(X)
    X = tf.keras.layers.Conv2D(
        FILTERS[-1] * 2,
        1,
        padding="same",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)

    X = tf.keras.layers.Dropout(rate=0.1)(X)

    X = tf.keras.layers.Conv2D(
        classes,
        1,
        padding="same",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)

    # is this the same as global pooling????
    X = logmeanexp(X, axis=1, sharpness=5, keepdims=False)
    X = logmeanexp(X, axis=2, sharpness=5, keepdims=False)

    # X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(classes, activation="sigmoid", name="prediction")(X)
    model = tf.keras.Model(inputs=X_input, outputs=X, name="WRResNet")
    model.summary()
    return model


def logmeanexp(x, axis=None, keepdims=False, sharpness=5):
    return (
        tfp.math.reduce_logmeanexp(x * sharpness, axis=axis, keepdims=keepdims)
        / sharpness
    )


def wr_block(X, f, filters, kernel_size, stage, block, stride=1, depth=1):
    s_block = f"{block}0"
    sub_id = 0
    X = basic_block(X, f, kernel_size, filters, stage, s_block, sub_id, stride)
    for d in range(depth - 1):
        s_block = f"{block}{d+1}"
        sub_id += 1
        X = basic_block(X, f, kernel_size, filters, stage, s_block, sub_id, 1)
    return X


# BASIC block containing some suggestions from birdnet and
# Bag of Tricks for Image Classification with Convolutional Neural Networks
def basic_block(X, f, kernel_size, filters, stage, block, sub_id, stride=1):
    # defining name basis
    print(
        "Stage",
        stage,
        "sub",
        sub_id,
        "Filters",
        filters,
        "inshape",
        X.shape[1],
        " stride",
        stride,
    )
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Save the input value. You'll need this later to add back to the main path.

    X_shortcut = X
    # First component of main path
    if stride > 1:
        X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2a0")(X)
        X = tf.keras.layers.Activation("relu")(X)
        X = tf.keras.layers.Conv2D(
            filters=X.shape[1],
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            name=conv_name_base + "2a0",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
        )(X)

    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        filters=X.shape[1],
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        name=conv_name_base + "21",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)
    if stride > 1:
        X = tf.keras.layers.MaxPool2D(pool_size=(stride, stride))(X)
    X = tf.keras.layers.Dropout(rate=0.1)(X)
    # , training=istraining_ph)

    # Second component of main path
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        name=conv_name_base + "2b",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
    )(X)

    if X.shape[-1] == X_shortcut.shape[-1]:
        X_shortcut = tf.keras.layers.Identity()(X_shortcut)
    else:
        X_shortcut = tf.keras.layers.AveragePooling2D(
            pool_size=(stride, stride), strides=stride, padding="same"
        )(X_shortcut)
        X_shortcut = tf.keras.layers.Conv2D(
            filters,
            strides=1,
            kernel_size=1,
            padding="same",
        )(X_shortcut)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.keras.layers.Add()([X, X_shortcut])
    if stage + sub_id > 1:
        X = tf.keras.layers.Activation("relu")(X)
    return X


WRResNet()
