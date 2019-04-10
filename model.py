from keras.models import *
from keras.layers import *
from keras.optimizers import *


# This file is separate to prevent code clutter in the main
# script.
def fashion_net(input_size=(28, 28, 1), num_classes=10):
    inputs = Input(input_size)
    norm = BatchNormalization()(inputs)

    conv = Conv2D(32, 3, padding="same", activation="relu")(norm)
    maxp = MaxPool2D()(conv)

    conv = Conv2D(64, 3, padding="same", activation="relu")(maxp)
    spat = SpatialDropout2D(0.3)(conv)
    maxp = MaxPool2D()(spat)

    conv = Conv2D(128, 3, padding="same", activation="relu")(maxp)
    flat = Flatten()(conv)

    dens = Dense(num_classes*3)(flat)
    drop = Dropout(0.3)(dens)

    dens = Dense(num_classes*2)(drop)
    drop = Dropout(0.3)(dens)

    outputs = Dense(num_classes, activation="softmax")(drop)

    model = Model(input=inputs, output=outputs)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
