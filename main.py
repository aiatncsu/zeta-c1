import keras
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.callbacks import *

import numpy as np
import pandas as pd
from model import fashion_net


num_classes = 10
img_size = (28, 28, 1)
directory = "next/"

# First, read in the dataset because if you import it directly from
# Keras it fucks everything up.
test = pd.read_csv("./fashion-mnist_test.csv")
x_test = test[list(test.columns)[2:]].values
y_test = test['label'].values
x_test = x_test.reshape(-1, 28, 28)
train = pd.read_csv("./fashion-mnist_train.csv")
x_train = train[list(train.columns)[2:]].values
y_train = train['label'].values
x_train = x_train.reshape(-1, 28, 28)

# Set everything up so that it plays well with Keras. This is basically
# just expanding the image datasets to 4-dimensional tensors and
# changing the labels into categorical arrays.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Set up a Keras Data Generator to augment and standardize the images.
datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    vertical_flip=True)
datagen.fit(x_train)

# Set up the network hyperparameters and callbacks.
epochs = 25
batch_size = 16
cp = EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto',
                   baseline=None, restore_best_weights=False)
tb = keras.callbacks.TensorBoard(log_dir='./logdir/' + directory, batch_size=batch_size,
                                 write_graph=True, write_images=True, update_freq='epoch')
callbacks = [cp, tb]

# Load the model from the model.py file. This reduces clutter.
model = fashion_net(input_size=img_size, num_classes=num_classes)

# Either retrain the network or load the weights from the project
# directory.
train = 0
if train == 1:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
                        callbacks=callbacks, shuffle=True)
    model.save_weights('./logdir/' + directory + "model_weights.h5")
    model.save('./logdir/' + directory + "full_model.h5")
else:
    model.load_weights('./logdir/' + directory + "model_weights.h5")
    model.save('./logdir/' + directory + "full_model_loaded.h5")

# Evaluate the model on the testing data.
modeleval = model.evaluate_generator(datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False),
                                     steps=len(x_test) / batch_size)

# Generate the predictions, put them in the right format, and
# save them.
predictions = model.predict_generator(datagen.flow(x_test, batch_size=batch_size, shuffle=False),
                                      steps=len(x_test) / batch_size)
predictions_ind = predictions.argmax(axis=1).transpose()
d = pd.DataFrame({'Label': predictions_ind})
d.to_csv(path_or_buf=r"./submission_maybe.csv", index=True, index_label="ID")
