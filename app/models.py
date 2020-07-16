import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.activations import sigmoid, tanh, relu
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras as keras




def create_linear_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(3, activation=sigmoid))
    # model.add(Dense(4, activation=sigmoid))
    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model

def create_mlp_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(64, activation=tanh))
    model.add(Dense(64, activation=tanh))
    model.add(Dense(3, activation=sigmoid))
    # model.add(Dense(4, activation=sigmoid))
    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model


def create_convolutional_neural_network_model():
    model = Sequential()
    model.add(Conv2D(4, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(8, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation=tanh))
    model.add(Dense(3, activation=sigmoid))
    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model

def create_dense_res_nn_model():
    input_tensor = keras.layers.Input(((64, 64)[0], (64, 64)[1], 3))
    previous_tensor = Flatten()(input_tensor)
    next_tensor = Dense(64, activation=relu)(previous_tensor)
    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])
    next_tensor = Dense(64, activation=relu)(previous_tensor)
    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])
    next_tensor = Dense(64, activation=relu)(previous_tensor)
    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])
    next_tensor = Dense(64, activation=tanh)(previous_tensor)
    next_tensor = Dense(3, activation=sigmoid)(next_tensor)
    model = keras.models.Model(input_tensor, next_tensor)

    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model