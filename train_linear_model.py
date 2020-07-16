import operator
import os
from enum import Enum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image, ImageFile
from tensorflow.keras.activations import sigmoid, tanh, relu
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.utils import to_categorical


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# ImageFile.LOAD_TRUNCATED_IMAGES = True



class Vehicles(Enum):
    MOTO = [1, 0, 0]
    VOITURE = [0, 1, 0]
    AVION = [0, 0, 1]

def load_dataset():
    Ximgs = []
    y_train = []

    target_resolution = (64, 64)
    for file in os.listdir("./dataset/train/Moto/"):
        Ximgs.append(np.array(Image.open(f"./dataset/train/Moto/{file}").resize(target_resolution).convert('RGB')) / 255.0)
        y_train.append(Vehicles.MOTO.value)
    for file in os.listdir("./dataset/train/Voiture/"):
        Ximgs.append(np.array(Image.open(f"./dataset/train/Voiture/{file}").resize(target_resolution).convert('RGB')) / 255.0)
        y_train.append(Vehicles.VOITURE.value)
    for file in os.listdir("./dataset/train/Avion/"):
        Ximgs.append(np.array(Image.open(f"./dataset/train/Avion/{file}").resize(target_resolution).convert('RGB')) / 255.0)
        y_train.append(Vehicles.AVION.value)


    Ximgs_test = []
    y_test = []
    for file in os.listdir("./dataset/test/Moto/"):
        Ximgs_test.append(np.array(Image.open(f"./dataset/test/Moto/{file}").resize(target_resolution).convert('RGB')) / 255.0)
        # y_test.append([1, 0, 0, 0])
        y_test.append(Vehicles.MOTO.value)
    for file in os.listdir("./dataset/test/Voiture/"):
        Ximgs_test.append(np.array(Image.open(f"./dataset/test/Voiture/{file}").resize(target_resolution).convert('RGB')) / 255.0)
        # y_test.append([0, 1, 0, 0])
        y_test.append(Vehicles.VOITURE.value)
    for file in os.listdir("./dataset/test/Avion/"):
        Ximgs_test.append(np.array(Image.open(f"./dataset/test/Avion/{file}").resize(target_resolution).convert('RGB')) / 255.0)
        # y_test.append([0, 0, 1, 0])
        y_test.append(Vehicles.AVION.value)

    x_train = np.array(Ximgs)
    y_train = np.array(y_train)
    x_test = np.array(Ximgs_test)
    y_test = np.array(y_test)
    return (x_train, y_train), (x_test, y_test)

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


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_dataset()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # model = create_linear_model()
    model = create_mlp_model()
    # model = create_convolutional_neural_network_model()

    # model = load_model("cnnmodel.keras")

## confusion train matrix before training
    true_values = np.argmax(y_train, axis=1)
    preds = np.argmax(model.predict(x_train), axis=1)
    print("confusion train matrix before training")
    print(confusion_matrix(true_values, preds))

##confusion test matrix before training
    true_values = np.argmax(y_test, axis=1)
    preds = np.argmax(model.predict(x_test), axis=1)
    print("confusion test matrix before training")
    print(confusion_matrix(true_values, preds))

    print(f'Train acc {model.evaluate(x_train, y_train)[1]}')
    print(f'Test acc {model.evaluate(x_test, y_test)[1]}')

    logs = model.fit(x_train, y_train, batch_size=16, epochs=50, verbose=0, validation_data=(x_test, y_test))

##confusion train matrix after training
    true_values = np.argmax(y_train, axis=1)
    preds = np.argmax(model.predict(x_train), axis=1)
    print("confusion train matrix after training")
    print(confusion_matrix(true_values, preds))

##confusion test matrix after training
    true_values = np.argmax(y_test, axis=1)
    preds = np.argmax(model.predict(x_test), axis=1)
    print("confusion test matrix after training")
    print(confusion_matrix(true_values, preds))

    print(f'Train acc {model.evaluate(x_train, y_train)[1]}')
    print(f'Test acc {model.evaluate(x_test, y_test)[1]}')


    # Affichage de la courbe d'accuracy de l'apprentissage
    # plt.plot(logs.history['accuracy'])
    # plt.plot(logs.history['val_accuracy'])
    # plt.show()

    # Affichage de la courbes de loss de l'apprentissage
    # plt.plot(logs.history['loss'])
    # plt.plot(logs.history['val_loss'])
    # plt.show()
    # plt.savefig("image/" + plotAccName)
    #
    # model.save("linear.keras")

    #
    # model = load_model("linear.keras")
    y = []

    for file in os.listdir("./reconition/"):
        y.append(np.array(Image.open(f"./reconition/image.jpg").resize((64, 64)).convert('RGB')) / 255.0)

    res = model.predict(np.array(y))

    resul_dict = {}
    for e in res:
        resul_dict["moto"] = e[0]
        resul_dict["voiture"] = e[1]
        resul_dict["avion"] = e[2]


    vehicles_type = (max(resul_dict.items(), key=operator.itemgetter(1))[0])
    print("c'est un vehicule de type : %s" %(vehicles_type))