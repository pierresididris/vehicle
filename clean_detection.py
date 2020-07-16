import operator
import os
from enum import Enum

from app.analysis.analysis import analysis, learningAccuracyCurve, learningLossCurve
from app.models import create_mlp_model, create_convolutional_neural_network_model, create_linear_model, \
    create_dense_res_nn_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image, ImageFile



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


def imageClassification(image, model):
    y = []
    # for file in os.listdir("./reconition/"):
    # y.append(np.array(Image.open(f"./reconition/image.jpg").resize((64, 64)).convert('RGB')) / 255.0)
    y.append(np.array(image.resize((64, 64)).convert('RGB')) / 255.0)
    res = model.predict(np.array(y))
    resul_dict = {}
    for e in res:
        resul_dict["moto"] = e[0]
        resul_dict["voiture"] = e[1]
        resul_dict["avion"] = e[2]
    vehicles_type = (max(resul_dict.items(), key=operator.itemgetter(1))[0])
    classification_result = ("c'est un vehicule de type : %s" % (vehicles_type))
    return classification_result
# def imageClassification(image, model):
#     y = []
#     for file in os.listdir("./reconition/"):
#         y.append(np.array(Image.open(f"./reconition/image.jpg").resize((64, 64)).convert('RGB')) / 255.0)
#     res = model.predict(np.array(y))
#     resul_dict = {}
#     for e in res:
#         resul_dict["moto"] = e[0]
#         resul_dict["voiture"] = e[1]
#         resul_dict["avion"] = e[2]
#     vehicles_type = (max(resul_dict.items(), key=operator.itemgetter(1))[0])
#     print("c'est un vehicule de type : %s" % (vehicles_type))

def modelsDirectory(modelname):
    if not os.path.exists("./models/%s" %(modelname)):
        os.makedirs("./models/%s" %(modelname))
        if not os.path.exists("./models/%s/curve" % (modelname)):
            os.makedirs("./models/%s/curve" % (modelname))


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_dataset()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # model = create_linear_model()
    # model = create_mlp_model()
    # model = create_convolutional_neural_network_model()
    model = create_dense_res_nn_model()

    # model = load_model("cnnmodel.keras")


    print("before training")
    analysis(model, y_train, x_train, y_test, x_test)

    # logs = model.fit(x_train, y_train, batch_size=16, epochs=20000, verbose=0, validation_data=(x_test, y_test))
    logs = model.fit(x_train, y_train, batch_size=128, epochs=300, verbose=0, validation_data=(x_test, y_test))

    print("after training")
    analysis(model, y_train, x_train, y_test, x_test)


    modelname = "resNet_bach128_300epochs"


    modelsDirectory(modelname)

    learningAccuracyCurve(logs, modelname)
    learningLossCurve(logs, modelname)

    model.save("%s.keras" %("./models/%s/model.keras" %(modelname)))
    # model = load_model("linear.keras")

    # imageClassification(image, model)

    # y = []
    #
    # for file in os.listdir("./reconition/"):
    #     y.append(np.array(Image.open(f"./reconition/image.jpg").resize((64, 64)).convert('RGB')) / 255.0)
    #
    # res = model.predict(np.array(y))
    #
    # resul_dict = {}
    # for e in res:
    #     resul_dict["moto"] = e[0]
    #     resul_dict["voiture"] = e[1]
    #     resul_dict["avion"] = e[2]
    #
    #
    # vehicles_type = (max(resul_dict.items(), key=operator.itemgetter(1))[0])
    # print("c'est un vehicule de type : %s" %(vehicles_type))