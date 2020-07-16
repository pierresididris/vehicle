import operator
import os
from enum import Enum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def analysis(model, y_train, x_train, y_test, x_test):
    print("confusion train matrix" )
    true_values = np.argmax(y_train, axis=1)
    predictions = np.argmax(model.predict(x_train), axis=1)
    print(confusion_matrix(true_values, predictions))

    print("confusion test matrix")
    true_values = np.argmax(y_test, axis=1)
    predictions = np.argmax(model.predict(x_test), axis=1)
    print(confusion_matrix(true_values, predictions))

    print(f'Train Acc : {model.evaluate(x_train, y_train)[1]}')
    print(f'Test Acc : {model.evaluate(x_test, y_test)[1]}')


def learningAccuracyCurve(logs, modelname):
    print("Affichage de la courbe d'accuracy de l'apprentissage")
    plt.plot(logs.history['accuracy'])
    plt.plot(logs.history['val_accuracy'])
    plt.savefig("models/%s/curve/accuracy.png" % (modelname))
    plt.show()


def learningLossCurve(logs, modelname):
    print("Affichage de la courbes de loss de l'apprentissage")
    plt.plot(logs.history['loss'])
    plt.plot(logs.history['val_loss'])
    plt.savefig("models/%s/curve/loss.png" %(modelname))
    plt.show()
