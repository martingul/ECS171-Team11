import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import preprocessing
import models
import pickle

def save_model():
    train_X, train_y, test_X, test_y = preprocessing.main()

    clf = models.create_sklearn_MLP()
    clf.fit(train_X, train_y)
    with open("sklean_MLP.pkl","wb") as file:
        pickle.dump(clf,file)

    svm = models.create_svm()
    svm.fit(train_X, train_y)
    with open("svm.pkl","wb") as file:
        pickle.dump(svm,file)
    
    keras = models.create_keras_mlp(train_X, train_y)
    keras.fit(train_X, train_y,batch_size=32, epochs=300)
    keras.save('keras')
    
if __name__ == "__main__":
    save_model()
    