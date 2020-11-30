# -*- coding: utf-8 -*-
from sklearn.neural_network import MLPClassifier
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.svm import SVC
from time import time

def printModelScores(actual_y, pred_y):
  # print precision and recall scores for predictions of a model
  # Args:
  #  - actual_y: actual class attribute values
  #  - pred_y: predicted values from one of the models
  print('Train Precision =', precision_score(actual_y, pred_y))
  print('Train Recall =', recall_score(actual_y, pred_y))

def create_sklearn_MLP():
  # returns a new instance of MLPClassifier with the params we found worked best
  return MLPClassifier(hidden_layer_sizes=(24, 8), activation='logistic', solver='adam', learning_rate_init=0.005, max_iter=300)

def create_sklearn_MLP(hidden_layer_sizes=(24, 8), activation='logistic', solver='adam', learning_rate_init=0.005, max_iter=300):
    # create model with custom parameters
    # Args:
    #  - hidden_layer_sizes: tuple containing sizes of the hidden layers of the model
    #  - activation: activation function to use. can be one of {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
    #  - solver: optimizer to use with the model. one of {‘lbfgs’, ‘sgd’, ‘adam’}
    #  - learning_rate_init: value of learning rate to use
    #  - max_iter: maximum number of iterations to do if model has not yet converged
    # see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html for full details
  return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, learning_rate_init=learning_rate_init, max_iter=max_iter)


def train_eval_sklearn_mlp(clf, train_X, train_y, test_X, test_y):
  # train and evaluate the MLPClassifier
  # prints out the precision and recall for train and test sets, classification report, and 
  # the training time of the model
  # Args:
  #  - clf: MLPClassifier model to use
  #  - train_X: dataframe of training set with dependent attruibutes
  #  - train_y: dataframe of training set with class attribute only 
  #  - test_X: dataframe of test set with dependent attruibutes
  #  - test_y: dataframe of test set with class attribute only 
  # Returns:
  #  - mlpc_test_probs: list of the predictions of the model when run on test set
  mlpc_start_time = time()
  clf.fit(train_X, train_y)
  mlpc_train_time = time() - mlpc_start_time
  mlpc_train_probs = clf.predict(train_X)
  mlpc_train_preds = [0 if mlpc_train_probs[i] < 0.5 else 1 for i in range(len(mlpc_train_probs))]
  printModelScores(train_y, mlpc_train_preds)
  mlpc_test_probs = clf.predict(test_X)
  mlpc_test_preds = [0 if mlpc_test_probs[i] < 0.5 else 1 for i in range(len(mlpc_test_probs))]
  printModelScores(test_y, mlpc_test_preds)
  print('Training Time: {:.2f}s'.format(mlpc_train_time))

  print(classification_report(test_y, mlpc_test_preds))
  # return model predictions for further use
  return mlpc_test_probs

def create_keras_mlp(train_X, train_y):
  # returns a new instance of keras sequential model with the structure and params we found worked best
  model = keras.Sequential([layers.Dense(32, activation='sigmoid', input_dim=train_X.shape[1]),
                            layers.Dense(12, activation='sigmoid'),
                            layers.Dense(1, activation='sigmoid')])
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mean_squared_error', 
              metrics=['accuracy'])
  return model

def create_keras_mlp(learn_rate=None, input_dim=None):
  # returns a new instance of keras sequential model with the structure and params we found worked best
  # use with keras_gd because need input_dim to be passed as a parameter for sklearn KerasClassifier wrapper
  model = keras.Sequential([layers.Dense(32, activation='sigmoid', input_dim=input_dim),
                            layers.Dense(12, activation='sigmoid'),
                            layers.Dense(1, activation='sigmoid')])
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=learn_rate), loss='mean_squared_error', metrics=['accuracy'])
  return model

def create_keras_mlp(train_X, train_y, activation='sigmoid', learning_rate=0.005, loss='mean_squared_error'):
  # create and return an instance of keras sequential model with custom parameters
    # Args:
    #  - activation: activation function to use
    #  - learning_rate: value of learning rate to use
    #  - loss: loss function to use
    # see https://www.tensorflow.org/api_docs/python/tf/keras/Sequential for list of possible parameter options
  model = keras.Sequential([layers.Dense(32, activation=activation, input_dim=train_X.shape[1]),
                            layers.Dense(12, activation=activation),
                            layers.Dense(1, activation=activation)])
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, 
              metrics=['accuracy'])
  return model

def train_eval_keras_mlp(model, train_X, train_y, test_X, test_y, epochs=300, batch_size=32):
  # train and evaluate keras sequential model
  # prints out the precision and recall for train and test sets, classification report, and 
  # the training time of the model
  # Args:
  #  - model: keras sequential model to use
  #  - train_X: dataframe of training set with dependent attruibutes
  #  - train_y: dataframe of training set with class attribute only 
  #  - test_X: dataframe of test set with dependent attruibutes
  #  - test_y: dataframe of test set with class attribute only 
  # Returns:
  #  - keras_test_probs: list of the predictions of the model when run on test set
  keras_start_time = time()
  model_res = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs)
  keras_train_time = time() - keras_start_time

  # evaluate model on train set and test set
  keras_train_probs = model.predict(train_X)
  keras_train_preds = [0 if keras_train_probs[i] < 0.5 else 1 for i in range(len(keras_train_probs))]
  print("\n")
  printModelScores(train_y, keras_train_preds)
  keras_test_probs = model.predict(test_X)
  keras_test_preds = [0 if keras_test_probs[i] < 0.5 else 1 for i in range(len(keras_test_probs))]
  printModelScores(test_y, keras_test_preds)
  print('Training Time: {:.2f}s'.format(keras_train_time))

  print(classification_report(test_y, keras_test_preds))
  # return model predictions for further use
  return keras_test_probs

def create_svm():
  # returns a new instance of SVM classifer with the params we found worked best
  return SVC(kernel='linear', class_weight='balanced', gamma='scale')

def create_svm(kernel='linear', class_weight='balanced', gamma='scale'):
  # return model with custom parameters
    # Args:
    #  - kernel: kernel type to use. can be one of {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
    #  - class_weight: sets regularization parameter. can be 'balanced' or a dict where parameter C of class i is set to class_weight[i]*C
    #  - gamma: kernel coefficeint for {‘poly’, ‘rbf’, ‘sigmoid’}. can be {‘scale’, ‘auto’} or float value
    # see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html for full details
  return SVC(kernel=kernel, class_weight=class_weight, gamma=gamma) 

def train_eval_svm(svm_clf, train_X, train_y, test_X, test_y):
  # train and evaluate the SVM model
  # prints out the precision and recall for train and test sets, classification report, and 
  # the training time of the model
  # Args:
  #  - svm_clf: SVM to use
  #  - train_X: dataframe of training set with dependent attruibutes
  #  - train_y: dataframe of training set with class attribute only 
  #  - test_X: dataframe of test set with dependent attruibutes
  #  - test_y: dataframe of test set with class attribute only 
  # Returns:
  #  - svm_test_probs: list of the predictions of the model when run on test set
  svm_start_time = time()
  svm_clf.fit(train_X, train_y)
  svm_train_time = time() - svm_start_time
  svm_train_probs = svm_clf.predict(train_X)
  svm_train_preds = [0 if svm_train_probs[i] < 0.5 else 1 for i in range(len(svm_train_probs))]
  printModelScores(train_y, svm_train_preds)
  svm_test_probs = svm_clf.predict(test_X)
  svm_test_preds = [0 if svm_test_probs[i] < 0.5 else 1 for i in range(len(svm_test_probs))]
  printModelScores(test_y, svm_test_preds)
  print('Training Time: {:.2f}s'.format(svm_train_time))

  print(classification_report(test_y, svm_test_preds))
  return svm_test_probs

def mlpc_svm_cross_validation(model, train_X, train_y):
  # get cross validation scores for SVM or MLPCModel
  # Args:
  #  - model: model to use to calculate the cross validation scores
  #  - train_X: dataframe of training set with dependent attruibutes
  #  - train_y: dataframe of training set with class attribute only 
  return cross_val_score(model, train_X, train_y, scoring='f1_micro', n_jobs=-1)

def keras_cross_valdidation(model, X, y):
  # get cross validation scores for keras sequential model
  # Args:
  #  - model: keras model to be used
  #  - X: dataframe of dependent attributes of the dataset
  #  - y: dataframe of class attribute of the dataset
  KFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
  cv_scores = []

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mean_squared_error', 
                  metrics=['accuracy'])
  for train, test in KFold.split(X, y):
    model.fit(X.loc[train], y.loc[train], batch_size=32, epochs=300)
    
    # Evaluate model and append metric scores
    scores = model.evaluate(X.loc[test], y.loc[test])
    cv_scores.append(scores[1] * 100)

  # Print results for each fold of the metric
  print ('\n', model.metrics_names[1])
  for index, score in enumerate(cv_scores):
    print ('fold', index, ':', score)
  print ('mean:', np.mean(cv_scores), 'std:', np.std(cv_scores))

def gd_keras_mlp(model, train_X, train_y):
  # perform gridsearch for batch size, epochs, and learning rate for the keras model
  # Args:
  #  - model: keras sequential model to use for gridsearch
  #  - train_X: dataframe of training set with dependent attruibutes
  #  - train_y: dataframe of training set with class attribute only
  param_grid = dict(batch_size=[100, 150, 200, 300], epochs=[150, 200, 250, 300], learn_rate=[0.001, 0.005, 0.01, 0.05])
  grid_search(model, param_grid, train_X, train_y)

  model = KerasClassifier(build_fn=create_keras_mlp, input_dim=train_X.shape[1], verbose=0)
  gd_keras_mlp(model, train_X, train_y)

def gd_mlpc(model, train_X, train_y):
  # perform gridsearch for hidden layer size, activation, and learning rate, and max iterations
  # for the MLPClassifer model
  # Args:
  #  - model: MLPClassifier to use for gridsearch
  #  - train_X: dataframe of training set with dependent attruibutes
  #  - train_y: dataframe of training set with class attribute only
  param_grid = dict(hidden_layer_sizes=[(24, 8), (20, 5), (20), (10)], 
    activation=('logistic', 'tanh', 'relu'), learning_rate_init=[0.005, 0.001, 0.05], 
    max_iter=[300, 500, 800])
  grid_search(model, param_grid, train_X, train_y)

def gd_svm(model, train_X, train_y):
  # perform gridsearch for batch size, epochs, and learning rate for the MLPClassifer model
  # Args:
  #  - model: MLPClassifier to use for gridsearch
  #  - train_X: dataframe of training set with dependent attruibutes
  #  - train_y: dataframe of training set with class attribute only
  param_grid = dict(kernel=('linear', 'poly', 'rbf', 'sigmoid'), 
    class_weight=(None, 'balanced'), gamma=('scale', 'auto', 1.2, 0.8))
  grid_search(model, param_grid, train_X, train_y)

def grid_search(model, param_grid, train_X, train_y):
  # perform gridsearch on the given model with specfied paramters
  # prints the scores of each of the models using each of the  different combinations 
  # of parameters, and the best parameters and score of all the models tried.
  # Args:
  #  - model: model to use for gridsearch
  #  - param_grid: dict that specifies the grid of parameters to use 
  #  - train_X: dataframe of training set with dependent attruibutes
  #  - train_y: dataframe of training set with class attribute only
  # Returns:
  #  - grid_result: results of gridsearch
  grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='balanced_accuracy', 
    n_jobs=-1, cv=3)
  grid_result = grid.fit(train_X, train_y)
  
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))
  return grid_result



