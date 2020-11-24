from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.svm import SVC
from time import time

def printModelScores(actual_y, pred_y):
  print('Train Precision =', precision_score(actual_y, pred_y))
  print('Train Recall =', recall_score(actual_y, pred_y))

def create_sklearn_MLP():
  # create a MLPClassifier with params we found
  return MLPClassifier(hidden_layer_sizes=(24, 8), activation='logistic', solver='adam', learning_rate_init=0.005, max_iter=300)

def train_eval_sklearn_mlp(clf, train_X, train_y, test_X, test_y):
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

# try keras mlp
def create_keras_mlp(train_X, train_y):
  model = keras.Sequential([layers.Dense(32, activation='sigmoid', input_dim=train_X.shape[1]),
                            layers.Dense(12, activation='sigmoid'),
                            layers.Dense(1, activation='sigmoid')])
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mean_squared_error', 
              metrics=['accuracy'])
  return model

def train_eval_keras_mlp(model, train_X, train_y):
  keras_start_time = time()
  model_res = model.fit(train_X, train_y, batch_size=32, epochs=300)
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
  return SVC(kernel='sigmoid', class_weight='balanced', gamma='auto')

def train_eval_svm(svm_clf, train_X, train_y, test_X, test_y):
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
