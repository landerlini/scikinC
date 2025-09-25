import numpy as np 
import tensorflow as tf

# PyTest testing infrastructure
import pytest

# Local testing infrastructure
from wrap import deploy_keras 


################################################################################
## Test preparation
def make_classifier (layers):
  classifier_ = tf.keras.models.Sequential(layers) 
      
  X = np.concatenate (( 
      np.random.uniform (0,2,(1000, 10)), 
      np.random.uniform (1,3,(1000, 10)), 
      )) 
  y = np.array ( 
      [0] * 1000 + [1] * 1000 )

  classifier_.compile(loss="binary_crossentropy", optimizer="adam")
  classifier_.fit (X, y) 
  return classifier_


def eval_error (classifier, deployed):
  xtest = np.random.uniform (0,1, 10)
  py = classifier.predict (xtest[None])[0]
  c  = deployed.transform (len(py), xtest)

  return np.abs(py-c).max() 



################################################################################
### Dense layers 
@pytest.fixture
def classifier_dense():
  return make_classifier ([
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])


def test_dense (classifier_dense):
  deployed = deploy_keras("keras_dense", classifier_dense) 
  assert eval_error (classifier_dense, deployed) < 1e-5

 
################################################################################
### PReLU layer 
@pytest.fixture
def classifier_prelu():
  return make_classifier ([
        tf.keras.layers.Dense(16, activation='linear'),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])


def test_prelu (classifier_prelu):
  deployed = deploy_keras("keras_prelu", classifier_prelu) 
  assert eval_error (classifier_prelu, deployed) < 1e-5


################################################################################
### Softmax layer
@pytest.fixture
def classifier_softmax():
    return make_classifier([
        tf.keras.layers.Dense(16, activation='linear'),
        tf.keras.layers.Softmax(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def test_softmax(classifier_softmax):
    deployed = deploy_keras("keras_softmax", classifier_softmax)
    assert eval_error(classifier_softmax, deployed) < 1e-5


################################################################################
### LeakyReLU layer 
@pytest.fixture
def classifier_leakyrelu():
  return make_classifier ([
        tf.keras.layers.Dense(16, activation='linear'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])


def test_leakyrelu (classifier_leakyrelu):
  deployed = deploy_keras("keras_leakyrelu", classifier_leakyrelu) 
  assert eval_error (classifier_leakyrelu, deployed) < 1e-5

################################################################################
### Dropout layer (passthrough)
@pytest.fixture
def classifier_dropout():
    return make_classifier ([
        tf.keras.layers.Dense(16, activation='linear'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def test_dropout(classifier_dropout):
    deployed = deploy_keras("keras_dropout", classifier_dropout)
    assert eval_error (classifier_dropout, deployed) < 1e-5

################################################################################
### BatchNormalization layer
@pytest.fixture
def classifier_batch_normalization():
    return make_classifier ([
        tf.keras.layers.Dense(16, activation='linear'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def test_batch_normalization(classifier_batch_normalization):
    deployed = deploy_keras("keras_dropout", classifier_batch_normalization)
    assert eval_error (classifier_batch_normalization, deployed) < 1e-5

