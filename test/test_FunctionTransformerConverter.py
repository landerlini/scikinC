import numpy as np 
from sklearn.preprocessing import FunctionTransformer 

# PyTest testing infrastructure
import pytest

# Local testing infrastructure
from wrap import deploy_pickle 

################################################################################
## Test preparation
@pytest.fixture
def empty_transformer():
  transformer_ = FunctionTransformer(validate=True)
  X = np.random.uniform (20,30,(1000, 10))
  transformer_.fit (X) 
  return transformer_

@pytest.fixture
def log_transformer():
  transformer_ = FunctionTransformer(np.log, np.exp, validate=True)
  X = np.random.uniform (20,30,(1000, 10))
  transformer_.fit (X) 
  return transformer_

@pytest.fixture
def custom_transformer():
  transformer_ = FunctionTransformer(np.square, np.sqrt, validate=True)
  transformer_.func_inC = 'pow({x}, 2)'
  X = np.random.uniform (20,30,(1000, 10))
  transformer_.fit (X) 
  return transformer_


@pytest.fixture
def empty_transformer_wo_fit():
  transformer_ = FunctionTransformer()
  transformer_.n_features_in_ = 10
  return transformer_



scalers = [
    'empty_transformer',
    'log_transformer',
    'custom_transformer',
    'empty_transformer_wo_fit',
    ]


################################################################################
## Real tests
@pytest.mark.parametrize ('scaler', scalers)
def test_forward (scaler, request):
  scaler = request.getfixturevalue(scaler)
  deployed = deploy_pickle("functiontransformer", scaler)
  xtest = np.random.uniform (21,29, 10)
  py = scaler.transform (xtest[None])
  c  = deployed.transform (10, xtest)
  print (xtest, "->", c, " instead of: ", py)
  assert np.abs(py-c).max() < 1e-4
 

@pytest.mark.parametrize ('scaler', scalers)
def test_inverse (scaler, request):
  scaler = request.getfixturevalue(scaler)
  deployed = deploy_pickle("function_transformer", scaler)
  xtest = np.random.uniform (0,1, 10)
  py = scaler.inverse_transform (xtest[None])
  c  = deployed.transform_inverse (10, xtest)
  assert np.abs(py-c).max() < 1e-4
 



