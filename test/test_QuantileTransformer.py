import numpy as np 
from sklearn.preprocessing import QuantileTransformer 

# PyTest testing infrastructure
import pytest

# Local testing infrastructure
from wrap import deploy_pickle 

################################################################################
## Test preparation
@pytest.fixture
def scaler_uniform():
  scaler_ = QuantileTransformer()
  X = np.random.uniform (20,30,(1000, 10))
  scaler_.fit (X) 
  return scaler_

@pytest.fixture
def scaler_normal():
  scaler_ = QuantileTransformer(output_distribution='normal', n_quantiles=2)
  X = np.random.uniform (20,30,(1000, 10))
  scaler_.fit (X) 
  return scaler_

@pytest.fixture
def scaler_bool_uniform():
  scaler_ = QuantileTransformer(output_distribution='uniform')
  X = np.random.choice ([22.,27.], (1000, 10), (0.8, 0.2))
  scaler_.fit (X) 
  return scaler_



scalers = [
    'scaler_uniform', 
    'scaler_normal',
    'scaler_bool_uniform',
    ]


################################################################################
## Real tests
@pytest.mark.parametrize ('scaler', scalers)
def test_forward (scaler, request):
  scaler = request.getfixturevalue(scaler)
  deployed = deploy_pickle("quantiletransformer", scaler)
  xtest = np.random.uniform (21,29, 10)
  py = scaler.transform (xtest[None])
  c  = deployed.transform (10, xtest)
  print (xtest, "->", c, " instead of: ", py)
  assert np.abs(py-c).max() < 1e-4
 

@pytest.mark.parametrize ('scaler', scalers)
def test_inverse (scaler, request):
  scaler = request.getfixturevalue(scaler)
  deployed = deploy_pickle("quantiletransformer", scaler)
  xtest = np.random.uniform (0,1, 10)
  py = scaler.inverse_transform (xtest[None])
  c  = deployed.transform_inverse (10, xtest)
  assert np.abs(py-c).max() < 1e-4
 


