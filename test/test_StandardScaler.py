import numpy as np 
from sklearn.preprocessing import StandardScaler 

# PyTest testing infrastructure
import pytest

# Local testing infrastructure
from wrap import deploy_pickle 

################################################################################
## Test preparation
@pytest.fixture
def scaler():
  scaler_ = StandardScaler()
  X = np.random.uniform (20,30,(1000, 10))
  scaler_.fit (X) 
  return scaler_


@pytest.fixture
def deployed(scaler):
  return deploy_pickle("standardscaler", scaler)


################################################################################
## Real tests
def test_forward (scaler, deployed):
  xtest = np.random.uniform (20,30, 10)
  py = scaler.transform (xtest[None])
  c  = deployed.transform (10, xtest)
  assert np.abs(py-c).max() < 1e-5
 

def test_inverse (scaler, deployed):
  xtest = np.random.uniform (0,1, 10)
  py = scaler.inverse_transform (xtest[None])
  c  = deployed.transform_inverse (10, xtest)
  assert np.abs(py-c).max() < 1e-5
 

