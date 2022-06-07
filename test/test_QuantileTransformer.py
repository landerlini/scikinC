import os.path
import pickle

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
  X = np.random.uniform (20,30,(1000, 20))
  scaler_.fit (X) 
  return scaler_

@pytest.fixture
def scaler_normal():
  scaler_ = QuantileTransformer(output_distribution='normal', n_quantiles=100)
  X = np.random.uniform (20,30,(1000, 20))
  scaler_.fit (X) 
  return scaler_

@pytest.fixture
def scaler_bool_uniform():
  scaler_ = QuantileTransformer(output_distribution='uniform')
  X = np.random.choice ([22.,27.], (1000, 20), (0.8, 0.2))
  scaler_.fit (X) 
  return scaler_

@pytest.fixture
def scaler_bool_normal():
  scaler_ = QuantileTransformer(output_distribution='normal')
  X = np.random.choice ([22.,27.], (1000, 20), (0.8, 0.2))
  scaler_.fit (X) 
  return scaler_

@pytest.fixture
def scaler_delta_normal():
  scaler_ = QuantileTransformer(output_distribution='normal')
  X = np.full((10000,20), np.pi)
  scaler_.fit (X) 
  return scaler_

def read_file(filename):
  dir = os.path.dirname(__file__)
  with open(os.path.join(dir, "pathologies", filename), 'rb') as f:
    return pickle.load(f)

@pytest.fixture
def pathology_1():
  return read_file('column_with_quantile_1.pkl')

@pytest.fixture
def pathology_2():
  return read_file('column_with_quantile_2.pkl')

@pytest.fixture
def pathology_3():
  return read_file('column_with_quantile_3.pkl')

@pytest.fixture
def pathology_4():
  return read_file('column_with_quantile_4.pkl')





scalers = [
    'scaler_uniform', 
    'scaler_normal',
    'scaler_bool_uniform',
    'scaler_bool_normal',
    'scaler_delta_normal',
    'pathology_1',
    'pathology_2',
    'pathology_3',
    'pathology_4',
    ]


################################################################################
## Real tests
@pytest.mark.parametrize ('scaler', scalers)
def test_forward (scaler, request):
  scaler = request.getfixturevalue(scaler)
  n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 20

  deployed = deploy_pickle("quantiletransformer", scaler)
  results = []
  try:
    for iAttempt in range(100):
      xtest = np.random.uniform (-1000,-990, n_features)
      py = scaler.transform (xtest[None])
      c  = deployed.transform (n_features, xtest)
      results.append ([py[0].flatten(), c.flatten(), np.abs(py[0]-c).flatten() > 1e-5 ])
      assert np.abs(py-c).max() < 1e-4
  finally:
    array = np.array(results)
    print (array.T)

@pytest.mark.parametrize ('scaler', scalers)
def test_inverse (scaler, request):
  if hasattr(scaler, 'transform_inverse'):
    scaler = request.getfixturevalue(scaler)
    deployed = deploy_pickle("quantiletransformer", scaler)
    xtest = np.random.uniform (0,1, 20)
    py = scaler.inverse_transform (xtest[None])
    c  = deployed.transform_inverse (20, xtest)
    assert np.abs(py-c).max() < 1e-4



