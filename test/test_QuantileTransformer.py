import os.path
import pickle

import numpy as np
from sklearn.preprocessing import QuantileTransformer 

# PyTest testing infrastructure
import pytest

# Local testing infrastructure
from wrap import deploy_pickle
from fixture_registry import fixtures

################################################################################
## Test preparation
@fixtures.register()
def scaler_uniform():
  scaler_ = QuantileTransformer()
  X = np.random.uniform (20,30,(1000, 20))
  scaler_.fit (X) 
  return scaler_

@fixtures.register()
def scaler_normal():
  scaler_ = QuantileTransformer(output_distribution='normal', n_quantiles=100)
  X = np.random.uniform (20,30,(1000, 20))
  scaler_.fit (X) 
  return scaler_

@fixtures.register()
def scaler_bool_uniform():
  scaler_ = QuantileTransformer(output_distribution='uniform')
  X = np.random.choice ([22.,27.], (1000, 20), (0.8, 0.2))
  scaler_.fit (X) 
  return scaler_

@fixtures.register()
def scaler_bool_normal():
  scaler_ = QuantileTransformer(output_distribution='normal')
  X = np.random.choice ([22.,27.], (1000, 20), (0.8, 0.2))
  scaler_.fit (X) 
  return scaler_

@fixtures.register()
def scaler_delta_normal():
  scaler_ = QuantileTransformer(output_distribution='normal')
  X = np.full((10000,20), np.pi)
  scaler_.fit (X) 
  return scaler_


################################################################################
## Real tests
@fixtures.test()
def test_forward (scaler):
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

@fixtures.test()
def test_inverse (scaler):
  if hasattr(scaler, 'transform_inverse'):
    deployed = deploy_pickle("quantiletransformer", scaler)
    xtest = np.random.uniform (0,1, 20)
    py = scaler.inverse_transform (xtest[None])
    c  = deployed.transform_inverse (20, xtest)
    assert np.abs(py-c).max() < 1e-4



