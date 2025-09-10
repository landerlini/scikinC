import numpy as np 
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

# PyTest testing infrastructure
import pytest

# Local testing infrastructure
from wrap import deploy_pickle
from fixture_registry import fixtures

################################################################################
## Test preparation
@fixtures.register('invertible')
def passthrough_transformer():
  transformer_ = ColumnTransformer([], remainder='passthrough')
  X = np.random.uniform (20,30,(1000, 10))
  transformer_.fit (X) 
  return transformer_


@fixtures.register()
def double_passthrough_transformer():
  transformer_ = ColumnTransformer([
    ('keep1', 'passthrough', [0,2]),
    ('keep2', 'passthrough', [3,5]),
    ])
  X = np.random.uniform (20,30,(1000, 10))
  transformer_.fit (X) 
  return transformer_


@fixtures.register('invertible')
def ss_and_passthrough_transformer():
  transformer_ = ColumnTransformer([
    ('ss', StandardScaler(), [1,2,3]),
    ], remainder='passthrough')
  X = np.random.uniform (20,30,(1000, 10))
  transformer_.fit (X) 
  return transformer_


@fixtures.register('invertible')
def qt_and_passthrough_transformer():
  transformer_ = ColumnTransformer([
    ('qt', QuantileTransformer(output_distribution='normal'), [0,2]),
    ], remainder='passthrough')
  X = np.random.uniform (20,30,(1000, 10))
  transformer_.fit (X) 
  return transformer_


@fixtures.register('invertible')
def double_qt_and_passthrough_transformer():
  transformer_ = ColumnTransformer([
    ('qt1', QuantileTransformer(n_quantiles=100, output_distribution='normal'), [3,4]),
    ('qt2', QuantileTransformer(n_quantiles=500, output_distribution='normal'), [1,2]),
    ], remainder='passthrough')
  X = np.random.uniform (20,30,(1000, 10))
  transformer_.fit (X) 
  return transformer_


@fixtures.register('invertible')
def qt_and_ss_and_passthrough_transformer():
  transformer_ = ColumnTransformer([
    ('qt', QuantileTransformer(output_distribution='normal'), [0,1]),
    ('ss', StandardScaler(), [3,4]),
    ], remainder='passthrough')
  X = np.random.uniform (20,30,(1000, 10))
  transformer_.fit (X) 
  return transformer_


@fixtures.register('invertible')
def qt_and_ft_transformer_only():
  transformer_ = ColumnTransformer([
    ('qt', QuantileTransformer(output_distribution='normal'), [0,1,2,3,4]),
    ('ft', FunctionTransformer(), [5,6,7,8,9]),
    ])
  X = np.random.uniform (20,30,(1000, 10))
  transformer_.fit (X) 
  return transformer_


@fixtures.register('invertible')
def double_qt_transformer_only():
  transformer_ = ColumnTransformer([
    ('qt1', QuantileTransformer(n_quantiles=100, output_distribution='normal'), [5,6,7,8,9]),
    ('qt2', QuantileTransformer(n_quantiles=500, output_distribution='normal'), [0,1,2,3,4]),
    ])
  X = np.random.uniform (20,30,(1000, 10))
  transformer_.fit (X) 
  return transformer_


@fixtures.register()
def qt_and_ft_transformer_dropping():
  transformer_ = ColumnTransformer([
    ('qt', QuantileTransformer(output_distribution='normal'), [0,2]),
    ('ft', FunctionTransformer(), [3,5]),
    ], remainder='drop')
  X = np.random.uniform (20,30,(1000, 10))
  transformer_.fit (X) 
  return transformer_


################################################################################
## Real tests
@fixtures.test()
def test_forward (scaler):
  deployed = deploy_pickle("functiontransformer", scaler)
  xtest = np.random.uniform (21,29, 10)
  py = scaler.transform (xtest[None])
  print (py.shape)
  c  = deployed.transform (py.shape[1], xtest)
  print (xtest, "->", c, " instead of: ", py)
  assert np.abs(py-c).max() < 1e-4
 

@fixtures.test('invertible')
def test_inverse (scaler):
  deployed = deploy_pickle("function_transformer", scaler)
  xtest = np.random.uniform (0,1, 10)
  py = np.empty (10)
  counter = 0
  for _, transform, columns in scaler.transformers_:
    inputs = xtest[counter:counter+len(columns)]
    counter += len(columns)
    if transform == 'passthrough':
      py[columns] = inputs
    else: 
      py[columns] = transform.inverse_transform ([inputs])[0]

  c  = deployed.transform_inverse (len(py), xtest)

  print (np.c_ [xtest, c, py])
  assert np.abs(py-c).max() < 1e-4
 
