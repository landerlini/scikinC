import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline 

# PyTest testing infrastructure
import pytest

# Local testing infrastructure
from wrap import deploy_pickle 

################################################################################
## Test preparation
@pytest.fixture
def pipeline():
  X = np.concatenate (( 
      np.random.uniform (0,2,(1000, 10)), 
      np.random.uniform (1,3,(1000, 10)), 
      np.random.uniform (2,4,(1000, 10)), 
      )) 

  step1 = MinMaxScaler ()
  X2 = step1.fit_transform (X)
  step2 = StandardScaler()
  step2.fit(X2)

  return Pipeline(steps = ((("minmax", step1), ("standard", step2))))


@pytest.fixture
def deployed(pipeline):
  return deploy_pickle("pipeline", pipeline) 


################################################################################
## Real tests
def test_pipeline (pipeline, deployed):
  xtest = np.random.uniform (0,1, 10)
  py = pipeline.transform (xtest[None])[0]
  c  = deployed.transform (len(py), xtest)
  c_back  = pipeline.inverse_transform (py[None])
  py_back = deployed.transform_inverse (len(py), py)

  assert np.abs(py-c).max() < 1e-5
  assert np.abs(py_back - c_back).max() < 1e-5

 

 


