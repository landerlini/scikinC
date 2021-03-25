import numpy as np 
from sklearn.ensemble import GradientBoostingClassifier 

# PyTest testing infrastructure
import pytest

# Local testing infrastructure
from wrap import deploy_pickle 

################################################################################
## Test preparation
@pytest.fixture
def classifier():
  classifier_ = GradientBoostingClassifier()
  X = np.concatenate (( 
      np.random.uniform (0,2,(1000, 10)), 
      np.random.uniform (1,3,(1000, 10)), 
      np.random.uniform (2,4,(1000, 10)), 
      )) 
  y = np.array ( 
      [0] * 1000 + [1] * 1000 + [2] * 1000 )
  classifier_.fit (X, y) 
  return classifier_


@pytest.fixture
def deployed(classifier):
  return deploy_pickle("gbdtc", classifier) 


################################################################################
## Real tests
def test_normalization (classifier, deployed):
  xtest = np.random.uniform (0,1, 10)
  py = classifier.predict_proba (xtest[None])[0]
  c  = deployed.transform (len(py), xtest)

  assert np.abs(np.sum(c)-1).max() < 1e-5

def test_predict (classifier, deployed):
  xtest = np.random.uniform (0,1, 10)
  py = classifier.predict_proba (xtest[None])[0]
  c  = deployed.transform (len(py), xtest)

  print (np.c_[py, c]) 
  assert np.abs(py-c).max() < 1e-5
 

 

