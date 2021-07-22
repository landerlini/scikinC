import numpy as np 
from sklearn.dummy import DummyClassifier

################################################################################
def array2c (array, fmt = None):
  "Converts an array in a C string. fmt can be a %format, a callable or None"
  if fmt is None: 
    fmt_ = lambda x: "%.20f" % x 
  elif isinstance (fmt, str): 
    fmt_ = lambda x: fmt % x 
  else:
    fmt_ = fmt

  if isinstance (array, (list, tuple, set)):
    array = np.asarray(array) 

  if isinstance (array, (int,float,str)) or len(array.shape) == 0: 
    return fmt_(array) 

  return "{%s}"%(", ".join([array2c(row,fmt) for row in array]))


################################################################################
def get_n_features (algo):
  if hasattr(algo, 'n_features'):    return algo.n_features  
  elif hasattr(algo, 'n_features_'): return algo.n_features_  
  elif algo.__class__.__name__ == 'Sequential':
    return algo.layers[-1].kernel.shape[-1] 
  elif algo.__class__.__name__ == 'DecorrTransformer':
    return algo.eig.shape[-1] 
  elif algo.__class__.__name__ == 'StandardScaler':
    return algo.mean_.shape[-1] if algo.mean_ is not None else algo.var_.shape[-1]
  elif algo.__class__.__name__ == 'MinMaxScaler':
    return algo.data_min_.shape[-1] 
  elif algo.__class__.__name__ == 'QuantileTransformer':
    return algo.quantiles_.shape[-1] 
  elif algo.__class__.__name__ == 'Pipeline':
    return get_n_features (algo.steps[-1]) 

  raise TypeError ("Cannot determine output features for %s" % type(algo))


################################################################################
def retrieve_prior (bdt):
  "Retrieve the prior for BDT classifiers"
  if bdt.init_ == 'zero':
    return np.zeros(bdt.n_classes_)
  elif isinstance (bdt.init_, DummyClassifier):
    X = np.empty([1, bdt.n_classes_])
    return np.asarray(bdt.loss_.get_init_raw_predictions(X, bdt.init_)).ravel()
  raise NotImplementedError (
      "Cannot convert initializer %s" % str(bdt.init_)
      )


