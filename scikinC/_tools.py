import numpy as np 

def array2c (array, fmt = None):
  "Converts an array in a C string. fmt can be a %format, a callable or None"
  if fmt is None: 
    fmt = lambda x: "%.20f" % x 
  elif isinstance (fmt, str): 
    fmt = lambda x: fmt % x 

  if isinstance (array, (int,float,str)) or len(array.shape) == 0: 
    return fmt(array) 

  return "{%s}"%(", ".join([array2c(row,fmt) for row in array]))


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
