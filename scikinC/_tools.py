import numpy as np 
from sklearn.dummy import DummyClassifier
from scikinC import get_converters, InvertibleConverter
import sys
import sklearn
from packaging import version

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
  elif hasattr(algo, 'n_output_features_'): return algo.n_output_features_
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
  elif algo.__class__.__name__ == 'ColumnTransformer':
    return 1+max([max(c) for _, _, c in algo.transformers_])

  raise TypeError ("Cannot determine output features for %s" % type(algo))


################################################################################
def retrieve_prior (bdt):
  "Retrieve the prior for BDT classifiers"
  if bdt.init_ == 'zero':
    return np.zeros(bdt.n_classes_)
  elif isinstance (bdt.init_, DummyClassifier) and hasattr(bdt, 'loss_'):
    X = np.empty([1, bdt.n_classes_])
    return np.asarray(bdt.loss_.get_init_raw_predictions(X, bdt.init_)).ravel()
  elif isinstance (bdt.init_, DummyClassifier) and hasattr(bdt.init_, 'predict_proba'):
    X = np.zeros([1, bdt.n_features_in_], dtype=np.float32)

    def inverse_softmax(probs):
      probs = np.clip(probs, 1e-15, 1 - 1e-15)
      return np.log(probs)

    ret = inverse_softmax(bdt.init_.predict_proba(X)).ravel()
    return ret
  raise NotImplementedError (
      "Cannot convert initializer %s" % str(bdt.init_)
      )


################################################################################
def get_interpolation_function (func_name):
  return """
    extern "C"
    FLOAT_T %(func_name)s ( FLOAT_T x, FLOAT_T *xs, FLOAT_T *ys, int N )
    {
      int min = 0;
      int max = N; 
      int n;  
      
      if (N<=1) return ys[0]; 

      if (x <= xs[0]) return ys[0]; 
      if (x >= xs[N-1]) return ys[N-1]; 


      for (;;) 
      {
        n = (min + max)/2; 
        if ( x < xs[n] ) 
          max = n; 
        else if ( x >= xs[n+1] )
          min = n; 
        else
          break; 
      } 

      return (x - xs[n])/(xs[n+1]-xs[n])*(ys[n+1]-ys[n]) + ys[n]; 
    }
    """ % dict(func_name=func_name); 

################################################################################
def is_invertible (model):
  import sys 
  converters = get_converters()
  converter = converters[model.__class__.__name__]
  module = __import__ ( "scikinC.%s" % converter, fromlist = [converter])
  ret = getattr(module, converter).INVERTIBLE
  return ret


def sklearn_min_version(req_version):
  return version.parse(sklearn.__version__) >= version.parse(req_version)