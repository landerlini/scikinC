"""
Set of tools to convert scikit learn and tensorflow into plain C functions
"""

from scikinC.BaseConverter import BaseConverter, InvertibleConverter
from scikinC               import ModelLoader 
from copy                  import copy

version = '0.1'

__CONVERTERS = {
      ## Scikit-learn 
      'GradientBoostingClassifier': 'GBDTC_Converter', 
      'MinMaxScaler': 'MinMaxScalerConverter', 
      'StandardScaler': 'StandardScalerConverter', 
      'QuantileTransformer': 'QuantileTransformerConverter',
      'WeightedQuantileTransformer': 'QuantileTransformerConverter',
      'DecorrTransformer': 'DecorrTransformerConverter',
      'Pipeline': 'PipelineConverter', 
      'FastQuantileLayer': 'FastQuantileLayerConverter',
      'FunctionTransformer': 'FunctionTransformerConverter',
      'ColumnTransformer': 'ColumnTransformerConverter',
      'PolynomialFeatures': 'PolynomialFeaturesConverter',

    ## Keras
      'Sequential': 'KerasSequentialConverter', 
    }

def get_converters ():
  """Return the dictionary of converters"""
  global __CONVERTERS
  return copy(__CONVERTERS)


def convert ( some_object, *args, **kwargs ):

  if isinstance (some_object, dict) and len (some_object) > 1:
    some_object = [{k:v} for k,v in some_object.items()] 
  elif not isinstance ( some_object, (list,tuple) ):
    some_object = (some_object,) 


  objs = []
  for obj in some_object:
    if isinstance (obj, str):
      objs += ModelLoader.load_from_string (obj)
    elif isinstance (obj, dict):
      objs.append(obj) 
    else:
      objs.append({obj.__class__.__name__ : obj}) 

  ret = [] 
  for kobj in objs: 
    k, obj = next(iter(kobj.items())) 
    if obj.__class__.__name__ in __CONVERTERS.keys(): 
      converter = __CONVERTERS[obj.__class__.__name__] 
      cnv = __import__ ( "scikinC.%s" % converter, fromlist = [converter] ) 
      ret.append(
          getattr(cnv, converter)(*args, **kwargs).convert (obj, name=k) 
          )

    else:
      raise NotImplementedError (
          "No converter is known for class %s" % obj.__class__.__name__ 
          )

  return "\n\n".join (ret) 

