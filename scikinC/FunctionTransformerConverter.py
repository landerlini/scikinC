import numpy as np

from scikinC import InvertibleConverter
from ._tools import array2c


class FunctionTransformerConverter (InvertibleConverter):
  def convert(self, model, name=None):
    lines = self.header()

    if not hasattr(model, 'n_features_in_'):
      raise NotImplementedError(
          "Conversion requires its n_features_in_ attribute to be set")

    nFeatures = model.n_features_in_

    func_dict = {
        None: '{x}',
        np.log1p: 'log(1+{x})',
        np.expm1: 'exp({x})-1',
        np.arcsin: 'asin({x})',
        np.arccos: 'acos({x})',
        np.arctan: 'atan({x})',
        np.abs: 'fabs({x})',
        }

    if model.func is not None or model.inverse_func is not None:
      lines.append("#include <math.h>")

    c_funcs = ('sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log', 'log10', 'sqrt', 'ceil', 'floor')
    func_dict.update({getattr(np, f): "%s({x})"%f for f in c_funcs})
    
    if hasattr(model, 'func_inC'):
      fwd = model.func_inC
    elif hasattr(model.func, 'inC'):
      fwd = model.func.inC
    elif model.func in func_dict.keys():
      fwd = func_dict[model.func]
    else:
      raise NotImplementedError(
          "Translation of function %s not implemented nor defined as func_inC argument" 
          % str(model.func))


    if hasattr(model, 'inverse_func_inC'):
      bwd = model.inverse_func_inC
    elif hasattr(model.inverse_func, 'inC'):
      bwd = model.inverse_func.inC
    elif model.inverse_func in func_dict.keys():
      bwd = func_dict[model.inverse_func]
    else:
      raise NotImplementedError(
          "Translation of function %s not implemented nor defined as inverse_func_inC argument" 
          % str(model.inverse_func))


    ## Input sanitization
    if any([banned in fwd for banned in (';', '//', '/*', '*/')]):
      raise ValueError("Invalid implementation: %s" % fwd);
    if any([banned in bwd for banned in (';', '//', '/*', '*/')]):
      raise ValueError("Invalid implementation: %s" % bwd);


    lines.append("""
    extern "C"
    FLOAT_T* %(name)s (FLOAT_T* ret, const FLOAT_T *input)
    {
      int c;

      for (c = 0; c < %(nFeatures)d; ++c)
        ret [c] = %(func)s;

      return ret;
    }
      """ % dict(
        name=name,
        nFeatures=nFeatures,
        func=fwd.format(x='input[c]'),
        )
      )

    lines.append ( """
    extern "C"
    FLOAT_T * %(name)s_inverse(FLOAT_T * ret, const FLOAT_T * input)
    {
      int c;

      for (c=0; c < %(nFeatures)d; ++c)
        ret [c]= %(func)s;

      return ret;
    }
      """ % dict (
        name=name, 
        nFeatures = nFeatures,
        func=bwd.format(x='input[c]'),
        )
      )


    return "\n".join(lines)
