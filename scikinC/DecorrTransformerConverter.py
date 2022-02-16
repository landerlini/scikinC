import numpy as np 
from scikinC import InvertibleConverter 
from scipy import stats
from ._tools import array2c 

class DecorrTransformerConverter (InvertibleConverter):
  """
  Converter for a decorrelation step not defined in scikit-learn.
  This is only present for compatibility with existing applications 
  in the LHCb Software stack. 
  """

  def convert (self, model, name = None): 
    lines = self.header() 

    eig = model.eig
    nFeatures = eig.shape[0] 

    lines.append ( """
    extern "C"
    FLOAT_T * %(name)s (FLOAT_T *ret, const FLOAT_T *x)
    {
      int i, j; 
      FLOAT_T e[%(nFeatures)d][%(nFeatures)d] = %(eString)s; 

      for (i = 0; i < %(nFeatures)d; ++i)
        ret [i] = 0;

      for (i = 0; i < %(nFeatures)d; ++i)
        for (j = 0; j < %(nFeatures)d; ++j)
          ret [i] += x[j] * e[j][i]; 


      return ret; 
    }

    extern "C"
    FLOAT_T * %(name)s_inverse (FLOAT_T *ret, const FLOAT_T *x)
    {
      int i, j; 
      FLOAT_T e[%(nFeatures)d][%(nFeatures)d] = %(eString)s; 

      for (i = 0; i < %(nFeatures)d; ++i)
        ret [i] = 0;

      for (i = 0; i < %(nFeatures)d; ++i)
        for (j = 0; j < %(nFeatures)d; ++j)
          ret [i] += x[j] * e[i][j]; 

      return ret; 
    }
    """ % dict (
      name = name, 
      nFeatures = nFeatures,
      eString = array2c(eig), 

      ))

    return "\n".join (lines) 






