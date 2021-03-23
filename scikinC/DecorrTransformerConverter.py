import numpy as np 
from scikinC import BaseConverter 
from scipy import stats
from ._tools import array2c 

class DecorrTransformerConverter (BaseConverter):
  def __init__ (self):
    pass 


  def convert (self, model, name = None): 
    lines = self.header() 

    eig = model.eig
    nFeatures = eig.shape[0] 

    lines.append ( """
    extern "C"
    double * %(name)s (double *ret, const double *x)
    {
      int i, j; 
      double e[%(nFeatures)d][%(nFeatures)d] = %(eString)s; 

      for (i = 0; i < %(nFeatures)d; ++i)
        ret [i] = 0;

      for (i = 0; i < %(nFeatures)d; ++i)
        for (j = 0; j < %(nFeatures)d; ++j)
          ret [i] += x[j] * e[j][i]; 


      return ret; 
    }

    extern "C"
    double * %(name)s_inverse (double *ret, const double *x)
    {
      int i, j; 
      double e[%(nFeatures)d][%(nFeatures)d] = %(eString)s; 

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






