import numpy as np 
from scikinC import BaseConverter 
from scipy import stats
from ._tools import array2c 

class QuantileTransformerConverter (BaseConverter):
  def __init__ (self):
    pass 


  def convert (self, model, name = None): 
    lines = self.header() 

    distr = model.output_distribution
    if distr not in ['normal', 'uniform']:
      raise NotImplementedError ("Unexpected distribution %s" % distr)

    lines . append ( """
    extern "C"
    double qtc_interpolate_for_%(name)s ( double x, double *xs, double *ys, int N )
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
    """ % dict(name = name)); 

    q = model.quantiles_ 
    nQuantiles = model.quantiles_.shape[0] 
    nFeatures   = model.quantiles_.shape[1] 
    y = np.linspace (1e-7, 1.-1e-7, nQuantiles) 
    if distr == 'normal':
      y = stats.norm.ppf(y) 

    lines.append ("""
    extern "C"
    double *%(name)s (double *ret, const double *x)
    {
      int c; 
      double q[%(nFeatures)d][%(nQuantiles)d] = %(qString)s; 
      double y[%(nQuantiles)d] = %(yString)s; 

      for (c = 0; c < %(nFeatures)d; ++c)
        ret[c] = qtc_interpolate_for_%(name)s (x[c], q[c], y, %(nQuantiles)d ); 

      return ret; 
    }
    """ % dict (
      name = name, 
      nQuantiles = nQuantiles, 
      nFeatures  = nFeatures,
      qString = array2c ( q.T ), #", ".join ([
        #"{%s}"%(", ".join ([str(x) for x in ql])) for ql in q.T]) ,
      yString = array2c ( y ), #", ".join ([str(x) for x in y]) 
      )); 

    lines.append ("""
    extern "C"
    double *%(name)s_inverse (double *ret, const double *x)
    {
      int c; 
      double q[%(nFeatures)d][%(nQuantiles)d] = %(qString)s; 
      double y[%(nQuantiles)d] = %(yString)s; 

      for (c = 0; c < %(nFeatures)d; ++c)
        ret[c] = qtc_interpolate_for_%(name)s ( x[c], y, q[c], %(nQuantiles)d ); 

      return ret; 
    }
    """ % dict (
      name = name, 
      nQuantiles = nQuantiles, 
      nFeatures  = nFeatures,
      qString = array2c (q.T), #", ".join ([
        #"{%s}"%(", ".join ([str(x) for x in ql])) for ql in q.T]) ,
      yString = array2c (y), #", ".join ([str(x) for x in y]) 
      )); 

    return "\n".join (lines) 





