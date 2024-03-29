import numpy as np 
import sys
from scikinC import BaseConverter 
from scipy import stats
from ._tools import array2c 

class QuantileTransformerConverter (BaseConverter):
  def convert (self, model, name = None): 
    lines = self.header() 

    distr = model.output_distribution
    if distr not in ['normal', 'uniform']:
      raise NotImplementedError ("Unexpected distribution %s" % distr)

    lines . append ( """
    extern "C"
    FLOAT_T qtc_interpolate_for_%(name)s ( FLOAT_T x, FLOAT_T *xs, FLOAT_T *ys, int N )
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

    nSamples = 512
    xAxis = np.linspace (1e-7, 1.-1e-7, nSamples)
    yAxis = stats.norm.ppf (xAxis)
    yAxis [0] = stats.norm.ppf (1e-7 + np.spacing(1))
    yAxis [-1] = stats.norm.ppf (1.-1e-7 + np.spacing(1))

    #print (np.c_[y], file=sys.stderr)
#    print (stats.norm.ppf(model.references_), file=sys.stderr)
#    print (np.c_[xAxis, yAxis], file=sys.stderr)

    uniform_to_normal_string = """
      FLOAT_T u[] = %(xAxis)s;
      FLOAT_T norm[] = %(yAxis)s;

      for (c = 0; c < %(nFeatures)d; ++c)
        ret[c] = qtc_interpolate_for_%(name)s (ret[c], u, norm, %(n)d); 
    """ % dict (
          name = name, 
          xAxis = array2c (xAxis), 
          yAxis = array2c (yAxis), 
          nFeatures = nFeatures,
          n = nSamples,
        )

    normal_to_uniform_string = """
      FLOAT_T u[] = %(xAxis)s;
      FLOAT_T norm[] = %(yAxis)s;

      for (c = 0; c < %(nFeatures)d; ++c)
        x[c] = qtc_interpolate_for_%(name)s (x[c], norm, u, %(n)d); 
    """ % dict (
          name = name, 
          xAxis = array2c (xAxis), 
          yAxis = array2c (yAxis), 
          nFeatures = nFeatures,
          n = nSamples,
        )


    lines.append ("""
    extern "C"
    FLOAT_T *%(name)s (FLOAT_T *ret, const FLOAT_T *x)
    {
      int c; 
      FLOAT_T q[%(nFeatures)d][%(nQuantiles)d] = %(qString)s; 
      FLOAT_T y[%(nQuantiles)d] = %(yString)s; 

      for (c = 0; c < %(nFeatures)d; ++c)
        ret[c] = qtc_interpolate_for_%(name)s (x[c], q[c], y, %(nQuantiles)d ); 
      
      %(to_normal_string)s

      return ret; 
    }
    """ % dict (
      name = name, 
      nQuantiles = nQuantiles, 
      nFeatures  = nFeatures,
      qString = array2c ( q.T ), #", ".join ([
        #"{%s}"%(", ".join ([str(x) for x in ql])) for ql in q.T]) ,
      yString = array2c ( y ), #", ".join ([str(x) for x in y]) 
      to_normal_string = uniform_to_normal_string if distr=='normal' else '',
      )); 

    lines.append ("""
    extern "C"
    FLOAT_T *%(name)s_inverse (FLOAT_T *ret, const FLOAT_T *input)
    {
      int c; 
      FLOAT_T x[%(nFeatures)d]; 
      FLOAT_T q[%(nFeatures)d][%(nQuantiles)d] = %(qString)s; 
      FLOAT_T y[%(nQuantiles)d] = %(yString)s; 

      for (c = 0; c < %(nFeatures)d; ++c)
        x[c] = input[c]; 

      %(to_uniform_string)s

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
      to_uniform_string = normal_to_uniform_string if distr=='normal' else '',
      )); 

    return "\n".join (lines) 





