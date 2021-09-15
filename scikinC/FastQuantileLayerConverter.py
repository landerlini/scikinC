import numpy as np 
from scikinC import BaseConverter 
from scipy import stats
from ._tools import array2c 



class FastQuantileLayerConverter (BaseConverter):
  def convert (self, model, name = None): 
    lines = self.header() 

    n_vars = len (model.fwdTransforms_)
    xmin = [t.x_min for t in model.fwdTransforms_]
    xmax = [t.x_max for t in model.fwdTransforms_]
    yvalues = [t.tf_y_values.numpy() for t in model.fwdTransforms_]

    ymin = [t.x_min for t in model.bwdTransforms_]
    ymax = [t.x_max for t in model.bwdTransforms_]
    xvalues = [t.tf_y_values.numpy() for t in model.bwdTransforms_]


    lines . append ( """
    #include <math.h>
    #include <stdlib.h>
    #include <stdio.h>

    extern "C"
    FLOAT_T ql_interpolate_for_%(name)s ( FLOAT_T x, FLOAT_T xmin, FLOAT_T xmax, const FLOAT_T *ys, int N )
    {
      if (xmax <= xmin) return 0./0.; 
      const FLOAT_T range = xmax - xmin;
      const FLOAT_T id = (x / range - xmin / range) * (N - 1);
      const FLOAT_T x_0f = floor(id);
      const int x_0 = int((x_0f < 0) ? 0 : (x_0f > (N-2)) ? (N-2) : x_0f);
      const int x_1 = x_0 + 1; 
      const FLOAT_T y_0 = ys[x_0];
      const FLOAT_T y_1 = ys[x_1];

      const FLOAT_T dx = (id < x_0f) ? 0. : (id > x_0f + 1) ? 1. : (id - x_0f);

      return y_0 + dx * (y_1 - y_0); 
    }


    extern "C"
    FLOAT_T *%(name)s (FLOAT_T *ret, const FLOAT_T *x)
    {
      int c; 
      const FLOAT_T xmin[%(n_vars)d] = %(xmin)s; 
      const FLOAT_T xmax[%(n_vars)d] = %(xmax)s; 
      const FLOAT_T y[%(n_vars)d][%(n_y)d] = %(yvalues)s; 

      for (c = 0; c < %(n_vars)d; ++c)
        ret[c] = ql_interpolate_for_%(name)s (x[c], xmin[c], xmax[c], y[c], %(n_y)d ); 

      return ret; 
    }


    extern "C"
    FLOAT_T *%(name)s_inverse (FLOAT_T *ret, const FLOAT_T *y)
    {
      int c; 
      const FLOAT_T ymin[%(n_vars)d] = %(ymin)s; 
      const FLOAT_T ymax[%(n_vars)d] = %(ymax)s; 
      const FLOAT_T x[%(n_vars)d][%(n_x)d] = %(xvalues)s; 

      for (c = 0; c < %(n_vars)d; ++c)
        ret[c] = ql_interpolate_for_%(name)s (y[c], ymin[c], ymax[c], x[c], %(n_x)d ); 

      return ret; 
    }
    """ % dict(
      name=name,
      n_vars=n_vars,
      n_y=len(yvalues[0]),
      xmin=array2c(xmin),
      xmax=array2c(xmax),
      yvalues=array2c(yvalues),

      n_x=len(xvalues[0]),
      ymin=array2c(ymin),
      ymax=array2c(ymax),
      xvalues=array2c(xvalues),
      )); 

    return "\n".join (lines) 

