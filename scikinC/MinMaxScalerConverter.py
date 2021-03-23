from scikinC import BaseConverter 
from ._tools import array2c 

class MinMaxScalerConverter (BaseConverter):
  def __init__ (self):
    pass 



  def convert (self, model, name = None): 
    lines = self.header() 

    nFeatures = len (model.data_min_)

    lines.append ( """
    extern "C" 
    double* %(name)s (double* ret, const double *input)
    {
      int c; 
      double input_min[] = %(inputMin)s; 
      double input_max[] = %(inputMax)s; 
      double output_min = %(outputMin)f; 
      double output_max = %(outputMax)f; 

      for (int c = 0; c < %(nFeatures)d; ++c)
        ret [c] = (input[c] - input_min[c]) / (input_max[c] - input_min[c]) 
                  * (output_max - output_min) + output_min;

      return ret;
    }
      """ % dict (
        name = name, 
        nFeatures = nFeatures,
        inputMin = array2c(model.data_min_), 
        inputMax = array2c(model.data_max_),
        outputMin = model.feature_range[0], 
        outputMax = model.feature_range[1], 
        )
      )



    lines.append ( """
    extern "C" 
    double* %(name)s_inverse (double* ret, const double *input)
    {
      int c; 
      double input_min = %(inputMin)f; 
      double input_max = %(inputMax)f; 
      double output_min[] = %(outputMin)s; 
      double output_max[] = %(outputMax)s; 

      for (int c = 0; c < %(nFeatures)d; ++c)
        ret [c] = (input[c] - input_min) / (input_max - input_min) 
                  * (output_max[c] - output_min[c]) + output_min[c];

      return ret;
    }
      """ % dict (
        name = name, 
        nFeatures = nFeatures,
        inputMin = model.feature_range[0], 
        inputMax = model.feature_range[1], 
        outputMin = array2c(model.data_min_),
        outputMax = array2c(model.data_max_),
        )
      )

    return "\n".join(lines)

