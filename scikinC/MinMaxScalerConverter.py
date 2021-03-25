from scikinC import BaseConverter 
from ._tools import array2c 

class MinMaxScalerConverter (BaseConverter):

  def convert (self, model, name = None): 
    lines = self.header() 

    nFeatures = len (model.data_min_)

    lines.append ( """
    extern "C" 
    FLOAT_T* %(name)s (FLOAT_T* ret, const FLOAT_T *input)
    {
      int c; 
      FLOAT_T input_min[] = %(inputMin)s; 
      FLOAT_T input_max[] = %(inputMax)s; 
      FLOAT_T output_min = %(outputMin)f; 
      FLOAT_T output_max = %(outputMax)f; 

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
    FLOAT_T* %(name)s_inverse (FLOAT_T* ret, const FLOAT_T *input)
    {
      int c; 
      FLOAT_T input_min = %(inputMin)f; 
      FLOAT_T input_max = %(inputMax)f; 
      FLOAT_T output_min[] = %(outputMin)s; 
      FLOAT_T output_max[] = %(outputMax)s; 

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

