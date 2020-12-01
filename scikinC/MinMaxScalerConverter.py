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
    float* %(name)s (float* ret, const float *input)
    {
      int c; 
      float input_min[] = %(inputMin)s; 
      float input_max[] = %(inputMax)s; 
      float output_min = %(outputMin)f; 
      float output_max = %(outputMax)f; 

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
    float* %(name)s_inverse (float* ret, const float *input)
    {
      int c; 
      float input_min = %(inputMin)f; 
      float input_max = %(inputMax)f; 
      float output_min[] = %(outputMin)s; 
      float output_max[] = %(outputMax)s; 

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

