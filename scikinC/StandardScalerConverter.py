from scikinC import InvertibleConverter 
from ._tools import array2c 

class StandardScalerConverter (InvertibleConverter):
  def convert (self, model, name = None): 
    lines = self.header() 

    nFeatures = len (model.mean_)

    lines.append ( """
    extern "C" 
    FLOAT_T* %(name)s (FLOAT_T* ret, const FLOAT_T *input)
    {
      int c; 
      FLOAT_T mean [] = %(mean)s; 
      FLOAT_T scale[] = %(scale)s; 

      for (c = 0; c < %(nFeatures)d; ++c)
        ret [c] = (input[c] - mean[c]) / scale[c];

      return ret;
    }
      """ % dict (
        name = name, 
        nFeatures = nFeatures,
        mean = array2c (model.mean_ if model.mean_ is not None else np.zeros(nFeatures)),
        scale = array2c (model.scale_ if model.scale_ is not None else np.ones(nFeatures)),
        )
      )


    lines.append ( """
    extern "C" 
    FLOAT_T* %(name)s_inverse (FLOAT_T* ret, const FLOAT_T *input)
    {
      int c; 
      FLOAT_T mean [] = %(mean)s; 
      FLOAT_T scale[] = %(scale)s; 

      for (c = 0; c < %(nFeatures)d; ++c)
        ret [c] = (input[c] * scale[c]) + mean[c]; 

      return ret;
    }
    """ % dict (
      name = name, 
      nFeatures = nFeatures,
      mean = array2c (model.mean_ if model.mean_ is not None else np.zeros(nFeatures)),
      scale = array2c (model.scale_ if model.scale_ is not None else np.ones(nFeatures)),
      )
    )

    return "\n".join(lines)

