from scikinC.layers.BaseLayerConverter import BaseLayerConverter
from scikinC._tools import array2c 

class Dense (BaseLayerConverter):
  """
  Dense Layer converter
  """

  def definition(self):
    """Return the definition of the layer function"""
    ret = []

    nX = self.layer.kernel.shape[0] 
    nY = self.layer.kernel.shape[-1] 

    kernel, bias = self.layer.get_weights()

    ret += ["""
        extern "C"
        FLOAT_T* %(layername)s (FLOAT_T* ret, const FLOAT_T* input)
        {
            int i, j; 
            const FLOAT_T kernel[%(nX)d][%(nY)d] = %(kernel_values)s;
            const FLOAT_T bias[%(nY)d] = %(bias_values)s;

            for (i=0; i < %(nY)d; ++i)
            {
              ret[i] = bias[i]; 
              for (j=0; j<%(nX)d; ++j)
                ret[i] += input[j] * kernel[j][i];
              
              ret[i] = %(activate)s;
            }

            return ret; 
        }
        """ % dict(
          layername = self.name, 
          nX = nX,
          nY = nY,
          kernel_values = array2c (kernel),
          bias_values = array2c (bias),
          activate = self.activate('ret[i]'),
        )]


    return "\n".join(ret)

  def call(self, obuffer, ibuffer):
    """Return the call to the layer function""" 
    return "%(layername)s ( %(obuffer)s, %(ibuffer)s);" % dict (
        layername=self.name, obuffer=obuffer, ibuffer=ibuffer )
