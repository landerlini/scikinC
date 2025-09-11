from scikinC.layers.BaseLayerConverter import BaseLayerConverter
from scikinC._tools import array2c 

class LeakyReLU (BaseLayerConverter):
  """
  Leaky ReLU converter
  """

  def definition(self):
    """Return the definition of the layer function"""
    ret = []

    nX = self.layer.output_shape[1] if hasattr(self.layer, 'output_shape') else self.layer.output.shape[1]

    ret += ["""
        extern "C"
        FLOAT_T* %(layername)s (FLOAT_T* ret, const FLOAT_T* input)
        {
            int i; 
            const FLOAT_T alpha = %(alpha).10f;

            for (i=0; i < %(nX)d; ++i)
                ret[i] = input[i] > 0 ? input[i] : alpha * input[i];

            return ret; 
        }
        """ % dict(
          layername = self.name, 
          nX = nX,
          alpha = self.layer.alpha if hasattr(self.layer, 'alpha') else self.layer.negative_slope
        )]

    return "\n".join(ret)

  def call(self, obuffer, ibuffer):
    """Return the call to the layer function""" 
    return "%(layername)s ( %(obuffer)s, %(ibuffer)s);" % dict (
        layername=self.name, obuffer=obuffer, ibuffer=ibuffer )


