from scikinC.layers.BaseLayerConverter import BaseLayerConverter
from scikinC._tools import array2c 

class PReLU (BaseLayerConverter):
  """
  Programmable ReLU converter
  """

  def definition(self):
    """Return the definition of the layer function"""
    ret = []

    alpha, = self.layer.get_weights()
    nX = alpha.shape[0] 

    ret += ["""
        extern "C"
        FLOAT_T* %(layername)s (FLOAT_T* ret, const FLOAT_T* input)
        {
            int i; 
            const FLOAT_T alpha[%(nX)d] = %(alpha)s;

            for (i=0; i < %(nX)d; ++i)
                ret[i] = input[i] > 0 ? input[i] : alpha[i] * input[i];

            return ret; 
        }
        """ % dict(
          layername = self.name, 
          nX = nX,
          alpha = array2c (alpha),
        )]

    return "\n".join(ret)

  def call(self, obuffer, ibuffer):
    """Return the call to the layer function""" 
    return "%(layername)s ( %(obuffer)s, %(ibuffer)s);" % dict (
        layername=self.name, obuffer=obuffer, ibuffer=ibuffer )

