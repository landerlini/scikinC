from scikinC.layers.BaseLayerConverter import BaseLayerConverter
from scikinC._tools import array2c

class Softmax (BaseLayerConverter):
    """
    Softmax activation function
    """

    def definition(self):
        """Return the definition of the layer function"""
        ret = []

        nX = self.layer.output_shape[1]

        ret += ["""
        extern "C"
        FLOAT_T* %(layername)s (FLOAT_T* ret, const FLOAT_T* input)
        {
            int i; 

            double denom = 0;
            
            for (i=0; i < %(nX)d; ++i)
                denom += exp((double)input[i]);
            
            for (i=0; i < %(nX)d; ++i)
                ret[i] = (FLOAT_T) exp((double)input[i] - log(denom));

            return ret; 
        }
        """ % dict(
            layername=self.name,
            nX=nX
        )]

        return "\n".join(ret)

    def call(self, obuffer, ibuffer):
        """Return the call to the layer function"""
        return "%(layername)s ( %(obuffer)s, %(ibuffer)s);" % dict (
            layername=self.name, obuffer=obuffer, ibuffer=ibuffer )

