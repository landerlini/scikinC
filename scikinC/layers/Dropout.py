from scikinC.layers.BaseLayerConverter import BaseLayerConverter


class Dropout(BaseLayerConverter):
    """
    Dropout Layer converter
    """

    def definition(self):
        """Return the definition of the layer function"""
        ret = []

        ret += ["""
        extern "C"
        FLOAT_T* %(layername)s (FLOAT_T* ret, const FLOAT_T* input)
        {
            int i;
            for (i = 0; i < %(nX)d; ++i)
              ret[i] = input[i];
              
            return ret; 
        }
        """ % dict(
            layername=self.name,
            nX=self.layer.output_shape[1] if hasattr(self.layer, 'output_shape') else self.layer.output.shape[1],
        )]

        return "\n".join(ret)

    def call(self, obuffer, ibuffer):
        """Return the call to the layer function"""
        return "%(layername)s ( %(obuffer)s, %(ibuffer)s);" % dict(
            layername=self.name, obuffer=obuffer, ibuffer=ibuffer)
