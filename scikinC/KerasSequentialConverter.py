from scikinC import BaseConverter 
from scikinC import layers 
from ._tools import array2c 

class KerasSequentialConverter (BaseConverter):
  """
  Converter for Keras Sequential Model 
  """

  def convert (self, model, name = None): 
    if name is None:
      name = "keras_model"

    lines = self.header() 
    lines += ["""
    #include <math.h>

    """] 

    converters = [] 

    for layer in model.layers:
      class_ = layer.__class__.__name__
      if not hasattr (layers, class_):
        raise NotImplementedError (
            "No implementation found for layer %s (%s)" % (layer.name, class_)
            )

      converters.append (getattr(layers, class_) (name, layer))


    for converter in converters:
      lines += [ converter.definition() ] 

    nX = model.layers[0].kernel.shape[0] 
    nY = model.layers[-1].output_shape[-1] 
    nMax = max (*[l.output_shape[1] for l in model.layers]+[nX]) 

    lines.append ("""
    extern "C"
    FLOAT_T* %(name)s (FLOAT_T* ret, const FLOAT_T *input)
    {
      int i;
      FLOAT_T ibuf [%(nMax)d]; 
      FLOAT_T obuf [%(nMax)d]; 

      FLOAT_T *b1 = ibuf;
      FLOAT_T *b2 = obuf;
      FLOAT_T *b3 = 0x0; 

      for (i=0; i<%(nX)d; ++i)
        b1[i] = input[i]; 

      """ % dict(
        name = name,
        nMax = nMax,
        nX = nX,
        ))

    for converter in converters:
      lines.append ("""
      %(call)s
      b3=b2; b2=b1; b1=b3; b3=0x0; 
      """ % dict(call=converter.call("b2", "b1")))

    lines.append ("""
      for (i=0; i<%(nY)d; ++i)
        ret [i] = b1[i]; 
        
      return ret;
    }
    """ % dict(nY=nY))


    return "\n".join(lines) 

