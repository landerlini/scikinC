from scikinC import BaseConverter 

class KerasConverter (BaseConverter):
  def __init__ (self):
    pass 



  def convert (self, model, name = None): 
    lines = self.header() 
    lines += ["""
    #include <math.h>

    """] 

    for iLayer, layer in enumerate(model.layers):
      kernel, bias = layer.get_weights()
      lines.append ("inline float activation_%d (float x) " % iLayer);
      activation =  layer.get_config()['activation'] 
      if activation == 'sigmoid':
        lines.append ("{ return 1./(1 + exp(-x)); }")
      elif activation == 'tanh':
        lines.append ("{ return tanh(x);}")
      elif activation == 'relu':
        lines.append ("{ return x > 0 ? x : 0;}")
      elif activation == 'linear':
        lines.append ("{ return x;}")
      else:
        raise KeyError ("Unexpected activation %s"%activation)
      

      
    lines.append ("""
    extern "C" 
    float* %s (float* ret, const float *input)
    {
    """)

    nX = model.layers[0].kernel.shape[0] 
    nY = model.layers[-1].kernel.shape[-1] 
    nMax = max (*[l.kernel.shape[1] for l in model.layers]+[nX]) 

    for iLayer, layer in enumerate(model.layers):
      lines.append ("  // Declare the arrays in the stack")
      kernel, bias = layer.get_weights()
       
      lines.append ("  // Bias shape: " + str(bias.shape))
      kernel_values = "{%s}"%(',\n   '.join(["{%s}"%(','.join(["%18.13f"%x for x in row])) for row in kernel]))
      bias_values   = "{%s}"% ( ",".join(["%18.13f"%x for x in bias]))
      lines.append ("  const float kernel_%d[%d][%d] = \n  %s;" % (iLayer, kernel.shape[0], kernel.shape[1],kernel_values))
      lines.append ("  const float bias_%d[%d] = %s;" % (iLayer, bias.shape[0], bias_values))
      
    lines.append ("  float buffer_in[%d];" % nMax)
    lines.append ("  float buffer_out[%d];" % nMax)

    lines.append ("  unsigned int i,j,c; ")

    lines.append ("\n\n\n")
    lines.append ("  // Load the input in the buffer")
    lines.append ("  for (c = 0; c < 64; ++c) \n  buffer_in[c] = input[c];")

    for iLayer, layer in enumerate(model.layers):
      kernel, bias = layer.get_weights()

      lines.append ( "  // Processing layer %i " % iLayer )
      lines.append ( """
      for (c = 0; c < {n_out}; ++c ) 
        buffer_out[c] = bias_{iLayer}[c];
        
      for (c = 0; c < {n_out}; ++c )
        for (i = 0; i < {n_in}; ++i)
          buffer_out[c] += buffer_in[i] * kernel_{iLayer}[i][c];
      
      // Prepares for next layer 
      for (c = 0; c < {n_out}; ++c )
        buffer_in[c] = activation_{iLayer}(buffer_out[c]);
        
      """.format (
          n_in = kernel.shape[0],
          n_out = kernel.shape[1],
          iLayer = iLayer,
      ))
      
    last_kernel, last_bias = model.layers[-1].get_weights()
    lines.append ("""
      i = 0;
      for (c = 0; c < {n_out}; ++c)
        ret[c] = buffer_in[c]; 
      
      return ret;
    """.format(n_out = nY))

    lines.append ("}")
    return "\n".join(lines) 

