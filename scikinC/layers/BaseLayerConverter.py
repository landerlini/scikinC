class BaseLayerConverter:
  def __init__ (self, netname, layer):
    self.netname = netname
    self.layer = layer

  @property
  def name (self):
    return "%s_%s" % (self.netname, self.layer.name) 

  def activate (self, x):
    activation = self.layer.activation
    if not isinstance(activation, str):
      activation = activation.__name__

    if activation == 'sigmoid':
      return "%(x)s = 1. / (1+exp(-%(x)s));" % {'x':x} 
    elif activation == 'tanh':
      return "%(x)s = tanh(%(x)s);" % {'x':x} 
    elif activation == 'relu':
      return "%(x)s = %(x)s > 0. ? %(x)s : 0.;" % {'x':x} 
    elif activation == 'linear':
      return ""
    else:
      raise KeyError ("Unexpected activation %s"%activation)
