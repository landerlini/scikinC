import sys

import numpy as np

from scikinC import BaseConverter 
from scikinC.GBDTUnrollingConverter import GBDTUnrollingConverter 
from scikinC.GBDTTraversalConverter import GBDTTraversalConverter 


class GBDTC_Converter (BaseConverter):
  """
  GradientBoostingDecision Tree converter
  """
  def __init__ (self, *args, **kwargs):
    BaseConverter.__init__ (self, *args, **kwargs)
    self.args = args
    self.kwargs = kwargs 

  def convert(self, bdt, name=None):
    if bdt.max_depth <= 5:
      return GBDTUnrollingConverter (*self.args, **self.kwargs).convert(bdt, name)

    return GBDTTraversalConverter (*self.args, **self.kwargs).convert(bdt, name)

