import os 
from datetime import datetime 
class BaseConverter:
  def __init__(self):
    pass

  def header(self):
    return [
    "/***************************************************************************/\n"
    "/* File automatically generated with scikinC (github.com/landerli/scikinC) */\n"
    "/*                                                                         */\n"
    "/*                       D O   N O T   E D I T   ! ! !                     */\n"
    "/*                                                                         */\n"
    "/*  File generated on %(date)-26s                           */\n"
    "/*  by %(user)-26s                                          */\n"
    "/*  using %(conv)-36s as converter                */\n"
    "/*                                                                         */\n"
    "/***************************************************************************/\n" % 
    dict (
      user = os.environ["USER"],
      date = str(datetime.now())[:16],
      conv = self.__class__.__name__,
      )]
