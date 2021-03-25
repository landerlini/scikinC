import os 
from datetime import datetime 
class BaseConverter:
  def __init__(self, 
      float_t='float', 
      copyright=os.environ["USER"], 
      ):

    self.float_t = float_t
    self.copyright = copyright

  def header(self):
    return [
    "/***************************************************************************/\n"
    "/* File automatically generated with scikinC (github.com/landerli/scikinC) */\n"
    "/*                                                                         */\n"
    "/*                       D O   N O T   E D I T   ! ! !                     */\n"
    "/*                                                                         */\n"
    "/*  File generated on %(date)-26s                           */\n"
    "/*  by %(cprt)-46s                     */\n"
    "/*  using %(conv)-46s as converter      */\n"
    "/*                                                                         */\n"
    "/***************************************************************************/\n" 
    "#define FLOAT_T %(floatt)s" % 
    dict (
      cprt = self.copyright,
      date = str(datetime.now())[:16],
      conv = self.__class__.__name__,
      floatt = self.float_t, 
      ),
    ]
