import os 
from datetime import datetime 
class BaseConverter:
  """
  BaseConverter providing header and float datatype to all converters
  """
  INVERTIBLE = False

  def __init__(self, 
      float_t='float', 
      copyright=None
      ):
    if copyright is None:
      copyright = os.environ["USER"] if "USER" in os.environ.keys() else 'scikinC'



    self.float_t = float_t
    try: 
      self.copyright = copyright or os.environ["USER"] 
    except KeyError:
      self.copyright = "scikinC"


  def header(self):
    "Return the header for the generated C file"
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


class InvertibleConverter (BaseConverter):
  INVERTIBLE = True


