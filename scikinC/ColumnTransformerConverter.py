import numpy as np

from sklearn.preprocessing import FunctionTransformer

import scikinC
from scikinC import BaseConverter
from ._tools import array2c

import sys


class ColumnTransformerConverter (BaseConverter):
  def convert(self, model, name=None):
    lines = self.header()

    index_mapping = []
    keys = []
    transformers = []
    for key, transformer, columns in model.transformers_:
      if transformer == 'drop' or len(columns) == 0: 
        continue 

      if not all([isinstance(c, int) or int(c) == c for c in columns]):

        raise NotImplementedError ("Columns can only be indexed with integers, got", 
            [type(c) for c in columns])

      index_mapping += columns

      if key is None:
        key = "Preprocessor"
      if key in keys:
        key.append (str(1+len(keys)))

      if isinstance(transformer, (FunctionTransformer,)):
        if transformer.func is None and transformer.inverse_func is None:
          transformer = 'passthrough'
        else:
          transformer.n_features_in_ = len(columns)

      transformers.append (('colcnv_%s_%s' % (name, key), transformer, columns))


    if len([t for _, t, _ in transformers if t != 'passthrough']):
      lines.append( 
          scikinC.convert({k: t for k,t,_ in transformers if t != 'passthrough'})
          )
    
    mapping = {k: c for k,_,c in transformers}

    nFeatures = 1+max(index_mapping)

    lines.append("""
    extern "C"
    FLOAT_T* %(name)s (FLOAT_T* ret, const FLOAT_T *input)
    {
      int c;
      FLOAT_T bufin[%(nFeatures)d], bufout[%(nFeatures)s];

    """ % dict(
        name=name,
        nFeatures=nFeatures,
        )
      )

    for key, transformer, columns in transformers:
      lines.append("// Transforming %s columns" % key)
      if transformer == 'passthrough':
        for column in columns:
          lines.append("""
          ret [%(output)d] = input[%(column)d];
          """%dict(output=index_mapping.index(column), column=column))
      else: 
        for iCol, column in enumerate(columns):
          lines.append("""         bufin [%(iCol)d] = input[%(column)d];"""%
              dict(iCol=iCol, column=column))
        lines.append ("""          %(name)s (bufout, bufin);""" 
            % dict(name=key))
        for iCol, column in enumerate(columns):
          lines.append("""         ret[%(index_out)d] = bufout[%(iCol)d];"""% 
              dict(index_out=index_mapping.index(column), iCol=iCol))

    lines.append ("""
      return ret;
    }
    """)

    ## Check for not-invertible models
    ##  Any dropped columns? 
    if any([t == 'drop' for _, t, _ in model.transformers_]):
      return "\n".join(lines)

    ##  Any columns appearing twice? 
    if any([index_mapping.count(c)>1 for c in index_mapping]):
      return "\n".join(lines)

    ##  Any transformer not implementing an inverse transform?
    if not all([t == 'passthrough' or hasattr(t, 'inverse_transform')] for _,t,_ in transformers):
      return "\n".join(lines)

    index_mapping = [index_mapping.index(c) for c in range(len(index_mapping))]

    lines.append("""
    extern "C"
    FLOAT_T* %(name)s_inverse (FLOAT_T* ret, const FLOAT_T *input)
    {
      int c;
      FLOAT_T bufin[%(nFeatures)d], bufout[%(nFeatures)s];

    """ % dict(
        name=name,
        nFeatures=nFeatures,
        )
      )

    for key, transformer, columns in transformers:
      lines.append("// Transforming %s columns" % key)
      if transformer == 'passthrough':
        for column in columns:
          lines.append("""
          ret [%(output)d] = input[%(column)d];
          """%dict(output=index_mapping.index(column), column=column))
      else: 
        for iCol, column in enumerate(columns):
          lines.append("""          bufin [%(iCol)d] = input[%(column)d];"""%
              dict(iCol=iCol, column=column))
        lines.append  ("""          %(name)s_inverse (bufout, bufin);"""%
            dict(name=key))
        for iCol, column in enumerate(columns):
          lines.append("""          ret[%(index_out)d] = bufout[%(iCol)d]; """ %
              dict(index_out=index_mapping.index(column), iCol=iCol))

    lines.append ("""
      return ret;
    }
    """)

    return "\n".join(lines)

