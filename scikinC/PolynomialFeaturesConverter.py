from sklearn.preprocessing import PolynomialFeatures
from textwrap import indent
import sys

from scikinC import BaseConverter
from ._tools import array2c 

class PolynomialFeaturesConverter (BaseConverter):

  def convert (self, model: PolynomialFeatures, name = None):
    lines = self.header() 

    nOutputFeatures = model.n_output_features_
    nInputFeatures = model.n_features_in_
    order = model.order
    include_bias = model.include_bias

    powers = ['']
    for output_feature, features_in_powers in enumerate(model.powers_):
        if (all([p == 0 for p in features_in_powers])):
            powers.append("ret[%d] = 1;" % output_feature)
        else:
            powers.append(
                ("ret[%d] = " % output_feature) +
                ('*'.join(["input[%d]" % i_var for i_var, pow in enumerate(features_in_powers) for _ in range(pow)])) +
                ";"
            )


    lines.append ( """
    extern "C"
    FLOAT_T* %(name)s (FLOAT_T* ret, const FLOAT_T *input)
    {
      %(powers)s;
      return ret;
    }
      """ % dict (
        name=name,
        powers=indent('\n'.join(powers), ' '*7)
        )
      )

    return '\n'.join(lines)
