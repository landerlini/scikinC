import sys
from scikinC import BaseConverter
import numpy as np


class SmartFloat2Int:
  def __init__(self, low, high):
    self.low = low
    self.high = high

  def __call__(self, val):
    low, high = self.low, self.high
    if val > high: return 0x4000
    if val < low : return -0x4000
    return int((val - low)/(high - low) * 0x7FFF - 0x3FFF) 

  def toC (self, x):
    return (
        "%(x)s < %(low).20f ? -0x4000 : "
        "( %(x)s > %(high).20f ? 0x4000 : "
        "int ((%(x)s - %(low).20f)/(%(high).20f - %(low).20f) * 0x7FFF - 0x3FFF))"
        ) % dict(x=x, low=self.low, high=self.high)


class GBDTC_Converter (BaseConverter):
  """
  GradientBoostingDecision Tree converter
  """

  def _singletree(self, tree, node):
    "Single-tree traversal"
    if tree.feature[node] >= 0:
      return "(x[%d] <= %d ? %s : %s)" % (tree.feature[node],
          self.converter[tree.feature[node]](tree.threshold[node]),
          self._singletree(tree, tree.children_left[node]),
          self._singletree(tree, tree.children_right[node]))
    else:
      return str(tree.value[node][0][0])


  @ staticmethod
  def _get_limits(bdt):
    mins=[None] * bdt.n_features_
    maxs=[None] * bdt.n_features_

    for treeset in bdt.estimators_:
      for tree in treeset:
        for feature in range(bdt.n_features_):
          features=tree.tree_.feature
          if feature not in features: continue
          min_=np.min(tree.tree_.threshold[features == feature])
          if mins[feature] is None or min_ < mins[feature]:
            mins[feature]=min_

          max_=np.max(tree.tree_.threshold[features == feature])
          if maxs[feature] is None or max_ > maxs[feature]:
            maxs[feature]=max_

    return mins, maxs




  def convert(self, bdt, name=None):
    n_classes=bdt.n_classes_ if bdt.n_classes_ > 2 else 1
    lines=self.header()

    if n_classes > 1:
      for iClass in range(n_classes):
        lines.append("/*  ret [ %d ]   is the probability for category:  %-15s */" %
            (iClass,  str(bdt.classes_[iClass])))

    min_, max_=self._get_limits(bdt)
    self.converter = [SmartFloat2Int(l, h) for l, h in zip(min_, max_)]

    nX = bdt.n_features_ 

    retvar="FLOAT_T ret[%d]" % n_classes
    invar="FLOAT_T inp[%d]" % nX
    lines += [
        "#include <math.h>",
        "extern \"C\"",
        "FLOAT_T *%s (%s, const %s)" % (name or "bdt", retvar, invar),
        "{",
        "  int i; ",
        "  int x[%d];" % nX, 
        "  for (short i=0; i < %d; ++i) ret[i] = 0.f;" % n_classes,
      ]

    # preprocessing
    for feature in range(nX):
      lines.append ( "  x[%d] = %s; " % (feature, self.converter[feature].toC('inp[%d]' % feature)) )


    for iTree, tree in enumerate(bdt.estimators_):
      lines += [" /** TREE %03d **/" % iTree]
      for iClass in range(n_classes):
        lines += [
           "  ret[%d] += %f * (%s); " % (iClass, bdt.learning_rate,
                               self._singletree(tree[iClass].tree_, 0))
         ]




    if n_classes > 1:
      lines += [
          "  short argmax = 0; ",
          "  for (int i = 0; i < %d; ++i) if (ret[i] > ret[argmax]) argmax = i; " % n_classes,
          "  if (ret[argmax] > 1e10) { ",
          "    for (int i = 0; i < %d; ++i) ret[i] = (i==argmax ? 1.: 0.); " % n_classes,
          "    return ret; ",
          "  }",
          "  for (short i=0; i < %d; ++i) ret[i] = exp(ret[i]);" % n_classes,
          "  for (short i=0; i < %d; ++i) ret[i] = (ret[i] > 1e300?1e300:ret[i]);" % n_classes,
          "  long double sum = 0;",
          "  for (short i=0; i < %d; ++i) sum += ret[i];" % n_classes,
          "  for (short i=0; i < %d; ++i) ret[i] /= sum;" % n_classes,
        ]
    else:
      lines += [
        "  if (ret[0] > 1e10) ret[0] = 1.;",
        "  ret[0] = 1. / (1 + exp(-ret[0]));"
      ]


    lines += ["  return ret;", "}"]

    return "\n".join(lines)
