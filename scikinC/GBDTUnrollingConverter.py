import sys
from scikinC import BaseConverter
import numpy as np

from scikinC._tools import array2c, retrieve_prior, sklearn_min_version



class GBDTUnrollingConverter(BaseConverter):
    """
    Converts GradientBoostingClassifiers with explicit conversion of
    each tree in C language. Resulting C takes longer to compile, but it is
    slightly faster in inference, and does not require pointer algebra.
    """

    def _singletree(self, tree, node):
        "Single-tree traversal"
        if tree.feature[node] >= 0:
            return "(inp[%d] <= %.20f ? %s : %s)" % (tree.feature[node],
                                                     tree.threshold[node],
                                                     self._singletree(tree, tree.children_left[node]),
                                                     self._singletree(tree, tree.children_right[node]))
        else:
            return str(tree.value[node][0][0])

    @staticmethod
    def _get_limits(bdt):
        mins = [None] * bdt.n_features_in_
        maxs = [None] * bdt.n_features_in_

        for treeset in bdt.estimators_:
            for tree in treeset:
                for feature in range(bdt.n_features_in_):
                    features = tree.tree_.feature
                    if feature not in features: continue
                    min_ = np.min(tree.tree_.threshold[features == feature])
                    if mins[feature] is None or min_ < mins[feature]:
                        mins[feature] = min_

                    max_ = np.max(tree.tree_.threshold[features == feature])
                    if maxs[feature] is None or max_ > maxs[feature]:
                        maxs[feature] = max_

        return mins, maxs

    def convert(self, bdt, name=None):
        n_classes = bdt.n_classes_ if bdt.n_classes_ > 2 else 1
        lines = self.header()

        if n_classes > 1:
            for iClass in range(n_classes):
                lines.append(
                    "/*  ret [ %d ]   is the probability for category:  %-15s */" %
                    (iClass, str(bdt.classes_[iClass]))
                    )

        min_, max_ = self._get_limits(bdt)

        nX = bdt.n_features_in_
        n_output = max(2, n_classes) if sklearn_min_version("1.0") else n_classes

        retvar = "FLOAT_T ret[%d]" % n_output
        invar = "FLOAT_T inp[%d]" % nX
        lines += [
            "#include <math.h>",
            "extern \"C\"",
            "FLOAT_T *%s (%s, const %s)" % (name or "bdt", retvar, invar),
            "{",
            "  const FLOAT_T init[] = %s;" % array2c(retrieve_prior(bdt)),
            "  int i; ",
            "  for (i=0; i < %d; ++i) ret[i] = init[i];" % n_output,
        ]

        for iTree, tree in enumerate(bdt.estimators_):
            lines += [" /** TREE %03d **/" % iTree]
            for iClass in range(n_classes):
                class_id = 1 if n_classes == 1 else iClass
                lines += [
                    "  ret[%d] += %f * (%s); " % (class_id, bdt.learning_rate,
                                                  self._singletree(tree[iClass].tree_, 0))
                ]


        if n_output > 1:
            lines += [
                "  short argmax = 0; ",
                "  for (int i = 0; i < %d; ++i) if (ret[i] > ret[argmax]) argmax = i; " % n_output,
                "  if (ret[argmax] > 1e10) { ",
                "    for (int i = 0; i < %d; ++i) ret[i] = (i==argmax ? 1.: 0.); " % n_output,
                "    return ret; ",
                "  }",
                "  for (short i=0; i < %d; ++i) ret[i] = exp(ret[i]);" % n_output,
                "  for (short i=0; i < %d; ++i) ret[i] = (ret[i] > 1e300?1e300:ret[i]);" % n_output,
                "  long double sum = 0;",
                "  for (short i=0; i < %d; ++i) sum += ret[i];" % n_output,
                "  for (short i=0; i < %d; ++i) ret[i] /= sum;" % n_output,
            ]
        else:
            lines += [
                "  if (ret[0] > 1e10) ret[0] = 1.;",
                "  else ret[0] = 1. / (1 + exp(-ret[0]));"
            ]

        lines += ["  return ret;", "}"]

        return "\n".join(lines)
