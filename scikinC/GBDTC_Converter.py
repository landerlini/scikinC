from scikinC import BaseConverter 
class GBDTC_Converter (BaseConverter):
  def __init__ (self):
    pass 

  @staticmethod 
  def _singletree (tree, node):
    "Single-tree traversal"
    if tree.feature [node] >= 0:
      return "(x[%d] <= %f ? %s : %s)" % (tree.feature[node], 
          tree.threshold[node],
          GBDTC_Converter._singletree ( tree, tree.children_left[node] ), 
          GBDTC_Converter._singletree ( tree, tree.children_right[node] ) )  
    else:
      return str(tree.value [node][0][0]) 


  def convert (self, bdt, name = None): 
    n_classes = bdt.n_classes_ if bdt.n_classes_ > 2 else 1 
    lines = self.header() 

    if n_classes > 1: 
      for iClass in range(n_classes):
        lines.append ( "/*  ret [ %d ]   is the probability for category:  %-15s */" %
            ( iClass,  str(bdt.classes_[iClass]) ) )
    
    retvar = "double ret[%d]" % n_classes 
    invar  = "double x[%d]" % bdt.n_features_ 
    lines += [
        "#include <math.h>",
        "extern \"C\"",
        "double *%s (float *ret, const float *in)" % (name or "bdt", retvar, invar), 
        "{", 
        "  for (short i=0; i < %d; ++i) ret[i] = 0.f;" % n_classes, 
        "  double %s;" % (", ".join("y%02d" % d for d in range(n_classes)) )
      ]

    for iTree, tree in enumerate(bdt.estimators_):
      lines += [" /** TREE %03d **/" % iTree ]
      for iClass in range(n_classes):
        lines += [
           "  y%02d = %s; " % ( iClass, GBDTC_Converter._singletree(tree[iClass].tree_,0) ) , 
         ]
      for iClass in range(n_classes):
        lines += [
           "  ret[%d] += %f * y%02d; " % ( iClass, bdt.learning_rate, iClass) 
         ]

    lines += [
        "  short argmax = 0; ", 
        "  for (int i = 0; i < %d; ++i) if (ret[i] > ret[argmax]) argmax = i; " % n_classes, 
        "  if (ret[argmax] > 1e10) { " ,
        "    for (int i = 0; i < %d; ++i) ret[i] = (i==argmax ? 1.: 0.); " % n_classes, 
        "    return ret; ",
        "  }", 
        "  for (short i=0; i < %d; ++i) ret[i] = exp(ret[i]);" % n_classes, 
        "  for (short i=0; i < %d; ++i) ret[i] = (ret[i] > 1e300?1e300:ret[i]);" % n_classes, 
        "  long double sum = 0;", 
        "  for (short i=0; i < %d; ++i) sum += ret[i];" % n_classes, 
        "  for (short i=0; i < %d; ++i) ret[i] /= sum;" % n_classes, 
      ] 


    lines += ["  return ret;", "}"]

    return "\n".join ( lines ) 
