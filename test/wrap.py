import numpy as np 
import subprocess
from functools import partial
import os 

import pickle 
import string


class DeployedModel:
    def __init__(self, filename, compiled = 'test.so'):
        self.filename = filename
        self.compiled = compiled

        self.compile() 
        self.funcnames = self.get_funcnames() 

        for func in self.funcnames:
            setattr(self, func, partial (self.call_function, funcname=func))

    def __del__ (self):
      os.system ( "rm %s" % (self.compiled, ) )

    def compile (self):
      output = subprocess.check_output(
          ["gcc", self.filename, "-o", self.compiled, "--shared", "-fPIC", "-lm"]
          )
      if str(output, 'ASCII') not in ["", "\n"]:
        raise Exception("Compilation error %s" % str(output, 'ASCII'))


    def get_funcnames(self):
      output = subprocess.check_output(
          ["nm", '-D', self.compiled]
      )

      ret = []
      for line in str(output, 'ASCII').split('\n'):
        tokens = [a for a in line.split(' ') if len(a)] 
        if len(tokens) != 3: continue  
        addr, type_, name = tokens
        if type_ in "T":
          ret.append (name) 

      return ret 


    def call_function(self, nArgs, args, funcname):
        path = os.path.dirname(os.path.realpath(__file__))

        output = subprocess.check_output(
            [path+"/wrap.exe", self.compiled, funcname, str(nArgs)] +
            [str(x) for x in args]
        )

        return np.array([float(x) for x in output.decode('ASCII').split(" ")])


def deploy_pickle (name, obj, float_t = "float"):
  ### Randomize UID 
  s = string.ascii_letters 
  uid = [s[np.random.randint(len(s))] for _ in range(16)]
  tmpfile = name + ''.join(uid)

  with open(tmpfile+".pkl", "wb") as f:
    pickle.dump (obj, f) 

  os.system (
      "gcc test/wrap.C -o test/wrap.exe -ldl -lm "
      "-DFLOAT_T=%s" % float_t
      )

  os.system (
      "python3 -m scikinC --float_t %(floatt)s transform=%(tmpfile)s.pkl > %(tmpfile)s.C" %
      {'tmpfile': tmpfile, 'floatt': float_t} 
      ) 


  ret = DeployedModel(tmpfile+".C", compiled = './%s.so' % tmpfile) 

  os.system ("rm %(tmpfile)s.pkl %(tmpfile)s.C" % {'tmpfile': tmpfile} )

  return ret 

def deploy_keras (name, obj, float_t = "float"):
  ### Randomize UID 
  s = string.ascii_letters 
  uid = [s[np.random.randint(len(s))] for _ in range(16)]
  tmpfile = name + ''.join(uid)

  obj.save(tmpfile)

  os.system (
      "gcc test/wrap.C -o test/wrap.exe -ldl -lm "
      "-DFLOAT_T=%s" % float_t
      )

  os.system (
      "python3 -m scikinC --float_t %(floatt)s transform=%(tmpfile)s > %(tmpfile)s.C" %
      {'tmpfile': tmpfile, 'floatt': float_t} 
      ) 


  ret = DeployedModel(tmpfile+".C", compiled = './%s.so' % tmpfile) 

  os.system ("rm -r %(tmpfile)s %(tmpfile)s.C" % {'tmpfile': tmpfile} )

  return ret 


  

if __name__ == '__main__':
    import numpy as np
    d = DeployedModel("../test.C") 
    ret = d.a_minmaxscaler(5, np.ones(5))
    print (ret)
    ret = d.a_minmaxscaler_inverse(5, ret) 
    print(ret)

