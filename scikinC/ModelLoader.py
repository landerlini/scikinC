import os 
import re 
from glob import glob 
import pickle 

def _basename (filename):
  if "." not in filename: 
    return filename
  return re.findall ( "([A-Za-z0-9_\-]*)\.[A-Za-z0-9_\-\.]",
      os.path.basename(filename)) [0] 

def load_from_string ( string ):
  what = None
  if os.path.isfile (string):
    try:
      with open ( string, 'rb' ) as f:
        ## it is a pickled object 
        return ({_basename(string): pickle.load (f)},)
    except Exception as e:
      if ".pkl" in string:
        print (e) 

  ## Add here loaders for additional persistency formats 

  if os.path.isdir (string):
    if os.path.isfile (os.path.join(string, 'saved_model.pb')):
      ## it is a tensorflow model 
      from tensorflow.keras.models import load_model 
      return ({_basename(string): load_model (string, compile=False)},) 

    ret = [] 
    for fname in glob (os.path.join ( string, "*.pkl" )):
      try: 
        with open ( fname, 'rb' ) as f:
          return ({_basename(fname): pickle.load (f)},)
      except Exception as e:
        print ("Trying to load %s. Failed for %s" % 
            (fname, str(e)))

    return ret 

  if "*" in string: 
    ret = [] 
    for fname in glob (string):
      try: 
        with open ( fname, 'rb' ) as f:
          return ({_basename(fname): pickle.load (f)},)
      except Exception as e:
        print ("Trying to load %s. Failed for %s" % 
            (fname, str(e)))

    return ret 


  raise RuntimeError (
      "Resolution of string '%s' failed." % string 
      ) 
