import argparse 
parser = argparse.ArgumentParser ( "scikinC" ) 
parser.add_argument ( "files", nargs = "+", help = "Files with scikit-learn objects to convert in C function" )

args = parser.parse_args() 

from scikinC import convert 
print (convert (args.files))

