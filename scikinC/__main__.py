import argparse 
from scikinC import convert 

def main():
  parser = argparse.ArgumentParser ( "scikinC" ) 
  parser.add_argument ( "files", nargs = "+", help = "Files with scikit-learn objects to convert in C function" )

  args = parser.parse_args() 

  print (convert (args.files))

if __name__ == '__main__':
    main() 
