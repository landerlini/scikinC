import argparse 
import scikinC 


def main():
  parser = argparse.ArgumentParser ( "scikinC" ) 
  parser.add_argument ( "files", nargs = "*", help = "Files with scikit-learn objects to convert in C function" )
  parser.add_argument ( "--version", "-v", action='store_true', help = "Display the version and exits" )
  parser.add_argument ( "--copyright", "-c", type=str, default=None, help = "Copyright indication in generated files" )
  parser.add_argument ( "--float_t", "-f", type=str, default="float", help = "C type for floating point variables" )

  args = parser.parse_args() 

  if args.version:
    print (scikinC.version)
    exit(0)

  if len(args.files) == 0:
    print ("Missing input file")
    exit(0)

  cfg_args = ['copyright', 'float_t']
  config = {a:getattr(args,a) for a in cfg_args if hasattr(args,a) and getattr(args,a) is not None}

  print (scikinC.convert (args.files, **config))

if __name__ == '__main__':
    main() 
