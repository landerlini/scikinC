/*******************************************************************************
 * wrap.C - testing utility for compiled models 
 * 
 * Copyright - INFN Firenze (2020)
 * Refer to the LICENCE at
 * https://github.com/landerlini/scikinC/blob/main/LICENSE
 ******************************************************************************/
#include  <dlfcn.h>
#include  <stdlib.h>
#include  <stdio.h>

#define ERROR_SO_NOT_FOUND     1001
#define ERROR_FUNC_NOT_FOUND   1002

#ifndef FLOAT_T
#define FLOAT_T float
#endif 

typedef FLOAT_T *(*mlfunc)(FLOAT_T *, const FLOAT_T*);

int main (int argc, char *argv[])
{
  int i; 
  const char* libname = argv[1];
  const char* funcname = argv[2];
  const int nY = atoi (argv[3]); 
  const size_t in0 = 4;
  const size_t nX = argc-in0; 
  FLOAT_T iBuf[1064], oBuf[1064];

  void *handle = dlopen (libname, RTLD_LAZY);
  if (!handle)
  {
    printf( "dlerror: %s", dlerror());
    exit(ERROR_FUNC_NOT_FOUND); 
  }

  mlfunc func = mlfunc (dlsym (handle, funcname)); 

  for (i = 0; i < nX; ++i)
    iBuf[i] = atof(argv[in0 + i]);

  func (oBuf, iBuf); 

  for (i = 0; i < nY-1; ++i)
    printf ("%f ", oBuf[i]); 
  printf ("%f\n", oBuf[i]); 

  return 0; 
}
