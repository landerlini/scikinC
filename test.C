#include  <dlfcn.h>

void test (void)
{
  void *handle = dlopen ( "./Cfile.so", RTLD_LAZY );
  if (!handle)
    std::cout << "dlerror: " << dlerror() << std::endl; 

  typedef float *(*mlfunc)(float *, const float*);

  mlfunc minmax = mlfunc(dlsym (handle, "a_minmaxscaler")); 

  float ret [5];
  for (auto i: ret) std::cout << i << "\t"; std::cout << std::endl; 

  minmax ( ret, ret ); 

  for (auto i: ret) std::cout << i << "\t"; std::cout << std::endl; 

  mlfunc minmax_inverse = mlfunc(dlsym (handle, "a_minmaxscaler_inverse")); 

  minmax_inverse ( ret, ret ); 
  for (auto i: ret) std::cout << i << "\t"; std::cout << std::endl; 

  std::cout << "dlerror: " << dlerror() << std::endl; 

}
