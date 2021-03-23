# scikinC
`scikinC` is a simple tool intended for deployment of simple Machine Learning 
algorithms as shared objects. 
We consider as a target scikit-learn and keras neural networks. 

There are many other options to deploy machine learning algorithms in C and C++ 
environments, but they usually involve either specific compilation environments 
or require complicated threading structures that may make it difficult to 
integrate the developed models into existing frameworks. 

Besides, in large distributed-computing environments it may be interestring to 
distribute new models without the need to recompile the entiere software stack.
Some libraries (e.g. TMVA or PMML or LWTNN) allow to export trained models 
into portable formats, that can then be converted at run-time in a sequence 
of function calls providing the expected results. 
While very effective, these libraries add a bit of overhead to function calls 
and requires specific compilation environment that may be uneasy to reproduce 
in the target environment. 

The *scikinC* project aims at replacing these intermediate file formats, with 
C files, and the run-time interpretation of these files with a ahead-of-time 
compilation into dynamically linked shared objects. 

Using C instead of C++ allows to deploy machine learning function as plugin
function which can be easily binded to other languages and invoked with minimal
overhead. The compiled shared object do not make use of multithreading letting
the larger code infrastructure to deal with parallelization without introducing 
overhead.

Finally, the portable C files can be included as header files in other 
programs and statically compiled for less-conventional architectures such 
as microcontroller and FPGAs. 

As in many other circumstances, distributing binaries hinder software security, 
exposing clients to more severe risks than dedicated ML format. Users should be 
aware that plugging untrusted shared objects to their program may result in 
severe security breechs. 



## CLI

Create the C file with the exported model
```
scikinC some_model.pkl > Cfile.C
```

Compile the C file for dynamic loading 
```
gcc -o Cfile.so Cfile.C -shared -fPIC -Ofast
```

Use it everywhere
```
#include  <dlfcn.h>

typedef float *(*mlfunc)(float *, const float*);

void somewhere_in_your_code (void)
{
  void *handle = dlopen ( "./Cfile.so", RTLD_LAZY );
  if (!handle)
    std::cout << "dlerror: " << dlerror() << std::endl; 

  mlfunc minmax = mlfunc(dlsym (handle, "some_model")); 
  float *inp [] = { /* your input goes here */ };
  float *out [ /*output n_features goes here*/ ];
  minmax ( out, inp ); 

  dlclose(handle); 
}
```
**Note**: the symbol to load through dlsym is the name of the pickle file, 
stripped of its extension, if any. In this case `some_model.pkl` gets compiled 
in the symbol `some_model`. 

## Implemented converters

scikit-learn:
 * GradientBoostingClassifier (binary classification and multiclass)
 * MinMaxScaler 
 * QuantileTransformer
 * Pipeline (except for pipelines including other pipelines)

keras:
 * Sequential model
 * Dense layer 
 * tanh activation function
 * relu activation function 
 * sigmoid activation function 

other:
 * DecorrTransform from TrackPar (LHCb internal) 

## Related projects
LTWNN is a more mature software package that is based on the same philosophy. The main difference is that lwtnn compiles the network architecture at compile-time and loads the weights at runtime, while with scikinC both the architecture and the weights to be loaded at runtime. As a side benefit, scikinC is a C rather than a C++ project which has less requirements. On the other hand, LTWNN supports input and output tensors of various shapes, while scikinC only supports scalar inputs and scalar outputs.
