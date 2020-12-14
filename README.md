# scikinC
Set of tools to translate scikit learn and keras neural network in plain C. 

[![Write in C parody](https://img.youtube.com/vi/1S1fISh-pag/0.jpg)](https://www.youtube.com/watch?v=1S1fISh-pag)


## CLI

Create the C file with the exported model
```
python3 -m scikinC some_model.pkl > Cfile.C
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
LTWNN is a more mature software package that is based on the same philosophy. The main difference is that lwtnn compiles the network architecture at compile-time and loads the weights at runtime, while with scikinC both the architecture and the weights to be loaded at runtime. As a side benefit, scikinC is a C rather than a C++ project which has less requirements. On the other hand, LTWNN supports input and output tensors of various shapes, while scikinC only supports scalar inputs.
