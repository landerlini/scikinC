# scikinC

[![](https://img.shields.io/pypi/pyversions/scikinC)](https://pypi.python.org/pypi/scikinC/)
[![](https://img.shields.io/pypi/v/scikinC)](https://pypi.python.org/pypi/scikinC/)
[![](https://img.shields.io/pypi/status/scikinC)](https://pypi.python.org/pypi/scikinC/)
[![](https://img.shields.io/pypi/dm/scikinC)](https://pypi.python.org/pypi/scikinC/)
[![](https://img.shields.io/github/issues/landerlini/scikinC)](https://github.com/landerlini/scikinC/issues)
[![](https://img.shields.io/github/issues-pr/landerlini/scikinC)](https://github.com/landerlini/scikinC/pulls)
<!--
[![](https://badgen.net/github/forks/landerlini/scikinC)](https://github.com/landerlini/scikinC/network/members)
[![](https://img.shields.io/github/stars/landerlini/scikinC)](https://github.com/landerlini/scikinC/stargazers/)
-->

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
severe security breachs. 

## Logic
`scikinC` is a transpiler for scikit-learn and keras models generating
C files with `extern "C"` functions sharing the same signature:
```
FLOAT_T* <function_name> (FLOAT_T* output, const FLOAT_T* input);
```
Everything which is not either the input or the output is hardcoded in 
the C function, including:
 * the shape of the input and output tensors;
 * the structure of the ML method (number of trees in a forest o number of
   layers in a DNN);
 * the weights of the ML method.

The generated C function is inteded for immediate compilation with `gcc`, 
but most C/C++ compiler should be supported. 

Once compiled, the binary file contains everything that is needed to 
evaluate the ML function and with no external dependency beyond standard 
C libraries.

`scikinC` is designed to be as modular as possible in order to make it 
easy to extend it by adding converters for additional scikit-learn
models and keras layers.

## Command Line Interface
The easiest way to use scikinC is through its Command Line Interface (CLI).
To provide an example, let's consider the following simple python script
that train a preprocessing step from scikit learn and dumps it into a 
pickle file. 

```python
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler 

minmax = MinMaxScaler()
minmax.fit ( np.random.normal(0,5, (2,1000) )
   
with open("example_scaler.pkl", 'wb') as f:
  pickle.dump (minmax, f)
```

Once the file is created, one can convert the scaler 
into a C file, as 
```bash
scikinC example_scaler.pkl > Cfile.C
```

Finally you can compile the C file for dynamic loading 
```bash
gcc -o deployed_scaler.so Cfile.C -shared -fPIC -Ofast
```

## Python Interface
Sometimes it may be useful to include the conversion in C
directly in the Python script where the training procedure 
is defined. This is made possible by importing the `scikinC`
function and calling the convert method.
For example,
```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scikinC

minmax = MinMaxScaler()
minmax.fit ( np.random.normal(0,5, (2,1000) )

c_string = scikinC.convert({
  'myMinMaxScaler': minmax
})
```
`c_string` contains the text that describe `MinMaxScaler`
transform in C. It can be stored in a text file and compiled 
with `gcc` and outlined above.


## Using the compiled models in C/C++ applications
Considering the example producing the shared object 
`deployed_object.so` above, one can easily evaluate 
it from a C program, linking the shared object at
run-time and then pointing to the function:
```C
// C Library for dynamic linking
#include  <dlfcn.h>

// Define the type for generic machine learning functions
typedef float *(*mlfunc)(float *, const float*);

void somewhere_in_your_code (void)
{
  // Open the shared object library 
  void *handle = dlopen ( "./deployed_scaler.so", RTLD_LAZY );
  if (!handle)
    exit(1);

  // Load the scaler by name (by default, the pickle file name is used as name)
  mlfunc minmax = mlfunc(dlsym (handle, "example_scaler")); 

  // Prepares the input and output buffer and evaluate the function
  float *inp [] = { /* your input goes here */ };
  float *out [ /*output n_features goes here*/ ];
  minmax ( out, inp ); 

  // Optionally, closes the linked library file
  dlclose(handle); 
}
```
A few notes:
 1. the function prototype (`FLOAT_T* <name> (FLOAT_T* output, const FLOAT_T*)`)
    is the same for all the models converted by scikinC. This is basically the
    only strict requirement on what models can be converted.
 2. The floating point type, `float` by default, can be updgraded for
    numerically instable models (`scikinC --float_t double` or scikinC --float_t "long double"`)
 3. the symbol to load through dlsym is the name of the pickle file, 
    stripped of its extension, if any. In this case `some_model.pkl` gets compiled 
    in the symbol `some_model`. The compiled function name can be specified as
    ```bash
    scikinC desired_name=example_scaler.pkl > Cfile.C
    ```
    this is especially useful when the pickle name contains non alphanumeric
    characters which would break the C compilation (consider for example a 
    pickle file named "example-scaler.pkl"
 4. More than one model can be compiled in a single shared object
    ```bash
    gcc -o deployed_scaler.so Cfile1.C Cfile2.C Cfile3.C -shared -fPIC -Ofast
    ```
    and this considered good practice for bundling together preprocessing 
    and machine learning steps. 


## Implemented converters

#### Scikit-Learn preprocessing
  | Model                  | Implementation  | Test      | Notes                             |
  | ---------------------- | --------------- | --------- |-----------------------------------|
  | `MinMaxScaler`         | Available       | Available |                                   |
  | `StandardScaler`       | Available       | Available |                                   |
  | `QuantileTransformer`  | Available       | Available |                                   |
  | `FunctionTransformer`  | Available       | Available | Supports user-defined C functions |
  | `ColumnTransformer`    | Available       | Available | Only integer column indices       |
  | `PolynomialFeatures`   | Available       | Available |                                   |
  | `Pipeline`             | Available       | Partial   | Pipelines of pipelines break      |

#### Scikit-Learn models
  | Model                        | Implementation  | Test      | Notes                         |
  | ---------------------------- | --------------- | --------- | ----------------------------- |
  | `GradientBoostingClassifier` | Available       | Available |                               |

#### Keras Models
  | Model                        | Implementation  | Test      | Notes                         |
  | ---------------------------- | --------------- | --------- | ----------------------------- |
  | `Sequential`                 | Available       | Available |                               |

#### Keras Layers
  | Model                        | Implementation  | Test      | Notes                         |
  | ---------------------------- | --------------- | --------- | ----------------------------- |
  | `Dense`                      | Available       | Available |                               |
  | `PReLU`                      | Available       | Available |                               |
  | `LeakyReLU`                  | Available       | Available |                               |
  | `BatchNormalization`         | Available       | Available |                               |

#### Keras Activation functions
  | Model                        | Implementation  | Test      | Notes                         |
  | ---------------------------- | --------------- | --------- | ----------------------------- |
  | `tanh`                       | Available       | Available |                               |
  | `sigmoid`                    | Available       | Available |                               |
  | `relu`                       | Available       | Available |                               |


## Running tests
In order to install the full dependencies needed to test the whole package, 
install with the tag `fql`.
```
python3 setup.py bdist_wheel 
pip install dist/scikinC*.whl[fql]
```

Then run the tests with
```
pytest test
```


## Related projects
  * [LWTNN](https://github.com/lwtnn/lwtnn)
  * [SimpleNN](https://gitlab.cern.ch/mschille/simplenn)
  * [TensorFlow C API](https://www.tensorflow.org/install/lang_c)
  * [GaudiTensorFlow](https://gitlab.cern.ch/lhcb/LHCb/-/tree/master/Tools/GaudiTensorFlow)
 
## Citing scikinC
L. Anderlini, M. Barbetti, *"scikinC: a tool for deploying machine learning as binaries"*, [Proceeding of Science (**CompTools2021**) 034](https://pos.sissa.it/409/034)
```bibtex
@article{Anderlini:2022ltm,
    author = "Anderlini, Lucio and Barbetti, Matteo",
    title = "{scikinC: a tool for deploying machine learning as binaries}",
    doi = "10.22323/1.409.0034",
    journal = "PoS",
    volume = "CompTools2021",
    pages = "034",
    year = "2022"
}
```

