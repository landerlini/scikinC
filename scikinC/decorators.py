"""
Standard decorators to ease marking functions with special meaning for scikinC

**Example.**

The following code snippets adds a C implementation to a function that can then be used to define
FunctionTransformer in scikinC.

``` python
from scikinC.decorators import inline_c

@inline_c("1 / (1 + {x})")
def sigmoid(x):
    return 1 / (1 + x)
```
"""

from typing import Callable


def inline_c(c_implementation: str):
    """Simple decorator to associate a custom inline C implementation to a Python function"""
    def decorator(f: Callable):
        f.inC = c_implementation
        return f

    return decorator
