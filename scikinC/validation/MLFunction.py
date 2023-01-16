import ctypes
import numpy as np


class MLFunction:
    """
    A Python wrapper around a shared library obtained with scikinC to ease evaluation from Python

    Arguments:
        * lib_path (str): path to the shared object
        * function_name (str): name of a double*(double*, double*) function defined in the shared object
        * n_inputs (int): number of inputs variables
        * n_outputs (int): number of output variables
    """

    def __init__(self, lib_path, function_name, n_inputs, n_outputs, float_type=np.float32):
        if not (lib_path.startswith("/") or lib_path.startswith("./")):
            lib_path = "./" + lib_path

        self._lib = ctypes.CDLL(lib_path)

        self._f = getattr(self._lib, function_name)
        self._f.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32) for _ in (1, 2)]

        self._n_inputs = n_inputs
        self._n_outputs = n_outputs

        self._float_type = np.float32

    @property
    def n_inputs(self):
        "Number of input nodes"
        return self._n_inputs

    @property
    def n_outputs(self):
        "Number of output nodes"
        return self._n_outputs

    @property
    def float_type(self):
        "Type of float (np.float32 or np.float64)"
        return self._float_type

    def __call__(self, data_in):
        """
        Call the wrapped function.

        Arguments:
            * data_in (np.ndarray):
                if a 1d array, it is interpreted as input of the wrapped function, output is a 2d array with 1 row
                if a list of 1d arrays, the wrapped function is evaluated once for each input, output is a 2d array
                if a 2d array, each row is considered as an input to the wrapped function, output is a 2d array
        """
        if not isinstance(data_in, (np.ndarray, list)):
            raise TypeError

        data_in_f = np.asarray(data_in).astype(self.float_type)

        if len(data_in_f.shape) == 1:
            data_in_f = np.array([data_in_f])

        obuf = np.empty(self.n_outputs, dtype=np.float32)
        output_rows = []

        for data_row in data_in_f:
            self._f(obuf, data_row)
            output_rows.append(obuf.copy())

        if isinstance(data_in, np.ndarray) and len(data_in.shape) == 1:
            return output_rows[0]

        return np.stack(output_rows)

