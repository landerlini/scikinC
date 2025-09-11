import os.path
import pickle

import numpy as np

# PyTest testing infrastructure
import pytest

# Local testing infrastructure
from wrap import deploy_pickle
from fixture_registry import fixtures


def read_file(filename):
    dir = os.path.dirname(__file__)
    with open(os.path.join(dir, "pathologies", filename), 'rb') as f:
        return pickle.load(f)

@fixtures.register()
def pathology_1():
    return read_file('column_with_quantile_1.pkl')

@fixtures.register()
def pathology_2():
    return read_file('column_with_quantile_2.pkl')

@fixtures.register()
def pathology_3():
    return read_file('column_with_quantile_3.pkl')

@fixtures.register()
def pathology_4():
    return read_file('column_with_quantile_4.pkl')

################################################################################
## Real tests
@fixtures.test()
def test_forward (scaler):
    n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 20

    deployed = deploy_pickle("quantiletransformer", scaler)
    results = []
    try:
        for iAttempt in range(100):
            xtest = np.random.uniform (-1000,-990, n_features)
            try:
                py = scaler.transform (xtest[None])
            except (AttributeError,):
                pytest.skip(f"Failed running the pickled pathological example. Probably generated with old sklearn.")

            c  = deployed.transform (n_features, xtest)
            results.append ([py[0].flatten(), c.flatten(), np.abs(py[0]-c).flatten() > 1e-5 ])
            assert np.abs(py-c).max() < 1e-4
    finally:
        array = np.array(results)
        print (array.T)
