import numpy as np
from sklearn.preprocessing import StandardScaler

# PyTest testing infrastructure
import pytest

import scikinC
from scikinC.validation import MLFunction

# Local testing infrastructure
from wrap import deploy_pickle


################################################################################
## Test preparation
@pytest.fixture
def scaler():
    scaler_ = StandardScaler()
    X = np.random.uniform(20, 30, (1000, 10))
    scaler_.fit(X)
    return scaler_


@pytest.fixture
def mlfun(scaler):
    deplyed_model = deploy_pickle("standardscaler", scaler)
    return MLFunction(deplyed_model.compiled, "transform", 10, 10)


################################################################################
## Real tests
def test_1d(scaler, mlfun):
    xtest = np.random.uniform(20, 30, 10)
    py = scaler.transform(xtest[None])
    c = mlfun(xtest)
    assert np.abs(py - c).max() < 1e-5

def test_list_of_1d(scaler, mlfun):
    xtest = [np.random.uniform(20, 30, 10) for _ in range(3)]
    py = scaler.transform(xtest)
    c = mlfun(xtest)
    assert np.abs(py - c).max() < 1e-5

def test_2d(scaler, mlfun):
    xtest = np.random.uniform(20, 30, (3,10))
    py = scaler.transform(xtest)
    c = mlfun(xtest)
    assert np.abs(py - c).max() < 1e-5
