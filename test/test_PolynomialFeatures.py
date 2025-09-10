import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# PyTest testing infrastructure
import pytest

# Local testing infrastructure
from wrap import deploy_pickle
from fixture_registry import fixtures


################################################################################
## Test preparation
@fixtures.register()
def basic_pf():
    pf_ = PolynomialFeatures(degree=2)
    X = np.arange(10)[None, :]
    pf_.fit(X)
    return pf_

@fixtures.register()
def large_pf():
    pf_ = PolynomialFeatures(degree=10)
    X = np.arange(2)[None, :]
    pf_.fit(X)
    return pf_

@fixtures.register()
def no_bias_pf():
    pf_ = PolynomialFeatures(include_bias=False, degree=2)
    X = np.arange(10)[None, :]
    pf_.fit(X)
    return pf_

@fixtures.register()
def fortran_order_pf():
    pf_ = PolynomialFeatures(order='F', degree=2)
    X = np.arange(10)[None, :]
    pf_.fit(X)
    return pf_


################################################################################
## Real tests
@fixtures.test()
def test_forward(pf):
    xtest = np.arange(pf.n_features_in_)
    py = pf.transform(xtest[None])
    deployed = deploy_pickle("polynomialfeatures", pf)
    c = deployed.transform(n_outputs=pf.n_output_features_, args=xtest)

    print(np.c_[py.ravel(), c.ravel()])

    assert np.abs(py - c).max() < 1e-5, 'Result inconsistent with expectation'


