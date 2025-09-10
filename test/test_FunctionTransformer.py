import numpy as np
from sklearn.preprocessing import FunctionTransformer

from scikinC.decorators import inline_c

# Local testing infrastructure
from wrap import deploy_pickle
from fixture_registry import fixtures


################################################################################
## Test preparation
@fixtures.register()
def empty_transformer():
    transformer_ = FunctionTransformer(validate=True)
    X = np.random.uniform(20, 30, (1000, 10))
    transformer_.fit(X)
    return transformer_


@fixtures.register()
def log_transformer():
    transformer_ = FunctionTransformer(np.log, np.exp, validate=True)
    X = np.random.uniform(20, 30, (1000, 10))
    transformer_.fit(X)
    return transformer_


@fixtures.register()
def custom_transformer():
    transformer_ = FunctionTransformer(np.square, np.sqrt, validate=True)
    transformer_.func_inC = 'pow({x}, 2)'
    X = np.random.uniform(20, 30, (1000, 10))
    transformer_.fit(X)
    return transformer_


@fixtures.register()
def empty_transformer_wo_fit():
    transformer_ = FunctionTransformer()
    transformer_.n_features_in_ = 10
    return transformer_


################################################################################
## Real tests
@fixtures.test()
def test_forward(ft):
    deployed = deploy_pickle("functiontransformer", ft)
    xtest = np.random.uniform(21, 29, 10)
    py = ft.transform(xtest[None])
    c = deployed.transform(10, xtest)
    print(xtest, "->", c, " instead of: ", py)
    assert np.abs(py - c).max() < 1e-4


@fixtures.test()
def test_inverse(ft):
    deployed = deploy_pickle("functiontransformer", ft)
    xtest = np.random.uniform(0, 1, 10)
    py = ft.inverse_transform(xtest[None])
    c = deployed.transform_inverse(10, xtest)
    assert np.abs(py - c).max() < 1e-4
