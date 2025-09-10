import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

import pytest

# Local testing infrastructure
from wrap import deploy_pickle
from fixture_registry import fixtures


################################################################################
## Test preparation
@fixtures.register()
def pipeline():
    X = np.concatenate(
        (
            np.random.uniform(0, 2, (1000, 10)),
            np.random.uniform(1, 3, (1000, 10)),
            np.random.uniform(2, 4, (1000, 10)),
        )
    )

    step1 = MinMaxScaler()
    X2 = step1.fit_transform(X)
    step2 = StandardScaler()
    step2.fit(X2)

    return Pipeline(steps=((("minmax", step1), ("standard", step2))))


@fixtures.register()
def composition():
    X = np.concatenate(
        (
            np.random.uniform(0, 2, (1000, 10)),
            np.random.uniform(1, 3, (1000, 10)),
            np.random.uniform(2, 4, (1000, 10)),
        )
    )

    return Pipeline([
        ("cols", ColumnTransformer([
            ('log', FunctionTransformer(np.log), [0]),
            ('passthrough', 'passthrough', [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ])),
        ("standard", StandardScaler())
    ]).fit(X)

################################################################################
## Real tests
@fixtures.test()
def test_pipeline(pipeline):
    deployed = deploy_pickle("pipeline", pipeline)
    xtest = np.random.uniform(0, 1, 10)
    py = pipeline.transform(xtest[None])[0]
    c = deployed.transform(len(py), xtest)

    assert np.abs(py - c).max() < 1e-5


################################################################################
## Real tests
@fixtures.test()
def test_inverted_pipeline(pipeline):
    if not hasattr(pipeline, 'inverse_transform'):
        return pytest.skip("Will not test inversion of not-invertible pipeline")

    deployed = deploy_pickle("pipeline", pipeline)
    xtest = np.random.uniform(0, 1, 10)
    py = pipeline.transform(xtest[None])[0]
    c_back = pipeline.inverse_transform(py[None])
    py_back = deployed.transform_inverse(len(py), py)

    assert np.abs(py_back - c_back).max() < 1e-5

