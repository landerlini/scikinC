import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# PyTest testing infrastructure
import pytest

# Local testing infrastructure
from wrap import deploy_pickle
from fixture_registry import fixtures


################################################################################
## Test preparation
@fixtures.register()
def binary_classifier():
    classifier_ = GradientBoostingClassifier(n_estimators=10, init='zero')
    X = np.concatenate(
        (
            np.random.normal(0, 2, (1000, 10)),
            np.random.normal(1, 3, (1000, 10)),
        )
    )
    y = np.array(
        [0] * 1000 + [1] * 1000
    )
    classifier_.fit(X, y)
    return classifier_

@fixtures.register()
def deep_binary_classifier():
    classifier_ = GradientBoostingClassifier(n_estimators=10, init='zero', max_depth=8)
    X = np.concatenate(
        (
            np.random.normal(0, 2, (1000, 10)),
            np.random.normal(1, 3, (1000, 10)),
        )
    )
    y = np.array(
        [0] * 1000 + [1] * 1000
    )
    classifier_.fit(X, y)
    return classifier_

@fixtures.register()
def multiclass_classifier():
    classifier_ = GradientBoostingClassifier(n_estimators=10)
    X = np.concatenate(
        (
            np.random.normal(0, 2, (1000, 10)),
            np.random.normal(1, 3, (100, 10)),
            np.random.normal(2, 4, (10, 10)),
        )
    )
    y = np.array(
        [0] * 1000 + [1] * 100 + [2] * 10
    )
    classifier_.fit(X, y)
    return classifier_

@fixtures.register()
def zero_init():
    classifier_ = GradientBoostingClassifier(n_estimators=10, init='zero')
    X = np.concatenate(
        (
            np.random.normal(0, 2, (1000, 10)),
            np.random.normal(1, 3, (100, 10)),
            np.random.normal(2, 4, (10, 10)),
        )
    )
    y = np.array(
        [0] * 1000 + [1] * 100 + [2] * 10
    )
    classifier_.fit(X, y)
    return classifier_


@fixtures.register()
def deep_classifier():
    classifier_ = GradientBoostingClassifier(max_depth=8)
    X = np.concatenate(
        (
            np.random.normal(0, 2, (1000, 10)),
            np.random.normal(1, 3, (100, 10)),
            np.random.normal(2, 4, (10, 10)),
        )
    )
    y = np.array(
        [0] * 1000 + [1] * 100 + [2] * 10
    )
    classifier_.fit(X, y)
    return classifier_



################################################################################
## Real tests
@fixtures.test()
def test_normalization(classifier):
    deployed = deploy_pickle("gbdtcD", classifier)
    xtest = np.random.uniform(0, 1, 10)
    py = classifier.predict_proba(xtest[None])[0]
    c = deployed.transform(len(py), xtest)

    print (py, c)

    assert np.abs(np.sum(c) - 1).max() < 1e-5


@fixtures.test()
def test_predict(classifier):
    deployed = deploy_pickle("gbdtcD", classifier)
    xtest = np.random.uniform(0, 1, 10)
    py = classifier.predict_proba(xtest[None])[0]
    c = deployed.transform(len(py), xtest)

    print(np.c_[py, c])
    assert np.abs(py - c).max() < 1e-5

