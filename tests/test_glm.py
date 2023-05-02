# work out paths
import os
import sys

# this is intended to be used by pytest from parent directory

up = os.path.normpath(os.path.join(os.getcwd(), "src"))
sys.path.append(up)
sys.path.append(os.getcwd())

import numpy as np
from maxmf.glm import *
import maxmf.autodiff.reverse as rev
import maxmf.autodiff.forward as fwd
from common_funcs import check_close
import pytest

ra = lambda *args: np.random.rand(*args)

X = ra(20, 5)
y = ra(20)
b = ra(5)


# Test the mean functions for value & jacobian


def fwd_val_jac(f, argno, *args):
    f = fwd.Diff(f, argno=argno)
    return f.trace(*args), f.jacobian(gc=True)


MFlist = [
    (linearMF, lambda b, X: X @ b),
    (logitMF, lambda b, X: 1 / (1 + np.exp(-X @ b))),
    (expMF, lambda b, X: np.exp(X @ b)),
]


@pytest.mark.parametrize("f1,f2", MFlist)
def test_meanfunc(f1, f2):
    # we only test forward derivatives for these
    v1, j1 = fwd_val_jac(f1, 0, b, X)
    v2, j2 = fwd_val_jac(f2, 0, b, X)
    check_close(v1, v2)
    check_close(j1, j2)


# Test the built-in GLMs


def fwd_val_jac_hess(f, argno, *args):
    f = fwd.Diff(f, argno=argno)
    return f.trace(*args), f.jacobian(), f.hessian()


def rev_val_jac_hess(f, argno, *args):
    f = rev.Diff(f, argno=argno)
    return f.trace(*args), f.jacobian(), f.hessian()


def mylinearGLM(b, y, X):
    return -0.5 * np.sum((y - X @ b) ** 2)


def mylogisticGLM(b, y, X):
    mu = 1 / (1 + np.exp(-X @ b))
    cmu = np.clip(mu, 0.001, 0.999)
    return np.sum(y * np.log(cmu) + (1 - y) * np.log(1 - cmu))


def mypoissonGLM(b, y, X):
    mu = np.clip(np.exp(X @ b), 0, None)
    return np.sum(y * np.log(mu) - mu)


GLMlist = [
    (linearGLM, mylinearGLM),
    (logisticGLM, mylogisticGLM),
    (poissonGLM, mypoissonGLM),
]


@pytest.mark.parametrize("f1,f2", GLMlist)
def test_GLM_rev(f1, f2):
    v1, j1, h1 = rev_val_jac_hess(f1, 0, b, y, X)
    v2, j2, h2 = rev_val_jac_hess(f2, 0, b, y, X)
    check_close(v1, v2)
    check_close(j1, j2)
    check_close(h1, h2)


@pytest.mark.parametrize("f1,f2", GLMlist)
def test_GLM_fwd(f1, f2):
    v1, j1, h1 = fwd_val_jac_hess(f1, 0, b, y, X)
    v2, j2, h2 = fwd_val_jac_hess(f2, 0, b, y, X)
    check_close(v1, v2)
    check_close(j1, j2)
    check_close(h1, h2)
