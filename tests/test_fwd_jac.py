# work out paths
import os
import sys

# this is intended to be used by pytest from parent directory

up = os.path.normpath(os.path.join(os.getcwd(), "src"))
sys.path.append(up)
sys.path.append(os.getcwd())


import numpy as np
from fitany.forward import jacobian
import pytest

from common_funcs import check_close, nd_jacobian
from common_params import ufuncs, unary, binary


@pytest.mark.parametrize("fn, x", ufuncs)
def test_ufuncs(fn, x):
    j = jacobian(fn)(x)
    nj = nd_jacobian(fn, x)
    check_close(j, nj)


@pytest.mark.parametrize("fn, x", unary)
def test_unary(fn, x):
    j = jacobian(fn)(x)
    nj = nd_jacobian(fn, x)
    check_close(j, nj)

# binary ops


@pytest.mark.parametrize("fn,a,b", binary)
def test_binary_1(fn, a, b):
    # test varnode in both places
    f = lambda b: fn(a, b)
    j = jacobian(f)(b)
    nj = nd_jacobian(f, b)
    check_close(j, nj)
        
@pytest.mark.parametrize("fn,a,b", binary)
def test_binary_0(fn, a, b):
    if type(a) is np.array:
        f = lambda a: fn(a, b)
        j = jacobian(f)(a)
        nj = nd_jacobian(f, a)
        check_close(j, nj)

# we haven't tested np.einsum because it has already been used in the above tests.


# Integration tests - just a bunch of more complex functions
X = np.random.rand(30, 10)
y = np.random.rand(30)
b = np.random.rand(10) + 1
A = np.random.rand(10, 10)
AA = np.random.rand(4, 4)

# functions for jacobian tests


def LL(x):
    # binomial log-likelihood
    p = 1 / (1 + np.exp(-X @ x))
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def lr(b):
    # linear regression
    r = y - X @ b
    return np.dot(r, r)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: 1,  # this returns 0 in jacobian
        lambda x: x,
        lambda x: 1 / (1 + x),
        lambda x: 10 * x + 3,
        lambda x: 2 ** (A @ x + 1),
        lambda x: 1 / (1 + np.dot(x, x)),
        lambda x: A @ x,
        lambda x: X @ x,
        lambda x: x @ A @ x,
        lambda x: np.einsum("i,ij,j->", x, A, x),
        lambda x: 1 / (1 + np.sum(x)),
        lambda x: np.cos(np.sin(np.log(np.exp(np.sum(x))))),
        lambda x: (x @ A @ x) / (1 + np.dot(x, x)),
        LL,
        lr,
        lambda x: (1 - np.exp(-x)) / (1 + np.exp(-x)),
        lambda x: np.sum(np.ones((10,)) * A @ x),
        lambda x: np.dot(x[0:3], x[3:6]),
        lambda x: AA @ x[0:4],
        lambda x: x[0:4] @ AA @ x[4:8],
    ],
)
def test_jacobian(fn):
    # test the jacobian of fn at x
    j = jacobian(fn)(b)
    nj = nd_jacobian(fn, b)
    check_close(j, nj)
