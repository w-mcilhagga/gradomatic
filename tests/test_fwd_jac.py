# work out paths
import os
import sys

# this is intended to be used by pytest from parent directory

up = os.path.normpath(os.path.join(os.getcwd(), "src"))
sys.path.append(up)
sys.path.append(os.getcwd())


import numpy as np
from maxmf.autodiff.forward import jacobian
import pytest

from common_funcs import check_close, nd_jacobian
from common_params import ufuncs, unary, binary, jacfns


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


@pytest.mark.parametrize("fn,b", jacfns)
def test_jacobian(fn,b):
    # test the jacobian of fn at x
    j = jacobian(fn)(b)
    nj = nd_jacobian(fn, b)
    check_close(j, nj)
