# work out paths
import os
import sys

# this is intended to be called from ..

"""
"""

up = os.path.normpath(os.path.join(os.getcwd(), "src"))
sys.path.append(up)
sys.path.append(os.getcwd())


import numpy as np
from fitany.tracing import VarNode
import pytest

from common_funcs import check, check_close
from common_params import ufuncs, unary, binary


def test_varnode():
    value = np.random.rand(10, 5)
    varnode = VarNode(value)
    check(varnode, value)


@pytest.mark.parametrize("fn, x", ufuncs)
def test_ufuncs(fn, x):
    check_close(fn(x), fn(VarNode(x)))


@pytest.mark.parametrize("fn, x", unary)
def test_unary(fn, x):
    check_close(fn(x), fn(VarNode(x)))


@pytest.mark.parametrize("fn,a,b", binary)
def test_binary(fn, a, b):
    # test varnode in both places
    check_close(fn(a, b), fn(a, VarNode(b)))
    check_close(fn(a, b), fn(VarNode(a), b))


# we haven't tested np.einsum because it has already been used in the above tests.
