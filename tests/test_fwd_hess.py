# work out paths
import os
import sys

# this is intended to be used by pytest from parent directory

up = os.path.normpath(os.path.join(os.getcwd(), "src"))
sys.path.append(up)
sys.path.append(os.getcwd())


from maxmf.autodiff.forward import hessian
from numdifftools import Hessian as nd_hessian
import pytest

from common_funcs import check_close
from common_params import hessfns


@pytest.mark.parametrize("fn,b", hessfns)
def test_jacobian(fn, b):
    # test the jacobian of fn at x
    j = hessian(fn)(b)
    nj = nd_hessian(fn)(b)
    check_close(j, nj)
