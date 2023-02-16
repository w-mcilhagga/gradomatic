# work out paths
import os
up = os.path.normpath(os.path.join(os.getcwd(),".."))
import sys
sys.path.append(up)
sys.path.append(os.getcwd())

# imports

import numpy as np
import numdifftools as nd
import pytest
from functionlist import j_funcs, h_funcs, j_funcs_elem, b as x
from autostat.grad.fwd import jacobian, gradient_and_hessian

    
@pytest.mark.parametrize('fn', j_funcs)
def test_jacobian(fn):
    # test the jacobian of fn at x
    j = jacobian(fn)(x)
    nj = np.squeeze(nd.Jacobian(fn)(x))
    if j is not 0:
        assert(j.shape == nj.shape)
    assert( np.allclose(j, nj) )

@pytest.mark.parametrize('fn', h_funcs)
def test_hessian(fn):
    # test the jacobian of fn at x
    j = gradient_and_hessian(fn)(x)[1]
    nj = np.squeeze(nd.Hessian(fn)(x))
    if j is not 0:
        assert(j.shape == nj.shape)
    assert( np.allclose(j, nj) )
    
@pytest.mark.parametrize('fn', j_funcs_elem)
def test_jacobian_elementwise(fn):
    # test the jacobian of fn at x
    j = jacobian(fn)(x)
    nj = np.squeeze(nd.Jacobian(fn)(x))
    if j is not 0 and j.shape!=nj.shape:
        nj = np.diag(nj)
    if j is not 0:
        assert(j.shape == nj.shape)
    assert( np.allclose(j, nj) )