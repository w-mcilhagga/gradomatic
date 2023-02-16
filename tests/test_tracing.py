# work out paths
import os
import sys

# this is intended to be called from ..

'''
failed tests:
    trace, diag only work on square matrices.
    outer only works on vectors; it flattens matrices and
        it gives a different shape for scalar x vector
'''

up = os.path.normpath(os.path.join(os.getcwd(),"src"))
sys.path.append(up)
sys.path.append(os.getcwd())


import numpy as np
from fitany.tracing import *
import pytest

def test_varnode():
    value = np.random.rand(10,5)
    varnode = VarNode(value)
    assert(varnode.ndim==value.ndim)
    assert(varnode.shape==value.shape)
    assert(np.all(value_of(varnode)==value))

# operator tests

# ufunc tests

unary =   [np.negative, 
           np.exp, 
           np.log,
           np.sqrt,
           np.abs,
           np.sign,
           np.cos, 
           np.sin, 
           np.tan,
           np.transpose,
           #np.trace, 
           #np.diag,
           ]

@pytest.mark.parametrize('fn', unary)
def test_unary(fn):
    value = np.random.rand(10,5)
    f_value = fn(value)
    opnode = fn(VarNode(value))
    assert(opnode.ndim==f_value.ndim)
    assert(opnode.shape==f_value.shape)
    assert(np.all(value_of(opnode)==f_value))

@pytest.mark.parametrize('ax', [{}, {'axis':0}, {'axis':1}])
def test_sum(ax):
    value = np.random.rand(10,5)
    f_value = np.sum(value, **ax)
    opnode = np.sum(VarNode(value), **ax)
    assert(opnode.ndim==f_value.ndim)
    assert(opnode.shape==f_value.shape)
    assert(np.allclose(value_of(opnode), f_value, 1e-10, 1e-10))

# dot, inner, outer are complicated by the number of different args

value = np.random.rand(10)
mat1 = np.random.rand(5,10)
mat2 = np.random.rand(10,3)
mat3 = np.random.rand(5,3,10)

@pytest.mark.parametrize('a,b', [(3,value), (value,value), 
                                 (mat1, value), (mat1, mat2), (mat3, mat2)])
def test_dot(a,b):
    f_value = np.dot(a,b)
    opnode = np.dot(a, VarNode(b))
    assert(opnode.ndim==f_value.ndim)
    assert(opnode.shape==f_value.shape)
    assert(np.allclose(value_of(opnode), f_value, 1e-10, 1e-10))
    
@pytest.mark.parametrize('a,b', [(3,value), (value,value), 
                                 (mat1, value), (mat1, mat3)])
def test_inner(a,b):
    f_value = np.inner(a,b)
    opnode = np.inner(a, VarNode(b))
    assert(opnode.ndim==f_value.ndim)
    assert(opnode.shape==f_value.shape)
    assert(np.allclose(value_of(opnode), f_value, 1e-10, 1e-10))


@pytest.mark.parametrize('a,b', [(value, value)])
def test_outer(a,b):
    f_value = np.outer(a,b)
    opnode = np.outer(a, VarNode(b))
    assert(opnode.ndim==f_value.ndim)
    assert(opnode.shape==f_value.shape)
    assert(np.allclose(value_of(opnode), f_value, 1e-10, 1e-10))


@pytest.mark.parametrize('a,shape', [(value, (5,2)), (mat3, (150,))])
def test_reshape(a,shape):
    f_value = np.reshape(a, shape)
    opnode = np.reshape(VarNode(a), shape)
    assert(opnode.ndim==f_value.ndim)
    assert(opnode.shape==f_value.shape)
    assert(np.all(value_of(opnode)==f_value))
    
# TODO

# indexing

n_nary =  [np.add, np.subtract, np.divide, 
           np.matmul,
           np.tensordot,
           np.einsum,
           np.broadcast_to,
           np.clip]