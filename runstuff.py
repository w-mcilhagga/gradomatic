# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:23:26 2023

@author: willi
"""
import os
import sys

up = os.path.normpath(os.path.join(os.getcwd(), "src"))
sys.path.append(up)
sys.path.append(os.getcwd())
import numdifftools as nd
from fitany.tracing import value_of, dims, shape
import fitany.forward as fwd
import fitany.reverse as rev
    
# now do tests

import numpy as np

def nd_jacobian(fn, x):
    '''the jacobian of any function not just vector->vector'''
    xshape = shape(x)
    yshape = shape(fn(x))
    if len(xshape) <= 1 and len(yshape) <= 1:
        # the standard jacobian is fine
        return nd.Jacobian(fn)(x)
    else:
        # flstten has to pass through scalars; np.ravel doesn't
        flatten = lambda x: x if len(shape(x)) == 0 else np.ravel(x)
        # make a vector->vector function
        f2 = lambda x: flatten(fn(np.reshape(x, xshape)))
        # work out the jacobian of the vector->vector function
        j = nd.Jacobian(f2)(np.ravel(x))
        # reshape j to the correct x->y shape
        return np.reshape(np.squeeze(j), (*yshape, *xshape))
    
    
value2 = np.random.rand(5, 1)

mat1 = np.random.rand(5, 10)

mul = lambda x: mat1 * x
mul2 = lambda x: mat1 * np.broadcast_to(x, mat1.shape)

print(fwd.jacobian(mul)(value2).shape)
print(rev.jacobian(mul2)(value2).shape)