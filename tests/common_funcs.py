# -*- coding: utf-8 -*-
"""
test utilities
"""

import numpy as np
import numdifftools as nd

# attribute queries


def dims(x):
    return getattr(x, "ndim", 0)


def shape(x):
    return getattr(x, "shape", ())


def value_of(x):
    return getattr(x, "value", x)

# check that two arrays are the same

def check(a, b):
    assert dims(a) == dims(b)
    assert shape(a) == shape(b)
    assert np.all(value_of(a) == value_of(b))

# check that two arrays are close

def check_close(a, b):
    if dims(a)!=dims(b):
        a = np.squeeze(a)
        b = np.squeeze(b)
    if dims(a)!=dims(b):
        # check all equal to the same value 
        # TODO: and/or one is a diagonal identity.
        a = np.atleast_1d(a)
        assert np.all(a==a[0]) and np.all(b==a[0])
        return
    assert dims(a) == dims(b)
    assert shape(a) == shape(b)
    assert np.allclose(value_of(a), value_of(b), 1e-10, 1e-10)
    
    
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