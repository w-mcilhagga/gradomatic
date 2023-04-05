# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:56:49 2023

@author: willi
"""

import numpy as np

value_of = lambda x: getattr(x, 'value', x)

class Subgrad:
    @staticmethod
    def isgrad(n):
        """checks if a subgrad object n is really a grad"""
        return isinstance(n, Subgrad) and not np.all(np.isclose(n.lo, n.hi))
    
    @staticmethod
    def assubgrad(n):
        return n if isinstance(n, Subgrad) else Subgrad(n,n)

    def __init__(self, a, b):
        # [a,b] is an interval
        self.lo = np.minimum(a, b)
        self.hi = np.maximum(a, b)
        self.ndim = getattr(a, 'ndim', 0)

    # operator overrides for operations involving python objects

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __sub__(self, other):
        return np.subtract(self, other)

    def __rsub__(self, other):
        return np.subtract(other, self)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return np.multiply(other, self)

    def __truediv__(self, other):
        return np.true_divide(self, other)

    def __rtruediv__(self, other):
        return np.true_divide(other, self)

    def __neg__(self):
        return np.negative(self)

    # def __matmul__(self, other):
    #    return np.matmul(self, other)

    # def __rmatmul__(self, other):
    #    return np.matmul(other, self)

    def __pow__(self, other):
        return np.power(self, other)

    def __rpow__(self, other):
        return np.power(other, self)

    def __getitem__(self, key):
        return Subgrad(self.lo[key], self.hi[key])

    def __hash__(self):
        # required because __eq__ was overridden
        return id(self)

    def __len__(self):
        return self.lo.shape[0]

    def asarray(self):
        # converts to a numpy array - if bracketing zero, returns zero,
        # otherwise returns the value closest to zero.
        L = value_of(self.lo)
        H = value_of(self.hi)
        pos = np.logical_and(L > 0, H > 0)
        neg = np.logical_and(L < 0, H < 0)
        arr = np.zeros(L.shape)
        arr[pos] = np.minimum(L[pos], H[pos])
        arr[neg] = np.maximum(L[neg], H[neg])
        return arr

    def __array__(self, dtype=None):
        return self.asarray()
    
    # numpy dispatch for when one of the objects is a numpy array

    handled_funcs = {}

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # dispatcher for numpy ufuncs
        if method != "__call__" or ufunc not in self.handled_funcs:
            return NotImplemented
        s = self.handled_funcs[ufunc](*args, **kwargs)
        return s

    def __array_function__(self, func, types, args, kwargs):
        # dispatcher for other numpy funcs
        if func not in self.handled_funcs:
            return NotImplemented
        return self.handled_funcs[func](*args, **kwargs)

    @classmethod
    def add_handler(cls, f, g):
        # declares that we should use g when numpy function f is called
        cls.handled_funcs[f] = g

    @classmethod
    def register_handler(cls, *flist):
        # a decorator for add_handler
        def decorator(g):
            for f in flist:
                cls.add_handler(f, g)
            return g

        return decorator


@Subgrad.register_handler(np.add)
def s_add(a, b):
    # either a, b or both are subgrads
    a = Subgrad.assubgrad(a)
    b = Subgrad.assubgrad(b)
    return Subgrad(a.lo + b.lo, a.hi + b.hi)


@Subgrad.register_handler(np.subtract)
def s_subtract(a, b):
    # either a, b or both are subgrads
    a = Subgrad.assubgrad(a)
    b = Subgrad.assubgrad(b)
    return Subgrad(a.lo - b.lo, a.hi - b.hi)


@Subgrad.register_handler(np.multiply)
def s_mul(a, b):
    # either a, b or both are subgrads
    a = Subgrad.assubgrad(a)
    b = Subgrad.assubgrad(b)
    return Subgrad(a.lo * b.lo, a.hi * b.hi)


@Subgrad.register_handler(np.true_divide)
def s_div(a, b):
    # either a, b or both are subgrads
    a = Subgrad.assubgrad(a)
    b = Subgrad.assubgrad(b)
    return Subgrad(a.lo / b.lo, a.hi / b.hi)


@Subgrad.register_handler(np.power)
def s_pow(a, b):
    # either a, b or both are subgrads
    a = Subgrad.assubgrad(a)
    b = Subgrad.assubgrad(b)
    return Subgrad(a.lo**b.lo, a.hi**b.hi)


@Subgrad.register_handler(np.negative)
def s_neg(a):
    return Subgrad(-a.hi, -a.lo)


@Subgrad.register_handler(np.einsum)
def s_einsum(script, *args, **kwargs):
    # an implementation, but probably not the right one.
    args = list(args)
    argno = next(filter(lambda p: isinstance(p[1], Subgrad), enumerate(args)))[0]
    sg = args[argno]
    args[argno] = sg.lo
    low = np.einsum(script, *args, **kwargs)
    args[argno] = sg.hi
    high = np.einsum(script, *args, **kwargs)
    return Subgrad(low, high)

def signum(x):
    # returns the subgrad of abs(x)
    x = value_of(x) # not differentiable
    sx = np.sign(x)
    sz = sx==0
    return Subgrad(sx - sz, sx + sz)
