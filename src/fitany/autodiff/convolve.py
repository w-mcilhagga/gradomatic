# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 15:05:30 2023

@author: willi
"""

import numpy as np
from itertools import product

shape = lambda x: getattr(x, "shape", ())
dims = lambda x: getattr(x, "ndim", 0)


def corrtensor_valid(a, b, axes=None):
    # Works out a tensor A such that
    # tensordot(A,b,axes) == tensorcorrel(a,b,axes,mode='valid')
    if axes is None:
        axes = dims(b)
    # corrsz is the size of the b filter over the axes.
    corrsz = shape(b)[:axes]
    # validshape is the shape of the a*b results over the valid shifts
    validshape = np.array(shape(a)[-axes:]) - corrsz + 1
    # create a zero tensor to hold the correlation tensor
    # values with mode='valid' which has shape:
    # (unused `a` dimensions, validshape, corrsz)
    ctensor = np.zeros((*shape(a)[:-axes], *validshape, *corrsz))
    # do the loop
    nda = dims(a)
    a_slice = [slice(None)] * nda
    ctensor_slice = [slice(None)] * dims(ctensor)
    for indices in product(*[range(c) for c in validshape]):
        for i, ai, bi in zip(range(axes), indices, corrsz):
            a_slice[nda - axes + i] = slice(ai, ai + bi)
            ctensor_slice[
                nda - axes + i
            ] = ai  # nb these singleton indices get compressed
        ctensor[tuple(ctensor_slice)] = a[tuple(a_slice)]
    return ctensor


def corrtensor_full(a, b, axes=None):
    # Works out a tensor A such that
    # tensordot(A,b,axes) == tensorcorrel(a,b,axes,mode='full')
    if axes is None:
        axes = dims(b)
    # corrsz is the size of the b filter over the axes.
    corrsz = np.array(shape(b)[:axes])
    # fullshape is the shape of the a*b results over the full shifts
    fullshape = np.array(shape(a)[-axes:]) - corrsz + 1 + 2 * (corrsz - 1)
    # create a zero tensor to hold the correlation tensor
    # values with mode='full' which has shape:
    # (unused `a` dimensions, fullshape, corrsz)
    ctensor = np.zeros((*shape(a)[:-axes], *fullshape, *corrsz))
    # do the loop
    nda = dims(a)
    a_slice = [slice(None)] * nda
    ctensor_slice = [slice(None)] * dims(ctensor)
    for indices in product(*[range(c) for c in fullshape]):
        for i, ai, bi in zip(range(axes), indices, corrsz):
            # ai is the index into the fullshape & corresponds
            # to the a range [ai-bi ... ai] inclusive
            # If this falls outside the indices of a in that axis,
            # we need to restrict the size of it, and the same within
            # the corrsz axes of ctensor
            lo = ai - bi
            hi = ai
            ctensor_slice[nda - axes + i] = ai
            if lo < 0:
                a_slice[nda - axes + i] = slice(0, hi + 1)
                ctensor_slice[nda + i] = slice(-lo - 1, None)
            elif hi >= a.shape[nda - axes + i]:
                shift = hi - a.shape[nda - axes + i]
                a_slice[nda - axes + i] = slice(
                    lo + 1, a.shape[nda - axes + i]
                )
                ctensor_slice[nda + i] = slice(0, bi - shift - 1)
            else:
                a_slice[nda - axes + i] = slice(lo + 1, hi + 1)
                ctensor_slice[nda + i] = slice(None)
        try:
            ctensor[tuple(ctensor_slice)] = a[tuple(a_slice)]
        except:
            print("fail", ctensor_slice, a_slice)
    return ctensor


def corrtensor_same(a, b, axes=None):
    # corrtensor but for mode=same
    if axes is None:
        axes = dims(b)
    # corrsz is the size of the b filter over the axes.
    # corrsz = np.array(shape(b)[:axes])
    # fullshape is the shape of the a*b results over the full shifts
    # fullshape = np.array(shape(a)[-axes:]) - corrsz + 1 + 2 * (corrsz - 1)
    ctensor = corrtensor_full(a, b, axes=axes)
    # extract the a-shaped part of ct
    ctensor_slice = [slice(None)] * dims(ctensor)
    for i in range(dims(a)):
        ctsz = ctensor.shape[i]
        asz = a.shape[i]
        start = (ctsz - asz) // 2
        ctensor_slice[i] = slice(start, start + asz)
    ctensor = ctensor[tuple(ctensor_slice)]
    return ctensor


from weakref import WeakValueDictionary

Tcache = WeakValueDictionary()


def tensorcorrelate(a, v, mode="full", axes=-1):
    # tensor correlation; it keeps the correlation tensor
    # weakly because it's often reused
    if axes == -1:
        axes = dims(v)
    key = (id(a), shape(v), mode, axes)
    if key in Tcache:
        ct = Tcache[key]
    else:
        if mode == "full":
            ct = corrtensor_full(a, v, axes)
        elif mode == "same":
            ct = corrtensor_same(a, v, axes)
        elif mode == "valid":
            ct = corrtensor_valid(a, v, axes)
        Tcache[key] = ct
    return np.tensordot(ct, v, axes=axes)


def tensorconvolve(a, v, mode="full", axes=-1):
    # tensor convolution
    v = v[tuple([slice(None, None, -1)] * dims(v))]
    return tensorcorrelate(a, v, mode=mode, axes=axes)
