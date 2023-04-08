# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 15:05:30 2023

@author: willi
"""

import numpy as np
from itertools import product

shape = lambda x: getattr(x, "shape", ())
dims = lambda x: getattr(x, "ndim", 0)


def _corrtensor_valid(a, b, axes=None):
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


def _corrtensor_full(a, b, axes=None):
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


def _corrtensor_same(a, b, axes=None):
    # corrtensor but for mode=same
    if axes is None:
        axes = dims(b)
    # corrsz is the size of the b filter over the axes.
    # corrsz = np.array(shape(b)[:axes])
    # fullshape is the shape of the a*b results over the full shifts
    # fullshape = np.array(shape(a)[-axes:]) - corrsz + 1 + 2 * (corrsz - 1)
    ctensor = _corrtensor_full(a, b, axes=axes)
    # extract the a-shaped part of ct
    ctensor_slice = [slice(None)] * dims(ctensor)
    for i in range(dims(a)):
        ctsz = ctensor.shape[i]
        asz = a.shape[i]
        start = (ctsz - asz) // 2
        ctensor_slice[i] = slice(start, start + asz)
    ctensor = ctensor[tuple(ctensor_slice)]
    return ctensor


def _tensorcorrelate(a, v, mode="full", axes=-1, returntype='value'):
    """correlation of two tensors

    Args:
        a (numpy array): the tensor to correlate over
        v (numpy array): the filter tensor
        mode (string): either 'full', 'same', or 'valid'
        axes: the number of axes to correlate. If not given, it is all
            the axes of v.
            tensorcorrelate(a,v,axes=n) will correlate
            the last n axes of a with the first n axes of v.
        returntype (string)): 'value' (default) 'callable', or 'tensor'

    Returns:
        The return value depends on returntype. 
        * 'value': returns the convolution of a with v
        * 'callable': returns a function of v which does the correlation
        * 'tensor': returns the tensor which can do the correlation using
          np.tensordot(tensor, v, axes=..) NB the default axes for tensordot
          is not the same as the default for tensorcorrelate.
    """
    if axes == -1:
        axes = dims(v)
    if mode == "full":
        ct = _corrtensor_full(a, v, axes)
    elif mode == "same":
        ct = _corrtensor_same(a, v, axes)
    elif mode == "valid":
        ct = _corrtensor_valid(a, v, axes)
    if returntype=='callable':
        return lambda v: np.tensordot(ct, v, axes=axes)
    if returntype=='value':
        return np.tensordot(ct, v, axes=axes)
    if returntype=='tensor':
        return ct


def _tensorconvolve(a, v, mode="full", axes=-1, returntype='value'):
    """convolution of two tensors

    Args:
        a (numpy array): the tensor to correlate over
        v (numpy array): the filter tensor
        mode (string): either 'full', 'same', or 'valid'
        axes: the number of axes to correlate. If not given, it is all
            the axes of v.
            tensorconvolve(a,v,axes=n) will convolve
            the last n axes of a with the first n axes of v.
        returntype (string)): 'value' (default) 'callable', or 'tensor'

    Returns:
        The return value depends on returntype. 
        * 'value': returns the convolution of a with v
        * 'callable': returns a function of v which does the correlation
        * 'tensor': returns the tensor which can do the convolution using
          np.tensordot(tensor, v[:,:,-1], axes=..) NB the default axes for tensordot
          is not the same as the default for tensorconvolve
    """
    reverse = tuple([slice(None, None, -1)] * dims(v))
    if returntype == 'value':
        # reverse v
        v = v[reverse]
    result = _tensorcorrelate(a, v, mode=mode, axes=axes, returntype=returntype)
    if returntype=='callable':
        return lambda v: result(v[reverse])
    else:
        return result
