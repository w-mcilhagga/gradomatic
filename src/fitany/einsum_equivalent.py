#!/usr/bin/env python
# coding: utf-8

# # `einsum` equivalents
#
# The code returns einsum equivalents for numpy functions.


import numpy as np

indices = "ijklmnopqrstuvwxyz"

equivalents = {}


def equivalent_of(f):
    def dec(g):
        equivalents[f] = g
        return g

    return dec


def getindices(*dims):
    # returns distinct index subsets for dimensions
    start = 0
    out = []
    for d in dims:
        out.append(indices[start : start + d])
        start += d
    return out if len(out) > 1 else out[0]


@equivalent_of(np.dot)
def dotscript(a, b, **kwargs):
    # einsum script for numpy.dot. We can be sure that
    # at least one of the arguments has a .ndim attribute
    adim = getattr(a, "ndim", 0)
    bdim = getattr(b, "ndim", 0)
    ai, bi = getindices(adim, bdim)
    if adim == 0 or bdim == 0:
        return ai + "," + bi + "->" + ai + bi, a, b
    last = bi if len(bi) < 2 else bi[-2]
    bi = bi.replace(last, ai[-1])
    return ai + "," + bi + "->" + (ai + bi).replace(ai[-1], ""), a, b


@equivalent_of(np.matmul)
def matmulscript(a, b, **kwargs):
    # einsum script for numpy.matmul. We can be sure that
    # all the arguments have .ndim
    adim = a.ndim
    bdim = b.ndim
    if adim <= 2 and bdim <= 2:
        ai, bi = getindices(adim, bdim)
        last = bi if len(bi) < 2 else bi[-2]
        bi = bi.replace(last, ai[-1])
        return ai + "," + bi + "->" + (ai + bi).replace(ai[-1], ""), a, b
    # otherwise "stacks of matrices"
    # first indices are the same, last two are altered
    ai, b_end = getindices(adim, 1)
    bi = ai[0:-2] + ai[-1] + b_end
    result = ai[0:-1] + bi[-1]
    return ai + "," + bi + "->" + result, a, b


@equivalent_of(np.multiply)
def multiplyscript(a, b, **kwargs):
    # einsum script for numpy.multiply
    adim = getattr(a, "ndim", 0)
    bdim = getattr(b, "ndim", 0)
    if np.isscalar(a) or np.isscalar(b):
        # dealt with by multiply
        return None
    if adim == 0 or bdim == 0:
        # this has to be an einsum in case a or b is a Node
        ai, bi = getindices(adim, bdim)
        return ai + "," + bi + "->" + ai + bi, a, b
    ai = getindices(adim)
    if adim == bdim:
        ash = np.array(a.shape)
        bsh = np.array(b.shape)
        # it might be better to broadcast the objects rather than 
        # use an einsum, as the einsum doesn't work for reverse mode
        # when some of the dimensions are zero.
        if np.all(
            np.logical_or(ash == bsh, np.logical_or(ash == 1, bsh == 1))
        ):
            return ai + "," + ai + "->" + ai, a, b
        else:
            raise ValueError("operands could not be broadcast together")
    # broadcasting required
    if adim > bdim:
        ai = result = getindices(adim)
        bi = ai[-adim + 1 :]
    else:
        bi = result = getindices(bdim)
        ai = bi[:adim]
    return ai + "," + bi + "->" + result, a, b


@equivalent_of(np.sum)
def sumscript(a, axis=None, keepdims=np._NoValue):
    adim = getattr(a, "ndim", 0)
    if adim == 0:
        return "", a
    ai = out = getindices(adim)
    if axis is None:
        axis = [*range(adim)]
    if type(axis) not in [tuple, list]:
        axis = (axis,)
    oldout = out
    lhs = [ai]
    args = [a]
    if keepdims == True:
        available = sorted(set(indices) - set(ai))
    for i, ax in enumerate(axis):
        if ax < 0:
            ax = ax + a.ndim
        if keepdims == True:
            out = out.replace(oldout[ax], available[i])
            lhs.append(available[i])
            args.append(np.ones((1,)))
        else:
            out = out.replace(oldout[ax], "")
    return ",".join(lhs) + "->" + out, *args


@equivalent_of(np.inner)
def innerscript(a, b, **kwargs):
    adim = getattr(a, "ndim", 0)
    bdim = getattr(b, "ndim", 0)
    ai, bi = getindices(adim, bdim)
    if adim == 0 or bdim == 0:
        return ai + "," + bi, a, b  # no -> needed
    # otherwise multiply & sumover the last dimension of a & b
    bi = bi[0:-1] + ai[-1]
    result = ai[0:-1] + bi[0:-1]
    return ai + "," + bi + "->" + result, a, b


@equivalent_of(np.outer)
def outerscript(a, b, **kwargs):
    # np.outer always returns an array, so watch out
    # for this when just two scalars
    a = np.ravel(a)
    b = np.ravel(b)
    adim = getattr(a, "ndim", 0)
    bdim = getattr(b, "ndim", 0)
    ai, bi = getindices(adim, bdim)
    return "i,j->ij", a, b


@equivalent_of(np.transpose)
def transposescript(a, axes=None):
    # transpose will transpose a scalar to an array
    # so einsum not completely equivalent
    if axes is None:
        axes = range(a.ndim)[::-1]
    adim = getattr(a, "ndim", 0)
    if adim == 0:
        return "", a
    ai = getindices(adim)
    result = ""
    for ax in axes:
        result += ai[ax]
    return ai + "->" + result, a


@equivalent_of(np.tensordot)
def tensordotscript(a, b, axes=2):
    # will assume that a & b have dimensions
    # i.e. not scalar
    ai, bi = getindices(a.ndim, b.ndim)
    if type(axes) is int:
        # sum over last n of a and first n of n
        if axes == 0:
            result = ai + bi
        else:
            bi = ai[-axes:] + bi[axes:]
            result = ai[:-axes] + bi[axes:]
    if type(axes) in [list, tuple]:
        a_axes, b_axes = axes
        result = ai + bi
        for ax, bx in zip(a_axes, b_axes):
            result = result.replace(ai[ax], "")
            result = result.replace(bi[bx], "")
            bi = bi.replace(bi[bx], ai[ax])
    return ai + "," + bi + "->" + result, a, b


@equivalent_of(np.broadcast_to)
def broadcastscript(a, shape, **kwargs):
    # equivalent of broadcast_to, assumes shape
    # is a valid broadcast target
    adim = getattr(a, "ndim", 0)
    ashape = getattr(a, "shape", ())
    ai, si, subs = getindices(adim, len(shape), len(shape))
    extras = []
    args = []
    # work from back to front
    ashape = ashape[::-1]
    shape = shape[::-1]
    ai = ai[::-1]
    si = si[::-1]
    for i in range(len(ashape)):
        if ashape[i] == shape[i]:
            # these dimensions are copied over
            ai = ai.replace(ai[i], si[i])
        else:
            # create a new dimension for ashape==1
            extras.append(si[i])
            args.append(np.ones((shape[i],)))
    for i in range(len(ashape), len(shape)):
        # create new dimensions
        extras.append(si[i])
        args.append(np.ones((shape[i],)))
    ai = ai[::-1]
    si = si[::-1]
    return ",".join([ai, *extras]) + "->" + si, a, *args


# ## The lookup function


def einsum_equivalent(fn, *args, **kwargs):
    try:
        return equivalents[fn](*args, **kwargs)
    except:
        return None
