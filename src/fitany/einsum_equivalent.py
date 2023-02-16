#!/usr/bin/env python
# coding: utf-8

# # `einsum` equivalents 
# 
# The code returns einsum equivalents for numpy functions.


import numpy as np

indices = 'ijklmnopqrstuvwxyz'

def getindices(*dims):
    # returns distinct index subsets for dimensions
    start = 0
    out = []
    for d in dims:
        out.append(indices[start:start+d])
        start += d
    return out if len(out)>1 else out[0]


def dotscript(a, b):
    # einsum script for numpy.dot. We can be sure that
    # at least one of the arguments has a .ndim attribute
    adim = getattr(a,'ndim', 0)
    bdim = getattr(b,'ndim', 0)
    ai, bi = getindices(adim, bdim)
    if adim==0 or bdim==0:
        return ai+','+bi+'->'+ai+bi, a, b
    last = bi if len(bi)<2 else bi[-2]
    bi = bi.replace(last, ai[-1])
    return ai+','+bi+'->'+(ai+bi).replace(ai[-1],''), a, b


def matmulscript(a, b):
    # einsum script for numpy.matmul. We can be sure that
    # all the arguments have .ndim
    adim = a.ndim
    bdim = b.ndim
    if adim<=2 and bdim<=2:
        ai, bi = getindices(adim, bdim)
        last = bi if len(bi)<2 else bi[-2]
        bi = bi.replace(last, ai[-1])
        return ai+','+bi+'->'+(ai+bi).replace(ai[-1],''), a, b
    # otherwise "stacks of matrices"
    # first indices are the same, last two are altered
    ai, b_end = getindices(adim, 1)
    bi = ai[0:-2]+ai[-1]+b_end
    result = ai[0:-1]+bi[-1]
    return ai+','+bi+'->'+result, a, b


def multiplyscript(a,b):
    # einsum script for numpy.multiply 
    adim = getattr(a,'ndim', 0)
    bdim = getattr(b,'ndim', 0)
    if (np.isscalar(a) or np.isscalar(b)):
        return None
    if adim==0 or bdim==0:
        # this has to be an einsum in case a or b is a Node
        ai, bi = getindices(adim, bdim)
        return ai+','+bi+'->'+ai+bi, a, b
    ai = getindices(adim)
    if adim == bdim and a.shape==b.shape:
        return ai+','+ai+'->' +ai, a, b
    # broadcasting required
    if adim>bdim:
        ai = result = getindices(adim)
        bi = ai[-adim+1:]
    else:
        bi = result = getindices(bdim)
        ai = bi[:adim]
    return ai+','+bi+'->' +result, a, b


def sumscript(a, axis=None, keepdims=np._NoValue):
    adim = getattr(a,'ndim', 0)
    if adim==0:
        return '', a
    ai = out = getindices(adim)
    if axis is None:
        axis = [*range(adim)]
    if type(axis) not in [tuple, list]:
        axis = (axis,)
    oldout = out
    lhs = [ai]
    args = [a]
    if keepdims == True:
        available = sorted(set(indices)-set(ai))
    for i, ax in enumerate(axis):
        if ax<0: ax = ax+a.ndim
        if keepdims == True:
            out = out.replace(oldout[ax],available[i])
            lhs.append(available[i])
            args.append(np.ones((1,)))
        else:
            out = out.replace(oldout[ax],'')
    return ','.join(lhs)+'->'+out, *args


def innerscript(a,b):
    adim = getattr(a,'ndim', 0)
    bdim = getattr(b,'ndim', 0)
    ai, bi = getindices(adim, bdim)
    if adim==0 or bdim==0:
        return ai+','+bi, a, b # no -> needed
    # otherwise multiply & sumover the last dimension of a & b
    bi = bi[0:-1]+ai[-1]
    result = ai[0:-1]+bi[0:-1]
    return ai+','+bi+'->'+result, a, b


def outerscript(a,b):
    # np.outer always returns an array, so watch out
    # for this when just two scalars
    adim = getattr(a,'ndim', 0)
    bdim = getattr(b,'ndim', 0)
    # need to flatten the arrays if adim or bdim>1
    ai, bi = getindices(adim, bdim)
    if adim==0 or bdim==0:
        # the shape changes if a or b is a vector,
        # which requires a reshape operation,
        # so not entirely an einsum thing here
        return ai+','+bi+'->'+ai+bi, a, b
    # if these are matrices, they must be ravelled, but this
    # isn't implemented.
    if adim>1:
        pass
    if bdim>1:
        pass
    return 'i,j->ij', a, b


def tracescript(a, axis1=0, axis2=1, **kwargs):
    adim = getattr(a,'ndim', 0)
    if adim==0:
        return '', a # fails in np.trace
    ai = getindices(adim)
    ai = ai.replace(ai[axis1], ai[axis2])
    result = ai.replace(ai[axis2], '')
    return ai+'->'+result, a

def diagscript(a, **kwargs):
    adim = getattr(a,'ndim', 0)
    if adim==0:
        return '' # fails in np.trace
    return 'ii->i', a


def transposescript(a, axes=None):
    # transpose will transpose a scalar to an array
    # so einsum not completely equivalent
    if axes is None:
        axes = range(a.ndim)[::-1]
    adim = getattr(a,'ndim', 0)
    if adim==0:
        return '', a
    ai = getindices(adim)
    result = ''
    for ax in axes:
        result += ai[ax]
    return ai+'->'+result, a


def tensordotscript(a, b, axes=2):
    # will assume that a & b have dimensions 
    # i.e. not scalar
    ai, bi = getindices(a.ndim, b.ndim)
    if type(axes) is int:
        bi = ai[-axes:]+bi[axes:]
        result = ai[:-axes]+bi[axes:]
    if type(axes) in [list, tuple]:
        a_axes, b_axes = axes
        result = ai+bi
        for ax, bx in zip(a_axes, b_axes):
            result = result.replace(ai[ax], '')          
            result = result.replace(bi[bx], '')          
            bi = bi.replace(bi[bx], ai[ax])
    return ai+','+bi+'->'+result, a, b


def broadcastscript(a, shape):
    # equivalent of broadcast_to, assumes shape 
    # is a valid broadcast target
    adim = getattr(a,'ndim', 0)
    ashape = getattr(a, 'shape', ())
    ai, si, subs = getindices(adim, len(shape), len(shape))
    extras = []
    args = []
    # work from back to front
    ashape = ashape[::-1]
    shape = shape[::-1]
    ai = ai[::-1]
    si = si[::-1]
    for i in range(len(ashape)):
        if ashape[i]==shape[i]:
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
    return ','.join([ai, *extras])+'->'+si, a, *args

# ## The lookup function


def einsum_equivalent(fn, *args, **kwargs):
    # finds the einsum equivalent, returns
    # the script and args (no kwargs)
    # a dict lookup is not used because the numpy functions
    # *might* be overloaded.
    if fn==np.dot:
        return dotscript(*args)
    if fn==np.matmul:
        return matmulscript(*args)
    if fn==np.multiply:
        return multiplyscript(*args)   
    if fn==np.sum:
        return sumscript(*args, **kwargs)   
    if fn==np.inner:
        return innerscript(*args)   
    if fn==np.outer:
        return outerscript(*args)   
    if fn==np.trace:
        return tracescript(*args, **kwargs)   
    if fn==np.diag:
        return diagscript(*args)   
    if fn==np.transpose:
        return transposescript(*args, **kwargs)
    if fn==np.tensordot:
        return tensordotscript(*args, **kwargs)
    # if no equivalent,
    return None







