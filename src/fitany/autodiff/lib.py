# -*- coding: utf-8 -*-
"""
matrix creation for diff & convolve
"""

import numpy as np
from scipy.linalg import toeplitz

# misc routines for tracing


def diffmat(d, n=1, pad=False, npcompat=True):
    # creates a diff matrix for n fold diff, padding with zeros
    #
    # d : size of axis to differentiate
    # n: number of times to differentiate
    # pad: pad with zeros
    # npcompat: if True, pads with zeros once (numpy behaviour when pre/append=0.0).
    #    If false, pads every time the derivative is taken.
    d_impulse = np.diff(np.array([*[0] * n, 1.0, *[0] * n]), n=n)
    col = np.zeros((d + n,))
    col[: n + 1] = d_impulse
    row = np.zeros((d,))
    row[0] = col[0]
    dmat = toeplitz(col, row)
    if not pad:
        dmat = dmat[n:d]
    elif npcompat:
        # we have to remove some rows of dmat
        dmat = dmat[n - 1 : dmat.shape[0] - (n - 1)]
    return dmat


def np_correlate(a, v, mode='valid'):
    # pad a if mode is same or full
    vlen = len(v)
    if mode=='full':
        pad = [0.0]*(vlen-1)
        pad2 = [0.0]*(vlen)
        a = np.hstack( (pad, a, pad))
    if mode=='same':
        pad1 = [0.0]*(vlen//2)
        pad2 = [0.0]*(vlen-vlen//2-1)
        a = np.hstack( (pad1, a, pad2))
    result = []
    for i in range(len(a)-len(v)+1):
        result.append(np.reshape(np.sum(a[i:i+vlen]*v),(1,)))
    # doesn't work in tracing.py because concatenate doesn't like it
    print(result[0], getattr(result[0], 'value', ''))
    return np.hstack(result)

def np_convolve(a, v, mode='full'):
    return np_correlate(a, v[::-1], mode=mode)