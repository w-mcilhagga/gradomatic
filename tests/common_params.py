# -*- coding: utf-8 -*-
"""
common parameters for all tests
"""


import numpy as np
import scipy.signal as sig
from maxmf.autodiff.convolve import tensorcorrelate, tensorconvolve

ra = lambda *args: np.random.rand(*args)

a = ra(10, 5)
b = ra(10, 10)
v = ra(5)

# for clipping
b_clip = ra(10, 10)
b_clip[np.logical_and(b > 0.125, b < 0.175)] = 0.125
b_clip[np.logical_and(b > 0.875, b < 0.925)] = 0.925

# all the ufuncs that we have

ufuncs = [
    (np.negative, a),
    (np.negative, v),
    (np.exp, a),
    (np.exp, v),
    (np.log, a + 1),
    (np.log, v + 1),
    (np.sqrt, a + 1),
    (np.sqrt, v + 1),
    (np.abs, a),
    (np.abs, v),
    (np.sign, a),
    (np.sign, v),
    (np.cos, a),
    (np.cos, v),
    (np.sin, a),
    (np.sin, v),
    (np.tan, a),
    (np.tan, v),
    (np.max, a),
    (lambda x: np.max(x, axis=0), a),
    (lambda x: np.min(x, axis=0), a),
    (lambda x: np.max(x, axis=1), a),
    (lambda x: np.max(x, axis=0, keepdims=True), a),
    (lambda x: np.min(x, axis=0, keepdims=True), a),
]

# unary operations. Keywords or non-node parameters are added in by
# wrapping the unary function in a lambda

value = ra(10)
value2 = ra(5, 1)
value3 = ra(3)
value4 = ra(4)

v10 = ra(1, 10)
mat1 = ra(5, 10)
stacker = ra(2, 10)
mat2 = ra(10, 3)
mat3 = ra(5, 3, 10)
ssig = ra(15, 10)
sfilt = ra(2, 3)

# for clipping
b_clip = ra(10, 8)
b_clip[np.logical_and(b_clip > 0.125, b_clip < 0.175)] = 0.125
b_clip[np.logical_and(b_clip > 0.875, b_clip < 0.925)] = 0.925

unary = [
    (np.transpose, a),
    (np.diag, a),
    (np.diag, v),
    (np.ravel, a),
    (np.ravel, v),
    (np.diagonal, a),
    (np.diagonal, b),
    (lambda x: np.flip(x, axis=0), mat1),
    (lambda x: np.flip(x, axis=1), mat1),
    (np.fliplr, mat1),
    (np.flipud, mat1),
    (lambda x: np.clip(x, 0.15, 0.9), b_clip),
    (lambda x: x[1:5], b),
    (lambda x: x[::-1], b),
    (lambda x: x[:, 1:5], b),
    (lambda x: x[(1, 2, 3), (4, 5, 6)], b),
    (lambda x: np.sum(x, axis=0), v),
    (lambda x: np.sum(x, axis=0), a),
    (lambda x: np.sum(x, axis=1), a),
    (lambda x: np.sum(x), a),
    (np.trace, a),
    (np.trace, b),
    (lambda x: np.reshape(x, (5, 2)), value),
    (lambda x: np.reshape(x, (150,)), mat3),
    (np.squeeze, ra(1, 3, 1, 45)),
    (lambda x: np.broadcast_to(x, (2, 5)), v),
    (lambda x: np.broadcast_to(x, (5, 3)), value2),
    (lambda x: np.concatenate((v10, x, mat2.T), axis=0), mat1),
    (lambda x: np.concatenate((v10, x), axis=1), v10),
    (lambda x: np.concatenate((v10, x, value), axis=None), mat1),
    (lambda x: np.vstack((value, x, stacker)), mat1),
    (lambda x: np.vstack((x[0, :], x, stacker)), mat1),
    (lambda x: np.vstack((value, x**2, stacker)), mat1),
    (lambda x: np.diff(x, axis=0), mat1),
    (lambda x: np.diff(x, axis=0, prepend=0.0), mat1),
    (lambda x: np.diff(x, axis=1), mat1),
    (lambda x: np.diff(x, n=2, axis=1), mat1),
    (lambda x: np.correlate(value, x), value3),
    # (lambda x: np.correlate(x, x), value3), fails due to numdifftools
    (lambda x: np.convolve(value, x), value3), 
    (lambda x: np.correlate(value, x), value4),
    (lambda x: np.convolve(value, x), value4),
    (lambda x: tensorcorrelate(mat1, x), value3),
    (lambda x: tensorconvolve(mat1, x), value3),
    (lambda x: sig.correlate(ssig, x), sfilt),
    (lambda x: sig.convolve(ssig, x), sfilt),
    (lambda x: np.maximum(x, 0.5), value),
    (lambda x: np.minimum(x, 0.5), value),    
    (lambda x: np.maximum(x, v10.flatten()), value),
    (lambda x: np.minimum(x, v10.flatten()), value),
    (lambda x: np.maximum(x, 1-x), value),
    (lambda x: np.minimum(x, 1-x), value),
]

binary = [
    (np.dot, 3, value),
    (np.dot, value, value),
    (np.dot, mat1, value),
    (np.dot, mat1, mat2),
    (np.dot, mat3, mat2),
    (np.inner, 3, value),
    (np.inner, value, value),
    (np.inner, mat1, value),
    (np.inner, mat1, mat3),
    (np.outer, value, value),
    (np.outer, mat1, mat1),
    (np.outer, 5.0, value),
    # operators are tested inside a function
    # rather than calling np.add, np.matmul, etc.
    (lambda a, b: a + b, value, value),
    (lambda a, b: a + b, mat1, mat1),
    (lambda a, b: a + b, mat1, value),
    (lambda a, b: a + b, 5.0, value),
    (lambda a, b: a - b, value, value),
    (lambda a, b: a - b, mat1, mat1),
    (lambda a, b: a - b, mat1, value),
    (lambda a, b: a - b, 5.0, value),
    (lambda a, b: a * b, value, value),
    (lambda a, b: a * b, mat1, mat1),
    (lambda a, b: a * b, mat1, value),
    (lambda a, b: a * b, mat1, value2),  # fails
    (lambda a, b: a * b, 5.0, value),
    (lambda a, b: a / b, value, value + 1),
    (lambda a, b: a / b, mat1, mat1 + 1),
    (lambda a, b: a / b, mat1, value + 1),
    (lambda a, b: a / b, mat1, value2 + 1),  # fails
    (lambda a, b: a / b, 5.0, value + 1),
    (lambda a, b: a**b, value, value),
    (lambda a, b: a**b, mat1, mat1),
    (lambda a, b: a**b, mat1, value),
    (lambda a, b: a**b, mat1, value2),  # fails
    (lambda a, b: a**b, 5.0, value),
    (lambda a, b: a**b, 1, value),
    (lambda a, b: a @ b, value, mat2),
    (lambda a, b: a @ b, mat1, value),
    (lambda a, b: a @ b, mat1, mat2),
    (
        lambda a, b: a @ b,
        np.arange(2 * 2 * 4).reshape((2, 2, 4)),
        np.arange(2 * 2 * 4).reshape((2, 4, 2)),
    ),
    (
        lambda a, b: np.tensordot(a, b, axes=1),
        ra(10),
        ra(10),
    ),
    (
        lambda a, b: np.tensordot(a, b, axes=0),
        ra(10),
        ra(10),
    ),
    (
        lambda a, b: np.tensordot(a, b, axes=1),
        ra(10, 5),
        ra(5, 10),
    ),
    (
        lambda a, b: np.tensordot(a, b, axes=2),
        ra(6, 10, 5),
        ra(10, 5, 3),
    ),
    (
        lambda a, b: np.tensordot(a, b, axes=([1, 0], [0, 1])),
        np.arange(60.0).reshape(3, 4, 5),
        np.arange(24.0).reshape(4, 3, 2),
    ),
]

# Integration tests - just a bunch of more complex functions
X = ra(30, 10)
y = ra(30)
b = ra(10) + 1
A = ra(10, 10)
AA = ra(4, 4)


def LL(x):
    # binomial log-likelihood
    p = 1 / (1 + np.exp(-X @ x))
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def lr(b):
    # linear regression
    r = y - X @ b
    return np.dot(r, r)


jacfns = [
    (lambda x: 1, b),  # this returns 0 in jacobian
    (lambda x: x, b),
    (lambda x: 1 / (1 + x), b),
    (lambda x: 10 * x + 3, b),
    (lambda x: 2 ** (A @ x + 1), b),
    (lambda x: 1 / (1 + np.dot(x, x)), b),
    (lambda x: A @ x, b),
    (lambda x: X @ x, b),
    (lambda x: x @ A @ x, b),
    (lambda x: np.einsum("i,ij,j->", x, A, x), b),
    (lambda x: 1 / (1 + np.sum(x)), b),
    (lambda x: np.cos(np.sin(np.log(np.exp(np.sum(x))))), b),
    (lambda x: (x @ A @ x) / (1 + np.dot(x, x)), b),
    (LL, b),
    (lr, b),
    (lambda x: (1 - np.exp(-x)) / (1 + np.exp(-x)), b),
    (lambda x: np.sum(np.ones((10,)) * A @ x), b),
    (lambda x: np.dot(x[0:3], x[3:6]), b),
    (lambda x: AA @ x[0:4], b),
    (lambda x: x[0:4] @ AA @ x[4:8], b),
]

def pr(x):
    return 1/(1+np.exp(-X@x))

hessfns = [
    (lambda x: 1, b),
    (lambda x: 1.0 / np.sum(x), b),
    (lambda x: np.sum(x**2), b),
    (lambda x: x @ A @ x, b),
    (lambda x: 1 / (1 + np.sum(x)), b),
    (lambda x: np.cos(np.sin(np.log(np.exp(np.sum(x))))), b),
    (lambda x: (x @ A @ x) / (1 + np.dot(x, x)), b),
    (lambda x: np.dot(x, x), b),
    (np.sum, b),
    (lambda x: np.sum(np.exp(x)), b),
    (lambda x: np.sum(A @ x), b),
    (lambda x: np.sum(x @ A @ x), b),
    (lambda x: np.einsum("i,ij,j->", x, A, x), b),
    (lambda x: np.sum(pr(x)), b),
    (LL, b),
    (lr, b),
    (lambda x: np.dot(x[0:3], x[3:6]), b),
    (lambda x: x[0:4] @ AA @ x[4:8], b),
]
