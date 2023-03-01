# -*- coding: utf-8 -*-
"""
common parameters for all tests
"""

import numpy as np

a = np.random.rand(10, 5)
b = np.random.rand(10, 10)
v = np.random.rand(5)

# for clipping
b_clip = np.random.rand(10, 10)
b_clip[np.logical_and(b > 0.125, b < 0.175)] = 0.125
b_clip[np.logical_and(b > 0.875, b < 0.925)] = 0.925

# all the ufuncs that we have

ufuncs = [
    (np.negative, a),
    (np.negative, v),
    (np.exp, a),
    (np.exp, v),
    (np.log, a+1),
    (np.log, v+1),
    (np.sqrt, a),
    (np.sqrt, v),
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
]

# unary operations. Keywords or non-node parameters are added in by
# wrapping the unary function in a lambda

value = np.random.rand(10)
value2 = np.random.rand(5, 1)

mat1 = np.random.rand(5, 10)
mat2 = np.random.rand(10, 3)
mat3 = np.random.rand(5, 3, 10)

# for clipping
b_clip = np.random.rand(10,8)
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
    (lambda x: np.clip(x, 0.15, 0.9), b_clip),
    (lambda x: x[1:5], b),
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
    (lambda x: np.broadcast_to(x, (2, 5)), v),
    (lambda x: np.broadcast_to(x, (5, 3)), value2),
    (np.transpose, a),
    (np.ravel, a),
    (np.ravel, v),
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
    (lambda a, b: a * b, mat1, value2), # fails
    (lambda a, b: a * b, 5.0, value),
    (lambda a, b: a / b, value, value+1),
    (lambda a, b: a / b, mat1, mat1+1),
    (lambda a, b: a / b, mat1, value+1),
    (lambda a, b: a / b, mat1, value2+1), # fails
    (lambda a, b: a / b, 5.0, value+1),
    (lambda a, b: a**b, value, value),
    (lambda a, b: a**b, mat1, mat1),
    (lambda a, b: a**b, mat1, value),
    (lambda a, b: a**b, mat1, value2), # fails
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
        np.random.rand(10),
        np.random.rand(10),
    ),
    (
        lambda a, b: np.tensordot(a, b, axes=0),
        np.random.rand(10),
        np.random.rand(10),
    ),
    (
        lambda a, b: np.tensordot(a, b, axes=1),
        np.random.rand(10, 5),
        np.random.rand(5, 10),
    ),
    (
        lambda a, b: np.tensordot(a, b, axes=2),
        np.random.rand(6, 10, 5),
        np.random.rand(10, 5, 3),
    ),
    (
        lambda a, b: np.tensordot(a, b, axes=([1, 0], [0, 1])),
        np.arange(60.0).reshape(3, 4, 5),
        np.arange(24.0).reshape(4, 3, 2),
    ),
]
