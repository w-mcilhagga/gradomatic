# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:26:16 2023

@author: willi
"""

# optimization test functions
# These have MINIMA, so use minimize

import numpy as np

def sphere(x):
    # min = (0,0,...0)
    return np.sum(x**2)

def rosenbrock(x):
    # min = (0,0,...0) because add 1 to the vector
    x = x+1.0
    return np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)

def absfn(x):
    return np.sum(np.abs(x))

def sumsquares(x):
    # min = (0,0,...0)
    i = np.arange(len(x))+1
    return np.sum(i*x**2)

def zakharov(x):
    # min = (0,0,...0)
    i = np.arange(len(x))+1
    return np.sum(x**2)+np.sum(0.5*i*x)**2.0+np.sum(0.5*i*x)**4.0

def sumpwr(x):
    # min = (0,0,...0)
    i = np.arange(len(x)) + 1
    return np.sum(np.abs(x) ** i)
    
def gauss(x, sd=1):
    # min = (0,0,...0)
    return 1-np.exp(-np.sum((x/sd)**2))

funclist = [
    (sphere, 1e-8, 1e-5),
    (rosenbrock, 1e-8, 1e-5),
    (absfn, 1e-8, 1e-5),
    (sumsquares, 1e-8, 1e-5),
    (zakharov, 1e-8, 1e-5),
    (sumpwr, 1e-15, 1e-3), # very difficult to do this accurately
    (gauss, 1e-8, 1e-5),
]