# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:23:26 2023

@author: willi
"""
import os
import sys

up = os.path.normpath(os.path.join(os.getcwd(), "src"))
sys.path.append(up)
sys.path.append(os.getcwd())

#import numdifftools as nd
#from fitany.tracing import value_of, dims, shape
import fitany.forward as fwd
import fitany.reverse as rev
    
# now do tests

import numpy as np
    
    
value2 = np.random.rand(10)

mat1 = np.random.rand(10,5)

mul = lambda x: np.sum(np.diff(x, n=2)**2)

d = fwd.Diff(mul)

print(d(value2))
print(d.jacobian())
print(d.hessian())
