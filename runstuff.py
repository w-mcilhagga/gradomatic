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

import numpy as np
from fitany.autodiff.subgrad import Subgrad
from fitany.autodiff.printtrace import graph_fwd
import fitany.autodiff.forward as fwd

fwd.use_subgrad = True

def lasso(x):
    # min = (0,0,...0)
    return np.sum(np.abs(x))   

ld = fwd.Diff(lasso)
val = ld.trace(np.array([-1.0,0,1]))
j = ld.jacobian()

graph_fwd(ld.fval)
print(ld.j.lo)