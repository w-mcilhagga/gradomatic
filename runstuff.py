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
from maxmf.autodiff.subgrad import Subgrad
from maxmf.autodiff.printtrace import graph_fwd
import maxmf.autodiff.forward as fwd

fwd.use_subgrad = False

def diff2(x):
    # min = (0,0,...0)
    return np.diff(x, n=2, prepend=[0,0], append=[0,0])

x = np.array([1,2,3,4])
print(diff2(x))
ld = fwd.Diff(diff2)
ld.trace(x)
print(ld.jacobian())