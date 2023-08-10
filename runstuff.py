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
from maxmf.fit import Fit

X = np.random.rand(300, 10)
x = np.random.rand(10) - 0.5
y = X @ x + 0.5 * (np.random.rand(300) - 0.5)


def ls(beta, X, y, **kwargs):
    return -np.sum((y - X @ beta) ** 2)


fitter = Fit(
    lambda x, X, y, wt: ls(x, X, y) - wt * np.sum(np.abs(x)),
    ls,
    wt=15.0,
    initx=np.zeros((10,)),
    fitparams={"maxiters": 500, 'fconv':1e-5},
)

fitter.fit(X, y)
print(fitter.fval_)
print(fitter.score(X, y))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(fitter, X, y, cv=5)
