# -*- coding: utf-8 -*-
"""
Fit API tryout
"""

import inspect
import numpy as np

# fit object api


def penalized(model, penalty):
    # returns a penalized model
    def penmodel(X, y, beta, **kwargs):
        kwargs = {**kwargs}
        penwt = kwargs["penaltywt"]
        del kwargs["penaltywt"]
        return model(X, y, beta, **kwargs) + penwt * penalty(beta)

    return penmodel

def lasso(model, wt=1.0, index=None):
    if index is None:
        return penalized(model, lambda x:np.sum(np.abs(x)*wt))
    else:
        return penalized(model, lambda x:np.sum(np.abs(x[index])*wt))

def ridge(model, wt=1.0, index=None):
    if index is None:
        return penalized(model, lambda x:np.sum(x**2*wt))
    else:
        return penalized(model, lambda x:np.sum(x[index]**2*wt))

class Fit:
    def __init__(
        self,
        model=None,
        maxiters=20,
        conv=1e-5,
        fconv=1e-5,
        method=None,
        **kwargs
    ):
        """create a model fitting object

        Args:
            model: a model function
            maxiters: the maximum number of iterations for fitting
            conv: the convergence criterion on beta for fitting
            fconv: the convergence criterion on model(beta)
            method: the minimization method, either newton_descent or
                subgrad_descent. If none, it is picked automatically
            **hyper: hyperparameters for the model, if any

        Notes:
            The model function has a signature
                model(X, y, beta, **hyperparams)
            where
                X is a tensor of features
                y is a tensor of observations
                beta is a tensor of values to fit
                hyperparams is a set of hyper parameters

            The first axis of y and X indexes the obsevations,
            so len(X)==len(y).

            The hyperparameters are for things like penalties,
            which can be varied by grid search. Default values *must*
            be supplied and will be used when scores are needed.
            
            That's actually a terrible idea, some of the hyperparams
            might be needed to get the score. 
        """
        self.model = model
        # get default-valued hyperparams from inspection
        modelparams = inspect.signature(model).parameters
        self.hyperkeys = []
        for k in modelparams.keys():
            if modelparams[k].default != inspect._empty:
                self.hyperkeys.append(k)
        self.method = method
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            # filter out invalid keys
            setattr(self, k, v)

    def fit(self, X, y, **hyper):
        """find beta to minimize model(y,X,beta, **hyper)

        Args:
            X: tensor of explanatory variables
            y: tensor of observations
            hyper: named hyperparameters, if any

        Note:
            the hyperparameters are blended with existing hyperparameters
        """
        self.set_params(**hyper)
        hyper = {}
        for k in self.hyperkeys:
            hyper[k] = getattr(self, k)
        func = lambda _coef: self.model(X, y, _coef, **hyper)
        pass

    def score(self, X, y, **hyper):
        """returns the value of model(X,y,self.coef_)"""
        # the model score always uses default hyperparams
        return self.model(X, y, self.coef_)
