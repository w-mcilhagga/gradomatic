# -*- coding: utf-8 -*-
"""
Fit API tryout
"""

# fit object api
import warnings
import numpy as np
from .minimize import maximize, minimize


class Fit:
    def __init__(
        self, f, scoref=None, initx=None, maximize=True, fitparams={}, **kwargs
    ):
        """create a model fitting object following the scikit learn API

        Args:
            f: a model function for fitting (penalized maybe)
            scoref: a scoring function
            initx: a method for working out the initial value or a fixed vector
            maximize: whether to maximize or minimize
            kwargs: keyword args for f or maximization

        Notes:
            The model and score functions have a signature
                model(x, X, y, **hyperparams)
            where
                x is a vector of values to fit
                X, y are tensors with 1st dimension = n_samples
                hyperparams is a set of hyper parameters. The score
                function can ignore it.

        """
        self.f = f
        self.scoref = scoref
        self.initx = initx
        self.maximize = maximize
        self.fitparams = fitparams
        self.coreparams = [
            "f",
            "scoref",
            "initx",
            "maximize",
            "fitparams",
            "coreparams",
        ]
        # copy the rest
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True, remove=[]):
        # get all name:value pairs that aren't set by fit
        items = [i for i in [*self.__dict__.items()] if i[0][-1] != "_"]
        # filter out remove keys
        items = [i for i in items if i[0] not in remove]
        return dict(items)

    def fit(self, X, y, **fitparams):
        """hyperparameters are both for minimization & the model function"""
        hyper = self.get_params(remove=self.coreparams)
        mf = lambda x: self.f(x, X, y, **hyper)
        # filter the kwargs for any fitting values
        fitter = maximize if self.maximize else minimize
        if callable(self.initx):
            x0 = self.initx(X)
        elif self.initx is None:
            x0 = np.random.rand(X.shape[-1]) - 0.5
        else:
            x0 = self.initx
        result = fitter(
            mf, x0, **{**{"report": -1}, **self.fitparams, **fitparams}
        )
        self.coef_ = result["x"]
        self.n_iter_ = result["iterations"]
        self.fval_ = result["fval"]
        self.converged_ = result["converged"]
        # if not converged
        if not self.converged_:
            warnings.warn(f"not converged after {self.n_iter_} iterations")
        return self

    def score(self, X, y):
        """returns the value of scoref"""
        hyper = self.get_params(remove=self.coreparams)
        return (self.scoref or self.f)(self.coef_, X, y, **hyper)
