# -*- coding: utf-8 -*-
"""
Fit API tryout
"""


# fit object api


class Fit:
    def __init__(
        self,
        modelfn=None,
        scorefn=None,
    ):
        """create a model fitting object

        Args:
            modelfn: a model function for fitting (penalized maybe)
            scorefn: a scoring function (or hyperparams for the model)

        Notes:
            The model function has a signature
                model(beta, *data **hyperparams)
            where
                beta is a tensor of values to fit
                data is a list of data values
                hyperparams is a set of hyper parameters
        """
        pass

    def set_params(self, **kwargs):
        pass

    def fit(self, *data, **hyper):
        """hyperparameters are both for minimization & the model function"""
        pass

    def score(self, *data, **hyper):
        """returns the value of scorefn"""
        pass
