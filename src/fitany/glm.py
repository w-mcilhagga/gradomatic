r"""Generalized Linear (and nonlinear) Models and functions to build them. 

## Theory.

The log-likelihood \(L(y,\beta)\) is a function of the observation vector \(y\) and 
the parameters \(\beta\). This can be broken down into a loss function (or error distribution)
\(Loss(y, \mu)\), where \(\mu\) is the expected value of \(y\), and a predictor 
function which computes \(\mu\) from \(\beta\), 

$$
L(y, \beta) = Loss(y,\mu(\beta)) 
$$

The gradient (or score \(s\)) is a vector with entries

$$
s_i = \frac{d L(y, \beta)}{d\beta_i} = \sum_j{\frac{d Loss(y, \mu)}{d\mu_j}\frac{d\mu_j}{d\beta_i}}
$$

This can be written as 

$$
s=DJ
$$

 where \(D\) is a diagonal matrix  
with entries \(D_{j,j}=d Loss(y, \mu)/d\mu_j\) and \(J\) is
the Jacobian for the predictor, with entries \(J_{j,i} = d\mu_j/d\beta_i\).

Instead of computing the Hessian when auto-differentiated, this module computes 
the Fisher information, which has entries

$$
F_{i,j} = E\left( \frac{d L(y, \beta)}{d\beta_i} \frac{d L(y, \beta)}{d\beta_j} \right)
$$

(This is the negative of the expected Hessian.)
Using the loss+predictor decomposition, this can (eventually) be written as

$$
F = J'VJ
$$

where \(V\) is a diagonal matrix with entries \(V_{i,i} = E(d Loss(y, \mu)/d\mu_i)^2\)

Note that standard GLM theory further decomposes the jacobian \(J\), but this isn't
necessary and in fact *not* doing so allows you to use nonlinear predictors.

## Usage.

There are three levels of usage:

### Level 1: Use Common GLMs

If you just want to fit an established generalized linear model (e.g. logistic)
then you would call 

```
maximize(lambda b: logisticGLM(b, y, X), initial_b_value)
```

This module provides linear, logistic, poisson, and negative binomial GLMs.

### Level 2: GLMs with non-standard predictors.

If you want to use an established GLM loss with a non-standard predictor 
function, you create the GLM by invoking the Loss with the given
predictor. For example, a Weibull function could be created with

```
weibull = Binomial(lambda b, X: 1-0.5*np.exp(-(X@b[:-1])**b[-1]))
```

Here the predictor is `lambda b, X: 1-0.5*np.exp(-(X@b[:-1])**b[-1])`.

This GLM could be maximized with

```
maximize(lambda b: weibull(b, y, X), initial_b_value)
```

This module provides Gaussian/Normal, Binomial, Poisson, and Negative Binomial
GLM errors.

### Level 3: New Loss Functions.

If you want to create new error models, you define them using GLM_Builder. 
"""

__pdoc__ = {}

import numpy as np

from .autodiff.tracing import notrace, value_of
from .autodiff import forward as fwd
from .autodiff import reverse as rev

def GLM_Builder(loss, dLdmu, varfunc):
    r"""creates a new Generalized Linear Error model.

    Args:
        loss (callable[y,mu]): the log-likelihood loss function
            'y' is a numpy array of observations
            `mu` is a numpy array of expected values `E(y)`
            
            This returns \(Loss(y,\mu)\)
        dLdmu (callable[y,mu]): the derivative of `loss` with respect
            to `mu`
            
            This returns a vector with entries \(d Loss(y, \mu)/d\mu_j\)
        varfunc (callable[y,mu]): the variance function. Usually, `y` doesn't matter.
            
            This returns a vector with entries \( E(d Loss(y, \mu)/d\mu_i)^2\)
    Returns:
        A factory function for generalized linear models. This function takes a mean
        function for the GLM and returns a GLM likelihood
        function which takes `b`, `y`, and data and returns the log-likelihood.
        
        The mean function (or predictor) takes the parameter vector `b` and any data,
        and returns `E(y)` the expected value of `y` given the parameters.
    """

    def GLM(meanfunc):
        # factory function to create likelihoods with derivatives
        # y is the data and meanfunc is a callable which takes (beta, X)
        # returns a function L(beta, y, X) which is notraced

        @notrace
        def L(b, y, *args, **kwargs):
            return loss(y, meanfunc(b, *args, **kwargs))

        # reverse mode

        @rev.register_deriv(L)
        def dLrev(dYdZ, argno, b, y, *args, **kwargs):
            # function to work out the meanfunc & jacobian then pass onto
            # the actual derivative. This step is necessary to memoize the
            # value of mu and J
            if argno != 0:
                return 0
            mfunc = fwd.Diff(meanfunc, argno=0)
            mu = mfunc.trace(value_of(b), *args, **kwargs)
            J = mfunc.jacobian(gc=True)
            # we have to keep b as a parameter here so that it will be traced when
            # computing higher derivatives.
            # NB dydz is guaranteed to be a scalar
            return dYdZ * _dLrev(b, y, mu, J)

        @notrace
        def _dLrev(b, y, mu, J):
            # works out the derivative of L given mu=meanfunc(X,b) & its jacobian
            # b is just there to make sure it's traced
            return np.einsum("i,i...->...", dLdmu(y, mu), J)

        @rev.register_deriv(_dLrev)
        @notrace
        def dL2rev(dYdZ, argno, b, y, mu, J):
            # this doesn't depend on b so has no higher derivatives
            # works out the hessian of L given y, mu=meanfunc(X,b) and its jacobian
            if argno != 0:
                return 0
            # if J has more than 2 dim, we need to allow for this.
            return -dYdZ @ np.einsum("i,ij,ik->jk", varfunc(y, mu), J, J, optimize=True)

        # forward mode

        @fwd.register_deriv(L)
        def dLfwd(b, y, *args, **kwargs):
            # work out the jacobian & store it
            mfunc = fwd.Diff(meanfunc, argno=0)
            mu = mfunc.trace(value_of(b), *args, **kwargs)
            J = mfunc.jacobian(gc=True)
            return _dLfwd(b, y, mu, J)

        @notrace
        def _dLfwd(b, y, mu, J):
            # works out the derivative of L given mu=meanfunc(X,b) & its jacobian
            return np.einsum("i,i...->...", dLdmu(y, mu), J)

        @fwd.register_deriv(_dLfwd)
        @notrace
        def dL2fwd(b, y, mu, J):
            # this doesn't depend on b so has no higher derivatives
            return -np.einsum("i,ij,ik->jk", varfunc(y, mu), J, J, optimize=True)

        return L

    return GLM


Gaussian = GLM_Builder(
    lambda y, mu: -0.5 * np.sum((y - mu) ** 2),
    lambda y, mu: y - mu,
    lambda y, mu: np.ones(y.shape),
)
__pdoc__[
   "Gaussian"
] = r"""function to build a GLM based on Gaussian or Normal loss.

Args:
    meanfunc (callable[b, args, kwargs]): the mean function which returns `E(y)`
Returns:
    a GLM function which takes `b`, `y`, and `X` and returns the log-likelihood.
Notes:
    The Gaussian loss is \(Loss(y,\mu)=-\sum_i{\frac{(y_i-\mu_i)^2}{2\sigma^2}}\)
    where \(\mu\) us supplied by the predictor. The value of \(\sigma\) is not important
    for parameter estimation in the GLM.
Example:
     `Gaussian(lambda b, X: X@b)` uses a linear mean function and returns a least-squares
     likelihood function. (In this case, however, you are better using linearGLM)
"""

Normal = Gaussian
__pdoc__["Normal"] = """a synonym for the Gaussian function"""

clip = [0.001, 0.999]


def _binomial_funcs():
    def logl(y, mu):
        cmu = np.clip(mu, *clip)
        return np.sum(y * np.log(cmu) + (1 - y) * np.log(1 - cmu))

    def dLdmu(y, mu):
        cmu = np.clip(mu, *clip)
        unclipped = cmu == mu
        return unclipped * (y - cmu) / (cmu * (1 - cmu))

    def varfunc(y, mu):
        cmu = np.clip(mu, *clip)
        unclipped = cmu == mu
        return unclipped / (mu * (1 - mu))

    return logl, dLdmu, varfunc


Binomial = GLM_Builder(*_binomial_funcs())
__pdoc__[
    r"Binomial"
] = r"""function to build a GLM based on Binomial loss.

Args:
    meanfunc (callable[b, args, kwargs]): the mean function which returns `E(y)`
Returns:
    a GLM function which takes `b`, `y`, and `X` and returns the log-likelihood.
Notes:
    The binomial loss is \(Loss(y,\mu)=\sum_i(y_i\log{\mu_i}+(1-y_i)\log{(1-\mu_i)})\)
Example:
     `Binomial(lambda b,X: 1/(1+np.exp(-X@b)))` uses a logit mean function and 
     returns a logistic regression likelihood function. (In this case, however, 
     you are better using logisticGLM)
"""


def _poisson_funcs():
    def logl(y, mu):
        unclipped = mu > 0
        cmu = mu * unclipped
        return np.sum(y * np.log(cmu) - cmu)

    def dLdmu(y, mu):
        unclipped = mu > 0
        return unclipped * (y - mu) / (1e-10 + mu * unclipped)

    def varfunc(y, mu):
        unclipped = mu > 0
        return unclipped / (1e-10 + mu * unclipped)

    return logl, dLdmu, varfunc


Poisson = GLM_Builder(*_poisson_funcs())
__pdoc__[
    "Poisson"
] = """function to build GLMs based on Poisson errors.

Args:
    meanfunc (callable[b, X]): the mean function which returns `E(y)`
Returns:
    a GLM function which takes `b`, `y`, and `X` and returns the log-likelihood.
Example:
     `Poisson(lambda b,X: np.exp(-X@b))` uses an exponential mean function and 
     returns a poisson regression likelihood function. (In this case, however, 
     you are better using poissonGLM)
"""


def negativeBinomial(alpha, meanfunc):
    """factory function to build GLMs based on Negative Binomial errors.

    The parameterization of this is the same as in statsmodels.

    Args:
        alpha (float): the reciprocal of the number of successes.
        meanfunc (callable[X,b]): the mean function which returns `E(y)`
    Returns:
        A factory function for generalized linear models. This function takes a mean
        function for the GLM. (A mean function takes observations `X` and parameters
        `b` and returns the expected value of `y`) and returns a GLM
        function which takes `y`, `X`, and `b` and returns the log-likelihood.
    Example:
        `NegativeBinomial(1/5.0)(lambda X,b:X@b)` uses `alpha=1/5` and a linear
        mean function to return a negative binomial likelihood.


        To use this to fit a negative binomial regression, you would call
        ```
        negbin = NegativeBinomial(1/5, meanfunc)
        maximize(lambda b:negbin(y, X, b), initial_b)
        ```

        The canonical mean function is `1/alpha*np.exp(nu)/(1-np.exp(nu))` where
        `nu = X@b`. This hasn't been implemented.
    """

    def logl(y, mu):
        unclipped = mu > 0
        cmu = mu * unclipped
        amu = alpha * cmu
        return np.sum(
            -(1 / alpha) * np.log(1 + amu) + y * np.log(amu / (1 + amu))
        )

    def dLdmu(y, mu):
        unclipped = mu > 0
        cmu = unclipped * mu
        return unclipped * (y - mu) / (1e-10 + cmu + alpha * cmu)

    def varfunc(y, mu):
        unclipped = mu > 0
        cmu = unclipped * mu
        return 1.0 / (1e-10 + cmu + alpha * cmu)

    return GLM_Builder(logl, dLdmu, varfunc)(meanfunc)


@notrace
def linearMF(b, X):
    """a linear mean function `E(y)=X@b`

    Args:
        b (numpy array): the parameter vector with shape (nparams,)
        X (numpy array): the observation matrix with shape (nparams, nobs)
    Returns:
        `E(y)` the expected value of the observations given `b`.
    Note:
        This is the canonical mean function for Gaussian errors.
    """
    return X @ b


@fwd.register_deriv(linearMF)
@notrace
def _dlinearMF(b, X):
    return X


linearGLM = Gaussian(linearMF)

__pdoc__[
    "linearGLM"
] = """a linear regression model

Args:
    b (numpy array): the parameter vector with shape (nparams,)
    y (numpy array): vector of observations with shape (nobs,)
    *args: just the observation matrix `X` with shape (nparams, nobs)
    **kwargs: not used
Returns:
    the log-likelihood 
Note:
    The log-likelihood is `-0.5*np.sum((y-X@b)**2)`

    
    To use this to fit a linear regression, you would call
    ```
    maximize(lambda b:linearGLM(b, y, X), initial_b)
    ```
"""


@notrace
def logitMF(b, X):
    """a logistic mean function `E(y)=1/(1+np.exp(-X@b))`

    Args:
        b (numpy array): the parameter vector with shape (nparams,)
        X (numpy array): the observation matrix with shape (nparams, nobs)
    Returns:
        `E(y)` the expected value of the observations given `b`.
    Notes:
        This is the canonical mean function for Binomial errors.
    """
    nu = X @ b
    return 1 / (1 + np.exp(-nu))


@fwd.register_deriv(logitMF)
@notrace
def _dlogitMF(b, X):
    exp_nu = np.exp(-X @ b)
    return np.einsum("i,ij->ij", exp_nu / (1 + exp_nu) ** 2, X)


logisticGLM = Binomial(logitMF)

__pdoc__[
    "logisticGLM"
] = """a logistic regression model

Args:
    b (numpy array): the parameter vector with shape (nparams,)
    y (numpy array): vector of observations with shape (nobs,)
    *args: just the observation matrix `X` with shape (nparams, nobs)
    **kwargs: not used
Returns:
    the log-likelihood 
Note:
    The log-likelihood is computed using code equivalent to:
    ```
    mu = 1/(1+np.exp(-X@b))
    return y*np.log(mu)+(1-y)*np.log(1-mu)
    ```
    except that `mu` (the probability) is clipped at 0.001 and 0.999
    
    To use this to fit a logistic regression, you would call
    ```
    maximize(lambda b:logisticGLM(b, y, X), initial_b)
    ```
"""


@notrace
def expMF(b, X):
    """an exponential mean function `E(y)=np.exp(X@b)`

    Args:
        b (numpy array): the parameter vector with shape (nparams,)
        X (numpy array): the observation matrix with shape (nparams, nobs)
    Returns:
        `E(y)` the expected value of the observations given `b`.
    Notes:
        This is the canonical mean function for Poisson errors.
    """
    return np.exp(X @ b)


@fwd.register_deriv(expMF)
@notrace
def _dexpMF(b, X):
    return np.einsum("i,ij->ij", np.exp(X @ b), X)


poissonGLM = Poisson(expMF)

__pdoc__[
    "poissonGLM"
] = """a poisson regression model

Args:
    b (numpy array): the parameter vector with shape (nparams,)
    y (numpy array): vector of observations with shape (nobs,)
    *args: just the observation matrix `X` with shape (nparams, nobs)
    **kwargs: not used
Returns:
    the log-likelihood 
Note:
    The log-likelihood is computed using code equivalent to:
    ```
    mu = np.exp(-X@b)
    return np.sum(y * np.log(mu) - mu)
    ```
    
    To use this to fit a poisson regression, you would call
    ```
    maximize(lambda b:poissonGLM(b, y, X), initial_b)
    ```
"""
