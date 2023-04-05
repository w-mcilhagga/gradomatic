'''minimization and maximization of functions.

Minimization uses Newton's method with autodiff created gradients and hessians.
However, if the gradient is a subgradient, then minimization just uses gradient
descent, as the function is probably not smooth enough for Newton.

The function to be minimized must have a single numpy array as a parameter. If not,
wrap it in a lambda. 

Example:
```
def f(x):
    return np.sum(x**2)

result = minimize(f, np.random.rand(10))
# result[0] should be zeros
```

With more than one parameter:
```
def f(x, p):
    return np.sum(x**p)

result = minimize(lambda x: f(x,2), np.random.rand(10))
# result[0] should be zeros
```

'''
import numpy as np
from .autodiff.reverse import (
    Diff,
)  # we use reverse mode by default because its faster
from .autodiff.subgrad import Subgrad


def minimize(model, beta, maxiters=20, gconv=1e-8, xconv=0, fconv=0, method=None):
    """find beta which minimizes a function

    Args:
        model (callable[numpy array]): the model function to minimize, having
            a single parameter beta.
        beta (numpy array): the initial value (a 1d vector)
        maxiters (int): the maximum number of iterations
        gconv (float): the convergence criterion for the gradient
        xconv (float): the convergence criterion on beta
        fconv (float): the convergence criterion on f = model(beta)
        method (callable): the minimization method, either newton_descent or
                subgrad_descent. If none, it is picked automatically

    Returns:
        A tuple `(beta, fval, converged_flag, number_of_iterations)`
        where `beta` minimizes `model(beta)`

    Notes:
        This chooses newton's method or subgradient descent,
        depending on whether the first gradient us a subgradient or not.
        Subgradients are generated when np.abs is used and
        autodiff.reverse.use_subgrad == True
    """
    model = Diff(model)
    fval = model.trace(beta)
    grad = model.jacobian()
    # choose the minimization routine based on the grad type
    if method is None:
        method = (
            subgrad_descent if isinstance(grad, Subgrad) else newton_descent
        )
    return method(model, beta, fval, grad, maxiters, gconv, xconv, fconv)


def maximize(model, beta, maxiters=20, gconv=1e-8, xconv=0, fconv=0, method=None):
    """find beta which maximizes a function

    Args:
        model (callable[numpy array]): the model function to minimize
        beta (numpy array): the initial value (a 1d vector)
        maxiters (int): the maximum number of iterations
        gconv (float): the convergence criterion for the gradient
        xconv (float): the convergence criterion on beta
        fconv (float): the convergence criterion on f = model(beta)
        method (callable): the minimization method, either newton_descent or
                subgrad_descent. If none, it is picked automatically

    Returns:
        A tuple `(beta, fval, converged_flag, number_of_iterations)`
        where `beta` minimizes `model(beta)`

    Notes:
        This simply calls `minimize(lambda b: -model(b), ...)`. See 
        minimize for further details.
    """
    neg_model = lambda x: -model(x)
    return minimize(neg_model, beta, maxiters, gconv, xconv, fconv, method)

eps = np.sqrt(np.finfo(float).eps)

def newton_descent(model, beta, fval, grad, maxiters, gconv, xconv, fconv):
    """find beta which minimizes a function using Newton's method. Called by
    maximize/minimize 

    Args:
        model: a Diff object containing the function
        beta: the initial value (a 1d vector)
        fval: the value of model(beta)
        grad: the gradient of the model at beta
        maxiters: the maximum number of iterations
        conv: the convergence criterion on beta
        fconv: the convergence criterion on model(beta)

    Returns:
        (beta, fval, converged_flag, number_of_iterations)
        where beta minimizes model(beta)
    """
    converged = False
    message = 'maxiters exceeded'
    for iter in range(maxiters):
        grad = np.asarray(grad)
        if np.max(np.abs(grad))<gconv:
            converged=True
            message = 'gradient small'
            break
        try:
            # try a newton step
            hess = model.hessian()
            delta, _, _, sing = np.linalg.lstsq(hess, -grad, rcond=None)
            if np.max(np.abs(sing))<eps:
                raise RuntimeError('singular')
            newbeta = beta + delta
            newfval = model.trace(newbeta)
            if newfval > fval or np.isnan(newfval):
                # the newton step went wrong somehow,
                # the raised exception will cause a descent step
                raise RuntimeError("newton failed")
        except (RuntimeError, np.linalg.LinAlgError):
            # try a descent step
            newbeta = _descent(model, beta, grad, fval)
            # one needless function evaluation here, in order to
            # trace the computation
            newfval = model.trace(newbeta)
            if np.isnan(newfval):
                raise RuntimeError("descent failed")
        # convergence flags
        better = newfval <= fval
        beta_conv = np.max(np.abs(newbeta - beta)) < xconv
        f_conv = abs(newfval - fval) < (1 + abs(fval)) * fconv
        # update
        beta, fval = newbeta, newfval
        # check convergence
        if beta_conv and better:
            converged = True
            message = 'small difference in coefficients'
            break        
        if f_conv and better:
            converged = True
            message = 'small difference in function value'
            break
        # compute grad for next iteration
        grad = model.jacobian()

    return {
        "fval": newfval,
        "beta": newbeta,
        "grad": grad,
        # maxiters = 1 is for linear regression problems
        "converged": converged or maxiters == 1,
        "message": message,
        "iterations": iter + 1,
    }


def _descent(func, beta, grad, fval, initstep=1.0, extend=True):
    """find `beta+a*grad` which roughly minimizes a function.
    Internal function - don't use

    Args:
        func: a Diff object containing the function
        beta: the initial value (a 1d vector)
        grad: the gradient of the model at beta
        fval: the value of model(beta)

    Returns:
        `newbeta = beta+a*grad` which sufficiently decreases `model(newbeta)`
        within limits set by Armijo conditions
    """
    step = initstep
    normg2 = np.dot(grad, grad)
    newfval = func(beta - step * grad)
    # extend step until armijo condition violated
    if extend:
        while (newfval - fval) < -0.5 * step * normg2:  # Armijo condition
            step = 2 * step
            newfval = func(beta - step * grad)
        step = step / 2  # overshot, come back
    newfval = func(beta - step * grad)
    while (newfval - fval) > -0.5 * step * normg2 or np.isnan(newfval):
        step = step / 2
        newfval = func(beta - step * grad)
    return beta - step * grad


def subgrad_descent(model, beta, fval, grad, maxiters, gconv, xconv, fconv):
    """find beta which minimizes a function using subgradient descent

    Args:
        model: a Diff object containing the function
        beta: the initial value (a 1d vector)
        fval: the value of model(beta)
        grad: the subgradient of the model at beta
        maxiters: the maximum number of iterations
        conv: the convergence criterion on beta
        fconv: the convergence criterion on model(beta)

    Returns:
        (beta, fval, converged_flag, number_of_iterations)
        where beta minimizes model(beta)
    """
    converged = False
    message = 'maxiters exceeded'
    for iter in range(maxiters):
        grad = np.asarray(grad) 
        if np.max(np.abs(grad))<gconv:
            converged=True
            message = 'gradient small'
            break
        newbeta = _descent(model, beta, grad, fval)
        # retract newbeta if it changes sign, assuming 
        # the grad will flip sign.
        signchange = np.sign(beta) != np.sign(newbeta)
        if np.any(signchange):
            b = beta[signchange]
            nb = newbeta[signchange]
            alpha = np.min(np.abs(b) / np.abs(nb - b))
            newbeta = beta + alpha * (newbeta - beta)
        newfval = model.trace(newbeta)
        # xconvergence flags
        better = newfval <= fval
        beta_conv = np.max(np.abs(newbeta - beta)) < xconv
        f_conv = abs(newfval - fval) < (1 + abs(fval)) * fconv
        # update
        beta, fval = newbeta, newfval
        # check convergence
        if beta_conv and better:
            converged = True
            message = 'small difference in coefficients'
            break        
        if f_conv and better:
            converged = True
            message = 'small difference in function value'
            break
        # compute grad for next iteration
        grad = model.jacobian()

    return {
        "fval": newfval,
        "beta": newbeta,
        "grad": grad,
        "converged": converged,
        "message": message,
        "iterations": iter + 1,
    }
