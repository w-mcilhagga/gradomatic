import numpy as np
from .reverse import (
    gradient_and_hessian,
)  # we use reverse mode by default because its faster


def newton(model, beta, maxiters=20, conv=1e-5, fconv=1e-5):
    """find beta which minimizes a function
    
    Args:
        model : the model function to minimize
        beta: the initial value (a 1d vector)
        maxiters: the maximum number of iterations
        conv: the convergence criterion on beta
        fconv: the convergence criterion on model(beta)
        
    Returns:
        (beta, fval, converged_flag, number_of_iterations)
        where beta minimizes model(beta)
    """
    fval = model(beta)
    converged = False
    jh = gradient_and_hessian(model)
    # newton steps
    for iter in range(maxiters):
        g, h = jh(beta)
        try:
            newbeta = beta + np.linalg.lstsq(h, -g, rcond=None)[0]
            newfval = model(newbeta)
            if newfval > fval or np.isnan(newfval):
                raise RuntimeError("newton failed")
        except (RuntimeError, np.linalg.LinAlgError):
            # something went wrong, try a descent step
            newbeta, newfval = descent(model, beta, g, fval)
        # convergence flags
        better = newfval <= fval
        beta_conv = np.linalg.norm(newbeta - beta) < conv * len(beta)
        f_conv = abs(newfval - fval) < (1 + abs(fval)) * fconv
        # update
        beta, fval = newbeta, newfval
        # check convergence
        if (beta_conv or f_conv) and better:
            converged = True
            break
    return {
        "beta": newbeta,
        "fval": newfval,
        "converged": converged or maxiters == 1,
        "iterations": iter + 1,
    }


def descent(func, beta, grad, fval):
    # quick & dirty descent step for when Newton fails
    # could probably be massively improved
    step = 1  # perfect for quadratic functions
    normg2 = np.dot(grad, grad)
    newfval = func(beta - step * grad)
    # extend step until armijo condition violated
    while (newfval - fval) < -0.5 * step * normg2:  # Armijo condition
        step = 2 * step
        newfval = func(beta - step * grad)
    step = step / 2  # overshot, come back
    newfval = func(beta - step * grad)
    while (newfval - fval) > -0.5 * step * normg2 or np.isnan(newfval):
        step = step / 2
        newfval = func(beta - step * grad)
    return beta - step * grad, newfval
