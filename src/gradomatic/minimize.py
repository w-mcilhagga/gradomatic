"""minimization and maximization of functions.

Minimization uses Newton's method with autodiff created gradients and hessians.
However, if the gradient is a subgradient, then minimization just uses gradient
descent, as the function is probably not smooth enough for Newton.

The function to be minimized must have a single numpy array as a parameter. 
If not, it must be wrapped it in a lambda. 

Example:
```
def f(x):
    return np.sum(x**2)

result = minimize(f, np.random.rand(10))
# result['x'] should be zeros
```

With more than one parameter:
```
def f(x, p):
    return np.sum(x**p)

result = minimize(lambda x: f(x,2), np.random.rand(10))
# result['x'] should be zeros
```

"""
import numpy as np
from .autodiff import forward, reverse # to set use_subgrad
from .autodiff.reverse import (
    Diff,
)  # we use reverse mode by default because its faster
from .autodiff.subgrad import Subgrad


def minimize(
    func,
    x,
    maxiters=20,
    gconv=1e-8,
    xconv=0,
    fconv=0,
    method=None,
    report=0,
    outsign=1
):
    """find x which minimizes a function

    Args:
        func (callable[numpy array]): the function to minimize, having
            a single parameter x.
        x (numpy array): the initial value (a 1d vector)
        maxiters (int): the maximum number of iterations
        gconv (float): the convergence criterion for the gradient
        xconv (float): the convergence criterion on x
        fconv (float): the convergence criterion on f = func(x)
        method (callable): the minimization method, either newton_descent or
                subgrad_descent. If None, it is picked automatically
        report (int): the reporting level, -1 is none, 0 is basic start/end,
               1 is every iteration, 2 is details
        outsign (float): the sign used to report function values

    Returns:
        An object containing {x, fval, gradient, converged, message, iterations}
        where `x` minimizes `func(x)`

    Notes:
        This chooses newton's method or subgradient descent,
        depending on whether the first gradient is a subgradient or not.
        However if the method is set to newton, subgradients are not used.
        (Subgradients are generated when np.abs is used and
        autodiff.reverse.use_subgrad == True)
    """
    reverse.set_config(use_subgrad = method!=newton_descent)
    forward.set_config(use_subgrad = method!=newton_descent)
    func = Diff(func)
    fval = func.trace(x)
    grad = func.jacobian()
    # choose the minimization routine based on the grad type
    if method is None:
        method = (
            subgrad_descent if isinstance(grad, Subgrad) else newton_descent
        )
    return method(
        func, x, fval, grad, maxiters, gconv, xconv, fconv, report, outsign=outsign
    )


def maximize(
    func,
    x,
    maxiters=20,
    gconv=1e-8,
    xconv=0,
    fconv=0,
    method=None,
    report=0
):
    """find x which maximizes a function

    See `minimize` for arguments & return value

    Notes:
        This simply calls `minimize(lambda b: -func(b), ...)`. See
        minimize for further details.
    """
    neg_func = lambda x: -func(x)
    return minimize(
        neg_func, x, maxiters, gconv, xconv, fconv, method, report, outsign=-1
    )


eps = np.sqrt(np.finfo(float).eps)


def _reporter(level, current, *args):
    if level >= current:
        print(*args)


def newton_descent(
    func, x, fval, grad, maxiters, gconv, xconv, fconv, report, outsign=1
):
    """find x which minimizes a function using Newton's method. Called by
    maximize/minimize

    See `minimize` for arguments & return value
    """
    reporter = lambda *args: _reporter(report, *args)

    newfval = fval
    newx = x
    converged = False
    message = "maxiters exceeded"
    reporter(0, "Newton's method")
    for iter in range(maxiters):
        steptype=''
        grad = np.asarray(grad)
        gradsz = np.linalg.norm(grad)/len(grad)
        if  gradsz < gconv:
            reporter(1, f"Iteration {iter+1:4d} fval {outsign*fval:8g} grad {gradsz:6e}")
            reporter(
                0,
                f"Gradient convergence in iteration {iter+1}, final value {outsign*newfval}",
            )
            converged = True
            message = "gradient small"
            break
        try:
            # try a newton step
            steptype='newton step'
            hess = func.hessian()
            delta, _, _, sing = np.linalg.lstsq(hess, -grad, rcond=None)
            if np.max(np.abs(sing)) < eps:
                raise RuntimeError("singular")
            newx = x + delta
            newfval = func(newx)
            # we should backtrack the step to satisfy "good" descent.
            if newfval < fval:
                step = 1.0
                dg = np.dot(delta, grad)
                dhd = 0.5 * np.einsum("i,ij,j->", delta, hess, delta)
                predicted = lambda step: 0.5*(step * dg + step * step * dhd)
                actual = lambda step: func(x+step*delta)-fval
                if (newfval-fval)>predicted(1.0):
                    # failed the prediction, shrink
                    step = _linesearch(actual, predicted)
                if step<1.0:
                    steptype = f'shrunken newton step {step}'
                newx = x + step * delta
                newfval = func(newx)
            if newfval > fval or np.isnan(newfval):
                # the newton step went wrong somehow,
                raise RuntimeError("newton failed")
        except (RuntimeError, np.linalg.LinAlgError):
            # try a descent step
            steptype='descent step'
            newx = _descent(func, x, grad, fval)
            if np.isnan(newfval):
                raise RuntimeError("descent failed")
        newfval = func.trace(newx)
        # convergence flags
        better = newfval <= fval
        x_conv = np.max(np.abs(newx - x)) < xconv
        f_conv = abs(newfval - fval) < (1 + abs(fval)) * fconv
        # update
        x, fval = newx, newfval
        gradsz = np.linalg.norm(grad)/len(grad)
        reporter(1, f"Iteration {iter+1:4d} fval {outsign*fval:8g} grad {gradsz:6e} : {steptype}")
        # check convergence
        if x_conv and better:
            reporter(
                0,
                f"Parameter convergence in iteration {iter+1}, final value {outsign*newfval}",
            )
            converged = True
            message = "small difference in coefficients"
            break
        if f_conv and better:
            reporter(
                0,
                f"Function convergence in iteration {iter+1}, final value {outsign*newfval}",
            )
            converged = True
            message = "small difference in function value"
            break
        # compute grad for next iteration
        grad = func.jacobian()

    if not converged:
        reporter(0, f"Did not converge in {iter+1} iterations.")

    return {
        "fval": outsign*newfval,
        "x": newx,
        "grad": grad,
        # maxiters = 1 is for linear regression problems
        "converged": converged or maxiters == 1,
        "message": message,
        "iterations": iter + 1,
    }

def _linesearch(actual, predicted, initstep=1.0, extend=True):
    '''line search. 
    
    Args:
        actual (callable[float]): returns the actual change
            in function value for a step of given length
        predicted (callable[float]): returns the predicted change
            in function value for a step of given length. This may be
            scaled by a certain value e.g. 0.5 is common.
        initstep (float): the initial step along the line
        extend (bool): whether to go through an extend step
        
    Returns:
        A step length for which actual(step)<=predicted(step) 
        (which is fine because both should be negative)
    
    Notes:
        both actual(step) and predicted(step) should be less than zero
        when the step is returned.
    '''
    step = initstep
    # ensure actual & predicted are negative so the step is a decrease
    while actual(step)>0 or predicted(step)>0:
        step = step/2
    # expand until you find a step where the 
    # actual drop is not as good as the predicted
    ok = lambda a,p: a<0 and p<0 and a<=p
    initstep = step
    while extend and ok(actual(step),predicted(step)): 
        step = 2*step
    if step!=initstep: 
        # last extension is not ok
        step = step/2
    # contract until you find a step where the actual drop
    # is better than the predicted
    poor = lambda a, p: a>0 or p>0 or a>p
    iters = 0
    while poor(actual(step), predicted(step)) and iters<10: 
        step = step/2
        iters += 1
    return step
    
def _descent(func, x, grad, fval, initstep=1.0, extend=True):
    """find `x+a*grad` which roughly minimizes a function.
    Internal function - don't use

    Args:
        func: a Diff object containing the function
        x: the initial value (a 1d vector)
        grad: the gradient of the func at x
        fval: the value of func(x)

    Returns:
        `newx = x+a*grad` which sufficiently decreases `func(newx)`
        within limits set by Armijo conditions
    """
    step = initstep
    normg2 = np.dot(grad, grad)
    # we step in the direction -step*grad
    predicted = lambda step: -0.5*step*normg2
    actual = lambda step: func(x-step*grad)-fval
    step = _linesearch(actual, predicted, initstep, extend)
    return x-step*grad


def subgrad_descent(
    func, x, fval, grad, maxiters, gconv, xconv, fconv, report, outsign
):
    """find x which minimizes a function using subgradient descent

    See `minimize` for arguments & return value
    """
    reporter = lambda *args: _reporter(report, *args)

    newfval = fval
    newx = x
    converged = False
    message = "maxiters exceeded"
    reporter(0, "Subgradient descent method")
    for iter in range(maxiters):
        grad = np.asarray(grad)
        gradsz = np.linalg.norm(grad)/len(grad)
        if gradsz < gconv:
            reporter(1, f"Iteration {iter+1:4d} fval {outsign*fval:8g} grad {gradsz:6e}")
            reporter(
                0,
                f"Gradient convergence in iteration {iter+1}, final value {outsign*newfval}",
            )
            converged = True
            message = "gradient small"
            break
        newx = _descent(func, x, grad, fval)
        # retract newx if it changes sign, assuming
        # the grad will flip sign.
        signchange = np.sign(x)* np.sign(newx)<0
        if np.any(signchange):
            b = x[signchange]
            nb = newx[signchange]
            alpha = np.min(np.abs(b) / np.abs(nb - b))
            newx = x + alpha * (newx - x)
        newfval = func.trace(newx)
        # xconvergence flags
        better = newfval <= fval
        x_conv = np.max(np.abs(newx - x)) < xconv
        f_conv = abs(newfval - fval) < (1 + abs(fval)) * fconv
        # update
        x, fval = newx, newfval
        reporter(1, f"Iteration {iter+1} fval {outsign*fval}")
        # check convergence
        if x_conv and better:
            reporter(
                0,
                f"Parameter convergence in iteration {iter+1}, final value {outsign*newfval}",
            )
            converged = True
            message = "small difference in coefficients"
            break
        if f_conv and better:
            reporter(
                0,
                f"Function convergence in iteration {iter+1}, final value {outsign*newfval}",
            )
            converged = True
            message = "small difference in function value"
            break
        # compute grad for next iteration
        grad = func.jacobian()

    if not converged:
        reporter(0, f"Did not converge in {iter+1} iterations.")

    return {
        "fval": outsign*newfval,
        "x": newx,
        "grad": grad,
        "converged": converged,
        "message": message,
        "iterations": iter + 1,
    }
