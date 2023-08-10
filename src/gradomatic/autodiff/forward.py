import numpy as np
from .tracing import (
    Node,
    VarNode,
    OpNode,
    dims,
    value_of,
    getroot,
    index,
    shape,
    concat_spread,
)
from .memoize import memoize

from .subgrad import signum

config = {
    'use_subgrad': False
    }


def set_config(**kwargs):
    global config
    config = {**config, **kwargs}

def tensor_eye(shape):
    """returns the identity tensor

    Args:
        shape: the shape for the identity

    Returns:
        an identity whose shape is (*shape, *shape)
    """
    sz = np.prod(shape)
    return np.reshape(np.eye(sz), [*shape, *shape])


# have to do my own caching because ndarray isn't hashable


@memoize
def deriv(Z):
    """calculates the derivative of the node in a computation tree.

    Args:
        Z: the node

    Returns:
        a node implementing dZ/dX, where X is the root varnode in the tree
    """
    if type(Z) is VarNode:
        return tensor_eye(Z.shape)
    if type(Z) is OpNode:
        return deriv_ops[Z.fn](*Z.args, **Z.kwargs)
    # otherwise, Z doesn't depend on the node. This should be
    # a zero tensor but the shape is sometimes unknown; 0 works fine.
    return 0.0


def derivative(Y):
    """wrapper for `deriv` to do necessary init/teardown

    Args:
        Y: the top node in the computation tree

    Returns:
        a node implementing dY/dX, where X is the root varnode in the tree

    Note:
        This does some necessary housekeeping before & after calling `deriv`
    """
    oldcache, deriv.cache = deriv.cache, {}
    dYdX = deriv(Y)
    deriv.cache = oldcache
    return dYdX


# dict to contain derivative operations
deriv_ops = {}


def register_deriv(f, df=None):
    """decorator to register a function's derivative"""
    if df is None:

        def inner(df):
            deriv_ops[f] = df
            return df

        return inner
    else:
        deriv_ops[f] = df


# simple unary & binary functions


@register_deriv(np.add)
def deriv_add(A, B):
    """for Z=A+B, returns dZdX = dAdX + dBdX"""
    dAdX, dBdX = deriv(A), deriv(B)
    if Node.iszero(dAdX):
        return dBdX
    if Node.iszero(dBdX):
        return dAdX
    return np.add(dAdX, dBdX)


@register_deriv(np.subtract)
def deriv_sub(A, B):
    """for Z=A-B, returns dZdX = dAdX - dBdX"""
    dAdX, dBdX = deriv(A), deriv(B)
    if Node.iszero(dAdX):
        return np.negative(dBdX)
    if Node.iszero(dBdX):
        return dAdX
    return np.subtract(dAdX, dBdX)


@register_deriv(np.negative)
def deriv_neg(A):
    """for Z=-A returns dZdX = -dAdX"""
    return np.negative(deriv(A))


@register_deriv(np.multiply)
def deriv_multiply(A, B):
    """for Z=A*B, returns dZdX=A*dBdX + dAdX*B.
    However, one of A, B is guaranteed to be a scalar with ddX=0
    Other forms of multiplication are converted to einsum
    """
    if np.isscalar(A):
        return np.multiply(A, deriv(B))
    else:
        return np.multiply(deriv(A), B)


r"""
#2.2 Tensor Product

If $Z_z=A_aB_b$ (with $z \subseteq a \cup b$) then

$$
\D{Z}{X} = A\D{B}{X}+\D{A}{X}B
$$

The indices are $(zx)=a(bx)+(ax)b$ and since $z$ is a contraction of $ab$, 
this all works out. Essentially all that happens when the jacobian is formed 
is that a new set of indices are added which trail and don't overlap the 
original ones. 

Einsum scripts can be optimized when one of the items is the varnode. In that 
case, the varnode's derivative is simply an identity matrix which, when 
inserted into the einsum script, just renames indices. For example

* script: `ij,j->i`
* derivative in 2nd place: `ij,ja->ia`  (`ja` is an identity matrix)
* simplified: `ij->ij` 

In all other terms, replace `a` with `j`, and this must happen in the rhs; 
then remove the derivative term. But `j` can't appear in the rhs, and must 
appear in the rest of the lhs (to be summed out).

`i->` becomes `ia->a` so replace `a` with `i` then  remove deriv term to 
give `->i` which is wrong. So can't do this if only one term.

`ij,jk->ik` becomes `ij,jkab->ikab` : the `k` can't be replaced with `b` in 
any other term

`ij,j->ij` becomes `ij,ja->ija`

Conditions:
1. There must be more than one term in the lhs
2. The varnode term's indices must all be summed out (i.e. not on the rhs, 
   present in the rest of the rhs)
"""


# einsums:

all_indices = set("abcdefghijklmnopqrstuvwxyz")


def parse_script(script):
    """parse an einsum script

    Args:
        script: the einsum script

    Returns:
        a tuple (lhs, rhs, available) where
            lhs is a list of indices for each component of the einsum
            rhs is the right hand side of the einsum
            available is a string giving the unused index letters

    Examples:
        parse_script('ij,j->') gives lhs = ['ij', 'j'] and rhs=''
        parse_script('ij,j') gives lhs = ['ij', 'j'] and rhs='i'

        In the latter case, the rhs is inferred from the lhs
    """
    script = script.split("->")
    lhs = script[0].split(",")
    if len(script) == 1:
        # rhs is implict summation, by removing from the
        # union any index that is repeated between any pair of
        # lhs elements
        rhs = lhs[0]
        for term in lhs[1:]:
            for idx in term:
                if idx in rhs:
                    rhs = rhs.replace(term, "")
                else:
                    rhs = rhs + term
    else:
        rhs = script[1]
    # get a set of unique indices for the derivative dimensions
    indices = set("".join(lhs))
    available = sorted(all_indices - indices)
    available = "".join(available)
    return lhs, rhs, available


def deriv_script(lhs, rhs, wrt, i, args):
    """returns the einsum script for the derivative of one of the lhs items

    Args:
        lhs, rhs: the left and right hand sides of the einsum script (see `parse_script`)
        wrt: the indices of the X tensor we are differentiating with respect to
        i: the item index (i.e. lhs[i]) to do the derivative
        args: the argument list to be passed to the einsum.

    Returns:
        an einsum script (string) which computes the derivative of the einsum with
        respect to X on the i-th item.

    Example:
        if Z = np.einsum(script, A, B), then
        dZdX = np.einsum(ds0,dAdX, B)+np.einsum(ds1, A, dBdX), where
        ds0 = deriv_script(lhs, rhs, Xindex, 0, (A,B)) and
        ds1 = deriv_script(lhs, rhs, Xindex, 1, (A,B))
    """
    lhs = [*lhs]
    lhs[i] = lhs[i] + wrt
    args = [*args]
    args[i] = deriv(args[i])
    return ",".join(lhs), rhs + wrt, args


def summate(nodes):
    """returns an opnode which sums a list of nodes

    Args:
        nodes: the list of nodes to summate

    Returns:
        a node which represents nodes[0]+nodes[1]+...
    """
    if len(nodes) == 0:
        return 0
    sum = nodes[0]
    for n in nodes[1:]:
        sum = np.add(sum, n)
    return sum


@register_deriv(np.einsum)
def deriv_einsum(script, *args, **kwargs):
    """computes the derivative of an einsum

    Args:
        script: the einsum script
        *args: the einsum arguments
        **kwargs: the einsum keyword arguments

    Returns:
        a node which computes the derivative.

        If Z = np.einsum(inscript, A, B) then this returns
        dZdX = np.einsum(ds0, dAdX, B)+np.einsum(ds1, A, dBdX)
        where ds0, ds1 come from deriv_script
    """
    lhs, rhs, available = parse_script(script)
    wrt = available[: len(getroot(*args).derivshape)]
    terms = []
    args = [*args]
    for i, arg in enumerate(args):
        if not Node.isNode(arg):
            continue  # derivative is zero
        i_lhs, i_rhs, d_args = deriv_script(lhs, rhs, wrt, i, args)
        # optimize out zeros
        if i_lhs == i_rhs:
            terms.append(d_args[0])
        elif i >= len(d_args) or not Node.iszero(d_args[i]):
            terms.append(np.einsum(i_lhs + "->" + i_rhs, *d_args, **kwargs))
    return summate(terms)


@register_deriv(index)
def deriv_index(A, idx):
    """the derivative of an indexing operation Z = A[idx]

    Args:
        A: the indexed node
        idx: the index expression

    Returns:
        If Z=A[idx], this returns dZdX = dAdX[idx, :, :, ...], because
        the X indices are the trailing ones in dAdX and so aren't affected
        by taking the index
    """
    return deriv(A)[idx]


@register_deriv(np.reshape)
def deriv_reshape(A, newshape):
    """the derivative of reshape operation

    Args:
        A: the indexed node
        newshape: the shape following reshaping

    Returns:
        If Z=np.reshape(A, ns), this returns
        dZdX = np.reshape(dAdX, [*ns, *Xshape]) where Xshape is the shape of
        the X variable
    """
    dAdX = deriv(A)
    newshape = [*newshape, *dAdX.shape[-len(A.root.derivshape) :]]
    return np.reshape(dAdX, newshape)


@register_deriv(concat_spread)
def deriv_concat(*args, axis=0):
    """the derivative of concat

    Args:
        *args: the matrices to concatenate
        axis: the axis to concatenate

    Returns:
        If Z = concat_spread(T1, T2, ..., axis=x), this returns
        dZdX = concat_spread(dT1dX, dT2dX, ... axis=x)
    """
    derivshape = getroot(*args).derivshape
    dTdX = []
    for a in args:
        if not Node.isNode(a):
            da = np.zeros((*shape(a), *derivshape))
        else:
            da = deriv(a)
        dTdX.append(da)
    return concat_spread(*dTdX, axis=axis)


r"""
#2.5 Unary functions.

A unary function is $f:\mathbb{R}\rightarrow\mathbb{R}$. When applied to a 
tensor $A$, it yields a tensor $Z=f(A)$ of the same shape as $A$ with elements

$$
Z_{(a)} = f(A_{(a)})
$$

The derivative is

$$
\D{Z}{X} = \D{Z}{A}\D{A}{X}
$$

The derivative $\d{Z}{A}$ is a "diagonal" tensor $dF_{a^*,a}$ where 
$dF_{(a^*,a)}=f^\prime(A_{(a)})$ if $(a^*)=(a)$, and zero otherwise. Thus 

$$
\D{Z}{X} = f^\prime(A) \circ \D{A}{X}
$$

where $f^\prime(A)$ is a tensor formed by applying the unary derivative 
$f^\prime$ elementwise to the elements of $A$, and $\circ$ is the elementwise 
product (Hadamard). In einsum script terms, it is 
`dZdX = einsum("a,ax->ax", df(A), dAdX)`

This einsum won't be created if we use standard multiplication, as it 
broadcasts from the last indices, whereas this broadcasts from the first. 
So we need to invent a new operation `premult` which broadcasts from the front.
"""


def premult(dZdA, dAdX):
    """computes dZdX = dZdA*dAdX"""
    bi = "ijklmnopqrstuvw"[: dAdX.ndim]
    if np.isscalar(dZdA):
        ai = ""
    else:
        ai = bi[: dZdA.ndim]
    return np.einsum(ai + "," + bi + "->" + bi, dZdA, dAdX)


@register_deriv(np.power)
def deriv_power(A, n):
    """the derivative of a power

    Args:
        A: the node
        n: the power, a non-node

    Returns:
        If Z = A**n, returns dZdX = n*dAdX**(n-1)

    Notes:
        This is handled differently from other ufuncs because
        of some common simplifications when n is 1 or 2
    """
    dAdX = deriv(A)
    if np.isscalar(n):
        if n == 1:
            return dAdX
        if n == 2:
            return premult(n * A, dAdX)
    return premult(n * np.power(A, n - 1), dAdX)


def ufunc_chainrule(f, df):
    """registers df as the derivative of the ufunc f"""

    @register_deriv(f)
    def dfunc(A, *args):
        return premult(df(A, *args), deriv(A))


ufunc_derivs = {
    np.negative: lambda x: -1.0,
    np.exp: np.exp,
    np.log: np.reciprocal,
    np.sqrt: lambda x: 0.5 * x ** (-0.5),
    np.abs: lambda x: signum(x) if config['use_subgrad'] else np.sign(x),
    np.sign: lambda x: 0,
    signum: lambda x: 0,
    np.sin: np.cos,
    np.cos: lambda x: -np.sin(x),
    np.tan: lambda x: np.cos(x) ** (-2.0)
    # add more here as required.
}

for f, df in ufunc_derivs.items():
    ufunc_chainrule(f, df)


def deriv_clip(A, amin=None, amax=None):
    """derivative of clip

    Args:
        A: a node
        amin, amax: the clipping values

    Returns:
        If Z = np.clip(A), this returns
        dZdA = 1 if not clipped, 0 if clipped.
        This is then fed into the usual ufunc derivative.
    """
    vA = value_of(A)
    ok = True
    if amin is not None:
        ok = np.logical_and(ok, vA >= amin)
    if amax is not None:
        ok = np.logical_and(ok, vA <= amax)
    if dims(ok) > 0:
        return ok * 1.0
    return np.ones_like(vA)


ufunc_chainrule(np.clip, deriv_clip)


r"""
#2.6 General functions.

A general tensor function $f$ takes one tensor and returns another. When 
applied to $A$, it yields $Z=f(A)$, where the dimensions of $A$ and $Z$ need 
not be the same.

The derivative is

$$
\D{Z}{X} = \D{Z}{A}\D{A}{X}
$$

where $\d{Z}{A}$ is a tensor with indices $za$. In einsum terms this is 
`dZdX = einsum('za,ax->zx', df(A), dAdZ)`

If $Z=f(A,B)$, then

$$
\D{Z}{X} = \D{Z}{A}\D{A}{X} + \D{Z}{B}\D{B}{X}
$$
"""


# func derivs should be no-traced


def chainrule(f, *df):
    """the derivative of a general node using the chainrule

    Args:
        f: the function
        *df: derivatives with respect to the arguments. Not all derivatives
             need to be supplied

    Returns:
        If Z = f(A,B) then dZdX = df[0](A,B)*dAdX + df[1](A,B)*dBdX
        where df[0] = df(A,B)/dA and df[1] = df(A,B)/dB
    """

    def getindices(*lengths):
        idx = "ijklmnopqrstuvwxyz"
        out = []
        start = 0
        for n in lengths:
            out.append(idx[start : start + n])
            start += n
        return out

    @register_deriv(f)
    def dfunc(*args, **kwargs):
        terms = []
        for i, a in enumerate(args):
            if Node.isNode(a):
                dZdA = df[i](*args, **kwargs)
                dAdX = deriv(a)
                zdim = dZdA.ndim - a.ndim
                xdim = dAdX.ndim - a.ndim
                Zidx, Aidx, Xidx = getindices(zdim, a.ndim, xdim)
                terms.append(
                    np.einsum(
                        Zidx + Aidx + "," + Aidx + Xidx + "->" + Zidx + Xidx,
                        dZdA,
                        dAdX,
                    )
                )
        return summate(terms)


def derivatives_of(fn, argno=0, value=False, jacobian=False, hessian=False):
    """returns a function to calculate derivatives of a function

    Args:
        fn : the function to calculate derivatives of
        argno: the argument number to calcuate the derivative with respect to
        value: True if you want the value returned
        jacobian: True if you want the jacobian (or gradient) returned
        hession: True if you want the hessian returned.

    Returns:
        a function to calculate the derivatives of fn

    Example:
        if fn is a function of 3 arguments fn(A,B,C) then
        dfn = derivatives_of(fn, argno=1, value=True, jacobian=True) is a function
        to calculate the value and gradient of the function with respect to B

        This function dfn(A,B,C) returns a tuple (value, jacobian).
    """

    def calc_d(*args, **kwargs):
        args = list(args)
        vn = VarNode(args[argno])
        args[argno] = vn
        result = []
        # tracing:
        f = fn(*args, **kwargs)
        # output:
        if value:
            result.append(value_of(f))
        if jacobian or hessian:
            jac = derivative(f)
            del f
            if jacobian:
                result.append(value_of(jac))
        if hessian:
            hess = derivative(jac)
            del jac
            result.append(value_of(hess))
            del hess
        # encourage GC:
        deriv.cache = {}
        vn.root.memo = {}
        return result[0] if len(result) == 1 else result

    return calc_d


class Diff:
    def __init__(self, func, argno=0):
        self.func = func
        self.argno = argno
        self.vn = None
        self.fval = None
        self.j = None
        self.h = None


    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def trace(self, *args, **kwargs):
        # call this if you want the function value &
        # will later want the jacobian.
        args = list(args)
        self.vn = args[self.argno] = VarNode(args[self.argno])
        self.fval = self.func(*args, **kwargs)
        return value_of(self.fval)

    def jacobian(self, gc=False):
        # works after trace is called
        self.j = derivative(self.fval)
        jac = value_of(self.j)
        if gc:
            self.gc()
        return jac

    def hessian(self, gc=True):
        # works after jacobian is called
        self.h = derivative(self.j)
        hess = value_of(self.h)
        if gc:
            self.gc()
        return hess

    def gc(self):
        # encourage GC:
        self.vn = None
        self.fval = None
        self.j = None
        self.h = None


# these are just for testing


def jacobian(f, argno=0):
    d = Diff(f, argno)

    def jac(*args, **kwargs):
        d.trace(*args, **kwargs)
        return d.jacobian()

    return jac


def hessian(f, argno=0):
    d = Diff(f, argno)

    def hess(*args, **kwargs):
        d.trace(*args, **kwargs)
        d.jacobian()
        return d.hessian()

    return hess
