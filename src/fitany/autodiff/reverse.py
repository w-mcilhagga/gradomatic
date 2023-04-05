import numpy as np
from .tracing import (
    Node,
    VarNode,
    shape,
    dims,
    value_of,
    index,
    backtrace,
    notrace,
    concat_spread,
)
from .memoize import memoize

from .subgrad import signum

use_subgrad = False


def tensor_eye(shape):
    """returns the identity tensor

    Args:
        shape: the shape for the identity

    Returns:
        an identity whose shape is (*shape, *shape)
    """
    sz = int(np.prod(shape))
    return np.reshape(np.eye(sz), [*shape, *shape])


# have to do my own caching because ndarray isn't hashable
# NB the caching could be smarter if backtrace is used
# because we only need to keep results that are called for >1 time

# memoizing has to be on the varnode, so derivatives are separate
# from one another.


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


@memoize
def deriv(A):
    """
    calculates the derivative of the node in a computation tree.

    Args:
        A: the node

    Returns:
        a node implementing dY/dA, where Y is the top opnode in the tree
    """
    if not Node.isNode(A):
        return 0
    elif A not in A.root.backrefs:
        # A is Y itself
        return tensor_eye(shape(A))
    else:
        # otherwise we work out dY/dA, where A is this node.
        derivs = []  # holds dY/dZi
        if True:  # id(n) not in deriv_memo:
            for Zi, argno in A.root.backrefs[A]:
                derivs.append(
                    pullback(Zi, argno)
                )  # this is dY/dZi*dZi/dA,argno
            derivs = summate(derivs)
        return derivs


def pullback(Z, argno):
    """this is dYdA = dY/dZ*dZ/dA when A is argno

    Args:
        Z : the opnode
        argno: indicates which of the args is A

    Returns:
        dZdA for the given argno
    """
    dYdZ = deriv(Z)
    return deriv_ops[Z.fn](dYdZ, argno, *Z.args, **Z.kwargs)


deriv.ydim = None


def derivative(Y):
    """wrapper for `deriv` to do necessary init/teardown

    Args:
        Y: the top node in the computation tree

    Returns:
        a node implementing dY/dX, where X is the root varnode in the tree

    Note:
        This does some necessary housekeeping before & after calling deriv
    """
    if not Node.isNode(Y):
        return 0
    # save attributes
    oldydim, deriv.ydim = deriv.ydim, getattr(Y, "ndim", 1)
    oldrefs = Y.root.backrefs
    oldcache, deriv.cache = deriv.cache, {}
    # calc deriv
    backtrace(Y)
    dYdX = deriv(Y.root)  # run the pullback from the varnode
    # restore attributes
    deriv.cache = oldcache
    Y.root.backrefs = oldrefs
    deriv.ydim = oldydim
    # and return
    return dYdX


# dict to contain derivative operations
deriv_ops = {}


def register_deriv(f, df=None):
    """decorator to declare the derivative df of function f

    Args:
        f: the function to register
        df: (optional) the derivative

    Returns:
        if called as a decorator @register_deriv(f) it regsiters the
        decorated function as the derivative of f
        If called as register_deriv(f, df) it does the same.

    A derivative function df of a node Z has the signature
    df(dYdZ, argno, *n.args, **n.kwargs) and returns dYdA where A is n.args[argno]
    """
    if df is None:
        # called as a decorator
        def inner(df):
            deriv_ops[f] = df
            return f

        return inner
    else:
        # called directly
        deriv_ops[f] = df


# simple unary & binary functions


@register_deriv(np.add)
def deriv_add(dYdZ, argno, *args):
    """for Z=A+B, returns
    dYdA = dYdZ*dZdA = dYdZ or
    dYdB = dYdZ*dZdB = dYdZ
    """
    return dYdZ


@register_deriv(np.subtract)
def deriv_sub(dYdZ, argno, *args):
    """for Z=A-B, returns
    dYdA = dYdZ*dZdA = dYdZ or
    dYdB = dYdZ*dZdB = -dYdZ
    """
    return dYdZ if argno == 0 else np.negative(dYdZ)


@register_deriv(np.negative)
def deriv_neg(dYdZ, argno, *args):
    """for Z=-A returns dYdA = dYdZ*dZdA = -dAdX"""
    return np.negative(dYdZ)


@register_deriv(np.multiply)
def deriv_mult(dYdZ, argno, A, B, **kwargs):
    """for Z=A*B, returns
    dYdA = dYdZ*dZdA = dYdZ*b or
    dYdB = dYdZ*dZdB = dYdZ*b

    However, one of A, B is guaranteed to eb a scalar with ddX=0
    Other forms of multiplication are einsummed
    """
    if argno == 0:
        return B * dYdZ
    if argno == 1:
        return A * dYdZ


# ## 3.2 Tensor Product
#
# If $Z_z=A_aB_b$ (and $a \ominus b \subseteq z$ so there is no summation outside of the shared indices of $A$ and $B$,
# where $\ominus$ is the symmetric difference: $a\cup b$ without those in both) then
#
# $$
# \D{Y}{A}  = \D{Y}{Z} \D{AB}{A} = \D{Y}{Z} B
# $$
#
# and
#
# $$
# \D{Y}{B}  = \D{Y}{Z} \D{AB}{B} = \D{Y}{Z} A
# $$
#
# There is a simple procedure for converting this to an einsum script. Let $z, a, b$ be the index sets for $Z, A, B$. Let $y$ be the index set for $Y$, entirely distinct from $a,b,c$. Then for example $d{Y}{Z}$ has indices $yz$.
#
# * The indices for $\d{Y}{A}$ are `(yz,b->ya, dYdZ, B)`
# * The indices for $\d{Y}{B}$ are `(yz,a->yb, dYdZ, A)`
#
# For example, putting in individual indices, let $Z_i = A_{ij}B_j$. Then the einsum for $\d{Y}{A}$ is `(yi,j->yij, dYdZ, A)`  and the einsum for $\d{Y}{B}$ is `(yi,ij->yj, dYdZ, A)`.
#
# In both cases, einsum will be able to figure out what's happening.

# Consider
#
# $$
# Z_k = A_{ij}B_{ijk}
# $$
#
# This has a script `ij,ijk->k`. The shared i,j are summed out.
#
# * If we work out the script for $\d{Y}{A}$ as above, we get `(yk,ijk->yij, dYdZ, B)` which works.
# * If we work out the script for $\d{Y}{B}$ as above, we get `(ij,yijk->yijk, A, dYdZ)` which works.
#
# Consider
#
# $$
# Z_k = A_{ijk}B_{k}
# $$
#
# This has a script `ijk,k->k`. The unshared i,j are summed out.
#
# * If we work out the script for $\d{Y}{A}$ as above, we get `(yk,k->yijk, dYdZ, B)` which doesn't work, since no i or j on the lhs.
# * If we work out the script for $\d{Y}{B}$ as above, we get `(ijk,yk->yk, A, dYdZ)` which works.
#
# The first one needs extra vectors to stretch it out. It would work as `(yk,k,i,j->yijk, dYdZ, B, ones, ones)` if the ones had the right size.

# ## 3.3 Summing along axes.
#
# Let $Z_z = \text{sum}(A_a, s)$ be the sum of tensor $A$ along axes $s \subseteq a$. It is implied here that $z=a-s$. This can be rewritten as a multiplication $Z_z=\mathbf{1}_sA_a$ where $\mathbf{1}_s$ is a tensor of ones over the given index set.
#
# Then
#
# $$
# \D{Y}{A} = \D{Y}{Z}\D{Z}{A} = \D{Y}{Z}\mathbf{1}_s
# $$
#
# Since $z=a-s$, this means that the jacobian $\d{Y}{Z}$ is broadcast along the summed axes $s$. The subsequent rearrangement of the axes from $yz,s$ on the right hand side into $ya$ on the left is part of numpy's einsum functionality, and works because $a$ and $zs$ have the same indices, just maybe in a different order.
#
# In numpy notation, this is `dYdA=einsum("yz,s->ya", dYdZ, ones(s))`, where `a` is a rearrangement of the elements of `zs`


# demo of np.einsum scripting changes for reverse mode


from .forward import parse_script


def get_sizes(lhs, args):
    # work out the size of each index in the einsum
    sizes = {}
    for indices, arg in zip(lhs, args):
        for idx, n in zip(indices, shape(arg)):
            sizes[idx] = n
    return sizes


def deriv_script(argno, script, args):
    # ydim is the dimension of the tensor Y
    # s is the script, x is the operands
    lhs, rhs, available = parse_script(script)
    y_indices = available[: deriv.ydim]
    sizes = get_sizes(lhs, args)
    rhs = y_indices + rhs

    newlhs = [*lhs]
    pullback = y_indices + newlhs[argno]
    newlhs[argno] = rhs
    # check to see if there is any extraneous summation
    # between the newlhs and the pullback
    extras = list(set(pullback) - set("".join(newlhs)))
    if len(extras) > 0:
        newlhs.extend(extras)
        shapes = [sizes[index] for index in extras]
    else:
        shapes = []
    dscript = ",".join(newlhs) + "->" + pullback
    return dscript, shapes


# einsums:


@register_deriv(np.einsum)
def deriv_einsum(dYdZ, argno, inscript, *args, **kwargs):
    # derivative of an einsum
    script, extras = deriv_script(argno, inscript, args)
    extras = [np.ones(ex) for ex in extras]
    dargs = [*args]
    dargs[argno] = dYdZ
    # need to do a bit of de-broadcasting if there was broadcasting
    # in the einsum
    return np.einsum(script, *dargs, *extras)


# trace uses index(obj, key) to do indexing.
from .einsum_equivalent import getindices


@register_deriv(index)
def deriv_index(dYdZ, argno, A, idx):
    """the derivative of an indexing operation Z = A[idx]

    Args:
        dYdZ: the derivative wrt Z
        argno: always 0
        A: the node
        idx: the index expression

    Returns:
        If Z=A[idx], this is equivalent to Z = IDX*A where IDX is
        an indexing tensor. Then dYdA = dYdZ*IDX'
    """
    # this creates an indexing tensor to expand the derivative
    IndexTensor = tensor_eye((A.shape))[idx]
    aidx, zidx, yidx = getindices(A.ndim, dYdZ.ndim - deriv.ydim, deriv.ydim)
    script = f"{yidx+zidx},{zidx+aidx}->{yidx+aidx}"
    return np.einsum(script, dYdZ, IndexTensor)


@register_deriv(np.reshape)
def deriv_reshape(dYdZ, argno, A, newshape):
    """the derivative of reshape operation Z = np.reshape(A, newshape)

    Args:
        dYdZ: the derivative wrt Z
        argno: always 0
        A: the node
        newshape: the new shape

    Returns:
        dYdA, where the non-Y dimensions of dYdZ have been reshaped to A's shape
    """
    yshape = dYdZ.shape[0 : deriv.ydim]
    newshape = [*yshape, *A.shape]
    return np.reshape(dYdZ, newshape)


@register_deriv(concat_spread)
def deriv_concat(dYdZ, argno, *args, axis=0):
    """the derivative of concat

    Args:
        dYdZ: the derivative wrt Z
        argno: the argument
        *args: the matrices to concatenate
        axis: the axis to concatenate

    Returns:
        dYdA = dYdZ*dZdA. The tensor dZdA picks out the appropriate
        submatrix of Z
    """
    rows = [len(value_of(a)) if dims(a) < 2 else shape(a)[axis] for a in args]
    Arange = np.cumsum([0, *rows])[argno : argno + 2]
    ydim = deriv.ydim
    idx = tuple([*[slice(None)] * (ydim + axis), slice(Arange[0], Arange[1])])
    return dYdZ[idx]


# ## 3.5 Unary functions.
#
# A unary function is $f:\mathbb{R}\rightarrow\mathbb{R}$. When applied to a tensor $A$, it yields a tensor $Z=f(A)$ of the same shape as $A$ with elements
#
# $$
# Z_{(a)} = f(A_{(a)})
# $$
#
# The pullback is
#
# $$
# \D{Y}{A} = \D{Y}{Z}\D{Z}{A}
# $$
#
# The derivative $\d{Z}{A}$ is a "diagonal" tensor $dF_{a^*,a}$ where $dF_{(a^*,a)}=f'(A_{(a)})$ if $(a^*)=(a)$, and zero otherwise. Thus
#
# $$
# \D{Y}{A} = \D{Y}{Z} \circ f^\prime(A)
# $$
#
# where $f^\prime(A)$ is a tensor formed by applying the unary derivative $f'$ elementwise to the elements of $A$, and $\circ$ is the elementwise product (Hadamard).
#

# In numpy terms, if `dYdZ` has indices `yz` and `A` has indices `z`, which it must, then the result is `einsum('yz,z', dYdZ, fA)` where `fA` is the elementwise derivative of $f$.


# ufunc (functions R->R applied elementwise) derivatives.


def postmult(dYdZ, dZdA):
    # used for ufuncs, does dYdZ*dZdA
    idx1 = "ijklmnopqrstuvw"[: dYdZ.ndim]
    if np.isscalar(dZdA):
        idx2 = ""
    else:
        idx2 = idx1[dYdZ.ndim - dZdA.ndim :]
    return np.einsum(idx1 + "," + idx2 + "->" + idx1, dYdZ, dZdA)


@register_deriv(np.power)
def deriv_power(dYdZ, argno, base, ex, **kwargs):
    # this is only called when exponent is not a Node.
    # it's a ufunc, but has some efficiencies
    # argno is always zero (or it should be)
    if np.isscalar(ex):
        if ex == 1:
            return dYdZ
        if ex == 2:
            return postmult(dYdZ, ex * base)
    return postmult(dYdZ, ex * np.power(base, ex - 1))


def ufunc_chainrule(f, df):
    """registers df as the derivative of the ufunc f.
    When df is called, argno is always zero."""

    @register_deriv(f)
    def dfunc(dYdZ, argno, *args, **kwargs):
        return postmult(dYdZ, df(*args, **kwargs))


ufunc_derivs = {
    np.negative: lambda x: -1.0,
    np.exp: np.exp,
    np.log: np.reciprocal,
    np.sqrt: lambda x: 0.5 * x ** (-0.5),
    np.abs: lambda x: signum(x) if use_subgrad else np.sign(x),
    np.sign: lambda x: 0,
    signum: lambda x: 0,
    np.sin: np.cos,
    np.cos: lambda x: -np.sin(x),
    np.tan: lambda x: np.cos(x) ** (-2),
}

for f, df in ufunc_derivs.items():
    ufunc_chainrule(f, df)


def deriv_clip(x, amin=None, amax=None):
    """derivative of clip"""
    vx = value_of(x)
    ok = True
    if amin is not None:
        ok = np.logical_and(ok, vx >= amin)
    if amax is not None:
        ok = np.logical_and(ok, vx <= amax)
    if dims(ok) > 0:
        return ok * 1.0
    return np.ones_like(vx)


ufunc_chainrule(np.clip, deriv_clip)


# ## 3.6 General functions.
#
# When we have $Z=f(A,B,C)$ then
#
# $$
# \D{Y}{A}=\D{Y}{Z}\D{Z}{A} = \D{Y}{Z}\D{f(A,B,C)}{A}
# $$
#
#
# The indices of $\d{Y}{Z}$ are `yz` and so the indices of $\d{Z}{A}$ are `za` for this to work out


# func derivs should be no-traced


def chainrule(f, *df):
    # df are the derivatives for each argument
    # which may be just one.
    notrace(f)

    @register_deriv(f)
    def dfunc(dYdZ, argno, *args, **kwargs):
        # work out indices
        yidx, zidx, aidx = getindices(
            deriv.ydim, dYdZ.ndim - deriv.ydim, args[argno].ndim
        )
        script = f"{yidx+zidx},{zidx+aidx}->{yidx+aidx}"
        return np.einsum(script, dYdZ, df[argno](*args, **kwargs))


def derivatives_of(
    fn, argno=0, value=False, jacobian=False, hessian=False, return_node=False
):
    # return some or all of the derivatives

    def d(*args, **kwargs):
        args = list(args)
        vn = args[argno] = VarNode(args[argno])
        result = []
        f = fn(*args, **kwargs)
        if value:
            result.append(value_of(f) if not return_node else f)
        if jacobian or hessian:
            jac = derivative(f)
            del f
            if jacobian:
                result.append(value_of(jac) if not return_node else jac)
        if hessian:
            hess = derivative(jac)
            del jac
            result.append(value_of(hess) if not return_node else hess)
            del hess
        # encourage GC:
        deriv.cache = {}
        vn.root.memo = {}
        return result[0] if len(result) == 1 else result

    return d


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


def jacobian(f, argno=0):
    # return derivatives_of(f, argno=argno, jacobian=True)
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
