"""Tracing.

Tracing a function call and returning a computation graph is the core of automatic 
differentiation. An important issue is to make sure the tracer can deal with 
`numpy` dispatch, and can convert `numpy` functions to `einsum` where 
appropriate, since we know how to take the derivative of `einsum`s.

# Nodes

A node is an object used to trace a computation and produce a computation tree 
for differentiation. Given a function `f(a,b,c,...)`, to trace the computation 
of, say, `b`, you would call
    `tree = f(a,VarNode(b),c,...)`.

The computation tree is composed of `VarNodes` which encapsulate a variable 
and `OpNodes` which encapsulate an operation. The return value `tree` is an OpNode.
For OpNodes, `tree.value` is the value `f(a,b,c,...)`. The tree can be operated on to 
get derivatives.

There should be only one VarNode.

# Operations.

When tracing, an operation is frequently changed for another equivalent one, 
to make differentiating easier. The changes are:
    
a/b -> a*b**(-1.0)
sum, inner, outer, dot, @, tensordot, broadcast_to, transpose -> einsum
* (where both are arrays) -> einsum
trace, ravel, diagonal -> replaced with equivalent indexing, reshaping
diag(v) -> replace with einsum when v is a vector

# Limitations.

some functions ignore additional keyword arguments. This needs to be
explicitly warned.
"""

# TODO:
# less greater, etc (as that's how the comparisons are implemented)
# pad, sort

from collections import defaultdict
from functools import wraps
import numpy as np
from .einsum_equivalent import einsum_equivalent
from .memoize import hash_or_id

# attribute queries


def dims(x):
    return getattr(x, "ndim", 0)


def shape(x):
    return getattr(x, "shape", ())


def value_of(x):
    # if type(x) in [list, tuple]:
    #    return type(x)(value_of(a) for a in x)
    return getattr(x, "value", x)


class Node:
    """A computational Node encapsulates a value and records a computational trace
    of a function. A Node overrides numpy operations.

    Args:
        value (np-array): the value of the node
        root (Node): the root node (VarNode) of the computational trace.

    Returns:
        a computational node with properties
            value: the given value
            ndim: the number of dimensions
            shape: the shape of the value
    """

    @staticmethod
    def isNode(n):
        """checks if an object is a Node"""
        return isinstance(n, Node)

    @staticmethod
    def iszero(n):
        """checks if an object is zero and not a Node"""
        return not Node.isNode(n) and np.all(n == 0)

    def __init__(self, value, root=None):
        self.value = value
        self.shape = shape(value)
        self.dims = dims(value)
        self.ndim = len(self.shape)
        self.root = root

    # operator overrides for operations involving python objects

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __sub__(self, other):
        return np.subtract(self, other)

    def __rsub__(self, other):
        return np.subtract(other, self)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return np.multiply(other, self)

    def __truediv__(self, other):
        return np.true_divide(self, other)

    def __rtruediv__(self, other):
        return np.true_divide(other, self)

    def __neg__(self):
        return np.negative(self)

    def __matmul__(self, other):
        return np.matmul(self, other)

    def __rmatmul__(self, other):
        return np.matmul(other, self)

    def __pow__(self, other):
        return np.power(self, other)

    def __rpow__(self, other):
        return np.power(other, self)

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __getitem__(self, key):
        # indexing the node; opnode is defined below
        return opnode(index, self, key)

    def __hash__(self):
        # required because __eq__ was overridden
        return id(self)

    def __len__(self):
        return self.shape[0]

    # numpy dispatch for when one of the objects is a numpy array

    handled_funcs = {}

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # dispatcher for numpy ufuncs
        if method != "__call__" or ufunc not in self.handled_funcs:
            return NotImplemented
        s = self.handled_funcs[ufunc](*args, **kwargs)
        return s

    def __array_function__(self, func, types, args, kwargs):
        # dispatcher for other numpy funcs
        if func not in self.handled_funcs:
            return NotImplemented
        return self.handled_funcs[func](*args, **kwargs)

    @classmethod
    def add_handler(cls, f, g):
        # declares that we should use g when numpy function f is called
        cls.handled_funcs[f] = g

    @classmethod
    def register_handler(cls, *flist):
        # a decorator for add_handler
        def decorator(g):
            for f in flist:
                cls.add_handler(f, g)
            return g

        return decorator


def index(obj, key):
    """a function which does indexing, used in Node.__getitem__

    Args:
        obj: the object to index
        key: the key to use for indexing

    Returns:
        obj[key]
    """
    return obj[key]


class VarNode(Node):
    """A VarNode encapsulates a variable

    Args:
        value: the value of the node

    Returns:
        A VarNode object with Node properties and:
            derivshape: the shape of the VarNode, which is used for
                forward-mode differentiation
            memo: used to memoize Nodes in the trace
            backrefs: used to store back references when preparing for
                reverse-mode differentiation
    """

    def __init__(self, value):
        super().__init__(value, root=self)
        self.derivshape = self.shape
        self.memo = {}
        self.backrefs = None


class OpNode(Node):
    """A node which encapsulates an operation/function call

    Args:
        fn: the function implementing the operation
        *args: the arguments to pass to the function
        root: the root node
        **kwargs: keyword arguments for the function

    Returns:
        a Node with the following properties:
            value: the value of fn(*args, **kwargs)
            fn: the function
            args: the arguments
            kwargs: the keyword arguments.

    """

    def __init__(self, fn, *args, root=None, **kwargs):
        # force optimization of long einsums
        if fn == np.einsum and len(args) > 4:
            kwargs["optimize"] = True
        super().__init__(fn(*[value_of(a) for a in args], **kwargs), root=root)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


def getroot(*args):
    """return the first root node in a list of arguments

    Args:
        *args: the argument list

    Returns:
        the root property of the first argument which is a Node
    """
    for a in args:
        if type(a) in [list, tuple]:
            g = getroot(*a)
        else:
            g = getattr(a, "root", False)
        if Node.isNode(g):
            return g
    return None


def opnode(fn, *args, **kwargs):
    """create an opnode or return a previous one with the same signature.

    Args:
        fn: the function for the op0node
        *args: the positional args of the function
        **kwargs: the keyword args of the function

    Returns:
        an OpNode

    Note:
        OpNodes should only be created using this function, because
        it
            a) checks to see if fn can be replaced by an einsum.
            b) checks to see if the opnode has already been created

        The latter case can occur if a variable is used more than once in a
        computation, and memoizing it can lead to a gain in speed.
    """
    # try and replace fn with an einsum
    eq = einsum_equivalent(fn, *args, **kwargs)
    if eq is not None:
        fn = np.einsum
        args = eq
        kwargs = {}  # these have already been consumed by einsum_equivalent

    # see if the opnode is already in existence
    key = tuple([fn, *[hash_or_id(a) for a in args]])
    try:
        root = getroot(*args)  # the root holds the node memo cache
        if key not in root.memo:
            root.memo[key] = OpNode(fn, *args, root=root, **kwargs)
        return root.memo[key]
    except Exception as e:
        # something went wrong - this shouldn't happen normally
        print("failure in opnode", fn.__name__, e)
        print("    with key", key)
        print("    with args", args)
        return OpNode(fn, *args, root=root, **kwargs)


def check_broadcasting(*args):
    """checks that opnodes or arrays can be broadcast. Used in addition

    Args:
        *args: the arguments to broadcast

    Returns:
        the arguments broadcast to the required shapes, or OpNodes which encapsulate
        broadcasting.
    """
    shapes = [getattr(a, "shape", ()) for a in args]
    bshape = np.broadcast_shapes(*shapes)
    # fix the a[0], a[1] in this expression - use a, s instead
    return [
        np.broadcast_to(a[0], bshape)
        if a[1] != bshape and Node.isNode(a[0])
        else a[0]
        for a in zip(args, shapes)
    ]


def makeaddnode(fn, a, b):
    # an add node is an opnode with fn=np.add or np.subtract;
    # it is separated out to check broadcasting.
    # kwargs is ignored.
    a, b = check_broadcasting(a, b)
    return opnode(fn, a, b)


# all of the handled functions are automatically not traced, an opnode is created instead.


Node.add_handler(
    np.add, lambda *args, **kwargs: makeaddnode(np.add, *args, **kwargs)
)
Node.add_handler(
    np.subtract,
    lambda *args, **kwargs: makeaddnode(np.subtract, *args, **kwargs),
)

# we should add new functions here at some time.
for fn in [
    np.negative,
    np.exp,
    np.log,
    np.sqrt,
    np.abs,
    np.fabs,
    np.absolute,  # we ignore complex case
    np.sign,
    np.cos,
    np.sin,
    np.tan,
    np.sum,
    np.dot,
    np.inner,
    np.outer,
    np.transpose,
    np.reshape,
    np.matmul,
    np.trace,
    np.tensordot,
    np.broadcast_to,
    np.einsum,
    np.clip,
]:
    (
        lambda fn: Node.add_handler(
            fn, lambda *args, **kwargs: opnode(fn, *args, **kwargs)
        )
    )(fn)


@Node.register_handler(np.reciprocal)
def oneover(x):
    # np.power is a good replacement for reciprocal; no noticeable speed deficit
    return np.power(x, -1.0)


@Node.register_handler(np.diagonal)
def do_diagonal(m):
    # ignore kwargs
    idx = tuple(range(np.min(m.shape)))
    return m[idx, idx]


@Node.register_handler(np.diag)
def do_diag(m):
    # ignore kwargs for now.
    if dims(m) >= 2:
        return do_diagonal(m)
    # ndims==1
    n = shape(m)[0]
    return np.einsum("ij,j->ij", np.eye(n), m)


@Node.register_handler(np.trace)
def do_trace(m):
    return np.sum(np.diagonal(m))


@Node.register_handler(np.ravel)
def do_flatten(m):
    sz = np.size(value_of(m))
    return np.reshape(m, (sz,))


@Node.register_handler(np.squeeze)
def do_squeeze(m):
    squeezed = [i for i in shape(m) if i > 1]
    return np.reshape(m, squeezed)


@Node.register_handler(np.flip)
def do_flip(m, axis=None):
    idx = [slice(None)] * dims(m)
    idx[axis] = slice(None, None, -1)
    return m[tuple(idx)]


@Node.register_handler(np.fliplr)
def do_fliplr(m, axis=None):
    return m[:, ::-1]


@Node.register_handler(np.flipud)
def do_flipud(m, axis=None):
    return m[::-1]


@Node.register_handler(np.multiply)
def mult(a, b):
    # multiply deals with a few special cases that turn up
    # quite frequently
    if np.isscalar(a):
        if a == 0:
            return 0
        if a == 1:
            return b
        if a == -1:
            return np.negative(b)
    if np.isscalar(b):
        if b == 0:
            return 0
        if b == 1:
            return a
        if b == -1:
            return np.negative(a)
    # we do need to broadcast here rather than leave it to einsum_equiv
    # because that doesn't always work
    a, b = check_broadcasting(a, b)
    return opnode(np.multiply, a, b)


@Node.register_handler(np.true_divide, np.divide)
def tdiv(a, b):
    # divide & true_divide are the same as far as I'm concerned.
    if np.isscalar(a):
        if a == 1:
            return np.power(b, -1.0)
        if a == -1:
            return -np.power(b, -1.0)
    # fallthrough case.
    # np.reciprocal doesn't work with integer b so use divide here
    return np.multiply(a, (np.power(b, -1.0) if Node.isNode(b) else 1.0 / b))


@Node.register_handler(np.power)
def powernode(a, b):
    # either a or b or both are Nodes
    if not Node.isNode(b):
        # a**b is simpler when b isn't a node object (often the case)
        if isinstance(a, OpNode) and a.fn == np.power:
            if a.args[1] * b == 0:
                return a.args[0]
            else:
                return opnode(np.power, a.args[0], a.args[1] * b)
        else:
            return opnode(np.power, a, b)
    else:
        return np.exp(np.multiply(np.log(a), b))


@Node.register_handler(np.maximum)
def maximum(a, b):
    # when equal, returns the average of a & b
    va = value_of(a)
    vb = value_of(b)
    eq = va == vb
    return ((va > vb) + 0.5 * eq) * a + ((vb > va) + 0.5 * eq) * b


@Node.register_handler(np.minimum)
def minimum(a, b):
    # when equal, returns the average of a & b
    va = value_of(a)
    vb = value_of(b)
    eq = va == vb
    return ((va < vb) + 0.5 * eq) * a + ((vb < va) + 0.5 * eq) * b


@Node.register_handler(np.amax)
def arraymax(a, axis=None, out=None, *, keepdims=np._NoValue):
    idx = np.argmax(value_of(a), axis=axis, keepdims=True)
    return pick(a, idx, axis, keepdims)


@Node.register_handler(np.amin)
def arraymin(a, axis=None, out=None, *, keepdims=np._NoValue):
    idx = np.argmin(value_of(a), axis=axis, keepdims=True)
    return pick(a, idx, axis, keepdims)


def pick(a, idx, axis, keepdims):
    # translates the argmax index to something reasonable
    if axis is None:
        return a[np.unravel_index(idx, shape=shape(a))]
    # otherwise ... complicated
    indices = np.meshgrid(*[range(n) for n in shape(a)], indexing="ij")
    indices[axis] = idx
    pic = a[tuple(indices)]
    # pic is the same size as a with the dimension along axis repeated,
    # so this has to be removed
    indices = [slice(None, None, None)] * dims(a)
    indices[axis] = 0
    pic = pic[tuple(indices)]
    # insert the 1-sized index if keepdims is True
    if keepdims is True:
        sh = list(shape(a))
        sh[axis] = 1
        pic = np.reshape(pic, sh)
    return pic


def notrace(f):
    """decorator to stop tracing inside a function

    Args:
        f: the function to be no-traced

    Returns:
        the function wrapped to return an OpNode instead
        of being evaluated.
    """
    if getattr(f, "untraced", False):
        return f  # it's already notraced

    @wraps(f)
    def nt(*args, **kwargs):
        if any(Node.isNode(a) for a in args):
            # use nt as the function so the notrace persists
            return opnode(nt, *args, **kwargs)
        else:
            # this is the call to work out the node value
            return f(*args, **kwargs)

    nt.untraced = True

    return nt


# numpy tuple args don't work well with
# the derivative operations, so we break them out


@Node.register_handler(np.vstack)
def do_vstack(T):
    T = list(T)
    for i, t in enumerate(T):
        if dims(t) == 1:
            T[i] = np.reshape(t, (1, shape(t)[0]))
    return opnode(concat_spread, *T, axis=0)


@Node.register_handler(np.hstack)
def do_hstack(T):
    # the numpy implementation of hstack for 1D arrays is
    # daft, but kept here anyway.
    return opnode(concat_spread, *T, axis=dims(T[0]) - 1)


@Node.register_handler(np.concatenate)
def do_concat(T, axis=0):
    # When axis is None, we just flatten everything &
    # concatenate along axis=0. This is so the derivative
    # works with this case.
    if axis is None:
        T = [np.ravel(a) for a in T]
        axis = 0
    return opnode(concat_spread, *T, axis=axis)


@notrace
def concat_spread(*args, axis=0):
    return np.concatenate(args, axis=axis)


from .lib import diffmat


@Node.register_handler(np.diff)
def do_diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    # diff is implemented by numpy as indexing and subtraction,
    # so we re-implement it here, changing some checks.
    # It would be nice to just use np.diff.__wrapped__ but this calls
    # asanyarray & this can't be wrapped inside np.diff - calls it without
    # the dispatch mechanism
    if n == 0:
        return a
    if n < 0:
        raise ValueError("order must be non-negative but got " + repr(n))
    nd = dims(a)
    if axis < 0:
        axis = nd + axis

    combined = []
    if prepend is not np._NoValue:
        if isinstance(prepend, (list, tuple)):
            prepend = np.array(prepend)
        if dims(prepend) == 0:
            shape_ = list(shape(a))
            shape_[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape_))
        combined.append(prepend)

    combined.append(a)

    if append is not np._NoValue:
        if isinstance(append, (list, tuple)):
            append = np.array(append)
        if dims(append) == 0:
            shape_ = list(shape(a))
            shape_[axis] = 1
            append = np.broadcast_to(append, tuple(shape_))
        combined.append(append)

    if len(combined) > 1:
        a = np.concatenate(combined, axis)

    # create the difference matrix directly because it will
    # come out anyway in the derivative
    dmat = diffmat(shape(a)[axis], n=n)
    didx = "ijklmnopqrstuvwxyz"[:2]
    idx = "ijklmnopqrstuvwxyz"[2 : dims(a) + 2]
    idx = idx.replace(idx[axis], didx[1])
    result = idx.replace(idx[axis], didx[0])
    return np.einsum(didx + "," + idx + "->" + result, dmat, a)

    # the code below was taken from np.diff & isn't used here
    # Might be reinstated if there are some efficiency problems with
    # the diff mat approach.

    # make slices
    slice1 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)

    slice2 = [slice(None)] * nd
    slice2[axis] = slice(None, -1)
    slice2 = tuple(slice2)

    for _ in range(n):
        a = a[slice1] - a[slice2]
    return a


from .lib import np_correlate, np_convolve
from .convolve import tensorcorrelate as tcorr, tensorconvolve as tconv


@Node.register_handler(np.correlate)
def _np_correlate(a, v, mode="valid"):
    if not Node.isNode(a):
        return tcorr(a, v, mode=mode)
    else:
        return np_correlate(a, v, mode=mode)


@Node.register_handler(np.convolve)
def _np_convolve(a, v, mode="full"):
    if not Node.isNode(a):
        return tconv(a, v, mode=mode)
    else:
        return np_convolve(a, v, mode=mode)

# numpy functions which aren't yet implemented will raise an error
def not_implemented(f):
    def fail(*args, **kwargs):
        raise NotImplementedError(
            f"{f.__name__} is not implemented for tracing"
        )

    return fail


# a selection of non-implemented functions, fill out the rest later

for fn in [
    np.stack,  # all arrays the same size
    np.roll,
    np.rot90,
    np.gradient,
    np.ediff1d,
    np.less,
    np.greater,  # etc.
    np.pad,
]:
    (
        lambda fn: Node.add_handler(
            fn,
            lambda *args, **kwargs: OpNode(
                not_implemented(fn), *args, **kwargs
            ),
        )
    )(fn)

# monkeypatching scipy.signal functions. This always works if tracing is imported
# before scipy.signal, and sometimes works if imported after, so long as
# convolve is not dereferenced until after importing tracing.py

from .monkey import override


@override("scipy.signal.correlate", np.ndarray, [VarNode, OpNode])
def scipy_correlate(in1, in2, mode="full", method="auto"):
    if dims(in1) != dims(in2):
        raise ValueError(
            "volume and kernel should have the same dimensionality"
        )
    return tcorr(in1, in2, mode=mode, axes=-1)


@override("scipy.signal.convolve", np.ndarray, [VarNode, OpNode])
def scipy_convolve(in1, in2, mode="full", method="auto"):
    if dims(in1) != dims(in2):
        raise ValueError(
            "volume and kernel should have the same dimensionality"
        )
    return tconv(in1, in2, mode=mode, axes=-1)


# ## Backwards References.
#
# for reverse mode derivatives.


def backtrace(node):
    # clear previous backtraces - they are no longer relevant.
    node.root.backrefs = defaultdict(set)
    traced = set()

    def run_backtrace(node):
        if node in traced:
            return
        traced.add(node)
        for i, a in enumerate(getattr(node, "args", [])):
            argno = i - 1 if node.fn == np.einsum else i
            if Node.isNode(a):
                node.root.backrefs[a].add((node, argno))
                run_backtrace(a)

    # notice that node doesn't get put into backrefs
    run_backtrace(node)
    return node.root.backrefs
