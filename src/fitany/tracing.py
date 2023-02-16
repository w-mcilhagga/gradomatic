'''Tracing.

Tracing a function call and returning a computation graph is the core of automatic 
differentiation. An important issue is to make sure the tracer can deal with 
`numpy` dispatch, and can convert `numpy` functions to `einsum` where 
appropriate, since we know how to take the derivative of `einsum`s.

Nodes

A node is an object used to trace a computation and produce a computation tree 
for differentiation. Given a function `f(a,b,c,...)`, to trace the computation 
of, say, `b`, you would call
    `tree = f(a,VarNode(b),c,...)`.

The computation tree is composed of `VarNodes` which encapsulate a variable 
and `OpNodes` which encapsulate an operation. The return value `tree` is an OpNode.
For OpNodes, `tree.value` is the value `f(a,b,c,...)`. The tree can be operated on to 
get derivatives.
'''

import operator as op
import numpy as np

def dims(x):
    '''the number of dimensions of x
    
    Args:
        x: the object
        
    Returns:
        x.ndim if x is a numpy array, otherwise 0
    '''
    return getattr(x, 'ndim', 0)
    
def shape(x):
    '''the shape of x
    
    Args:
        x: the object
        
    Returns:
        x.shape if x is a numpy array, otherwise ()
    '''
    return getattr(x, 'shape', ())

def value_of(x):
    '''return the value of a Node or object
    
    Args:
        x: the object
    
    Returns:
        x.value if it exists, otherwise x
    '''
    return getattr(x, 'value', x)


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
            
    Note:
        This is the base class for `VarNode`s and `OpNode`s.
    """
    
    @staticmethod
    def isNode(n):
        ''' checks if an object is a Node'''
        return isinstance(n, Node)
    
    @staticmethod
    def iszero(n):
        '''checks if an object is zero and not a Node'''
        return not Node.isNode(n) and np.all(n==0)
    
    def __init__(self, value, root=None):
        self.value = value 
        self.shape = shape(value)
        self.dims =  dims(value)
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
    
    # comparison overrides - these return a boolean rather than a node:
    
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
    
    # overrides for other python operations:
    
    def __getitem__(self, key):
        # indexing the node; opnode is defined below
        return opnode(index, self, key)
    
    def __hash__(self):
        # required because __eq__ was overridden
        return id(self)
    
    # numpy dispatch for when one of the objects is a numpy array
    
    handled_funcs = {}
    
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # dispatcher for numpy ufuncs
        if method != '__call__' or ufunc not in self.handled_funcs:
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
    def register_handler(cls, f):
        # a decorator for add_handler
        def decorator(g):
            cls.add_handler(f, g)
            return g
        return decorator


def index(obj, key):
    '''a function which does indexing, used in Node.__getitem__
    
    Args:
        obj: the object to index
        key: the key to use for indexing
        
    Returns:
        obj[key]
    '''
    return obj[key]


from .einsum_equivalent import einsum_equivalent

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
        if fn == np.einsum and len(args)>4:
            kwargs['optimize'] = True
        super().__init__(fn(*[value_of(a) for a in args], **kwargs), root=root)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        
from .memoize import hash_or_id 

def getroot(*args):
    '''return the first root node in a list of arguments
    
    Args:
        *args: the argument list
        
    Returns:
        the root property of the first argument which is a Node
    '''
    for a in args:
        g = getattr(a, 'root', False)
        if Node.isNode(g):
            return g
    return None

def opnode(fn, *args, **kwargs):
    '''create an opnode or return a previous one with the same signature.
    
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
        computation, and memoizing it can lead to a big gain in speed.
    '''
    # try and replace fn with an einsum
    eq = einsum_equivalent(fn, *args, **kwargs)
    if eq is not None:
        fn = np.einsum
        args = eq
        kwargs = {} # these have already been consumed by einsum_equivalent
        
    # see if the opnode is already in existence
    key = tuple([fn, *[hash_or_id(a) for a in args]])
    try:
        root = getroot(*args) # the root holds the node meo cache
        if key not in root.memo:
            root.memo[key] = OpNode(fn, *args, root=root, **kwargs)
        return root.memo[key]
    except Exception as e:
        # something went wrong - this shouldn't happen normally
        print('memo failure in opnode', fn.__name__, e)
        print('    with key', key)
        print('    with args', args)
        return OpNode(fn, *args, root=root, **kwargs)
    
def check_broadcasting(*args):
    '''checks that opnodes or arrays can be broadcast. Used in addition
    
    Args:
        *args: the arguments to broadcast
        
    Returns:
        the arguments broadcast to the required shapes, or OpNodes which encapsulate
        broadcasting.
    '''
    shapes = [getattr(a, 'shape', ()) for a in args]
    bshape = np.broadcast_shapes(*shapes)
    return [ np.broadcast_to(a[0], bshape) if a[1]!=bshape and Node.isNode(a[0]) else a[0] for a in zip(args, shapes)]

from collections import defaultdict
    
def makeaddnode(fn, a, b, **kwargs):
    # an add node is an opnode with fn=np.add or np.subtract;
    # it is separated out to check broadcasting. 
    # kwargs is ignored.
    a, b = check_broadcasting(a, b)
    return opnode(fn, a, b, **kwargs)

# all of the handled functions are automatically not traced, an opnode is created instead.

Node.add_handler(np.add, lambda *args, **kwargs: makeaddnode(np.add, *args, **kwargs))
Node.add_handler(np.subtract, lambda *args, **kwargs: makeaddnode(np.subtract, *args, **kwargs))

# we should add new functions here at some time.
for fn in [np.negative, 
           np.exp, 
           np.log,
           np.sqrt,
           np.abs,
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
           np.diag,
           np.tensordot,
           np.broadcast_to,
           np.einsum,
           np.clip]:
    (lambda fn: Node.add_handler(fn, lambda *args, **kwargs: opnode(fn, *args, **kwargs)))(fn)

# np.power is a good replacement for reciprocal; no noticeable speed deficit
@Node.register_handler(np.reciprocal)
def oneover(x):
    return np.power(x, -1.0)
    
# multiply deals with a few special cases that turn up
# quite frequently 
@Node.register_handler(np.multiply)
def mult(a,b):
    if np.isscalar(a):
        if a==1:
            return b
        if a==-1:
            return np.negative(b)
    if np.isscalar(b):
        if b==1:
            return a
        if b==-1:
            return np.negative(a)
    return opnode(np.multiply, a, b) # can't do np.multiply(a,b) because infinite recursion
        
# divide & true_divide are the same as far as I'm concerned.
def tdiv(a,b):
    if np.isscalar(a):
        if a==1:
            return np.power(b, -1.0)
        if a==-1:
            return -np.power(b, -1.0)
    else:
        # np.reciprocal doesn't work with integer b so use divide here
        return np.multiply(a, np.power(b,-1.0) if Node.isNode(b) else 1.0/b)

Node.add_handler(np.true_divide, tdiv)
Node.add_handler(np.divide, tdiv)

@Node.register_handler(np.power)
def powernode(a, b):
    # either a or b or both are Nodes
    if not Node.isNode(b):
        # a**b is simpler when b isn't a node object (often the case)
        if isinstance(a, OpNode) and a.fn == np.power:
            if a.args[1]*b==0:
                return a.args[0]
            else:
                return opnode(np.power, a.args[0], a.args[1]*b)
        else:
            return opnode(np.power, a, b)
    else:
        return np.exp(np.multiply(np.log(a), b))


# numpy functions which aren't yet implemented will raise an error
def not_implemented(f):
    def fail(*args, **kwargs):
        raise NotImplementedError(f'{f.__name__} is not implemented for tracing')
    return fail

for fn in [np.flip, np.flipud, np.fliplr, np.concatenate, np.hstack, np.vstack]:
    (lambda fn: Node.add_handler(fn, lambda *args, **kwargs: OpNode(not_implemented(fn), *args, **kwargs)))(fn)




from functools import wraps

def notrace(f):
    """decorator to stop tracing inside a function
    
    Args:
        f: the function to be no-traced
        
    Returns:
        the function wrapped to return an OpNode instead
        of being evaluated.
    """
    if getattr(f, 'untraced', False):
        return f # it's already notraced
    
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



# ## Backwards References.
# 
# Useful for reverse mode derivatives.


from collections import defaultdict

def backtrace(node):
    # clear previous backtraces - they are no longer relevant.
    node.root.backrefs = defaultdict(set)
    traced = set()
    
    def run_backtrace(node):
        if node in traced:
            return
        traced.add(node)
        for i, a in enumerate(getattr(node, 'args', [])):
            argno = i-1 if node.fn==np.einsum else i
            if Node.isNode(a):
                node.root.backrefs[a].add((node, argno))
                run_backtrace(a)
                
    # notice that node doesn't get put into backrefs
    run_backtrace(node)
    return node.root.backrefs
