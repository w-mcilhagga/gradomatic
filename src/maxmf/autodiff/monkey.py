# -*- coding: utf-8 -*-
"""
monkeypatching
"""

import importlib
from itertools import product

class Dispatcher:
    def __init__(self, defaultfn):
        self.signature = {}
        self.defaultfn = defaultfn
        
    def add(self, fn, *types):
        types = [t if type(t) in (list, tuple) else (t,) for t in types ]
        for p in product(*types):
            self.signature[p] = fn
        
    def __call__(self, *args, **kwargs):
        types = tuple(type(a) for a in args)
        return self.signature.get(types, self.defaultfn)(*args, **kwargs)
    

def patch(path, f, *types):
    # does a multi-dispatch on the signature 
    # for the function
    parts = path.split('.')
    module = importlib.import_module('.'.join(parts[:-1]))
    currentfn = getattr(module, parts[-1])
    if not isinstance(currentfn, Dispatcher):
        # first patch
        currentfn = Dispatcher(currentfn)
        setattr(module, parts[-1], currentfn)
    currentfn.add(f, *types)
    
def original(f):
    # returns the defaultfn 
    return getattr(f, 'defaultfn', f)

def restore(path):
    # returns the module to its original state with respect to the path.
    parts = path.split('.')
    module = importlib.import_module('.'.join(parts[:-1]))
    current = getattr(module, parts[-1])
    setattr(module, parts[-1], original(current))
    
def override(path, *types):
    # returns a decorator
    def decorate(f):
        patch(path, f, *types)
        return f
    return decorate


    
