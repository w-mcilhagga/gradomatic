#!/usr/bin/env python
# coding: utf-8


from functools import wraps

def memoize(f):
    # memoizes a function on the args, 
    # and if they aren't hashable, it 
    # just calls it
    @wraps(f)
    def memo_f(*args, **kwargs):
        try:
            return memo_f.cache[args]
        except:
            try:
                memo_f.cache[args] = f(*args, **kwargs)
                return memo_f.cache[args]
            except:
                return f(*args, **kwargs)
    memo_f.cache = {}
    return memo_f

def hash_or_id(x):
    # returns x if hashable otherwise the id
    try:
        if hash(x):
            return x
        else:
            return id(x)
    except:
        return id(x)
    
def memoize_by_id(f):
    # memoizes using id if necessary; assumes that
    # the object pointed to by id doesn't change
    @wraps(f)
    def memo_f(*args, **kwargs):
        args = tuple(hash_or_id(a) for a in args)
        try:
            return memo_f.cache[args]
        except:
            memo_f.cache[args] = f(*args, **kwargs)
            return memo_f.cache[args]
    memo_f.cache = {}
    return memo_f            







