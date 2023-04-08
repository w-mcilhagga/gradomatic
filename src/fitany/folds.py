# -*- coding: utf-8 -*-
"""
folds for cross-validation
"""
import numpy as np

def makefold(nfolds, foldno, *data, mode='interleaved'):
    '''splits data into training & test sets
    
    Args:
        nfolds (int): number of folds
        foldno (int): the fold number in the range 0...nfolds-1
        data: the data arrays. Every data array must have the same 
            shape[0] = number of observations
        mode (string): the fold mode (interleaved or random)
        
    Returns:
        train, test, which is the data tuple split into training set and test set
    '''
    n = len(data[0])
    foldidx = list(range(nfolds))*(n//nfolds+1)
    foldidx = np.array(foldidx[:n])
    if mode=='random':
        np.random.shuffle(foldidx)
    trainidx = foldidx!=foldno
    testidx = foldidx==foldno
    train = [d[trainidx] for d in data]
    test = [d[testidx] for d in data]
    return train, test
    
def folds(nfolds, *data, mode='interleaved', count=0):
    ''' generator for cross-validation data samples
    
    Args:
        nfolds (int): number of folds
        data: the data arrays. Every data array must have the same 
            shape[0] = number of observations
        mode (string): the fold mode (interleaved or random)    
        count (int): for random mode only, specifies the number
            of splist to perform, defaults to nfolds.
            
    Yields:
        train-test splits of the data until they are finished.
    '''
    if mode=='interleaved':
        count=0
    for i in range(count or nfolds):
        yield makefold(nfolds, i%nfolds, *data, mode=mode)
        
from itertools import product

def compare_cv(test1, test2):
    # compares the paired difference between the test results
    # should take account of the variance of the differences too.
    data = np.array(test1)-np.array(test2)
    print(data)
    actual = np.sum(data)/np.std(data)
    lt = eq = gt = 0
    count = 0
    results=[]
    for s in product(*[[1.0,-1.0]]*len(data)):
        x = np.sum(data*s)/np.std(data*s)
        results.append(x)
        if x>actual:
            gt += 1
        elif x==actual:
            eq += 1
        else:
            lt += 1
        count += 1
    # return the p-value of the best one-sided difference
    print(np.mean(results), np.std(results))
    return actual, min(lt+eq, gt+eq)/count

    
    