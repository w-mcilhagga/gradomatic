import numpy as np
import pickle as pkl

def load_pkl(fname):
    '''easy pickle loading'''
    with open(fname,'rb') as f:
        data = pkl.load(f)
    return data

def dump_pkl(fname, obj):
    '''easy pickle dumping'''
    with open(fname,'wb') as f:
        pkl.dump(obj, f)

def loaddata(subj, contrast, mode):
    '''loads a data file from the data folder
    Args:
        subj (string): the subject id
        contrast (int): the contrast percentage
        mode (string): either '5noise' or 'thresh' (0 noise)
        
    Returns:
        the dict in that data file, with keys
            target - (1D array) the target profile
            pedestal_test - (2D array) pedestal_test[i,:] is the pedestal+test profile on trial i
            pedestal - (2D array) pedestal[i,:] is the pedestal profile on trial i
            correct - (1D array) correct[i] is 1 if the observer made the correct choice
            phase - (1D array) phase[i] is the original phase of the pedestal, test on trial i 
                (the pedestal_test and pedestal data has all phases set the same by flipping left-to-right as needed)
            contrast - (1D array) contrast[i] is the contrast of the test on trial i
    '''
    fname = 'data/'+subj+'_'+str(contrast)+'_'+mode+'.pkl'
    data = load_pkl(fname)
    # fix the error where the target was saved normalized instead of peak=1
    data['target'] = data['target']/np.max(data['target'])
    return data