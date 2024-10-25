from os.path import isdir
import numpy as np
import numpyro


def SampleHPDIMean(arr, prob=0.5):
    hpdi = numpyro.diagnostics.hpdi(arr, prob=prob, axis=0)
    return np.mean(arr, axis=0, where=np.logical_and(hpdi[0]<arr, arr<hpdi[1]))

"""def SampleHPDIMedian(arr, prob=0.5):
    hpdi = numpyro.diagnostics.hpdi(arr, prob=prob, axis=0)
    return np.median(arr, axis=0, where=np.logical_and(hpdi[0]<arr, arr<hpdi[1]))"""

def SampleHPDIMiddle(arr, prob=0.5):
    return numpyro.diagnostics.hpdi(arr, prob=prob, axis=0).mean(axis=0)

def SampleHPDI(arr, prob=0.5):
    return numpyro.diagnostics.hpdi(arr, prob=prob, axis=0)

def AggregateSample(sample, mode, prob=0.5):
    assert mode in ['mean', 'median', 'mean_hpdi', 'mid_hpdi'] 
    if mode == 'mean':
        return np.mean(sample, axis=0)
    elif mode == 'median': 
        return np.median(sample, axis=0)
    elif mode == 'mean_hpdi':
        return SampleHPDIMean(sample, prob)
    elif mode == 'mid_hpdi':
        return SampleHPDIMiddle(sample, prob)
    
def FixDirPath(path: str):
    if path == "":
        return ""
    if path[-1] != '/':
        path += '/'
    assert isdir(path), "Wrong path"
    return path