from os.path import isdir
import numpy as np

    
def FixDirPath(path: str):
    if path == "":
        return ""
    if path[-1] != '/':
        path += '/'
    assert isdir(path), "Wrong path"
    return path

