import os
import pickle
import numpy as np

def mkfile(*args):
    os.makedirs(os.path.join(*args), exist_ok=True)

def npsave(item, *args):
    if len(args) > 1: mkfile(*args[:-1])
    np.save(os.path.join(*args), item)

def npload(*args): return np.load(os.path.join(*args))

def pklsave(item, *args):
    if len(args) > 1: mkfile(*args[:-1])
    f = open(os.path.join(*args), "wb")
    pickle.dump(item, f)
    f.close()

def pklload(*args):
    f = open(os.path.join(*args), "rb")
    data = pickle.load(f)
    f.close()
    return data

def write_logfile(f, message):
    with open(f, 'a') as logfile: 
        logfile.write(message + '\n')