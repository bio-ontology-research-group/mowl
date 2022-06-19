from os.path import exists
import pickle as pkl

def exist_files(*args):
    ex = True
    for arg in args:
        ex &= exists(arg)

    return ex

def load_pickles(*args):
    pkls = []
    for arg in args:
        with open(arg, "rb") as f:
            pkls.append(pkl.load(f))

    return tuple(pkls)

def save_pickles(*args):
    for data, filename in args:
        with open(filename, "wb") as f:
            pkl.dump(data, f)
