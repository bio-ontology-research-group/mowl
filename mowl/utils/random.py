import random
import torch as th
import numpy as np
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def seed_generators(seed=42):
    """Seed the generators of :mod:`random`, NumPy and PyTorch (the global ones) and nothing
    else: a run that draws from them with the same seed draws the same numbers."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
