from .base import RemoteDataset
import math
import random
import numpy as np
import gzip
import os


DATA_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast.tar.gz'
SLIM_DATA_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast_slim.tar.gz'

class PPIYeastDataset(RemoteDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(url=DATA_URL)

    def eval_classes(self, classes):
        """Classes that are used in evaluation
        """
        res = {}
        for k, v in classes.items():
            if k.startswith('<http://4932.'):
                res[k] = v
        return res

class PPIYeastSlimDataset(RemoteDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(url=SLIM_DATA_URL)

    def eval_classes(self, classes):
        """Classes that are used in evaluation
        """
        res = {}
        for k, v in classes.items():
            if k.startswith('<http://4932.'):
                res[k] = v
        return res
