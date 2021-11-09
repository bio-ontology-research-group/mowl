import pathlib

from .base import RemoteDataset, PathDataset
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

class PPIYeastLocalTestDataset(PathDataset):

    def __init__(self, *args, **kwargs):
        dataset_root = pathlib.Path('../data/ppi_yeast_localtest')
        self.data_root = pathlib.Path('../data/')
        self.dataset_name = 'ppi_yeast_localtest'
        super().__init__(
            os.path.join(dataset_root, 'ontology.owl'),
            os.path.join(dataset_root, 'valid.owl'),
            os.path.join(dataset_root, 'test.owl'))

    def eval_classes(self, classes):
        """Classes that are used in evaluation
        """
        res = {}
        for k, v in classes.items():
            if k.startswith('<http://4932.'):
                res[k] = v
        return res
