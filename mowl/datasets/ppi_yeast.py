import pathlib

from .base import RemoteDataset, PathDataset
import math
import random
import numpy as np
import gzip
import os
from java.util import HashSet

DATA_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast.tar.gz'
SLIM_DATA_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast_slim.tar.gz'

class PPIYeastDataset(RemoteDataset):

    def __init__(self, url=None):
        super().__init__(url=DATA_URL if not url else url)

    def get_evaluation_classes(self):
        """Classes that are used in evaluation
        """
        classes = super().get_evaluation_classes()
        eval_classes = HashSet()
        for owl_cls in classes:
            if owl_cls.toString().startsWith("<http://4932"):
                eval_classes.add(owl_cls)
        return eval_classes

class PPIYeastSlimDataset(PPIYeastDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(url=SLIM_DATA_URL)


class PPIYeastLocalTestDataset(PathDataset):

    def __init__(self, *args, **kwargs):
        dataset_root = pathlib.Path('../data/ppi_yeast_localtest')
        self.data_root = pathlib.Path('../data/')
        self.dataset_name = 'ppi_yeast_localtest'
        super().__init__(
            os.path.join(dataset_root, 'ontology.owl'),
            os.path.join(dataset_root, 'valid.owl'),
            os.path.join(dataset_root, 'test.owl'))

    def get_evaluation_classes(self):
        """Classes that are used in evaluation
        """
        classes = super().get_evaluation_classes()
        eval_classes = HashSet()
        for owl_cls in classes:
            if owl_cls.toString().startsWith("<http://4932"):
                eval_classes.add(owl_cls)
        return eval_classes
