#v0.0.30
import pathlib

from .base import RemoteDataset, PathDataset
import math
import random
import numpy as np
import gzip
import os
from java.util import HashSet
import warnings
from deprecated.sphinx import deprecated

DATA_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast.tar.gz'
SLIM_DATA_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast_slim.tar.gz'

@deprecated(
    reason = "Importing this dataset as `mowl.datasets.ppi_yeast.PPIYeastDataset` will be removed in version 1.0.0. Consider using :class:`mowl.datasets.builtin.PPIYeastDataset` instead",
    version = "0.1.0"
)
class PPIYeastDataset(RemoteDataset):
    
    def __init__(self, url=None):        
        super().__init__(url=DATA_URL if not url else url)
        self._evaluation_classes = None
        self._loaded_eval_data = False
        
    def get_evaluation_classes(self):
        """Classes that are used in evaluation
        """
        if self._loaded_eval_data:
            return self._evaluation_classes
        
        classes = super().get_evaluation_classes()
        proteins = set()
        for owl_cls in classes:
            if "http://4932" in owl_cls:
                proteins.add(owl_cls)
        self._evaluation_classes = proteins
        return proteins

    def get_evaluation_property(self):
        return "http://interacts_with"

@deprecated(
    reason = "Importing this dataset as `mowl.datasets.ppi_yeast.PPIYeastSlimDataset` will be removed in version 1.0.0. Consider using :class:`mowl.datasets.builtin.PPIYeastSlimDataset` instead",
    version = "0.1.0"
)
class PPIYeastSlimDataset(PPIYeastDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(url=SLIM_DATA_URL)
        
