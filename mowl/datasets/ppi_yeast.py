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
DATA_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast.tar.gz'
SLIM_DATA_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast_slim.tar.gz'

class PPIYeastDataset(RemoteDataset):
    warnings.warn("Importing this dataset as 'mowl.datasets.ppi_yeast.PPIYeastDataset' will be removed in version 1.0.0. Consider using 'mowl.datasets.builtin.PPIYeastDataset' instead", FutureWarning, stacklevel=2)
    def __init__(self, url=None):
        
            
        super().__init__(url=DATA_URL if not url else url)
        
    def get_evaluation_classes(self):
        """Classes that are used in evaluation
        """
        classes = super().get_evaluation_classes()
        proteins = set()
        for owl_cls in classes:
            if "http://4932" in owl_cls:
                proteins.add(owl_cls)
        return proteins

    def get_evaluation_property(self):
        return "http://interacts_with"
    
class PPIYeastSlimDataset(PPIYeastDataset):
    warnings.warn("Importing this dataset as 'mowl.datasets.ppi_yeast.PPIYeastSlimDataset' will be removed in version 1.0.0. Consider using 'mowl.datasets.builtin.PPIYeastSlimDataset' instead", FutureWarning, stacklevel=2)
    def __init__(self, *args, **kwargs):
        super().__init__(url=SLIM_DATA_URL)
        
