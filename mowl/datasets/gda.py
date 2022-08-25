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

from mowl.datasets.builtin.gda import DATA_HUMAN_URL, DATA_HUMAN_EL_URL, DATA_MOUSE_URL, DATA_MOUSE_EL_URL

class GDADataset(RemoteDataset):

    def __init__(self, url=None):
        super().__init__(url=url)

    def get_evaluation_classes(self):
        """Classes that are used in evaluation
        """
        classes = super().get_evaluation_classes()
        genes = set()
        diseases = set()
        for owl_cls in classes:
            if owl_cls[7:].isnumeric():
                genes.add(owl_cls)
            if "OMIM_" in owl_cls:
                diseases.add(owl_cls)
                
        return genes, diseases

    def get_evaluation_property(self):
        return "http://is_associated_with"

@deprecated(
    reason = "Importing this dataset as `mowl.datasets.gda.GDAHumanDataset` will be removed in version 1.0.0. Consider using `mowl.datasets.builtin.GDAHumanDataset`",
    version = "0.1.0"
)
class GDAHumanDataset(GDADataset):
    def __init__(self):
        super().__init__(url=DATA_HUMAN_URL)

@deprecated(
    reason = "Importing this dataset as `mowl.datasets.gda.GDAMouseDataset` will be removed in version 1.0.0. Consider using `mowl.datasets.builtin.GDAMouseDataset`",
    version = "0.1.0"
)
class GDAMouseDataset(GDADataset):
    def __init__(self):
        super().__init__(url=DATA_MOUSE_URL)
