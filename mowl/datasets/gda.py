import pathlib

from .base import RemoteDataset, PathDataset
import math
import random
import numpy as np
import gzip
import os
from java.util import HashSet

DATA_HUMAN_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/gda_human.tar.gz'
DATA_MOUSE_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/gda_mouse.tar.gz'

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
    
class GDAHumanDataset(GDADataset):
    def __init__(self):
        super().__init__(url=DATA_HUMAN_URL)

class GDAMouseDataset(GDADataset):
    def __init__(self):
        super().__init__(url=DATA_MOUSE_URL)
