from .base import RemoteDataset
import math
import random
import numpy as np
import gzip
import os


DATA_URL = 'https://bio2vec.cbrc.kaust.edu.sa/data/mowl/ppi_yeast.tar.gz'

class PPIYeastDataset(RemoteDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(url=DATA_URL)
