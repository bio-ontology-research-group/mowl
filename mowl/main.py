#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import pickle
import gzip
import os
import sys
import logging

sys.path.insert(0, '')

from mowl.datasets import PPIYeastDataset
from mowl.onto2vec.model import Onto2Vec
from mowl.elembeddings.model import ELEmbeddings

@ck.command()
def main():
    ds = PPIYeastDataset()
    model = ELEmbeddings(ds)
    model.train()


if __name__ == '__main__':
    main()
