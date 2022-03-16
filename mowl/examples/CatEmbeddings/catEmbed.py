#!/usr/bin/env python
import click as ck
import numpy as np
import pandas as pd
import pickle
import gzip
import os
import sys
import logging
import torch as th
logging.basicConfig(level=logging.DEBUG)   
sys.path.insert(0, '')
sys.path.append('../../../')

from mowl.datasets.base  import PathDataset
#from mowl.develop.catEmbeddings.modelWithFeats import CatEmbeddings
from mowl.develop.catEmbeddings.model import CatEmbeddings

@ck.command()
@ck.option(
    '--config', '-c', help="Configuration file in config/")

def main(config):
    logging.info(f"Number of cores detected: {os.cpu_count()}")


    go_slim = True

    if go_slim:
        ontology = "data/goslim_yeast.owl"
        val = "data/trans_yeast.owl"
    else:
        ontology = "data/go.owl"
        val = "data/trans.owl"
   
    ds = PathDataset(ontology, val, None)
    device = "cuda" if th.cuda.is_available() else "cpu"
    logging.info("Device: %s", device)

    
    model = CatEmbeddings(ds, 128, 128, device = device)

    model.train()
    model.evaluate()


if __name__ == '__main__':
    main()
