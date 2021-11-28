#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import pickle
import gzip
import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG)   
sys.path.insert(0, '')
sys.path.append('../../../')

from mowl.datasets.base  import PathDataset
from mowl.catEmbeddings.model import CatEmbeddings

@ck.command()
@ck.option(
    '--config', '-c', help="Configuration file in config/")

def main(config):
    logging.info(f"Number of cores detected: {os.cpu_count()}")


    ontology = "data/goslim_yeast.owl"
    val = "data/trans_yeast.owl"
    
#    ontology = "data/go.owl"
#    val = "data/trans.owl"
   
    ds = PathDataset(ontology, val, "")
        
    model = CatEmbeddings(ds, 32)

    model.train()
    model.evaluate()


if __name__ == '__main__':
    main()
