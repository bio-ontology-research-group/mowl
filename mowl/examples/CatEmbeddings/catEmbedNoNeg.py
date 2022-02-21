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

from mowl.datasets.ppi_yeast import PPIYeastSlimDataset, PPIYeastDataset
#from mowl.develop.catEmbeddings.modelWithFeats import CatEmbeddings
from mowl.develop.catEmbeddings.modelNoNeg import CatEmbeddings

@ck.command()
@ck.option(
    '--config', '-c', help="Configuration file in config/")

def main(config):
    logging.info(f"Number of cores detected: {os.cpu_count()}")


   
    ds = PPIYeastDataset()
    
    model = CatEmbeddings(ds, 4096*4, embedding_size = 1024)
    model.train()
#    model.evaluate()
#    model.evaluate_ppi()


if __name__ == '__main__':
    main()
