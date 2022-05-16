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
from mowl.develop.catEmbeddings.modelNF1 import CatEmbeddings

@ck.command()
@ck.option(
    '--species', '-sp', help="Nome of species",
    default = "yeast"
)

def main(species):
    logging.info(f"Number of cores detected: {os.cpu_count()}")


    
    ds = 'data/data-train/human-classes-normalized.owl', 'data/subsumption_data/valid_data.tsv', 'data/subsumption_data/test_data.tsv'
    lr = 5e-3

    milestones = [50, 100, 300, 1000,2000000]


    model = CatEmbeddings(
        ds, 
        4096*4, #4096*4, #bs 
        1024, #embeddings size
        lr, #lr ##1e-3 yeast, 1e-5 human
        1300, #epochs
        10, #num points eval ppi
        milestones,
        dropout = 0,
        decay = 0.00,
        eval_subsumption = True,
        sampling = True,
        nf1 = True,
        nf1_neg = True,
        margin = 5,
        seed = 0,
        early_stopping = 200000
    )

    model.train()
#    model.evaluate()
    model.evaluate_subsumption()


if __name__ == '__main__':
    main()
