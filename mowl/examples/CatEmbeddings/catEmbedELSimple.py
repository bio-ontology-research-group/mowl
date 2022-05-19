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

import mowl
mowl.init_jvm("10g")

from mowl.datasets.base import PathDataset
from mowl.datasets.ppi_yeast import PPIYeastSlimDataset, PPIYeastDataset
#from mowl.develop.catEmbeddings.modelWithFeats import CatEmbeddings
from mowl.develop.catEmbeddings.modelEL_simple import CatEmbeddings

@ck.command()
@ck.option(
    '--species', '-sp', help="Nome of species",
    default = "yeast"
)

def main(species):
    logging.info(f"Number of cores detected: {os.cpu_count()}")

    if species == "yeast":
#        ds = 'data/data-train/yeast-classes-normalized.owl', 'data/data-valid/4932.protein.links.v10.5.txt', 'data/data-test/4932.protein.links.v10.5.txt'        

#        ds = PPIYeastSlimDataset()
        ds = PathDataset("data_old/yeast-classes.owl", "data_old/valid.owl", "data_old/test.owl")
        lr = 5e-3
        embedding_size = 100
#        milestones = [ 150, 2000, 200112]
#        milestones = [50, 100, 150, 400,  6000, 20001001] #only_nf4
        gamma = 0.3
#        milestones = [150, 250, 450, 2000000]
        milestones = [8099999999]
        margin = 2
        epochs = 100

    elif species == "human":
        ds = 'data/data-train/human-classes-normalized.owl', 'data/data-valid/9606.protein.links.v10.5.txt', 'data/data-test/9606.protein.links.v10.5.txt'
        lr = 5e-2
        embedding_size =100
        milestones = [150, 200001111] #only_nf4
#        milestones = [150, 2000002]
        #milestones = [150,300, 450, 600,  2000000] #works with 1e-2 in lr but cannot rank negatives properly
        gamma = 0.1
        margin = 5
        epochs = 800

        
    model = CatEmbeddings(
        ds, 
        4096*8, #4096*4, #bs 
        embedding_size, #embeddings size
        lr, #lr ##1e-3 yeast, 1e-5 human
        epochs, #epochs
        1000, #num points eval ppi
        milestones,
        dropout = 0,
        decay = 0,
        gamma = gamma,
        eval_ppi = False,
        sampling = False,
        size_hom_set = 5,
        nf1 = True,
        nf1_neg = True,
        nf2 = False,
        nf2_neg = False,
        nf3 = True,
        nf3_neg = True,
        nf4 = False,
        nf4_neg = False,
        margin = margin,
        seed = 0,
        early_stopping = 200000,
        device = "cuda"
    )

    model.train()
#    model.evaluate()
    model.evaluate_ppi()


if __name__ == '__main__':
    main()
