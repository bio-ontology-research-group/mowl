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
    '--species', '-sp', help="Nome of species",
    default = "yeast"
)

def main(species):
    logging.info(f"Number of cores detected: {os.cpu_count()}")

    if species == "yeast":
        ds = 'data/data-train/yeast-classes-normalized.owl', 'data/data-valid/4932.protein.links.v10.5.txt', 'data/data-test/4932.protein.links.v10.5.txt'        
        lr = 5e-3
#        milestones = [68,129,200,300,400,500,600,700,800,900,1000,2000,2100]
#        milestones = [68,129,300,500,700,900,1000,2000,2001] #before activating other nfs
#        milestones = [15,21,26,161,227]
#        milestones = [5,12,36,45,49,54,92,137,180,250,300,350,400,20000]




        milestones = [150, 200000] #only_nf4
#        milestones = [150, 250, 450, 2000000]
        #milestones = [200000]
    elif species == "human":
        ds = 'data/data-train/human-classes-normalized.owl', 'data/data-valid/9606.protein.links.v10.5.txt', 'data/data-test/9606.protein.links.v10.5.txt'
        lr = 5e-3
#        milestones = [68,129,200,300,400,500,600,700,800,900,1000,2000]
#        milestones = [50, 100, 150, 200, 250, 300,20000]
        milestones = [150, 2000000]
#        milestones = [100*x for x in range(20)]

    model = CatEmbeddings(
        ds, 
        4096*4, #4096*4, #bs 
        1024, #embeddings size
        lr, #lr ##1e-3 yeast, 1e-5 human
        1024, #epochs
        1000, #num points eval ppi
        milestones,
        dropout = 0,
        decay = 0.00,
        eval_ppi = True,
        sampling = True,
        nf1 = False,
        nf1_neg = False,
        nf2 = False,
        nf2_neg = False,
        nf3 = False,
        nf3_neg = False,
        nf4 = True,
        nf4_neg = True,
        margin = 5,
        seed = 0,
        early_stopping = 200000
    )

    model.train()
#    model.evaluate()
    model.evaluate_ppi()


if __name__ == '__main__':
    main()
