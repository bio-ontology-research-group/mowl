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
from mowl.develop.catEmbeddings.modelEL import CatEmbeddings

@ck.command()
@ck.option(
    '--species', '-sp', help="Nome of species",
    default = "yeast"
)

def main(species):
    logging.info(f"Number of cores detected: {os.cpu_count()}")

    if species == "yeast":
#        ds = 'data/data-train/yeast-classes-normalized.owl', 'data/data-valid/4932.protein.links.v10.5.txt', 'data/data-test/4932.protein.links.v10.5.txt'        

#        ds = PPIYeastDataset()
        ds = PathDataset("data_old/yeast/yeast-classes.owl", "data_old/yeast/valid.owl", "data_old/yeast/test.owl")
#        ds = PathDataset("data_old/human/human-classes.owl", "data_old/human/valid.owl", "data_old/human/test.owl")
        lr = 1e-1
        embedding_size = 180
        
        #milestones = [20,50, 90,150, 180,400,  600, 800, 1000, 1300, 1600, 20001001] #only_nf4\
        gamma = 0.5
        margin = 0#0.5
        epochs = 1500
        step = 80
        milestones = [i*step for i in range(epochs//step)]
        milestones.append("70000000000")
    elif species == "human":
        ds = PathDataset("data_old/human/human-classes.owl", "data_old/human/valid.owl", "data_old/human/test.owl")
        lr = 1e-1
        embedding_size = 180
        
        #milestones = [20,50, 90,150, 180,400,  600, 800, 1000, 1300, 1600, 20001001] #only_nf4\
        gamma = 0.5
        margin = 0#0.5
        epochs = 1500
        step = 80
        milestones = [i*step for i in range(epochs//step)]
        milestones.append("80000000000")
 
        #ds = 'data/data-train/human-classes-normalized.owl', 'data/data-valid/9606.protein.links.v10.5.txt', 'data/data-test/9606.protein.links.v10.5.txt'
        #lr = 5e-3 #2 for ppi slim
        #embedding_size =100
        #milestones = [150, 200001111] #only_nf4
        #milestones = [150, 2000002]
        #milestones = [150,300, 450, 600,  2000000] #works with 1e-2 in lr but cannot rank negatives properly
        #gamma = 0.1
        #margin = 5
        #epochs = 1500

        
    model = CatEmbeddings(
        ds, 
        4096*8, #4096*4, #bs 
        embedding_size, #embeddings size
        lr, #lr ##1e-3 yeast, 1e-5 human
        epochs, #epochs
        1000, #num points eval ppi
        milestones,
        dropout = 0.3,
        decay = 0,
        gamma = gamma,
        eval_ppi = True,
        size_hom_set = 3,
        depth = 3,
        margin = margin,
        seed = 0,
        early_stopping = 20000,
        device = "cuda:0"
    )

    model.train()
#    model.evaluate()
    model.evaluate_ppi()


if __name__ == '__main__':
    main()
