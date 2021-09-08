#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import pickle
import gzip
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)   
sys.path.insert(0, '')
sys.path.append('../../')

from mowl.datasets import PPIYeastDataset
from mowl.gnn_sim.model import GNNSim

@ck.command()
def main():
    logging.info(f"Number of cores detected: {os.cpu_count()}")

    file_params = {
        "train_inter_file": "data/4932.train_interactions.pkl",
        "test_inter_file": "data/4932.test_interactions.pkl",
        "data_file": "data/swissprot.pkl",
        "output_model": "data/gnn_sim_model.pt"
    }

    ds = PPIYeastDataset()
    
    model = GNNSim(ds, # dataset
                   2, #n_hidden
                   0.1, #dropout
                   0.004, #learning_rate
                   1, #num_bases
                   32, #batch_size
                   32, #epochs
                   graph_generation_method = "taxonomy",
                   file_params = file_params
                   )


    model.train()
#    model.evaluate(relations)


if __name__ == '__main__':
    main()
