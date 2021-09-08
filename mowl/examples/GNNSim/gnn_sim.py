#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import pickle
import gzip
import os
import sys
import logging
import yaml
logging.basicConfig(level=logging.INFO)   
sys.path.insert(0, '')
sys.path.append('../../')

from mowl.datasets import PPIYeastDataset
from mowl.gnn_sim.model import GNNSim

@ck.command()
@ck.option(
    '--config', '-c', help="Configuration file in config/")

def main(config):
    logging.info(f"Number of cores detected: {os.cpu_count()}")


    params = parseYAML(config)

    n_hidden = params["rgcn-params"]["n-hidden"]
    dropout = params["rgcn-params"]["dropout"]
    lr = params["rgcn-params"]["lr"]
    n_bases = params["rgcn-params"]["num-bases"]
    batch_size = params["rgcn-params"]["batch-size"]
    epochs = params["rgcn-params"]["epochs"]
    graph_method = params["graph-generation"]["method"]
    ontology = params["rgcn-params"]["ontology"]
    
    file_params = params["files"]


    ds = PathDataset(ontology, "", "")
        
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


def parseYAML(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    return params

if __name__ == '__main__':
    main()
