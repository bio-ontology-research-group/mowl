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
sys.path.append('../../../')


from mowl.datasets.base  import PathDataset
from mowl.develop.GNNSim.gnn_sim.model_ppi import GNNSimPPI

@ck.command()
@ck.option(
    '--config', '-c', help="Configuration file in config/")

def main(config):
    logging.info(f"Number of cores detected: {os.cpu_count()}")


    params = parseYAML(config)

    parser = params["general"]["graph-gen-method"]
    ontology = params["general"]["ontology"]
    use_case = params["general"]["use-case"]
   
    
    
    lr = params["gnn-sim-params"]["lr"]
    batch_size = params["gnn-sim-params"]["bs"]
    epochs = params["gnn-sim-params"]["epochs"]
    regularization = params["gnn-sim-params"]["regularization"]
    normalize =  params["gnn-sim-params"]["normalization"]
    self_loop =  params["gnn-sim-params"]["self-loop"]
    seed =  params["gnn-sim-params"]["seed"]
    min_edges = params["gnn-sim-params"]["min-edges"]
    
   
    ppi_model_params = params["ppi-model-params"]
    file_params = params["files"]
    data_params = params["data-params"]


    ds = PathDataset(ontology, None, None)


    print(params)

    if use_case == "ppi":
        model = GNNSimPPI(ds, # dataset
                          epochs,
                          batch_size,
                          lr,
                          regularization,
                          parser,
                          normalize,
                          min_edges,
                          self_loop,
                          seed = seed,
                          ppi_model_params = ppi_model_params,
                          data_params = data_params,
                          file_params = file_params                         
                   )

    elif use_case == "gd":
        model = GNNSimGD(ds, # dataset
                       n_hidden,
                       dropout,
                       lr,
                       num_bases,
                       batch_size,
                       epochs,
                       use_case,
                       parser = graph_method,
                       normalize = normalize,
                       regularization = regularization,
                       self_loop = self_loop,
                       min_edges = min_edges,
                       seed = seed,
                       file_params = file_params
                   )



    model.train()
    model.evaluate()


def parseYAML(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    return params

if __name__ == '__main__':
    main()
