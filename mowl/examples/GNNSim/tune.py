from functools import partial
import click as ck
import numpy as np
import os
import torch as th
import torch.nn as nn
import dgl
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import logging
import sys

import os
curr_path = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO)   
sys.path.insert(0, '')
#sys.path.append("/ibex/scratch/zhapacfp/mowl/")
path = os.path.abspath(os.getcwd())
sys.path.append(path + "/../../../")

import yaml




@ck.command()
@ck.option(
    '--config', '-c', help="Configuration file in config/")
@ck.option(
    '--num-samples', '-ns', help="Number of samples", default=1)
@ck.option(
    '--max-num-epochs', '-e', help="Max number of epochs", default=10)
@ck.option(
    '--gpus-per-trial', '-g', help="GPUs per trial", default=1)

def main(config, num_samples, max_num_epochs, gpus_per_trial):

    params = parseYAML(config)
    #global g, annots
   
    feat_dim = 2
    
    tuning(params, num_samples, max_num_epochs, gpus_per_trial, feat_dim)

def parseYAML(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    return params

    
def train_tune(config, params=None, checkpoint_dir = None):

    sys.path.append(path + '/')
    sys.path.append(path + "/../../../")

    print("path in tune:", sys.path)

    from mowl.datasets.base  import PathDataset
    from mowl.gnn_sim.model import GNNSim



    
    batch_size = config["batch_size"]
    n_hidden = config["n_hid"]
    dropout = config["dropout"]
    lr = config["lr"]
    normalize = config["norm"]
    regularization = config["reg"]
    self_loop = config["loop"]
    
    
    ontology =  params["general"]["ontology"]
    ds = PathDataset(path + '/' + ontology, "", "")
    num_bases = params["rgcn-params"]["num-bases"]
    epochs = params["rgcn-params"]["epochs"]
    graph_method = params["general"]["graph-gen-method"]
    min_edges = params["rgcn-params"]["min-edges"]
    seed =  params["rgcn-params"]["seed"]
    file_params = params["files"]

    file_params = {k: path + '/' + v for k, v in file_params.items()}
    
    model = GNNSim(ds, # dataset
                   n_hidden,
                   dropout,
                   lr,
                   num_bases,
                   batch_size,
                   epochs,
                   graph_generation_method = graph_method,
                   normalize = normalize,
                   regularization = regularization,
                   self_loop = self_loop,
                   min_edges = min_edges,
                   seed = seed,
                   file_params = file_params
                   )

    model.train(tuning = True)




def tuning(params, num_samples, max_num_epochs, gpus_per_trial, feat_dim):


    config = {
        "n_hid": tune.choice([2, 3, 4]),
        "dropout": tune.choice([x/10 for x in range(1,6)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32]),
        "norm": tune.choice([True, False]),
        "reg": tune.loguniform(1e-6, 1e-2),
        "loop": tune.choice([True, False])
    }
    scheduler = ASHAScheduler(
        metric="auc",
        mode="max",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "auc"])
    result = tune.run(
        tune.with_parameters(train_tune,
                                params=params),
        resources_per_trial={"cpu": gpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("auc", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation auc: {}".format(
        best_trial.last_result["auc"]))



    sys.path.append(path + '/')
    sys.path.append(path + "/../../../")


    from mowl.datasets.base  import PathDataset
    from mowl.gnn_sim.model import GNNSim



    ontology =  params["general"]["ontology"]
    ds = PathDataset(path + '/' + ontology, "", "")
    num_bases = params["rgcn-params"]["num-bases"]
    epochs = params["rgcn-params"]["epochs"]
    graph_method = params["general"]["graph-gen-method"]
    min_edges = params["rgcn-params"]["min-edges"]
    seed =  params["rgcn-params"]["seed"]
    file_params = params["files"]
    
    file_params = {k: path + '/' + v for k, v in file_params.items()}


    best_trained_model = GNNSim(ds, # dataset
                                best_trial.config["n_hid"],
                                best_trial.config["dropout"],
                                best_trial.config["lr"],
                                num_bases,
                                best_trial.config["batch_size"],
                                epochs,
                                graph_generation_method = graph_method,
                                normalize = best_trial.config["norm"],
                                regularization = best_trial.config["reg"],
                                self_loop = best_trial.config["loop"],
                                min_edges = min_edges,
                                seed = seed,
                                file_params = file_params
                   )

    best_checkpoint_dir = best_trial.checkpoint.value
 
    test_loss, test_auc = best_trained_model.evaluate(tuning=True, best_checkpoint_dir = best_checkpoint_dir)

    print("Best trial test set loss: {}".format(test_loss))
    print("Best trial test set auc: {}".format(test_auc))
   


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:

    main()
