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

logging.basicConfig(level=logging.INFO)   
sys.path.insert(0, '')
sys.path.append('../../../')

from mowl.datasets.base  import PathDataset
from mowl.gnn_sim.model import GNNSim

import yaml

import os
curr_path = os.path.dirname(os.path.abspath(__file__))



@ck.command()
@ck.option(
    '--config', '-c', help="Configuration file in config/")
@ck.option(
    '--num-samples', '-ns', help="Number of samples", default=100)
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
    batch_size = config["batch_size"]
    n_hidden = config["n_hid"]
    dropout = config["dropout"]
    lr = config["lr"]
    normalize = config["norm"]
    regularization = config["reg"]
    self_loop = config["loop"]
    
    
    ontology = params["general"]["ontology"]
    ds = PathDataset(ontology, "", "")
    num_bases = params["rgcn-params"]["num-bases"]
    epochs = params["rgcn-params"]["epochs"]
    graph_method = params["general"]["graph-gen-method"]
    min_edges = params["rgcn-params"]["min-edges"]
    seed =  params["rgcn-params"]["seed"]
    file_params = params["files"]
    
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

    best_trained_model = PPIModel(feat_dim, num_rels, best_trial.config["num_bases"], num_nodes, best_trial.config["n_hid"], best_trial.config["dropout"])
    device = "cpu"
    if th.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = th.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_loss, test_auc = test(best_trial.config["n_hid"], best_trial.config["dropout"], best_trial.config["num_bases"], best_trial.config["batch_size"], file_params, model = best_trained_model)
    print("Best trial test set loss: {}".format(test_loss))
    print("Best trial test set auc: {}".format(test_auc))
   


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:

    main()
