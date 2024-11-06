import sys
sys.path.append("../")
import mowl
mowl.init_jvm("10g")
from mowl.models import GraphPlusPyKEENModel
from mowl.utils.random import seed_everything
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
from evaluators import PPIEvaluator
from datasets import PPIDataset

from utils import print_as_md

from pykeen.models import TransE
from tqdm import tqdm
import logging
import click as ck
import os
import torch as th
import torch.nn as nn
import numpy as np

import wandb
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@ck.command()
@ck.option("--embed_dim", "-dim", default=50, help="Embedding dimension")
@ck.option("--batch_size", "-bs", default=128, help="Batch size")
@ck.option("--learning_rate", "-lr", default=0.001, help="Learning rate")
@ck.option("--epochs", "-e", default=10, help="Number of epochs")
@ck.option("--device", "-d", default="cuda", help="Device to use")
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(embed_dim, batch_size, learning_rate, epochs, device,
         wandb_description, no_sweep, only_test):

    seed_everything(42)

    dataset_name = "ppi_yeast"
    evaluator_name = "ppi"
    
    wandb_logger = wandb.init(entity="zhapacfp_team", project="ontoem", group=f"owl2vecstar_kg_{dataset_name}", name=wandb_description)

    if no_sweep:
        wandb_logger.log({"dataset_name": dataset_name,
                          "embed_dim": embed_dim,
                          "epochs": epochs,
                          "batch_size": batch_size,
                          "learning_rate": learning_rate
                          })
    else:
        dataset_name = wandb.config.dataset_name
        embed_dim = wandb.config.embed_dim
        epochs = wandb.config.epochs
        batch_size = wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
                        
    root_dir, dataset = dataset_resolver(dataset_name)

    model_dir = f"{root_dir}/../models/"
    os.makedirs(model_dir, exist_ok=True)

    model_filepath = f"{model_dir}/{embed_dim}_{epochs}_{batch_size}_{learning_rate}.pt"
    
    model = OWL2VecStarModel(evaluator_name, dataset, batch_size,
                             learning_rate, embed_dim, model_filepath,
                             epochs, device, wandb_logger)

    if not only_test:
        model.train()
        
    metrics = model.test()
    print_as_md(metrics)

             
    wandb_logger.log(metrics)
        

def dataset_resolver(dataset_name):
    if dataset_name.lower() == "ppi_yeast":
        root_dir = "../use_cases/ppi_yeast/data/"
    elif dataset_name.lower() == "ppi_yeast_slim":
        root_dir = "../use_cases/ppi_yeast_slim/data/"
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return root_dir, PPIDataset(root_dir)

def evaluator_resolver(evaluator_name, *args, **kwargs):
    if evaluator_name.lower() == "ppi":
        return PPIEvaluator(*args, **kwargs)
    else:
        raise ValueError(f"Evaluator {evaluator_name} not found")

                                                             

class OWL2VecStarModel(GraphPlusPyKEENModel):
    def __init__(self, evaluator_name, dataset, batch_size,
                 learning_rate, embed_dim, model_filepath, epochs,
                 device, wandb_logger):

        classes = dataset.classes.as_str
        class_to_id = {class_: i for i, class_ in enumerate(classes)}
        
        super().__init__(dataset, model_filepath=model_filepath)

        self.embed_dim = embed_dim
        self.evaluator = evaluator_resolver(evaluator_name, dataset,
                                            device, batch_size=16)
        self.epochs = epochs
        self.device = device
        self.wandb_logger = wandb_logger

        self.set_projector(mowl.projection.OWL2VecStarProjector(include_literals=False))
        self.set_kge_method(TransE, embedding_dim=self.embed_dim, random_seed=42)
        self._kge_method = self._kge_method.to(self.device)
        self.optimizer = th.optim.Adam
        self.lr = learning_rate
        self.batch_size = batch_size

    def train(self):
        super().train(epochs = self.epochs)
        
    def test(self):
        self.from_pretrained(self.model_filepath)
        self._kge_method = self._kge_method.to(self.device)
        evaluation_module = EvaluationModel(self.kge_method, self.triples_factory, self.dataset, self.embed_dim, self.device)
        
        return self.evaluator.evaluate(evaluation_module)




class EvaluationModel(nn.Module):
    def __init__(self, kge_method, triples_factory, dataset, embedding_size, device):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device

        self.kge_method = kge_method
        
        classes = dataset.classes.as_str
        class_to_id = {class_: i for i, class_ in enumerate(classes)}

        ont_id_to_graph_id = dict()

        num_not_found = 0
        for class_ in classes:
            if class_ not in triples_factory.entity_to_id:
                ont_id_to_graph_id[class_to_id[class_]] = -1
                logger.warning(f"Class {class_} not found in graph")
                num_not_found += 1
            else:
                ont_id_to_graph_id[class_to_id[class_]] = triples_factory.entity_to_id[class_]

        logger.warning(f"Number of classes not found: {num_not_found}")
        assert list(ont_id_to_graph_id.keys()) == list(range(len(classes)))
        # for 
        
        self.graph_ids = th.tensor(list(ont_id_to_graph_id.values())).to(self.device)
        
        relation_id = triples_factory.relation_to_id["http://interacts_with"]
        self.rel_embedding = th.tensor(relation_id).to(self.device)
        
    def forward(self, data, *args, **kwargs):
        if data.shape[1] == 2:
            x = data[:, 0]
            y = data[:, 1]
        elif data.shape[1] == 3:
            x = data[:, 0]
            y = data[:, 2]
        else:
            raise ValueError(f"Data shape {data.shape} not recognized")

        x = self.graph_ids[x].unsqueeze(1)
        y = self.graph_ids[y].unsqueeze(1)

        x_unique = self.graph_ids[data[:, 0].unique()]
        y_unique = self.graph_ids[data[:, 1].unique()]
        assert th.min(x_unique) >= 0, f"sum: {(x_unique==-1).sum()} min: {th.min(x_unique)} len: {len(x_unique)}"
        assert th.min(y_unique) >= 0, f"sum: {(y_unique==-1).sum()} min: {th.min(y_unique)} len: {len(y_unique)}"
                        
        r = self.rel_embedding.expand_as(x)
        
        triples = th.cat([x, r, y], dim=1)
        assert triples.shape[1] == 3
        scores = - self.kge_method.score_hrt(triples)
        # print(scores.min(), scores.max())
        return scores
        
                     
if __name__ == "__main__":
    main()
