import sys
sys.path.append("../")
import mowl
mowl.init_jvm("10g")
from mowl.models import RandomWalkPlusW2VModel
from mowl.walking import Node2Vec
from mowl.utils.random import seed_everything
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
from evaluators import SubsumptionEvaluator
from datasets import SubsumptionDataset
from utils import print_as_md
from tqdm import tqdm
import logging
import click as ck
import os
import torch as th
import torch.nn as nn
import numpy as np
import random

import wandb
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@ck.command()
@ck.option("--dataset_name", "-ds", type=ck.Choice(["go", "foodon"]), default="go")
@ck.option("--embed_dim", "-dim", default=50, help="Embedding dimension")
@ck.option("--window_size", "-ws", default=5, help="Batch size")
@ck.option("--epochs", "-e", default=10, help="Number of epochs")
@ck.option("--walk_length", "-wl", default=20, help="Walk length")
@ck.option("--number_of_walks", "-nw", default=10, help="Number of walks per each node")
@ck.option("--p", "-p", required=True, type=float)
@ck.option("--q", "-q", required=True, type=float)
@ck.option("--evaluate_deductive", "-evalded", is_flag=True, help="Use deductive closure as positive examples for evaluation")
@ck.option("--filter_deductive", "-filterded", is_flag=True, help="Filter out examples from deductive closure")
@ck.option("--device", "-d", default="cuda", help="Device to use")
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(dataset_name, embed_dim, window_size, epochs, walk_length,
         number_of_walks, p, q, evaluate_deductive, filter_deductive,
         device, wandb_description, no_sweep, only_test):

    seed_everything(42)

    evaluator_name = "subsumption"
    
    wandb_logger = wandb.init(entity="zhapacfp_team", project="ontoem", group=f"owl2vecstar_sim_{dataset_name}", name=wandb_description)

    if no_sweep:
        wandb_logger.log({"dataset_name": dataset_name,
                          "embed_dim": embed_dim,
                          "window_size": window_size,
                          "epochs": epochs,
                          "walk_length": walk_length,
                          "number_of_walks": number_of_walks,
                          "p": p,
                          "q": q
                          })
    else:
        dataset_name = wandb.config.dataset_name
        embed_dim = wandb.config.embed_dim
        window_size = wandb.config.window_size
        epochs = wandb.config.epochs
        walk_length = wandb.config.walk_length
        number_of_walks = wandb.config.number_of_walks
        p = wandb.config.p
        q = wandb.config.q
        
    root_dir, dataset = dataset_resolver(dataset_name)

    model_dir = f"{root_dir}/../models/"
    os.makedirs(model_dir, exist_ok=True)

    corpora_dir = f"{root_dir}/../corpora/"
    os.makedirs(corpora_dir, exist_ok=True)
    
    model_filepath = f"{model_dir}/owl2vecstar_sim_{embed_dim}_{window_size}_{epochs}_{walk_length}_{number_of_walks}_{p}_{q}.pt"
    corpus_filepath = f"{corpora_dir}/owl2vecstar_sim_{embed_dim}_{window_size}_{epochs}_{walk_length}_{number_of_walks}_{p}_{q}.txt"
    
    model = OWL2VecStarModel(evaluator_name, dataset, window_size,
                             embed_dim, walk_length, number_of_walks,
                             p, q, model_filepath, corpus_filepath,
                             epochs, evaluate_deductive,
                             filter_deductive, device, wandb_logger)

    
    if not only_test:
        model.train()
        
        
    metrics = model.test()
    print_as_md(metrics)

             
    wandb_logger.log(metrics)
        
def dataset_resolver(dataset_name):
    if dataset_name.lower() == "go":
        root_dir = "../use_cases/go/data/"
    elif dataset_name.lower() == "foodon":
        root_dir = "../use_cases/foodon/data/"
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return root_dir, SubsumptionDataset(root_dir)
    
def evaluator_resolver(evaluator_name, *args, **kwargs):
    if evaluator_name.lower() == "subsumption":
        return SubsumptionEvaluator(*args, **kwargs)
    else:
        raise ValueError(f"Evaluator {evaluator_name} not found")

class OWL2VecStarModel(RandomWalkPlusW2VModel):
    def __init__(self, evaluator_name, dataset, window_size,
                 embed_dim, walk_length, number_of_walks, p, q,
                 model_filepath, corpus_filepath, epochs,
                 evaluate_deductive, filter_deductive, device,
                 wandb_logger):

        classes = dataset.classes.as_str
        class_to_id = {class_: i for i, class_ in enumerate(classes)}
        
        super().__init__(dataset, model_filepath=model_filepath)


        
        
        self.embed_dim = embed_dim
        self.evaluator = evaluator_resolver(evaluator_name, dataset,
                                            device, batch_size=16,
                                            evaluate_with_deductive_closure=evaluate_deductive,
                                            filter_deductive_closure=filter_deductive)
        self.epochs = epochs
        self.device = device
        self.wandb_logger = wandb_logger

        self.set_projector(mowl.projection.OWL2VecStarProjector(include_literals=True))
        self.set_walker(Node2Vec(number_of_walks, walk_length, p=p, q=q, workers=10, outfile=corpus_filepath))
        self.set_w2v_model(vector_size=embed_dim, workers=16, epochs=epochs, min_count=1, window=window_size, sg=1, negative=5)


    def train(self):
        super().train(epochs = self.epochs)
        self.w2v_model.save(self.model_filepath)

        
            
    def test(self):
        self.from_pretrained(self.model_filepath)
        evaluation_module = EvaluationModel(self.w2v_model, self.dataset, self.embed_dim, self.device)
        return self.evaluator.evaluate(evaluation_module)


class EvaluationModel(nn.Module):
    def __init__(self, w2v_model, dataset, embedding_size, device):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device
        
        self.embeddings = self.init_module(w2v_model, dataset)


    def init_module(self, w2v_model, dataset):
        classes = dataset.classes.as_str
        class_to_id = {class_: i for i, class_ in enumerate(classes)}

        w2v_vectors = w2v_model.wv
        embeddings_list = []
        for class_ in classes:
            if class_ in w2v_vectors:
                embeddings_list.append(w2v_vectors[class_])
            else:
                logger.warning(f"Class {class_} not found in w2v model")
                embeddings_list.append(np.random.rand(self.embedding_size))

        embeddings_list = np.array(embeddings_list)
        embeddings = th.tensor(embeddings_list).to(self.device)
        return nn.Embedding.from_pretrained(embeddings)
        
        
    def forward(self, data, *args, **kwargs):
        if data.shape[1] == 2:
            x = data[:, 0]
            y = data[:, 1]
        elif data.shape[1] == 3:
            x = data[:, 0]
            y = data[:, 2]
        else:
            raise ValueError(f"Data shape {data.shape} not recognized")

        
        x = self.embeddings(x)
        y = self.embeddings(y)

        dot_product = th.sum(x * y, dim=1)
        logger.debug(f"Dot product shape: {dot_product.shape}")
        return 1 - th.sigmoid(dot_product)

    
if __name__ == "__main__":
    main()
