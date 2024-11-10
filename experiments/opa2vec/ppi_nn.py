import sys
sys.path.append("../")
import mowl
mowl.init_jvm("10g")
from mowl.models import SyntacticPlusW2VModel
from mowl.projection import TaxonomyWithRelationsProjector
from mowl.utils.random import seed_everything
from mowl.utils.data import FastTensorDataLoader
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
from evaluators import PPIEvaluator
from datasets import PPIDataset
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
@ck.option("--embed_dim", "-dim", default=50, help="Embedding dimension")
@ck.option("--batch_size", "-bs", default=32, help="Batch size")
@ck.option("--learning_rate", "-lr", default=0.001, help="Learning rate")
@ck.option("--window_size", "-ws", default=5, help="Window size")
@ck.option("--epochs", "-e", default=10, help="Number of epochs")
@ck.option("--device", "-d", default="cuda", help="Device to use")
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(embed_dim, batch_size, learning_rate, window_size, epochs,
         device, wandb_description, no_sweep, only_test):

    seed_everything(42)

    dataset_name = "ppi_yeast"
    evaluator_name = "ppi"
    
    wandb_logger = wandb.init(entity="zhapacfp_team", project="ontoem", group=f"opa2vec_nn_{dataset_name}", name=wandb_description)

    if no_sweep:
        wandb_logger.log({"embed_dim": embed_dim,
                          "epochs": epochs,
                          "window_size": window_size,
                          "batch_size": batch_size,
                          "learning_rate": learning_rate,
                          })
    else:
        embed_dim = wandb.config.embed_dim
        epochs = wandb.config.epochs
        window_size = wandb.config.window_size
        batch_size = wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
        
    root_dir, dataset = dataset_resolver(dataset_name)

    model_dir = f"{root_dir}/../models/"
    os.makedirs(model_dir, exist_ok=True)

    corpora_dir = f"{root_dir}/../corpora/"
    os.makedirs(corpora_dir, exist_ok=True)
    
    model_filepath = f"{model_dir}/opa2vec_nn_{embed_dim}_{epochs}_{window_size}_{batch_size}_{learning_rate}.pt"
    corpus_filepath = f"{corpora_dir}/opa2vec_nn_{embed_dim}_{epochs}_{window_size}_{batch_size}_{learning_rate}.txt"

    model = OPA2VecModel(evaluator_name, dataset, batch_size,
                         learning_rate, window_size, embed_dim,
                         model_filepath, corpus_filepath, epochs,
                         device, wandb_logger)

    
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


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)
    
class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.3, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x

 
class PPINet(nn.Module):
    def __init__(self, embedding_layer, embed_dim):
        super().__init__()

        self.embedding_layer = embedding_layer

        net = []
        input_length = 2*embed_dim
        nodes = [embed_dim]
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, 1))
        net.append(nn.Sigmoid())
        self.ppi_net = nn.Sequential(*net)

    def forward(self, data, *args):
        if data.shape[1] == 2:
            h = data[:, 0]
            t = data[:, 1]
        elif data.shape[1] == 3:
            h = data[:, 0]
            t = data[:, 2]
        else:
            raise ValueError(f"Data shape not consistent: {data.shape}")
        
        head_embs = self.embedding_layer(h)
        tail_embs = self.embedding_layer(t)

        data = th.cat([head_embs, tail_embs], dim=1)
        return self.ppi_net(data)
    
class OPA2VecModel(SyntacticPlusW2VModel):
    def __init__(self, evaluator_name, dataset, batch_size,
                 learning_rate, window_size, embed_dim,
                 model_filepath, corpus_filepath, epochs, device,
                 wandb_logger):
        super().__init__(dataset, model_filepath=model_filepath, corpus_filepath=corpus_filepath)


        
        
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.evaluator = evaluator_resolver(evaluator_name, dataset,
                                            device, batch_size=32)
        self.nn_model_filepath = model_filepath + ".nn"
        self.epochs = epochs
        self.device = device
        self.wandb_logger = wandb_logger

        self.set_w2v_model(vector_size=embed_dim, min_count=1, window=window_size, epochs=self.epochs, sg=1, negative=5)
        if os.path.exists(self.corpus_filepath):
            logger.warning("Corpus already exists in file. Loading it...")
            self.load_corpus()
        else:
            self.generate_corpus(save=True, with_annotations=True)

        
    def load_protein_embeddings_data(self, load_w2v=False):
        logger.info(f"Initializing neural network")
        classes = self.dataset.classes.as_str
        class_to_id = {class_: i for i, class_ in enumerate(classes)}

        
        if load_w2v:
            self.w2v_model = self.w2v_model.load(self.model_filepath)
            
        w2v_vectors = self.w2v_model.wv
        
        
        embeddings_dict = {}
        for class_ in classes:
            if not "4932." in class_:
                continue
            if class_ in w2v_vectors:
                embeddings_dict[class_] = w2v_vectors[class_]
            else:
                logger.warning(f"Class {class_} not found in w2v model")
                embeddings_dict[class_] = np.random.rand(self.embed_dim)

        embedding_to_id = {cls: idx for idx, cls in enumerate(embeddings_dict.keys())}

        embeddings_list = np.array(list(embeddings_dict.values()))
        embeddings = th.tensor(embeddings_list, dtype=th.float).to(self.device)
        embedding_layer = nn.Embedding.from_pretrained(embeddings)
        embedding_layer.weight.requires_grad = False

        ppi_net = PPINet(embedding_layer, self.embed_dim)
        
        projector = TaxonomyWithRelationsProjector(relations=["http://interacts_with"])

        train_edges = projector.project(self.dataset.ontology)
        valid_edges = projector.project(self.dataset.validation)
        test_edges = projector.project(self.dataset.testing)

        train_indices = [(embedding_to_id[e.src], embedding_to_id[e.dst]) for e in train_edges]
        prot_ids = list(embedding_to_id.values())
        train_negatives = [(e[0], random.choice(prot_ids)) for e in train_indices]
        train_data = train_indices + train_negatives
        labels = [0]*len(train_indices) + [1]*len(train_negatives)
        valid_data = [(embedding_to_id[e.src], embedding_to_id[e.dst]) for e in valid_edges]
        test_data = [(embedding_to_id[e.src], embedding_to_id[e.dst]) for e in test_edges]

        train_data = th.tensor(train_data, dtype=th.long).to(self.device)
        train_labels = th.tensor(labels, dtype=th.long).to(self.device)
        valid_data = th.tensor(valid_data, dtype=th.long).to(self.device)
        test_data = th.tensor(test_data, dtype=th.long).to(self.device)
        
        return ppi_net, train_data, train_labels, valid_data, test_data

    def train(self):
        super().train()
        self.w2v_model.save(self.model_filepath)

        ppi_net, train_data, train_labels, valid_data, test_data = self.load_protein_embeddings_data()
        
        train_dataloader = FastTensorDataLoader(train_data, train_labels, batch_size=self.batch_size, shuffle=True)

        optimizer = th.optim.AdamW(ppi_net.parameters(), lr = self.learning_rate)

        ppi_net = ppi_net.to(self.device)

        criterion = nn.BCELoss()

        epochs = 10000
        best_mr = float("inf")
        best_mrr = 0
        tolerance = 5
        curr_tolerance = 5
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_data, batch_labels in train_dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                head = batch_data[:, 0]
                tail = batch_data[:, 1]
                data = th.vstack([head, tail]).T
                
                logits = ppi_net(data).squeeze()
                loss = criterion(logits, batch_labels.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_dataloader)
                
            

            ppi_net.eval()

            metrics = self.evaluator.evaluate(ppi_net, mode="valid")
            valid_mr = metrics["valid_mr"]
            valid_mrr = metrics["valid_mrr"]

            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                th.save(ppi_net.state_dict(), self.nn_model_filepath)
                curr_tolerance = tolerance
                logger.info(f"New best model found: {valid_mr:1f} - {valid_mrr:4f}")
            else:
                curr_tolerance -= 1
                
            logger.info(f"Epoch {epoch}: {epoch_loss} - Valid MR: {valid_mr:1f} - Valid MRR: {valid_mrr:4f}")

            if curr_tolerance == 0:
                logger.info("Early stopping")
                break
            
            
    def test(self):
        self.from_pretrained(self.model_filepath)
        ppi_net, _, _, _, _ = self.load_protein_embeddings_data()
        ppi_net.to(self.device)
        ppi_net.load_state_dict(th.load(self.nn_model_filepath))
        ppi_net.eval()

        return self.evaluator.evaluate(ppi_net, mode="test")
        
        
        # evaluation_module = EvaluationModel(self.w2v_model, self.dataset, self.embed_dim, self.device)
        
        # return self.evaluator.evaluate(evaluation_module)




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

        x = data[:, 0]
        y = data[:, 1]

        logger.debug(f"X shape: {x.shape}")
        logger.debug(f"Y shape: {y.shape}")
        
        x = self.embeddings(x)
        y = self.embeddings(y)

        logger.debug(f"X shape: {x.shape}")
        logger.debug(f"Y shape: {y.shape}")
        
        dot_product = th.sum(x * y, dim=1)
        logger.debug(f"Dot product shape: {dot_product.shape}")
        return 1 - th.sigmoid(dot_product)
        
class DummyLogger():
    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass
    
if __name__ == "__main__":
    main()
