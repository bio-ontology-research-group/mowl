import sys
sys.path.append("../")
import mowl
mowl.init_jvm("10g")
from mowl.models import GraphPlusPyKEENModel
from mowl.utils.random import seed_everything
from mowl.projection import Edge
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
from evaluators import SubsumptionEvaluator
from datasets import SubsumptionDataset

from pykeen.losses import PairwiseLoss

from utils import print_as_md
from data import create_graph_train_dataloader

from kg import OrderE
from tqdm import tqdm
import logging
import click as ck
import os
import torch as th
import torch.nn as nn
import numpy as np
import pickle as pkl

import wandb
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@ck.command()
@ck.option("--dataset_name", "-ds", type=ck.Choice(["go", "foodon"]), default="go")
@ck.option("--embed_dim", "-dim", default=50, help="Embedding dimension")
@ck.option("--batch_size", "-bs", default=128, help="Batch size")
@ck.option("--learning_rate", "-lr", default=0.001, help="Learning rate")
@ck.option("--num_negs", "-negs", default=1, help="Number of negative samples")
@ck.option("--margin", "-m", default=1.0, help="Margin for pairwise loss")
@ck.option("--epochs", "-e", default=10000, help="Number of epochs")
@ck.option("--evaluate_deductive", "-evalded", is_flag=True, help="Use deductive closure as positive examples for evaluation")
@ck.option("--filter_deductive", "-filterded", is_flag=True, help="Filter out examples from deductive closure")
@ck.option("--device", "-d", default="cuda", help="Device to use")
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(dataset_name, embed_dim, batch_size, learning_rate, num_negs,
         margin, epochs, evaluate_deductive, filter_deductive, device,
         wandb_description, no_sweep, only_test):

    seed_everything(42)

    evaluator_name = "subsumption"
    
    wandb_logger = wandb.init(entity="zhapacfp_team", project="ontoem", group=f"cate_kg_{dataset_name}", name=wandb_description)

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
    graph_filepath = f"{root_dir}/cat_projection.edgelist"
    
    model = CatEModel(evaluator_name, dataset, batch_size,
                      learning_rate, num_negs, embed_dim, model_filepath,
                      graph_filepath, margin, epochs, evaluate_deductive,
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


class CatEPairwiseLoss(PairwiseLoss):
        def __init__(self, margin: float, reduction: str = 'mean'):
                super().__init__(reduction)

                self.margin = margin
        
        def forward(self, positive_scores: th.FloatTensor, negative_scores: th.FloatTensor) -> th.FloatTensor:
            return -positive_scores.mean() + th.relu(self.margin + negative_scores).mean()

class CatEModel(GraphPlusPyKEENModel):
    def __init__(self, evaluator_name, dataset, batch_size,
                 learning_rate, num_negs, embed_dim, model_filepath,
                 graph_filepath, margin, epochs, evaluate_deductive,
                 filter_deductive, device, wandb_logger):

        self.classes = dataset.classes.as_str
        class_to_id = {class_: i for i, class_ in enumerate(self.classes)}
        
        super().__init__(dataset, model_filepath=model_filepath)

        self.graph_filepath = graph_filepath
        self.embed_dim = embed_dim
        self.num_negs = num_negs
        self.margin = margin
        self.evaluator = evaluator_resolver(evaluator_name, dataset,
                                            device, batch_size=16,
                                            evaluate_with_deductive_closure=evaluate_deductive,
                                            filter_deductive_closure=filter_deductive)
        self.epochs = epochs
        self.device = device
        self.wandb_logger = wandb_logger

        self.set_projector(mowl.projection.CategoricalProjector("str"))
        self.set_kge_method(OrderE, embedding_dim=self.embed_dim, random_seed=42)
        print(self._kge_method.loss)
        self._kge_method.loss = CatEPairwiseLoss(margin=1.0)
        print(self._kge_method.loss)
        self._kge_method = self._kge_method.to(self.device)
        self.optimizer = th.optim.Adam
        self.lr = learning_rate
        self.batch_size = batch_size

        ##### Properties #####

        self._ontology_classes_idxs = None
    
        
    def _load_edges(self):
        if self.projector is None:
            raise ValueError(msg.GRAPH_MODEL_PROJECTOR_NOT_SET)

        all_classes = set(self.dataset.classes.as_str)

        if os.path.exists(self.graph_filepath):
            logger.info(f"Loading graph from {self.graph_filepath}")
            edges = pkl.load(open(self.graph_filepath, "rb"))
            self._edges = [Edge(*tuple_) for tuple_ in edges]
            
        else:
            self._edges = list(self.projector.project(self.dataset.ontology))
            pkl.dump(self._edges, open(self.graph_filepath, "wb"))


        logger.info(f"Number of loaded edges: {len(self._edges)}")
        filtered_edges = list()
        for edge in self._edges:
            head = edge.src
            tail = edge.dst

            if "Nothing" in head and " " in tail:
                continue

            if "Thing" in tail and " " in head:
                continue

            filtered_edges.append(edge)

        self._edges = filtered_edges
        logger.info(f"Number of edges after filtering: {len(self._edges)}")
            
        nodes, relations = Edge.get_entities_and_relations(self._edges)
        nodes = set(nodes)

        missing_classes = all_classes - nodes
        logger.warning(f"There are {len(missing_classes)} classes not found in the graph. They might be ignored in the projection or they might be in the validation/testing set but not in the training set.")
        nodes = nodes.union(missing_classes)
        nodes = list(set(nodes))
        relations = list(set(relations))
        nodes.sort()
        relations.sort()
        self._graph_node_to_id = {node: i for i, node in enumerate(nodes)}
        self._graph_relation_to_id = {relation: i for i, relation in enumerate(relations)}

    @property
    def ontology_classes_idxs(self):
        if self._ontology_classes_idxs is not None:
            return self._ontology_classes_idxs
        
        class_to_id = {c: self._graph_node_to_id[c] for c in self.classes}
        ontology_classes_idxs = th.tensor(list(class_to_id.values()), dtype=th.long, device=self.device)
        self._ontology_classes_idxs = ontology_classes_idxs
        return self._ontology_classes_idxs

    def train(self):
        # super().train(epochs = self.epochs)

        logger.info(f"Number of model parameters: {sum(p.numel() for p in self._kge_method.parameters() if p.requires_grad)}")
                                                                                    
        optimizer = th.optim.Adam(self._kge_method.parameters(), lr=self.lr, weight_decay=0.0001)
        min_lr = self.lr/10
        max_lr = self.lr

        scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr,
                                                   max_lr=max_lr, step_size_up = 20,
                                                   cycle_momentum = False)

        criterion_bpr = nn.LogSigmoid()
        
        self._kge_method = self._kge_method.to(self.device)

        graph_dataloader = create_graph_train_dataloader(self._edges, self._graph_node_to_id, self._graph_relation_to_id, self.batch_size)

        initial_tolerance = 10
        tolerance = 0
        best_loss = float("inf")
        best_mr = 10000000
        best_mrr = 0
        ont_classes_idxs = th.tensor(list(self.ontology_classes_idxs), dtype=th.long,
                                     device=self.device)

        for epoch in tqdm(range(self.epochs), desc=f"Training..."):
            self._kge_method.train()

            graph_loss = 0
            for head, rel, tail in graph_dataloader:
                head = head.to(self.device)
                rel = rel.to(self.device)
                tail = tail.to(self.device)

                pos_logits = self._kge_method.forward(head, rel, tail)

                neg_logits = 0
                for i in range(self.num_negs):
                    neg_tail = th.randint(0, len(self._graph_node_to_id), (len(head),), device=self.device)
                    neg_logits += self._kge_method.forward(head, rel, neg_tail)
                neg_logits /= self.num_negs

 
                # if self.loss_type == "bpr":
                    # batch_loss = -criterion_bpr(self.margin + pos_logits).mean() - criterion_bpr(-neg_logits - self.margin).mean()
                # elif self.loss_type == "normal":
                batch_loss = -pos_logits.mean() + th.relu(self.margin + neg_logits).mean()

                    
                # batch_loss += self._kge_method.collect_regularization_term().mean()
                
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                graph_loss += batch_loss.item()
                

            graph_loss /= len(graph_dataloader)

            valid_every = 100
            if epoch % valid_every == 0:
                evaluation_module = EvaluationModel(self.kge_method, self.triples_factory, self.dataset, self.embed_dim, self.device)

                valid_metrics = self.evaluator.evaluate(evaluation_module, mode="valid")
                valid_mrr = valid_metrics["valid_mrr"]
                valid_mr = valid_metrics["valid_mr"]
                valid_metrics["train_loss"] = graph_loss
                                    
                # valid_mean_rank, valid_mrr = self.compute_ranking_metrics(mode="validate")
                if valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    th.save(self._kge_method, self.model_filepath)
                    tolerance = initial_tolerance+1
                    print("Model saved")
                else:
                    tolerance -= 1


                    
                    
                print(f"Training loss: {graph_loss:.6f}\tValidation mean rank: {valid_mr:.6f}\tValidation MRR: {valid_mrr:.6f}")
                self.wandb_logger.log({"epoch": epoch, "train_loss": graph_loss, "valid_mr": valid_mr, "valid_mrr": valid_mrr})
            
                                    
            if tolerance == 0:
                print("Early stopping")
                break

                



        
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
        self.graph_ids = th.tensor(list(ont_id_to_graph_id.values())).to(self.device)
        
        relation_id = triples_factory.relation_to_id["http://arrow"]
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
