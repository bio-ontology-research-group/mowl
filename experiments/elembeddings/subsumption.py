import sys
sys.path.append("../")
import mowl
mowl.init_jvm("10g")
from mowl.base_models.elmodel import EmbeddingELModel
from mowl.utils.random import seed_everything
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
from evaluators import SubsumptionEvaluator
from datasets import SubsumptionDataset
from utils import print_as_md
from tqdm import tqdm
from mowl.nn import ELEmModule
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from itertools import cycle
import logging
import math
import click as ck
import os
import wandb
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

th.autograd.set_detect_anomaly(True)

@ck.command()
@ck.option("--dataset_name", "-ds", type=ck.Choice(["go", "foodon"]), default="go")
@ck.option("--embed_dim", "-dim", default=50, help="Embedding dimension")
@ck.option("--batch_size", "-bs", default=78000, help="Batch size")
@ck.option("--module_margin", "-mm", default=0.1, help="Margin for the module")
@ck.option("--loss_margin", "-lm", default=0.1, help="Margin for the loss function")
@ck.option("--learning_rate", "-lr", default=0.001, help="Learning rate")
@ck.option("--epochs", "-ep", default=10000, help="Number of epochs")
@ck.option("--evaluate_every", "-every", default=50, help="Evaluate every n epochs")
@ck.option("--evaluate_deductive", "-evalded", is_flag=True, help="Use deductive closure as positive examples for evaluation")
@ck.option("--filter_deductive", "-filterded", is_flag=True, help="Filter out examples from deductive closure")
@ck.option("--device", "-d", default="cuda", help="Device to use")
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(dataset_name, embed_dim, batch_size, module_margin,
         loss_margin, learning_rate, epochs, evaluate_every,
         evaluate_deductive, filter_deductive, device,
         wandb_description, no_sweep, only_test):

    seed_everything(42)

    evaluator_name = "subsumption"
    
    wandb_logger = wandb.init(entity="zhapacfp_team", project="ontoem", group=f"elembeddings_{dataset_name}", name=wandb_description)


    if loss_margin == int(loss_margin):
        loss_margin = int(loss_margin)
    if module_margin == int(module_margin):
        module_margin = int(module_margin)
    
    if no_sweep:
        wandb_logger.log({"dataset_name": dataset_name,
                          "embed_dim": embed_dim,
                          "module_margin": module_margin,
                          "loss_margin": loss_margin,
                          "learning_rate": learning_rate
                          })
    else:
        dataset_name = wandb.config.dataset_name
        embed_dim = wandb.config.embed_dim
        module_margin = wandb.config.module_margin
        loss_margin = wandb.config.loss_margin
        learning_rate = wandb.config.learning_rate
 
    
    root_dir, dataset = dataset_resolver(dataset_name)

    model_dir = f"{root_dir}/../models/"
    os.makedirs(model_dir, exist_ok=True)

    model_filepath = f"{model_dir}/elembeddings_{embed_dim}_{module_margin}_{loss_margin}_{learning_rate}.pt"
    model = GeometricELModel(evaluator_name, dataset, batch_size,
                             embed_dim, module_margin, loss_margin,
                             learning_rate, model_filepath, epochs,
                             evaluate_every, evaluate_deductive,
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


class GeometricELModel(EmbeddingELModel):
    def __init__(self, evaluator_name, dataset, batch_size, embed_dim,
                 module_margin, loss_margin, learning_rate,
                 model_filepath, epochs, evaluate_every,
                 evaluate_deductive, filter_deductive, device,
                 wandb_logger):
        super().__init__(dataset, embed_dim, batch_size, model_filepath=model_filepath)

        self.module = ELEmModule(len(self.dataset.classes),
                                 len(self.dataset.object_properties),
                                 len(self.dataset.individuals),
                                 self.embed_dim,
                                 module_margin)

        self.evaluator = evaluator_resolver(evaluator_name, dataset, device, evaluate_with_deductive_closure=evaluate_deductive, filter_deductive_closure=filter_deductive)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.evaluate_every = evaluate_every
        self.loss_margin = loss_margin
        self.device = device
        self.wandb_logger = wandb_logger


    def tbox_forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

 
    def train(self):

        dls = {gci_name: DataLoader(ds, batch_size=self.batch_size, shuffle=True)
               for gci_name, ds in self.training_datasets.items() if len(ds) > 0}
        dls_sizes = {gci_name: len(ds) for gci_name, ds in self.training_datasets.items() if len(ds) > 0}
        total_dls_size = sum(dls_sizes.values())
        dls_weights = {gci_name: ds_size/total_dls_size for gci_name, ds_size in dls_sizes.items()}

        main_dl = dls["gci0"]
        logger.info(f"Training with {len(main_dl)} batches of size {self.batch_size}")
        dls = {gci_name: cycle(dl) for gci_name, dl in dls.items() if gci_name != "gci0"}
        logger.info(f"Dataloaders: {dls_sizes}")

        tolerance = 5
        curr_tolerance = tolerance

        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)

        best_mrr = 0
        best_loss = float("inf")
        
        num_classes = len(self.dataset.classes)

        self.module = self.module.to(self.device)
        for epoch in tqdm(range(self.epochs)):
            self.module.train()
            

            total_train_loss = 0
            
            for batch_data in main_dl:

                batch_data = batch_data.to(self.device)
                pos_logits = self.tbox_forward(batch_data, "gci0")
                neg_idxs = th.randint(0, num_classes, (len(batch_data),), device=self.device)
                neg_batch = th.cat([batch_data[:, :1], neg_idxs.unsqueeze(1)], dim=1)
                neg_logits = self.tbox_forward(neg_batch, "gci0")
                loss = - F.logsigmoid(-pos_logits + neg_logits - self.loss_margin).mean() * dls_weights["gci0"]

                for gci_name, gci_dl in dls.items():
                    if gci_name == "gci0":
                        continue

                    batch_data = next(gci_dl).to(self.device)
                    pos_logits = self.tbox_forward(batch_data, gci_name)
                    neg_idxs = th.randint(0, num_classes, (len(batch_data),), device=self.device)
                    neg_batch = th.cat([batch_data[:, :2], neg_idxs.unsqueeze(1)], dim=1)
                        
                    neg_logits = self.tbox_forward(neg_batch, gci_name)
                    loss += - F.logsigmoid(-pos_logits + neg_logits - self.loss_margin).mean() * dls_weights[gci_name]


                loss += self.module.regularization_loss()
                                                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                                    
            if epoch % self.evaluate_every == 0:
                valid_metrics = self.evaluator.evaluate(self.module, mode="valid")
                valid_mrr = valid_metrics["valid_mrr"]
                valid_mr = valid_metrics["valid_mr"]
                valid_metrics["train_loss"] = total_train_loss
                                    
                self.wandb_logger.log(valid_metrics)

                if valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    curr_tolerance = tolerance
                    th.save(self.module.state_dict(), self.model_filepath)
                else:
                    curr_tolerance -= 1

                if curr_tolerance == 0:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                logger.info(f"Epoch {epoch} - Train Loss: {total_train_loss:4f}  - Valid MRR: {valid_mrr:4f} - Valid MR: {valid_mr:4f}")

    def test(self):
        self.module.load_state_dict(th.load(self.model_filepath))
        self.module.to(self.device)
        self.module.eval()
        
        return self.evaluator.evaluate(self.module)

if __name__ == "__main__":
    main()
