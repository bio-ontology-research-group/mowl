import sys
sys.path.append("../")
import mowl
mowl.init_jvm("10g")
from mowl.models import SyntacticPlusW2VModel
from mowl.utils.random import seed_everything
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
from evaluators import SubsumptionEvaluator
from datasets import SubsumptionDataset
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
@ck.option("--dataset_name", "-ds", type=ck.Choice(["go", "foodon"]), default="go")
@ck.option("--evaluator_name", "-e", default="subsumption", help="Evaluator to use")
@ck.option("--embed_dim", "-dim", default=50, help="Embedding dimension")
@ck.option("--batch_size", "-bs", default=78000, help="Batch size")
@ck.option("--epochs", "-e", default=10, help="Number of epochs")
@ck.option("--evaluate_deductive", "-evalded", is_flag=True, help="Use deductive closure as positive examples for evaluation")
@ck.option("--filter_deductive", "-filterded", is_flag=True, help="Filter out examples from deductive closure")
@ck.option("--device", "-d", default="cuda", help="Device to use")
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(dataset_name, evaluator_name, embed_dim, batch_size, epochs,
         evaluate_deductive, filter_deductive, device,
         wandb_description, no_sweep, only_test):

    seed_everything(42)
    
    wandb_logger = wandb.init(entity="zhapacfp_team", project="ontoem", group="f{dataset_name}_{evaluate_deductive}_{filter_deductive}", name=wandb_description)

    if no_sweep:
        wandb_logger.log({"dataset_name": dataset_name,
                          "embed_dim": embed_dim,
                          "epochs": epochs
                          })
    else:
        dataset_name = wandb.config.dataset_name
        embed_dim = wandb.config.embed_dim
        epochs = wandb.config.epochs
                        
    root_dir, dataset = dataset_resolver(dataset_name)

    model_dir = f"{root_dir}/../models/"
    os.makedirs(model_dir, exist_ok=True)

    corpora_dir = f"{root_dir}/../corpora/"
    os.makedirs(corpora_dir, exist_ok=True)
    
    model_filepath = f"{model_dir}/{embed_dim}_{epochs}.pt"
    corpus_filepath = f"{corpora_dir}/{embed_dim}_{epochs}.txt"

    model = OPA2VecModel(evaluator_name, dataset, batch_size,
                         embed_dim, model_filepath, corpus_filepath, epochs,
                         evaluate_deductive, filter_deductive,
                         device, wandb_logger)

    
    if not only_test:
        model.train()
        model.w2v_model.save(model_filepath)
        
    metrics = model.test()
    print_as_md(metrics)

             
    wandb_logger.log(metrics)
        

def print_as_md(overall_metrics):
    
    metrics = ["test_mr", "test_mrr", "test_auc", "test_hits@1", "test_hits@3", "test_hits@10", "test_hits@50", "test_hits@100"]
    filt_metrics = [k.replace("_", "_f_") for k in metrics]

    string_metrics = "| Property | MR | MRR | AUC | Hits@1 | Hits@3 | Hits@10 | Hits@50 | Hits@100 | \n"
    string_metrics += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | \n"
    string_filtered_metrics = "| Property | MR | MRR | AUC | Hits@1 | Hits@3 | Hits@10 | Hits@50 | Hits@100 | \n"
    string_filtered_metrics += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | \n"
    
    string_metrics += "| Overall | "
    string_filtered_metrics += "| Overall | "
    for metric in metrics:
        if metric == "test_mr":
            string_metrics += f"{int(overall_metrics[metric])} | "
        else:
            string_metrics += f"{overall_metrics[metric]:.4f} | "
    for metric in filt_metrics:
        if metric == "test_f_mr":
            string_filtered_metrics += f"{int(overall_metrics[metric])} | "
        else:
            string_filtered_metrics += f"{overall_metrics[metric]:.4f} | "


    print(string_metrics)
    print("\n\n")
    print(string_filtered_metrics)
        
    

    
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


class OPA2VecModel(SyntacticPlusW2VModel):
    def __init__(self, evaluator_name, dataset, batch_size, embed_dim,
                 model_filepath, corpus_filepath, epochs,
                 evaluate_deductive, filter_deductive, device,
                 wandb_logger):
        super().__init__(dataset, model_filepath=model_filepath, corpus_filepath=corpus_filepath)


        
        
        self.embed_dim = embed_dim
        self.evaluator = evaluator_resolver(evaluator_name, dataset,
                                            device, batch_size=16,
                                            evaluate_with_deductive_closure=evaluate_deductive,
                                            filter_deductive_closure=filter_deductive)
        self.epochs = epochs
        self.device = device
        self.wandb_logger = wandb_logger

        self.set_w2v_model(vector_size=embed_dim, min_count=1, epochs=self.epochs)
        if os.path.exists(self.corpus_filepath):
            logger.warning("Corpus already exists in file. Loading it...")
            self.load_corpus()
        else:
            self.generate_corpus(save=True, with_annotations=True)
        
            
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
