import sys
sys.path.append("../../../")

import mowl
mowl.init_jvm("5g")
from os.path import exists

from mowl.datasets.ppi_yeast import PPIYeastSlimDataset, PPIYeastDataset
from mowl.datasets.base import PathDataset
from mowl.projection.factory import projector_factory, PARSING_METHODS
from mowl.projection.edge import Edge
from mowl.evaluation.rank_based import ModelRankBasedEvaluator

from mowl.evaluation.base import CosineSimilarity

import pickle as pkl
from mowl.visualization.base import TSNE as MTSNE
from mowl.benchmarking.scripts.util import exist_files, load_pickles, save_pickles
from mowl.embeddings.elembeddings.model import ELEmbeddings
import logging
logging.basicConfig(level=logging.INFO)


import click as ck

@ck.command()
@ck.option(
    "--test", "-t", default = "ppi", help = "Type of test: ppi or gda")
@ck.option(
    "--device", "-d", default = "cpu", help = "Device"
)
def main(test, device):
    global ROOT

    tsne = False

    if test == "ppi":
        ROOT = "tmp/"
        ds = PPIYeastDataset()
        tsne = True
        
    elif test == "gda_mouse":
        ROOT = "../gda/data_mouse/"
        ds = PathDataset(ROOT + "train_mouse.owl", ROOT + "valid_mouse.owl", ROOT + "test_mouse.owl")
    elif test == "gda_human":
        ROOT = "../gda/data_human/"
        ds = PathDataset(ROOT + "train_human.owl", ROOT + "valid_human.owl", ROOT + "test_human.owl")

    else:
        raise ValueError(f"Type of test not recognized: {test}")

    
    
    

    dummy_params  = {
        "vector_size" : 10,
        "epochs" : 10,
        "margin": -0.1,
        "batch_size": 32,
        "device": device
    }
    
    
    params = {
        "vector_size" : 50,
        "epochs" : 5000,
        "margin": -0.1,
        "batch_size": 32,
        "device": device
        
    }
    
    
    benchmark_case(ds, dummy_params, device, test)

def benchmark_case(dataset, params, device, test):

    eval_train_file = ROOT + "eval_data/training_set.pkl"
    eval_test_file = ROOT + "eval_data/test_set.pkl"
    eval_heads_file = ROOT + "eval_data/head_entities.pkl"
    eval_tails_file = ROOT + "eval_data/tail_entities.pkl"

    eval_data_files = [eval_train_file, eval_test_file, eval_heads_file, eval_tails_file]
    
    if exist_files(*eval_data_files):
        logging.info("Evaluation data found. Loading...")
        eval_train_edges, eval_test_edges, head_entities, tail_entities = load_pickles(*eval_data_files)
    else:
        logging.info("Evaluation data not found. Generating...")

        if test == "ppi":
            eval_projector = projector_factory("taxonomy_rels", relations = ["http://interacts_with"])

            eval_train_edges = eval_projector.project(dataset.ontology)
            eval_test_edges = eval_projector.project(dataset.testing)
            
            train_head_ents, _, train_tail_ents = Edge.zip(eval_train_edges)
            test_head_ents, _, test_tail_ents = Edge.zip(eval_test_edges)
            
            head_entities = list(set(train_head_ents) | set(test_head_ents))
            tail_entities = list(set(train_tail_ents) | set(test_tail_ents))
            
        else:
            eval_projector = projector_factory("taxonomy_rels", taxonomy = True, relations = ["http://is_associated_with", "http://has_annotation"])

            eval_train_edges = eval_projector.project(dataset.ontology)
            eval_test_edges = eval_projector.project(dataset.testing)

            print(f"number of test edges is {len(eval_test_edges)}")
            train_entities, _ = Edge.getEntitiesAndRelations(eval_train_edges)
            test_entities, _ = Edge.getEntitiesAndRelations(eval_test_edges)
                        
            all_entities = list(set(train_entities) | set(test_entities))
            print("\n\n")
            print(len(test_entities))
            head_entities = [e for e in all_entities if e[7:].isnumeric()]
            tail_entities = [e for e in all_entities if "OMIM_" in e]
    
        save_pickles(
            (eval_train_edges, eval_train_file),
            (eval_test_edges, eval_test_file),
            (head_entities, eval_heads_file),
            (tail_entities, eval_tails_file)
        )
        
    model = ELEmbeddings(
        dataset,
        epochs = params["epochs"],
        margin = params["margin"],
        device = params["device"],
        model_filepath = ROOT + "elem_model.th"
    )
    
    model.train()

        
            
    ### FINALLY, EVALUATION

    #model.evaluate_ppi()
    evaluator = ModelRankBasedEvaluator(
        model,
        device = device
    )

    evaluator.evaluate(show=False)

    log_file = ROOT + f"results_elembeddings.dat"

    with open(log_file, "w") as f:
        tex_table = ""
        for k, v in evaluator.metrics.items():
            tex_table += f"{v} &\t"
            f.write(f"{k}\t{v}\n")

        f.write(f"\n{tex_table}")


    ###### TSNE ############

    labels = dataset.get_labels()
        
    embeddings, _ = model.get_embeddings()

    entities = list(set(head_entities) | set(tail_entities))
    tsne = MTSNE(embeddings, labels, entities = entities)
    tsne.generate_points(5000, workers = 16, verbose = 1)
    tsne.savefig(ROOT + f'tsne/elembeddings.jpg')
        


if __name__ == "__main__":
    main()
