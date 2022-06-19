import sys
sys.path.append("../../")

import mowl
mowl.init_jvm("5g")
from os.path import exists

from mowl.datasets.ppi_yeast import PPIYeastSlimDataset, PPIYeastDataset
from mowl.projection.factory import projector_factory, PARSING_METHODS
from mowl.projection.edge import Edge
from mowl.evaluation.rank_based import ModelRankBasedEvaluator

from mowl.evaluation.base import CosineSimilarity

import pickle as pkl
from mowl.visualization.base import TSNE as MTSNE
from mowl.benchmarking.util import exist_files, load_pickles, save_pickles
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



    if test == "ppi":
        ROOT = "data_ppi/"
        ds = PPIYeastSlimDataset()
    elif test == "gda":
        ROOT = "data_gda/"
    else:
        raise ValueError(f"Type of test not recognized: {test}")

    parsing_methods = [m for m in PARSING_METHODS if not ("taxonomy" in m)]
    
    

    dummy_params  = {
        "vector_size" : 20,
        "epochs" : 501,
        "margin": 0.1,
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
    
    
    benchmark_case(ds, dummy_params, device)

def benchmark_case(dataset, params, device):

    eval_train_file = ROOT + "eval_training_set.pkl"
    eval_test_file = ROOT + "eval_test_set.pkl"
    eval_heads_file = ROOT + "eval_head_entities.pkl"
    eval_tails_file = ROOT + "eval_tail_entities.pkl"

    eval_data_files = [eval_train_file, eval_test_file, eval_heads_file, eval_tails_file]
    
    if exist_files(*eval_data_files):
        logging.info("Evaluation data found. Loading...")
        eval_train_edges, eval_test_edges, head_entities, tail_entities = load_pickles(*eval_data_files)
    else:
        logging.info("Evaluation data not found. Generating...")

        eval_projector = projector_factory("taxonomy_rels", relations = ["http://interacts_with"])

        eval_train_edges = eval_projector.project(dataset.ontology)
        eval_test_edges = eval_projector.project(dataset.testing)

        train_head_ents, _, train_tail_ents = Edge.zip(eval_train_edges)
        test_head_ents, _, test_tail_ents = Edge.zip(eval_test_edges)

        head_entities = list(set(train_head_ents) | set(test_head_ents))
        tail_entities = list(set(train_tail_ents) | set(test_tail_ents))

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

    


    model.evaluate_ppi_()

    model.evaluate_ppi()
#    evaluator = ModelRankBasedEvaluator(
#        model,
#        device = device
#    )

#    evaluator.evaluate(show=False)

#    log_file = ROOT + f"results_elembeddings.dat"

#    with open(log_file, "w") as f:
#        tex_table = ""
#        for k, v in evaluator.metrics.items():
#            tex_table += f"{v} &\t"
#            f.write(f"{k}\t{v}\n")

#        f.write(f"\n{tex_table}")


    ###### TSNE ############

    ec_num_file = ROOT + "ec_numbers_data"

    if exist_files(ec_num_file):
        ec_numbers, = load_pickles(ec_num_file)
    else:
        ec_numbers = {}
        with open("yeast_ec.tab", "r") as f:
            next(f)
            for line in f:
                it = line.strip().split('\t', -1)
                if len(it) < 5:
                    continue
                if it[3]:
                    prot_id = it[3].split(';')[0]
                    prot_id = '{0}'.format(prot_id)
                    ec_numbers[f"http://{prot_id}"] = it[4].split(".")[0]

        save_pickles((ec_numbers, ec_num_file))

    embeddings, _ = model.get_embeddings()

    entities = list(set(head_entities) | set(tail_entities))
    tsne = MTSNE(embeddings, ec_numbers, entities = entities)
    tsne.generate_points(5000, workers = 16, verbose = 1)
    tsne.savefig(ROOT + f'mowl_tsne_elembeddings.jpg')
        


if __name__ == "__main__":
    main()
