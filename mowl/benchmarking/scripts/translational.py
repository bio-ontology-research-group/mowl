import sys
sys.path.append("../../../")

import mowl
mowl.init_jvm("5g")
from os.path import exists

from mowl.datasets.ppi_yeast import PPIYeastSlimDataset, PPIYeastDataset
from mowl.datasets.base import PathDataset
from mowl.projection.factory import projector_factory, PARSING_METHODS
from mowl.projection.edge import Edge
from mowl.walking.factory import walking_factory, WALKING_METHODS
from mowl.evaluation.rank_based import EmbeddingsRankBasedEvaluator
from mowl.evaluation.base import CosineSimilarity
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pickle as pkl
from mowl.visualization.base import TSNE as MTSNE
from mowl.benchmarking.scripts.util import exist_files, load_pickles, save_pickles
from mowl.embeddings.translational.model import TranslationalOnt
import logging
logging.basicConfig(level=logging.INFO)


import click as ck

@ck.command()
@ck.option(
    "--test", "-t", default = "ppi", help = "Type of test: ppi or gda")
def main(test):
    global ROOT

    tsne = False
    if test == "ppi":
        ROOT = "../ppi/data/"
        ds = PPIYeastDataset()
        tsne = True
        
    elif test == "gda_mouse":
        ROOT = "../gda/data_mouse/"
        ds = PathDataset(ROOT + "train_mouse.owl", ROOT + "valid_mouse.owl", ROOT + "test_mouse.owl")
    elif test == "gda_human":
        ROOT = "../gda/data_human/"
        ds = PathDataset(ROOT + "train_human.owl", ROOT + "valid_human.owl", ROOT + "test_human.owl")

    parsing_methods = [m for m in PARSING_METHODS if not ("taxonomy" in m)]
    
    

    dummy_params  = {
        "bd" : True,
        "ot" : False,
        "il" : True,

        "vector_size" : 20,
        "epochs" : 2,
        "batch_size": 32,
        "device": "cuda:1"
    }
    
    
    params = {
        "bd" : True,
        "ot" : False,
        "il" : True,

        "vector_size" : 100,
        "epochs" : 20,
        "batch_size": 1024,
        "device": "cuda:1"
    }
    
    
    for pm in parsing_methods:
        for t in ["trans" + m for m in ["E", "R", "H", "D"]]:
            benchmark_case(ds,pm,t,params, test, tsne)

def benchmark_case(dataset, parsing_method, trans_method, params, test, tsne = False):

    graph_train_file = ROOT + f"graphs/{parsing_method}_train.pkl"
    graph_test_file = ROOT + f"graphs/{parsing_method}_test.pkl"
    graph_data_files = [graph_train_file, graph_test_file]

    eval_train_file = ROOT + "eval_data/training_set.pkl"
    eval_test_file = ROOT + "eval_data/test_set.pkl"
    eval_heads_file = ROOT + "eval_data/head_entities.pkl"
    eval_tails_file = ROOT + "eval_data/tail_entities.pkl"
    
                            
    if exist_files(*graph_data_files):
        logging.info("Graph found. Loading...")
        train_edges, test_edges, *_ = load_pickles(*graph_data_files)

    else:
        logging.info("Graph not found. Generating...")

        bd = params["bd"]
        il = params["il"]
        ot = params["ot"]
        
        projector = projector_factory(
            parsing_method,
            bidirectional_taxonomy = bd,
            include_literals = il,
            only_taxonomy = ot
        )

        train_edges = projector.project(dataset.ontology)
        test_edges = projector.project(dataset.testing)

        save_pickles((train_edges, graph_train_file), (test_edges, graph_test_file))

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
            eval_projector = projector_factory("taxonomy_rels", relations = ["http://is_associated_with"])

            eval_train_edges = eval_projector.project(dataset.ontology)
            eval_test_edges = eval_projector.project(dataset.testing)

            train_entities, _ = Edge.getEntitiesAndRelations(eval_train_edges)
            test_entities, _ = Edge.getEntitiesAndRelations(eval_test_edges)
                        
            all_entities = list(set(train_entities) | set(test_entities))
            
            head_entities = [e for e in all_entities if e.is_numeric()]
            tail_entities = [e for e in all_entities if e.contains("OMIM")]                                  

        save_pickles(
            (eval_train_edges, eval_train_file),
            (eval_test_edges, eval_test_file),
            (head_entities, eval_heads_file),
            (tail_entities, eval_tails_file)
        )

    transMethod = TranslationalOnt(
        train_edges,
        trans_method = trans_method,
        embedding_dim = params["vector_size"],
        epochs = params["epochs"],
        batch_size = params["batch_size"],
        device = params["device"]
    )

    transMethod.train()

    embeddings = transMethod.get_embeddings()
    ### FINALLY, EVALUATION

    evaluator = EmbeddingsRankBasedEvaluator(
        embeddings,
        eval_test_edges,
        CosineSimilarity,
        training_set=eval_train_edges,
        head_entities = head_entities,
        tail_entities = tail_entities,
        device = params["device"]
    )

    evaluator.evaluate(show=False)

    log_file = ROOT + f"results/{parsing_method}_{trans_method}.dat"

    with open(log_file, "w") as f:
        tex_table = ""
        for k, v in evaluator.metrics.items():
            tex_table += f"{v} &\t"
            f.write(f"{k}\t{v}\n")

        f.write(f"\n{tex_table}")


    ###### TSNE ############


    if tsne:
        if test == "ppi":
            ec_num_file = ROOT + "ec_numbers_data"

            if exist_files(ec_num_file):
                labels, = load_pickles(ec_num_file)
            else:
                labels = {}
                with open(ROOT + "yeast_ec.tab", "r") as f:
                    next(f)
                    for line in f:
                        it = line.strip().split('\t', -1)
                        if len(it) < 5:
                            continue
                        if it[3]:
                            prot_id = it[3].split(';')[0]
                            prot_id = '{0}'.format(prot_id)
                            labels[f"http://{prot_id}"] = it[4].split(".")[0]

                save_pickles((labels, ec_num_file))

            entities = list(set(head_entities) | set(tail_entities))
    
        tsne = MTSNE(embeddings, labels, entities = entities)
        tsne.generate_points(5000, workers = 16, verbose = 1)
        tsne.savefig(ROOT + f'tsne/{parsing_method}_{trans_method}.jpg')
        


if __name__ == "__main__":
    main()
