import sys
sys.path.append("../../../")

import mowl
mowl.init_jvm("10g")
from os.path import exists

from mowl.datasets.ppi_yeast import PPIYeastSlimDataset, PPIYeastDataset
from mowl.corpus.base import extract_and_save_axiom_corpus, extract_annotation_corpus
from mowl.projection.edge import Edge
from mowl.projection.factory import projector_factory
from mowl.reasoning.base import MOWLReasoner
from mowl.datasets.base import PathDataset
from mowl.benchmarking.scripts.util import exist_files, load_pickles, save_pickles
from mowl.walking.factory import walking_factory, WALKING_METHODS
from mowl.evaluation.rank_based import EmbeddingsRankBasedEvaluator
from mowl.evaluation.base import CosineSimilarity
from mowl.visualization.base import TSNE as MTSNE
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from org.semanticweb.elk.owlapi import ElkReasonerFactory

import pickle as pkl
import logging
import click as ck

ROOT = None

        
@ck.command()
@ck.option(
    "--test", "-t", default = "ppi", help = "Type of test: ppi or gda")
def main(test):

    global ROOT
    methods = ["opa2vec", "onto2vec"]

    tsne = False
    if test == "ppi":
        ROOT = "tmp/"
        #ROOT = "../ppi/data/"
        ds = PPIYeastDataset()
        tsne = True
        
    elif test == "gda_mouse":
        ROOT = "../gda/data_mouse/"
        ds = PathDataset(ROOT + "train_mouse.owl", ROOT + "valid_mouse.owl", ROOT + "test_mouse.owl")
    elif test == "gda_human":
        ROOT = "../gda/data_human/"
        ds = PathDataset(ROOT + "train_human.owl", ROOT + "valid_human.owl", ROOT + "test_human.owl")

    
                                

    dummy_params  = {
        "vector_size" : 20,
        "window" : 5,
        "epochs" : 5,
        "workers": 16
    }

    params = {
        "vector_size" : 100,
        "window" : 5,
        "epochs" : 40,
        "workers": 16
    }
    
    
    for m in methods:
        benchmark_case(ds,m,dummy_params, test, tsne)




def benchmark_case(dataset, method, params, test, tsne = False):

    corpus_file = ROOT +  f"corpus/{method}"

    if not exists(corpus_file) or True:
        reasoner_factory = ElkReasonerFactory()
        reasoner = reasoner_factory.createReasoner(dataset.ontology)
        reasoner.precomputeInferences()
        
        mowl_reasoner = MOWLReasoner(reasoner)
        mowl_reasoner.infer_subclass_axioms(dataset.ontology)
        mowl_reasoner.infer_equiv_class_axioms(dataset.ontology)
        
        extract_and_save_axiom_corpus(dataset.ontology, corpus_file)

    if method == "opa2vec":
        extract_annotation_corpus(dataset.ontology, corpus_file)
        
    ### WORD2VEC

    word2vec_file = ROOT + f"embeddings/{method}"

    if exists(word2vec_file):
        w2v = Word2Vec.load(word2vec_file)
        vectors = w2v.wv
    else:
        sentences = LineSentence(corpus_file)

        vector_size = params["vector_size"]
        window = params["window"]
        epochs = params["epochs"]
        workers = params["workers"]


        model = Word2Vec(
            sentences,
            sg = 1,
            min_count = 1,
            vector_size = vector_size,
            window = window,
            epochs = epochs,
            workers = workers
        )

        model.save(word2vec_file)

        vectors = model.wv

    
    ### FINALLY, EVALUATION
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


    ### FINALLY, EVALUATION

    evaluator = EmbeddingsRankBasedEvaluator(
        vectors,
        eval_test_edges,
        CosineSimilarity,
        training_set=eval_train_edges,
        head_entities = head_entities,
        tail_entities = tail_entities,
        device = "cuda"
    )

    evaluate = True
    if evaluate:
        evaluator.evaluate(show=False)

        log_file = ROOT + f"results/{method}.dat"

        with open(log_file, "w") as f:
            tex_table = ""
            for k, v in evaluator.metrics.items():
                tex_table += f"{v} &\t"
                f.write(f"{k}\t{v}\n")

            f.write(f"\n{tex_table}")


    ###### TSNE ############

    if tsne:
        if test == "ppi":
            labels = dataset.get_labels()
            entities = list(set(head_entities) | set(tail_entities))
        tsne = MTSNE(vectors, labels, entities = entities)
        tsne.generate_points(5000, workers = 16, verbose = 1)
        tsne.savefig(ROOT + f'tsne/{method}.jpg')





if __name__ == "__main__":
    main()
