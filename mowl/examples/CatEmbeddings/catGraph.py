import sys
sys.path.append('../../../')

import mowl
mowl.init_jvm("4g")

from org.mowl.Projectors import CatProjector
from mowl.datasets.ppi_yeast import PPIYeastSlimDataset, PPIYeastDataset
from mowl.projection.factory import projector_factory
from mowl.projection.edge import Edge
from mowl.walking.factory import walking_factory
from gensim.models import Word2Vec
from mowl.visualization.base import TSNE as MTSNE
from gensim.models.word2vec import LineSentence
from mowl.evaluation.rank_based import EmbeddingsRankBasedEvaluator
from mowl.evaluation.base import CosineSimilarity
import pickle as pkl
import logging
from os.path import exists


def exist_files(*args):
    ex = True
    for arg in args:
        ex &= exists(arg)

    return ex

def save_pickles(*args):
    for data, filename in args:
        with open(filename, "wb") as f:
            pkl.dump(data, f)

def load_pickles(*args):
    pkls = []
    for arg in args:
        with open(arg, "rb") as f:
            pkls.append(pkl.load(f))

    return tuple(pkls)



ROOT = "data/projection/"
dummy_params  = {
    "bd" : True,
    "ot" : False,
    "il" : True,
    
    "num_walks" : 10,
    "walk_length" : 10,
    "workers" : 16,
    "alpha" : 0.1,
    "p" : 10,
    "q" : 0.1,
    
    "vector_size" : 20,
    "window" : 5,
    "epochs" : 20
}

params = {
    "bd" : True,
    "ot" : False,
    "il" : False,
    
    "num_walks" : 20,
    "walk_length" : 20,
    "workers" : 16,
    "alpha" : 0.1,
    "p" : 10,
    "q" : 0.1,
    
    "vector_size" : 100, 
    "window" : 5 ,
    "epochs" : 10
}


params = params

test = "ppi"
dataset = PPIYeastDataset()
device = "cuda:1"
projector = CatProjector(dataset.ontology, True)

edges = projector.project()
edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
print(f"number of edges: {len(edges)}")
with open("data/projection/graph.pkl", "wb") as f:
    pkl.dump(edges, f)
walking_method = 'deepwalk'

walks_file = ROOT + f"walks_{walking_method}.txt"
workers = params["workers"]
    

logging.info("Walks not found. Generating...")
num_walks = params["num_walks"]
walk_length = params["walk_length"]

alpha = params["alpha"]
p = params["p"]
q = params["q"]
    
walker = walking_factory(
    walking_method,
    num_walks,
    walk_length,
    walks_file,
    workers = workers,
    alpha = alpha,
    p = p,
    q=q)

walker.walk(edges)


### WORD2VEC

word2vec_file = ROOT + f"embeddings_{walking_method}"

if exists(word2vec_file) and False:
    w2v = Word2Vec.load(word2vec_file)
    vectors = w2v.wv
else:
    sentences = LineSentence(walks_file)
    
    vector_size = params["vector_size"]
    window = params["window"]
    epochs = params["epochs"]
    
    
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



    
    evaluator = EmbeddingsRankBasedEvaluator(
        vectors,
        eval_test_edges,
        CosineSimilarity,
        training_set=eval_train_edges,
        head_entities = head_entities,
        device = device
    )

    evaluator.evaluate(show=False)

    log_file = ROOT + f"results_{walking_method}.dat"

    with open(log_file, "w") as f:
        tex_table = ""
        for k, v in evaluator.metrics.items():
            tex_table += f"{v} &\t"
            f.write(f"{k}\t{v}\n")

        f.write(f"\n{tex_table}")


    ###### TSNE ############
    tsne = True
    if tsne:
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

        entities = list(set(head_entities) | set(tail_entities))
        tsne = MTSNE(vectors, ec_numbers, entities = entities)
        tsne.generate_points(5000, workers = 16, verbose = 1)
        tsne.savefig(ROOT + f'mowl_tsne_{walking_method}.jpg')
        

