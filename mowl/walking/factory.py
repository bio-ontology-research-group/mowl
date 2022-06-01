from mowl.walking.deepwalk.model import DeepWalk
from mowl.walking.node2vec.model import Node2Vec
from mowl.walking.rdf2vec.model import RDF2Vec


WALKING_METHODS = ["deepwalk", "node2vec"]


def walking_factory(method_name, num_walks, walk_length, outfile, workers = 1, alpha = 0, p = 1, q=1):


    if method_name == "deepwalk":
        return DeepWalk(num_walks, walk_length, alpha, outfile, workers=workers)
    elif method_name == "node2vec":
        return Node2Vec(num_walks, walk_length, p, q, outfile, workers=workers)
    elif method_name == "walkrdfowl":
        return RDF2Vec([], num_walks, walk_length, outfile, workers=workers)
    else:
        raise Exception(f"Walking method unrecognized. Recognized methods are: {WALKING_METHODS}")
