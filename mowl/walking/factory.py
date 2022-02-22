from mowl.walking.deepwalk.model import DeepWalk
from mowl.walking.node2vec.model import Node2Vec
from mowl.walking.walkRdfAndOwl.model import WalkRDFAndOWL

def walking_factory(method_name, edges, num_walks, walk_length, outfile, workers = 1, alpha = 0, p = 1, q=1):

    methods = [
        "deepwalk",
        "node2vec",
        "walkrdfowl"
    ]

    if method_name == "deepwalk":
        return DeepWalk(edges, num_walks, walk_length, alpha, outfile, workers=workers)
    elif method_name == "node2vec":
        return Node2Vec(edges, num_walks, walk_length, p, q, outfile, workers=workers)
    elif method_name == "walkrdfowl":
        return WalkRDFAndOWL(edges, num_walks, walk_length, outfile, workers=workers)
    else:
        raise Exception(f"Walking method unrecognized. Recognized methods are: {methods}")
