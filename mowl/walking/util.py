from mowl.walking.deepwalk.model import DeepWalk
from mowl.walking.node2vec.model import Node2Vec


def walking_factory(method_name, edges, num_walks, walk_length, alpha, workers = 1, seed=0, p = 1, q=1):

    methods = [
        "deepwalk",
        "node2vec"
    ]

    if method_name == "deepwalk":
        return DeepWalk(edges, num_walks, walk_length, alpha, workers=workers, seed=seed)
    if method_name == "node2vec":
        return Node2Vec(edges, num_walks, walk_length, p, q, workers=workers)
    else:
        raise Exception(f"Walking method unrecognized. Recognized methods are: {methods}")
