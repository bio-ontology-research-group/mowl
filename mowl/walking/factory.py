from multiprocessing.sharedctypes import Value
from mowl.walking.deepwalk.model import DeepWalk
from mowl.walking.node2vec.model import Node2Vec
from mowl.error import INVALID_WALKER_NAME
WALKING_METHODS = ["deepwalk", "node2vec"]


def walker_factory(method_name, num_walks, walk_length, outfile=None, workers=1, alpha=0.,
                   p=1., q=1.):

    if method_name == "deepwalk":
        return DeepWalk(num_walks, walk_length, alpha=alpha, outfile=outfile, workers=workers)
    elif method_name == "node2vec":
        return Node2Vec(num_walks, walk_length, p=p, q=q, outfile=outfile, workers=workers)
    else:
        raise ValueError(INVALID_WALKER_NAME)
