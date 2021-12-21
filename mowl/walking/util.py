from mowl.walking.deepwalk.model import DeepWalk



def walking_factory(method_name, edges, num_walks, walk_length, alpha, num_workers = 1, seed=0, outfile=None, p = 1, q=1):

    methods = [
        "deepwalk",
        "node2vec"
    ]

    if method_name == "deepwalk":
        return DeepWalk(edges, num_walks, walk_length, alpha, num_workers=num_workers, seed=seed, outfile=outfile)
    if method_name == "node2vec":
        return Node2Vec(edges, num_walks, walk_length, p, q, num_workers=num_workers, seed=seed, outfile=outfile)
    else:
        raise Exception(f"Walking method unrecognized. Recognized methods are: {methods}")
