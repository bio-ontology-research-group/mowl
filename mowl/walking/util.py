from mowl.walking.deepwalk.model import DeepWalk



def walking_factory(method_name, edges, num_paths, path_length, alpha, num_workers = 1, seed=0, outfile=None):

    methods = [
        "deepwalk"
    ]

    if method_name == "deepwalk":
        return DeepWalk(edges, num_paths, path_length, alpha, num_workers=num_workers, seed=seed, outfile=outfile)
    else:
        raise Exception(f"Walking method unrecognized. Recognized methods are: {methods}")
