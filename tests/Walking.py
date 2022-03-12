import sys

sys.path.append("..")
import numpy as np
import time
import pandas as pd
from mowl.datasets.base import PathDataset
from mowl.graph.dl2vec.model import DL2VecParser
from mowl.graph.edge import Edge
from mowl.walking.factory import walking_factory
from mowl.walking.node2vec.model import Node2Vec as N2V
from mowl.walking.walkRdfAndOwl.model import WalkRDFAndOWL as WRDF


def testCase (dataset, num_walks, walk_length, alpha, p, q, workers):

    print("Parsing...")
    parser = DL2VecParser(dataset.ontology, bidirectional_taxonomy = True)
    edges = parser.parse()

    ents, _ = Edge.getEntitiesAndRelations(edges)

    ents_dict = {v: k for k, v in enumerate(ents)}
    
    srcs = []
    dsts = []

    for edge in edges:
        src, dst = edge.src(), edge.dst()
        srcs.append(ents_dict[src])
        dsts.append(ents_dict[dst])

    srcs = np.array(srcs)
    dsts = np.array(dsts)



    print("\n\nDeepWalk")

    start = time.time()

    walker = walking_factory(
        "deepwalk",
        edges,
        num_walks,
        walk_length,
        "data/walks_deepwalk",
        alpha = alpha,
        workers = workers
    )
    walker.walk()
    end = time.time()
    print(f"DeepWalk time: {end - start}")


    print("\n\nWalkRDFAndOWL")

    start = time.time()

    walker = walking_factory(
        "walkrdfowl",
        edges,
        num_walks,
        walk_length,
        "data/walks_walkrdfowl",
        workers = workers
    )
    walker.walk()
    end = time.time()
    print(f"WalkRDFAndOWL time: {end - start}")

    print("\n\nNode2Vec")

    start = time.time()
    walker = walking_factory(
        "node2vec",
        edges,
        num_walks,
        walk_length,
        "data/walks_node2Vec",
        p = p,
        q = q,
        workers = workers
    )
    walker.walk()
    end = time.time()
    print(f"Node2Vec time: {end - start}")

    

    

if __name__ == "__main__":
    data1 = "data/go.owl"
    data2 = "data/goslim_yeast.owl"
    ds = PathDataset(data2, None, None)
    walk_length = 100
    num_walks = 10
    p = 1
    q = 1
    alpha = 0
    workers = 16
    
    testCase(ds, num_walks, walk_length, alpha, p, q, workers)
