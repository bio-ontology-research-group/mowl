import sys

sys.path.insert(0, '')
sys.path.append('../../../')

import random

from mowl.datasets.base import PathDataset
from mowl.graph.taxonomy.model import TaxonomyParser


def get_closure(ont_file, out_file):
    dataset = PathDataset(ont_file, None, None)

    parser = TaxonomyParser(dataset.ontology)


    edges_trans_closure =  parser.parseWithTransClosure()
    with open(out_file, "w") as f:
        for e in edges_trans_closure:
            f.write(f"{e.src()}\t{e.dst()}\n")
        

def only_closure(ont_file, closure_file, out_file):

    dataset = PathDataset(ont_file, None, None)
    parser = TaxonomyParser(dataset.ontology)
    original_edges = parser.parse()

    original_edges = set([(e.src(), e.dst()) for e in original_edges])

    with open(closure_file, "r") as f:
        closure_edges = set()
        for line in f:
            line = line.rstrip("\n").split("\t")
            closure_edges.add((line[0], line[1]))

    only_closure_edges = closure_edges - original_edges

    with open(out_file, "w") as f:
        for s, d in list(only_closure_edges):
            f.write(f"{s}\t{d}\n")


def split_closure(train_file, closure_file, valid_out_file, test_out_file):
    
    dataset = PathDataset(train_file, None, None)
    parser = TaxonomyParser(dataset.ontology)
    original_edges = parser.parse()

    original_edges = set([(e.src(), e.dst()) for e in original_edges])


    with open(closure_file, "r") as f:
        closure_edges = set()
        for line in f:
            line = line.rstrip("\n").split("\t")
            closure_edges.add((line[0], line[1]))


    closure_edges = list(closure_edges)
    random.shuffle(closure_edges)

    num_closure = len(closure_edges)

    valid_data = closure_edges[:num_closure//2]
    test_data = closure_edges[num_closure//2:]

    with open(valid_out_file, "w") as f:
        for s, d in list(valid_data):
            f.write(f"{s}\t{d}\n")

    with open(test_out_file, "w") as f:
        for s, d in list(test_data):
            f.write(f"{s}\t{d}\n")
