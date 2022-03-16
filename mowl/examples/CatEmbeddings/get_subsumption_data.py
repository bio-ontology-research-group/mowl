import sys

sys.path.insert(0, '')
sys.path.append('../../../')

import random

from mowl.datasets.base import PathDataset
from mowl.graph.taxonomy.model import TaxonomyParser


def parse_with_closure(ont_file, out_train_edges_file, out_closure_edges_file):
    dataset = PathDataset(ont_file, None, None)

    parser = TaxonomyParser(dataset.ontology)

    train_edges = parser.parse()
    closure_edges =  parser.parseWithTransClosure()

    with open(out_train_edges_file, "w") as f:
        for e in train_edges:
            f.write(f"{e.src()}\t{e.dst()}\n")

    with open(out_closure_edges_file, "w") as f:
        for e in closure_edges:
            f.write(f"{e.src()}\t{e.dst()}\n")

    

def extract_only_closure(train_edges_file, closure_edges_file, out_only_closure_edges_file):

    train_edges = set()
    with open(train_edges_file, "r") as f:
        for line in f:
            line = line.rstrip("\n").split("\t")
            train_edges.add((line[0], line[1]))

    with open(closure_edges_file, "r") as f:
        closure_edges = set()
        for line in f:
            line = line.rstrip("\n").split("\t")
            closure_edges.add((line[0], line[1]))

    only_closure_edges = closure_edges - train_edges

    with open(out_only_closure_edges_file, "w") as f:
        for s, d in list(only_closure_edges):
            f.write(f"{s}\t{d}\n")


def split_closure(closure_file, valid_out_file, test_out_file):
    
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



if __name__ == "__main__":
    root = "data/"

    ont_file = root + "go.owl"
    train_edges_file = root + "train_pos_data.tsv"
    closure_edges_file = root + "closure_pos_data.tsv"
    only_closure_edges_file = root + "only_closure_pos_data.tsv"
    parse_with_closure(ont_file, train_edges_file, closure_edges_file)
    extract_only_closure(train_edges_file, closure_edges_file, only_closure_edges_file)
