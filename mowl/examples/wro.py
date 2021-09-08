#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import pickle
import gzip
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)   
sys.path.insert(0, '')
sys.path.append('../../')


from mowl.datasets import PPIYeastDataset
from mowl.graph.walking_rdf_and_owl.model import WalkRdfOwl

@ck.command()
def main():
    logging.info(f"Number of cores detected: {os.cpu_count()}")
    ds = PPIYeastDataset()
    model = WalkRdfOwl(ds, 'walk_rdf_corpus.txt', 'walk_rdf_embeddings.wordvectors', 
                       number_walks = 10, #500, 
                       length_walk = 5,# 40, 
                       embedding_size= 10, #256,
                       window = 5,
                       min_count = 5,
                       data_root = "../data")
    model.train()
#    relations = ['has_interaction']
#    model.evaluate(relations)


if __name__ == '__main__':
    main()
