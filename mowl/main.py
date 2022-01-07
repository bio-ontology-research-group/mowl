#!/usr/bin/env python

import click as ck
import sys
import os
sys.path.append('')
from mowl.datasets.ppi_yeast import PPIYeastSlimDataset, PPIYeastDataset
from mowl.onto2vec.model import Onto2Vec
from mowl.elembeddings.model import ELEmbeddings


@ck.command()
@ck.option(
    '--model', '-m', default='onto2vec',
    help='Method for generating embeddings')
@ck.option(
    '--dataset', '-d', default='ppi_yeast',
    help='Dataset name')
def main(model, dataset):
    dataset = PPIYeastDataset()
    model = ELEmbeddings(dataset)
    # model.train()
    model.evaluate_ppi()

if __name__ == '__main__':
    main()
