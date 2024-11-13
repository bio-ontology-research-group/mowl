#!/usr/bin/env python
import sys
sys.path.append("../")
import mowl
mowl.init_jvm("10g")

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from jpype import *
import jpype.imports
import os
import copy

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.reasoner import ConsoleProgressMonitor
from org.semanticweb.owlapi.reasoner import SimpleConfiguration
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from org.semanticweb.owlapi.util import InferredClassAssertionAxiomGenerator
from org.apache.jena.rdf.model import ModelFactory
from org.apache.jena.util import FileManager


# current directory is the mowl project root
# example parameters: --ont-file=../data/ppi_yeast_localtest/goslim_yeast.owl --data-file=../data/ppi_yeast_localtest/4932.protein.physical.links.v11.5.txt.gz --go-annots-file=../data/ppi_yeast_localtest/sgd.gaf.short --out-dir=../data/ppi_yeast_localtest


@ck.command()
@ck.option(
    '--ont-file', '-ont', default='data/go.owl',
    help='Ontology file (GO by default)')
@ck.option(
    '--org', '-org', type=ck.Choice(["yeast", "human"]),
    help='Organism')
@ck.option(
    '--go-annots-file', '-gf',
    help='as an alternative to --annots-file, specify the sdf.gaf from http://current.geneontology.org/products/pages/downloads.html, mapping uniprot proteins to GO annotations')
def main(ont_file, org, go_annots_file):
    if org == 'yeast':
        data_file = 'data/4932.protein.links.v11.5.txt.gz'
        annots_file = 'data/4932.annotations.tsv'
        out_dir = 'datasets/ppi_yeast'
        org_id = '4932'
    elif org == 'human':
        data_file = 'data/9606.protein.links.v12.0.txt.gz'
        annots_file = 'data/9606.annotations.tsv'
        out_dir = 'datasets/ppi_human'
        org_id = '9606'
    else:
        raise ValueError(f"Organism {org} not supported")
        
    train, valid, test = load_and_split_interactions(data_file)
    manager = OWLManager.createOWLOntologyManager()
    ont = manager.loadOntologyFromOntologyDocument(java.io.File(ont_file))
    factory = manager.getOWLDataFactory()
    interacts_rel = factory.getOWLObjectProperty(
        IRI.create("http://interacts_with"))
    has_function_rel = factory.getOWLObjectProperty(
        IRI.create("http://has_function"))

    # Add GO protein annotations to the GO ontology
    if go_annots_file:
        with open(go_annots_file) as f:
            for line in f:
                if not line.startswith('!'):
                    try:
                        items = line.strip().split('\t')
                        p_id = f'{org_id}.' + items[10] # e.g. '4932.YKL020C'
                        protein = factory.getOWLClass(IRI.create(f'http://{p_id}'))
                        go_id = items[4].replace(':', '_')
                        go_class = factory.getOWLClass(
                            IRI.create(f'http://purl.obolibrary.org/obo/{go_id}'))
                        axiom = factory.getOWLSubClassOfAxiom(
                            protein, factory.getOWLObjectSomeValuesFrom(
                                has_function_rel, go_class))
                        manager.addAxiom(ont, axiom)
                    except IndexError:
                        pass

    else:
        with open(annots_file) as f:
            for line in f:
                items = line.strip().split('\t')
                p_id = items[0]
                protein = factory.getOWLClass(IRI.create(f'http://{p_id}'))
                for go_id in items[1:]:
                    go_id = go_id.replace(':', '_')
                    go_class = factory.getOWLClass(
                        IRI.create(f'http://purl.obolibrary.org/obo/{go_id}'))
                    axiom = factory.getOWLSubClassOfAxiom(
                        protein, factory.getOWLObjectSomeValuesFrom(
                            has_function_rel, go_class))
                    manager.addAxiom(ont, axiom)

    # Add training set interactions to the ontology
    for inters in train:
        p1, p2 = inters[0], inters[1]  # e.g. 4932.YLR117C  and 4932.YPR101W
        protein1 = factory.getOWLClass(IRI.create(f'http://{p1}'))
        protein2 = factory.getOWLClass(IRI.create(f'http://{p2}'))
        axiom = factory.getOWLSubClassOfAxiom(
            protein1, factory.getOWLObjectSomeValuesFrom(
                interacts_rel, protein2))
        manager.addAxiom(ont, axiom)
        axiom = factory.getOWLSubClassOfAxiom(
            protein2, factory.getOWLObjectSomeValuesFrom(
                interacts_rel, protein1))
        manager.addAxiom(ont, axiom)

    # Save the files
    new_ont_file = os.path.join(out_dir, 'ontology.owl')
    manager.saveOntology(ont, IRI.create('file:' + os.path.abspath(new_ont_file)))

    valid_ont = manager.createOntology()
    for inters in valid:
        p1, p2 = inters[0], inters[1]
        protein1 = factory.getOWLClass(IRI.create(f'http://{p1}'))
        protein2 = factory.getOWLClass(IRI.create(f'http://{p2}'))
        axiom = factory.getOWLSubClassOfAxiom(
            protein1, factory.getOWLObjectSomeValuesFrom(
                interacts_rel, protein2))
        manager.addAxiom(valid_ont, axiom)
        axiom = factory.getOWLSubClassOfAxiom(
            protein2, factory.getOWLObjectSomeValuesFrom(
                interacts_rel, protein1))
        manager.addAxiom(valid_ont, axiom)

    valid_ont_file = os.path.join(out_dir, 'valid.owl')
    manager.saveOntology(valid_ont, IRI.create('file:' + os.path.abspath(valid_ont_file)))

    test_ont = manager.createOntology()
    for inters in test:
        p1, p2 = inters[0], inters[1]
        protein1 = factory.getOWLClass(IRI.create(f'http://{p1}'))
        protein2 = factory.getOWLClass(IRI.create(f'http://{p2}'))
        axiom = factory.getOWLSubClassOfAxiom(
            protein1, factory.getOWLObjectSomeValuesFrom(
                interacts_rel, protein2))
        manager.addAxiom(test_ont, axiom)
        axiom = factory.getOWLSubClassOfAxiom(
            protein2, factory.getOWLObjectSomeValuesFrom(
                interacts_rel, protein1))
        manager.addAxiom(test_ont, axiom)

    test_ont_file = os.path.join(out_dir, 'test.owl')
    manager.saveOntology(test_ont, IRI.create('file:' + os.path.abspath(test_ont_file)))


def load_and_split_interactions(data_file, ratio=(0.9, 0.05, 0.05)):
    inter_set = set()
    with gzip.open(data_file, 'rt') as f:
        next(f)
        for line in f:
            it = line.strip().split(' ')
            p1 = it[0]
            p2 = it[1]
            score = float(it[2])
            if score < 700:
                continue
            if (p2, p1) not in inter_set and (p1, p2) not in inter_set:
                inter_set.add((p1, p2))
    inters = np.array(list(inter_set))
    n = inters.shape[0]
    index = np.arange(n)
    np.random.seed(seed=0)
    np.random.shuffle(index)

    train_n = int(n * ratio[0])
    valid_n = int(n * ratio[1])
    train = inters[index[:train_n]]
    valid = inters[index[train_n: train_n + valid_n]]
    test = inters[index[train_n + valid_n:]]

    train_entities = set(train.flatten())
    valid_entities = set(valid.flatten())
    test_entities = set(test.flatten())

    only_valid = valid_entities - train_entities
    only_test = test_entities - train_entities

    if len(only_valid) > 0:
        initial_valid_n = len(valid)
        logger.warning(f"Valid entities not in train: {len(only_valid)}. Removing them")
        aux_valid = copy.deepcopy(valid)

        for (p1, p2) in aux_valid:
                if p1 in only_valid or p2 in only_valid:
                    valid = valid[~((valid[:, 0] == p1) & (valid[:, 1] == p2))]

        final_valid_n = len(valid)
        logger.info(f"Valid triples removed: {initial_valid_n - final_valid_n}. Remaining: {final_valid_n}")
        

    if len(only_test) > 0:
        initial_test_n = len(test)
        logger.warning(f"Test entities not in train: {len(only_test)}. Removing them")
        aux_test = copy.deepcopy(test)

        for (p1, p2) in aux_test:
            if p1 in only_test or p2 in only_test:
                test = test[~((test[:, 0] == p1) & (test[:, 1] == p2))]

        final_test_n = len(test)
        logger.info(f"Test triples removed: {initial_test_n - final_test_n}. Remaining: {final_test_n}")

    valid_entities = set(valid.flatten())
    test_entities = set(test.flatten())
    only_valid = valid_entities - train_entities
    only_test = test_entities - train_entities
        
    assert len(valid_entities - train_entities) == 0, f"Valid entities not in train: {len(valid_entities - train_entities)} out of {len(valid_entities)}"
    assert len(test_entities - train_entities) == 0, f"Test entities not in train: {len(test_entities - train_entities)} out of {len(test_entities)}"
    
    
    return train, valid, test

if __name__ == '__main__':
    main()
    shutdownJVM()
