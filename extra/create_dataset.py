#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from jpype import *
import jpype.imports
import os

logging.basicConfig(level=logging.INFO)

jars_dir = "gateway/build/distributions/gateway/lib/"
jars = f'{str.join(":", [jars_dir+name for name in os.listdir(jars_dir)])}'
startJVM(getDefaultJVMPath(), "-ea",  "-Djava.class.path=" + jars,  convertStrings=False)


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
    '--data-file', '-df', default='data/4932.protein.links.v11.5.txt.gz',
    help='STRING PPI file')
@ck.option(
    '--annots-file', '-af', default='data/annotations.tsv',
    help='Annotations file extracted from Uniprot (using uni2pandas.py and annotations.py)')
@ck.option(
    '--go-annots-file', '-gf',
    help='as an alternative to --annots-file, specify the sdf.gaf from http://current.geneontology.org/products/pages/downloads.html, mapping uniprot proteins to GO annotations')
@ck.option(
    '--out-dir', '-od', default='datasets/ppi_yeast',
    help='Dataset directory')
def main(ont_file, data_file, annots_file, go_annots_file, out_dir):
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
                        p_id = '4932.' + items[10] # e.g. '4932.YKL020C'
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
    return train, valid, test

if __name__ == '__main__':
    main()
    shutdownJVM()
