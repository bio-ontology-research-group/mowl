import mowl
mowl.init_jvm("20g")
import click as ck
from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter
import os
from tqdm import tqdm
import random
from org.semanticweb.owlapi.model import ClassExpressionType as CT
from org.semanticweb.owlapi.model import AxiomType, IRI
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.formats import RDFXMLDocumentFormat
from java.util import HashSet

import sys

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
    
@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
@ck.option('--min_number', "-min", default=3, help='Minimum number of axioms per class')
@ck.option("--percentage", "-p", type=float, default=0.3)
@ck.option("--random_seed", "-seed", type=int, default=0)
def main(input_ontology, min_number, percentage, random_seed):
    """Remove axioms from an ontology. It will remove subclass axioms of
        the form C subclassof D.  C and D are concept names and
        R is a role. The percentage value indicates the amount of
        axioms to be used for validation and testing set. By default, we set 
        percentage = 0.3 to get a split 70/10/20 for training/validation/testing.
    """

    if not input_ontology.endswith(".owl"):
        raise ValueError("The input ontology must be in OWL format")
    
    random.seed(random_seed)
    manager = OWLAPIAdapter().owl_manager

    parent_directory = os.path.dirname(input_ontology)
    
    train_file_name = os.path.join(parent_directory, "train.owl")
    valid_file_name = os.path.join(parent_directory, "valid.owl")
    test_file_name = os.path.join(parent_directory, "test.owl")
    
    ds = PathDataset(input_ontology)
    ontology = ds.ontology
    tbox_axioms = ontology.getTBoxAxioms(Imports.fromBoolean(True))
    logger.info("Number of initial axioms: {}".format(len(tbox_axioms)))
        
    selected_axioms = []
    other_axioms = []

    all_classes_to_axioms = dict()
    
    for axiom in tqdm(tbox_axioms, desc="Getting C subclassof D axioms"):
        classes_in_axiom = axiom.getClassesInSignature()
        for c in classes_in_axiom:
            if c not in all_classes_to_axioms:
                all_classes_to_axioms[c] = []
            all_classes_to_axioms[c].append(axiom)

        
        if axiom.getAxiomType() == AxiomType.SUBCLASS_OF:
            sub = axiom.getSubClass()
            sup = axiom.getSuperClass()

            if sub.getClassExpressionType() == CT.OWL_CLASS and sup.getClassExpressionType() == CT.OWL_CLASS:
                selected_axioms.append(axiom)

            else:
                other_axioms.append(axiom)
        else:
            other_axioms.append(axiom)

    logger.info("Number of selected axioms: {}".format(len(selected_axioms)))
    logger.info("Number of other axioms: {}".format(len(other_axioms)))


    random.shuffle(selected_axioms)
    split_index = 1 - int(len(selected_axioms)*percentage)
    num_testing_axioms = len(selected_axioms[split_index:])
    
    testing_axioms = []
    selected_axioms = iter(selected_axioms)
    while len(testing_axioms) < num_testing_axioms:
        axiom = next(selected_axioms)
        classes_in_axiom = axiom.getClassesInSignature()

        elegible = True
        for c in classes_in_axiom:
            if len(all_classes_to_axioms[c]) < min_number:
                elegible = False
                break

        if elegible:
            testing_axioms.append(axiom)
            for c in classes_in_axiom:
                all_classes_to_axioms[c].remove(axiom)
                                

    testing_classes_to_axioms = dict()
    for axiom in testing_axioms:
        classes_in_axiom = axiom.getClassesInSignature()
        for c in classes_in_axiom:
            if c not in testing_classes_to_axioms:
                testing_classes_to_axioms[c] = []
            testing_classes_to_axioms[c].append(axiom)
                
    diff_classes = set(testing_classes_to_axioms.keys()) - set(all_classes_to_axioms.keys())
    if len(diff_classes) > 0:
        logger.warning(f"{len(diff_classes)} classes in the test set that are not in the training set")
    
    axioms_to_remove = testing_axioms
    axioms_to_remove_j = HashSet()
    axioms_to_remove_j.addAll(axioms_to_remove)
    print(f"Removing {len(axioms_to_remove)} axioms from a total of {len(tbox_axioms)} axioms")
    manager.removeAxioms(ontology, axioms_to_remove_j)

    valid_axioms = HashSet()
    test_axioms = HashSet()
    num_axioms = len(axioms_to_remove)//3
    valid_axioms.addAll(axioms_to_remove[:num_axioms])
    test_axioms.addAll(axioms_to_remove[num_axioms:])
    
    num_axioms = len(ontology.getTBoxAxioms(Imports.fromBoolean(True)))
    print(f"Number of training axioms: {num_axioms}")
                     
    valid_ontology = manager.createOntology(valid_axioms)
    num_axioms = len(valid_ontology.getTBoxAxioms(Imports.fromBoolean(True)))
    print(f"Number of validation axioms: {num_axioms}")
    test_ontology = manager.createOntology(test_axioms)
    num_axioms = len(test_ontology.getTBoxAxioms(Imports.fromBoolean(True)))
    print(f"Number of testing axioms: {num_axioms}")
        
    print("Saving ontologies")
    manager.saveOntology(ontology, RDFXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(train_file_name)))
    manager.saveOntology(valid_ontology, RDFXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(valid_file_name)))
    manager.saveOntology(test_ontology, RDFXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(test_file_name)))
    
    print(f"Done.")

if __name__ == "__main__":
    main()
