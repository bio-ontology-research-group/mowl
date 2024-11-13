import mowl
mowl.init_jvm("10g")
import sys
import pandas as pd

from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter
from mowl.owlapi.defaults import BOT, TOP
import os

from java.util import HashSet
from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.formats import OWLXMLDocumentFormat


import tqdm

organism = sys.argv[1]

root_dir = "../use_cases/"

if organism == "yeast":
    org_id  = "4932."
    org_dir = "ppi_yeast/data"
elif organism == "human":
    org_id = "9606."
    org_dir = "ppi_human/data"


train_path = os.path.join(root_dir, org_dir, "ontology.owl")
valid_path = os.path.join(root_dir, org_dir, "valid.owl")
test_path = os.path.join(root_dir, org_dir, "test.owl")
out_path = os.path.join(root_dir, org_dir, "ontology_extended.owl")

adapter = OWLAPIAdapter()
owl_manager = adapter.owl_manager
data_factory = adapter.data_factory


# Load the ontology
dataset = PathDataset(train_path, valid_path, test_path)
ontology = dataset.ontology

classes = dataset.classes.as_str
proteins = [c for c in classes if org_id in c]
nodes = proteins
print("Found {} proteins".format(len(proteins)))


class_nodes = [adapter.create_class(node) for node in nodes]
relation = adapter.create_object_property("http://interacts_with")
existential_nodes = [data_factory.getOWLObjectSomeValuesFrom(relation, class_node) for class_node in class_nodes]

bot_class = adapter.create_class(BOT)
top_class = adapter.create_class(TOP)


bot_class_axioms = [data_factory.getOWLSubClassOfAxiom(bot_class, class_node) for class_node in class_nodes]
bot_ex_axioms = [data_factory.getOWLSubClassOfAxiom(bot_class, existential_node) for existential_node in existential_nodes]

top_class_axioms = [data_factory.getOWLSubClassOfAxiom(class_node, top_class) for class_node in class_nodes]
top_ex_axioms = [data_factory.getOWLSubClassOfAxiom(existential_node, top_class) for existential_node in existential_nodes]

axioms = bot_class_axioms + bot_ex_axioms + top_class_axioms + top_ex_axioms

java_set = HashSet()
java_set.addAll(axioms)

owl_manager.addAxioms(ontology, java_set)

owl_manager.saveOntology(ontology, OWLXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(out_path)))
