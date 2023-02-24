from tests.datasetFactory import FamilyDataset
from mowl.projection import DL2VecProjector
from mowl.owlapi.defaults import TOP
from mowl.owlapi import OWLAPIAdapter

from org.semanticweb.owlapi.model import IRI
from unittest import TestCase
from gensim.models import KeyedVectors

# Add entity: one or more (adding to the ontology: axioms/graph) 
# Train again 
# Infer: ensure the index after the training (the inference without training and the results should be the same)
# Test if the new entity/entities are there 
# worst case?

class InductiveMethodTest:

    def __init__(self, onto_path, model_path):
        self.onto_path = onto_path
        self.model_path = model_path
    
    def test(self):
        # Load the initial ontology
        onto = onto_path.ontology

        # Load the DL2Vec model
        model = KeyedVectors.load(self.model_path)

        # Add a new entity to the ontology
        """This should check if the projection result is correct"""
        projector = DL2VecProjector()
        edges = projector.project(self.ontology)
        edges = set([e.astuple() for e in edges])
                
        new_class = set()
        new_class.add(("http://Male", "http://subclassof", "http://Parent"))
        self.assertEqual(set(edges), ground_truth_edges)

        # Train the ontology again using DL2Vec
        model.wv.min_count = 0
        model.workers = 10
        model.hashfix = hash
        sentences = gensim.models.word2vec.LineSentence("_walks.txt")
        model.build_vocab(sentences, update=True, keep_raw_vocab=True)
        model.train(sentences, total_examples=len(model.wv), epochs=10)          
        for entity in onto.individuals():
            entity_embedding = model.wv[entity.name]
            entity.embedding = entity_embedding

        # Infer using the updated ontology
        with onto:
            inferred_classes = list(onto.search(subclass_of=onto.Thing))

        # Check if the new entity is present in the inferred classes
        assert new_class in inferred_classes, "New entity not present in inferred classes"


