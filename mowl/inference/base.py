from org.semanticweb.owlapi.model import OWLAxiom, OWLClass, OWLObjectProperty, OWLIndividual
from itertools import chain


class Inference():
    def __init__(self, method):
        self.method = method


class Inferrer():
    def __init__(self, score_method, axiom_template, *lists_of_entities):

        # Check if score_method is callable
        if not callable(score_method):
            raise TypeError("Parameter score_method must be a callable object (function, method, \
etc.)")

        if not isinstance(axiom_template, OWLAxiom):
            raise TypeError("Parameter axiom_template must be of type \
org.semanticweb.owlapi.model.OWLAxiom")

        if not all(isinstance(entity_list, list) for entity_list in lists_of_entities):
            raise TypeError("All parameters after axiom_template must be lists of entities \
(OWLClass, OWLObjectProperty or OWLIndividual)")

        if not all(isinstance(elem, (OWLClass, OWLObjectProperty, OWLIndividual)) for elem in
                   chain.from_iterable(lists_of_entities)):
            raise TypeError("All parameters after axiom_template must be lists of entities \
(OWLClass, OWLObjectProperty or OWLIndividual)")

        self.score_method = score_method
        self.axiom_template = axiom_template
        self.lists_of_entities = lists_of_entities
