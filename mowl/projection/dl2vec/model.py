from mowl.projection.base import ProjectionModel
from org.mowl.Projectors import DL2VecProjector as Projector
from org.semanticweb.owlapi.model import OWLOntology
from mowl.projection.edge import Edge
import logging


class DL2VecProjector(ProjectionModel):
    '''
    Implementation of projection rules defined in [chen2020]_.

    .. include:: extra/dl2vec.rst
    
    :param bidirectional_taxonomy: If true then per each SubClass edge one SuperClass
        edge will be generated.
    :type bidirectional_taxonomy: bool, optional
    '''

    def __init__(self, bidirectional_taxonomy: bool = False):
        super().__init__()

        if not isinstance(bidirectional_taxonomy, bool):
            raise TypeError("Optional parameter bidirectional_taxonomy must be of type boolean")
        self.projector = Projector(bidirectional_taxonomy)

    def project(self, ontology, with_individuals=False, verbose=False):
        r"""Generates the projection of the ontology.

        :param ontology: The ontology to be processed.
        :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
        :param with_individuals: If true then assertion axioms with named individuals
            will be included in the projection. Default is False.
        :type with_individuals: bool, optional
        :param verbose: If true then the warnings will be printed to the standard output.
            Default is False.
        :type verbose: bool, optional

        :rtype: list(:class:`mowl.projection.edge.Edge`)
        """

        if not isinstance(ontology, OWLOntology):
            raise TypeError(
                "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology")

        if not isinstance(with_individuals, bool):
            raise TypeError("Optional parameter with_individuals must be of type boolean")

        if not isinstance(verbose, bool):
            raise TypeError("Optional parameter verbose must be of type boolean")

        edges = self.projector.project(ontology, with_individuals, verbose)
        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        return edges
