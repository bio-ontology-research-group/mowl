from mowl.projection.base import ProjectionModel
from mowl.projection.edge import Edge
from org.mowl.Projectors import GDAProjector as Projector
from org.semanticweb.owlapi.model import OWLOntology

import jpype.imports
from java.util import ArrayList

OBO = "http://purl.obolibrary.org/obo/"

DEFAULT_SOURCE_PREFIXES = (OBO + "HP_", OBO + "MP_")
DEFAULT_TARGET_PREFIXES = (OBO + "GO_", OBO + "UBERON_")


class GDAProjector(ProjectionModel):
    '''
    Implementation of the projection rules defined in [zhapa2026]_.

    .. include:: extra/gda.rst

    This projector follows the OWL2Vec* rules and adds one rule: existential
    restrictions whose filler is not an atomic class are unfolded recursively,
    descending through intersections and nested existentials until a named class
    is reached. A single edge is generated whose relation is the composition of
    the roles traversed, minted under the ``http://mowl.borg/`` namespace.

    For example, given the axiom

    .. code-block:: text

        HP_0000123 SubClassOf ObjectSomeValuesFrom(has-part,
            ObjectIntersectionOf(PATO_0000001,
                ObjectSomeValuesFrom(inheres-in, UBERON_0000955)))

    the projector generates the edge

    .. code-block:: text

        HP_0000123  http://mowl.borg/has-part_inheres-in  UBERON_0000955

    which :class:`mowl.projection.owl2vec_star.model.OWL2VecStarProjector` drops
    entirely because the filler is not atomic.

    The unfolding is restricted by IRI prefix at both ends. The defaults target
    gene--disease association graphs: phenotype classes (HP, MP) as sources and
    function or anatomy classes (GO, UBERON) as targets. An empty tuple matches
    every class.

    :param bidirectional_taxonomy: If ``True`` then per each SubClass edge one SuperClass edge will be generated. Default is False.
    :type bidirectional_taxonomy: bool, optional
    :param only_taxonomy: If ``True``, the projection will only include subClass edges. Nested existentials are not unfolded. Default is False.
    :type only_taxonomy: bool, optional
    :param include_literals: If ``True`` the graph will also include triples involving data property assertions and annotations. Default is False.
    :type include_literals: bool, optional
    :param source_prefixes: IRI prefixes a class must match to be used as the source of an unfolded edge. Default is the HP and MP OBO prefixes.
    :type source_prefixes: tuple of str, optional
    :param target_prefixes: IRI prefixes a class must match to be used as the target of an unfolded edge. Default is the GO and UBERON OBO prefixes.
    :type target_prefixes: tuple of str, optional
    '''

    def __init__(self, bidirectional_taxonomy=False, only_taxonomy=False,
                 include_literals=False, source_prefixes=DEFAULT_SOURCE_PREFIXES,
                 target_prefixes=DEFAULT_TARGET_PREFIXES):
        super().__init__()

        if not isinstance(bidirectional_taxonomy, bool):
            raise TypeError("Optional parameter bidirectional_taxonomy must be of type boolean")
        if not isinstance(only_taxonomy, bool):
            raise TypeError("Optional parameter only_taxonomy must be of type boolean")
        if not isinstance(include_literals, bool):
            raise TypeError("Optional parameter include_literals must be of type boolean")
        source_prefixes = self._validate_prefixes(source_prefixes, "source_prefixes")
        target_prefixes = self._validate_prefixes(target_prefixes, "target_prefixes")

        self.bidirectional_taxonomy = bidirectional_taxonomy
        self.only_taxonomy = only_taxonomy
        self.include_literals = include_literals
        self.source_prefixes = source_prefixes
        self.target_prefixes = target_prefixes

        self.projector = Projector(self.bidirectional_taxonomy, self.only_taxonomy,
                                   self.include_literals,
                                   self._to_java_list(self.source_prefixes),
                                   self._to_java_list(self.target_prefixes))

    @staticmethod
    def _validate_prefixes(prefixes, name):
        if isinstance(prefixes, str) or not hasattr(prefixes, "__iter__"):
            raise TypeError(f"Optional parameter {name} must be an iterable of strings")
        prefixes = tuple(prefixes)
        if not all(isinstance(prefix, str) for prefix in prefixes):
            raise TypeError(f"Optional parameter {name} must be an iterable of strings")
        return prefixes

    @staticmethod
    def _to_java_list(prefixes):
        java_list = ArrayList()
        for prefix in prefixes:
            java_list.add(prefix)
        return java_list

    def project(self, ontology):
        r"""Generates the projection of the ontology.

        :param ontology: The ontology to be processed.
        :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
        """

        if not isinstance(ontology, OWLOntology):
            raise TypeError(
                "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology")
        edges = self.projector.project(ontology)

        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in
                 edges if str(e.dst()) != ""]

        return edges
