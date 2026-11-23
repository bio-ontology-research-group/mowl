"""The projection as a part: how an ontology becomes a graph.

Every method that is trained on a graph of the ontology (knowledge graph embeddings, random
walks) starts with a projection (:mod:`mowl.projection`): the ontology goes in, edges
``(source, label, target)`` come out. The projection is a part of the method like the class
model or the objective: it decides which axioms are read (``C subclassof D``, ``C r D`` for
``C ⊑ ∃r.D``, ``a type C``, ``r(a, b)``, nested existentials for the GDA rule) and nothing
else about the method depends on it. This module names the projections of the library and
turns a name, a class or an instance into a projector, so that a model built from parts takes
``projector="gda"`` as well as ``projector=GDAProjector()``.

The part is one class for every family of methods that starts from a graph: the walks of
``dl2vec``, ``owl2vec_star`` and ``taxonomy_walks`` read the edges of a :class:`ProjectionView`
and the knowledge graph embeddings (``graph_kge``, ``boxe``, ``expressive``, ``reshuffle``,
``gda_projection``) take their projector from one, whatever it is called
(:func:`projection_view`, :func:`projection`).

A model built from parts reads the graph through :class:`GraphView`: the nodes and the
labels of the edges with their ids, and the edges of the ontology as an integer array of
triples. The ids are those of :class:`~mowl.base_models.graph_model.GraphModel` (sorted, so
that two views of the same ontology agree).

Part (vocabulary value): the *view* ``graph`` (:class:`mowl.representation.View`).
"""

import numpy as np

from mowl.projection.base import ProjectionModel
from mowl.representation.sequence import OntologyView
from mowl.representation.vocabulary import Projection

__all__ = ["PROJECTIONS", "register_projection", "projection", "ProjectionView",
           "projection_view", "GraphView"]


def _taxonomy(bidirectional_taxonomy=False, **_):
    from mowl.projection import TaxonomyProjector
    return TaxonomyProjector(bidirectional_taxonomy)


def _taxonomy_relations(bidirectional_taxonomy=False, relations=None, **_):
    from mowl.projection import TaxonomyWithRelationsProjector
    return TaxonomyWithRelationsProjector(taxonomy=True,
                                          bidirectional_taxonomy=bidirectional_taxonomy,
                                          relations=relations)


def _dl2vec(bidirectional_taxonomy=False, **_):
    from mowl.projection import DL2VecProjector
    return DL2VecProjector(bidirectional_taxonomy)


def _owl2vec_star(bidirectional_taxonomy=False, include_literals=False, only_taxonomy=False,
                  **_):
    from mowl.projection import OWL2VecStarProjector
    return OWL2VecStarProjector(bidirectional_taxonomy=bidirectional_taxonomy,
                                include_literals=include_literals,
                                only_taxonomy=only_taxonomy)


def _gda(bidirectional_taxonomy=False, include_literals=False, only_taxonomy=False, **_):
    from mowl.projection import GDAProjector
    return GDAProjector(bidirectional_taxonomy=bidirectional_taxonomy,
                        include_literals=include_literals, only_taxonomy=only_taxonomy)


#: Name -> function ``(**options) -> projector``. Every projection has its short name
#: (``"owl2vec_star"``, ``"dl2vec"``, ``"taxonomy"``, ``"taxonomy with relations"``,
#: ``"gda"``) and the value of the vocabulary
#: (:class:`~mowl.representation.vocabulary.Projection`) as names.
PROJECTIONS = {
    "owl2vec_star": _owl2vec_star,
    "dl2vec": _dl2vec,
    "taxonomy": _taxonomy,
    "taxonomy with relations": _taxonomy_relations,
    "gda": _gda,
    Projection.TAXONOMY.value: _taxonomy,
    Projection.TAXONOMY_WITH_RELATIONS.value: _taxonomy_relations,
    Projection.DL2VEC.value: _dl2vec,
    Projection.OWL2VEC_STAR.value: _owl2vec_star,
    Projection.GDA.value: _gda,
}
#: Other names that are accepted.
_ALIASES = {"taxonomy_rels": Projection.TAXONOMY_WITH_RELATIONS.value,
            "owl2vecstar": Projection.OWL2VEC_STAR.value}
def _projector_parts():
    """Projector class -> vocabulary value (the classes of :mod:`mowl.projection`)."""
    from mowl.projection import (CategoricalProjector, DL2VecProjector, GDAProjector,
                                 OWL2VecStarProjector, TaxonomyProjector,
                                 TaxonomyWithRelationsProjector)
    return ((TaxonomyWithRelationsProjector, Projection.TAXONOMY_WITH_RELATIONS),
            (TaxonomyProjector, Projection.TAXONOMY), (DL2VecProjector, Projection.DL2VEC),
            (OWL2VecStarProjector, Projection.OWL2VEC_STAR),
            (CategoricalProjector, Projection.CATEGORICAL), (GDAProjector, Projection.GDA))


def register_projection(name):
    """Decorator: make a projector class (or a function without arguments returning one)
    available under ``name``."""
    def decorate(factory):
        if name in PROJECTIONS and PROJECTIONS[name] is not factory:
            raise ValueError(f"{name!r} is already a registered projection.")
        PROJECTIONS[name] = factory
        return factory
    return decorate


def projection(spec=None, default="owl2vec_star", **options):
    """The projector of ``spec``: ``None`` (the ``default`` projection), a name of
    :data:`PROJECTIONS` (with the ``options`` of the projector: ``bidirectional_taxonomy``,
    ...), a projector class, or a projector."""
    if spec is None:
        spec = default
    if isinstance(spec, str):
        spec = _ALIASES.get(spec, spec)
        if spec not in PROJECTIONS:
            raise ValueError(f"Unknown projection {spec!r}; available: {sorted(PROJECTIONS)}.")
        return PROJECTIONS[spec](**options)
    if isinstance(spec, type) and issubclass(spec, ProjectionModel):
        return spec()
    if isinstance(spec, ProjectionModel) or callable(getattr(spec, "project", None)):
        return spec
    raise TypeError("A projection is a name, a projector class or a projector "
                    "(mowl.projection).")


class ProjectionView(OntologyView):
    """A graph of the ontology, from a projector of :mod:`mowl.projection`.

    :param projector: A :class:`mowl.projection.base.ProjectionModel`, its class, or the name
        of a projection (:data:`PROJECTIONS`; see :func:`projection`).
    :param entailment: See :class:`~mowl.representation.sequence.OntologyView`.
    :param default: The projection when ``projector`` is ``None``.

    Used by ``dl2vec``, ``owl2vec_star`` and ``taxonomy_walks`` (the graph the walks run on)
    and by ``graph_kge``, ``boxe``, ``expressive``, ``reshuffle`` and ``gda_projection`` (the
    graph whose triples are embedded).
    """

    produces = "graph"

    def __init__(self, projector=None, entailment=None, default="owl2vec_star"):
        super().__init__(entailment)
        self.projector = projection(projector, default)

    @property
    def part(self):
        for cls, part in _projector_parts():
            if isinstance(self.projector, cls):
                return part.value
        return type(self.projector).__name__

    def read(self, dataset):
        """The edges (:class:`mowl.projection.Edge`) of the projected ontology."""
        return self.projector.project(self.ontology(dataset))


def projection_view(name, entailment=None, **options):
    """A :class:`ProjectionView` by name (a key of :data:`PROJECTIONS` or a short name:
    ``"taxonomy"``, ``"taxonomy_rels"``, ``"dl2vec"``, ``"owl2vec_star"``, ``"gda"``) with
    the options of the projector (``bidirectional_taxonomy``, ``include_literals``, ...)."""
    return ProjectionView(projection(name, **options), entailment=entailment)


class GraphView:
    """The graph of an ontology under a projection, as ids.

    :param edges: The edges of the projection.
    :param nodes: Node IRI -> id.
    :param relations: Edge label -> id.

    :ivar triples: ``(E, 3)`` integer array ``[source, label, target]`` of node and label ids.
    """

    def __init__(self, edges, nodes, relations):
        self.edges = edges
        self.nodes = dict(nodes)
        self.relations = dict(relations)
        self.triples = np.array([[nodes[e.src], relations[e.rel], nodes[e.dst]]
                                 for e in edges], dtype=np.int64).reshape(-1, 3)

    @classmethod
    def of(cls, model):
        """The view of a graph model (:class:`~mowl.base_models.graph_model.GraphModel`)."""
        return cls(model.edges, model.graph_node_to_id, model.graph_relation_to_id)

    def __len__(self):
        return len(self.triples)
