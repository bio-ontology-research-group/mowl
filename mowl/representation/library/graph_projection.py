"""The projection as a part: how an ontology becomes a graph.

Every method that is trained on a graph of the ontology (knowledge graph embeddings, random
walks) starts with a projection (:mod:`mowl.projection`): the ontology goes in, edges
``(source, label, target)`` come out. The projection is a part of the method like the class
model or the objective: it decides which axioms are read (``C subclassof D``, ``C r D`` for
``C ⊑ ∃r.D``, ``a type C``, ``r(a, b)``, nested existentials for the GDA rule) and nothing
else about the method depends on it. This module names the projections of the library and
turns a name, a class or an instance into a projector, so that a model built from parts takes
``projector="gda"`` as well as ``projector=GDAProjector()``.

A model built from parts reads the graph through :class:`GraphView`: the nodes and the
labels of the edges with their ids, and the edges of the ontology as an integer array of
triples. The ids are those of :class:`~mowl.base_models.graph_model.GraphModel` (sorted, so
that two views of the same ontology agree).

Part (vocabulary value): the *view* ``graph`` (:class:`mowl.representation.View`).
"""

import numpy as np

from mowl.projection import (DL2VecProjector, GDAProjector, OWL2VecStarProjector,
                             TaxonomyProjector, TaxonomyWithRelationsProjector)
from mowl.projection.base import ProjectionModel

__all__ = ["PROJECTIONS", "register_projection", "projection", "GraphView"]

#: Name -> function without arguments that makes the projector.
PROJECTIONS = {
    "owl2vec_star": OWL2VecStarProjector,
    "dl2vec": DL2VecProjector,
    "taxonomy": TaxonomyProjector,
    "taxonomy with relations": TaxonomyWithRelationsProjector,
    "gda": GDAProjector,
}


def register_projection(name):
    """Decorator: make a projector class (or a function without arguments returning one)
    available under ``name``."""
    def decorate(factory):
        if name in PROJECTIONS and PROJECTIONS[name] is not factory:
            raise ValueError(f"{name!r} is already a registered projection.")
        PROJECTIONS[name] = factory
        return factory
    return decorate


def projection(spec=None, default="owl2vec_star"):
    """The projector of ``spec``: ``None`` (the ``default`` projection), a name of
    :data:`PROJECTIONS`, a projector class, or a projector."""
    if spec is None:
        spec = default
    if isinstance(spec, str):
        if spec not in PROJECTIONS:
            raise ValueError(f"Unknown projection {spec!r}; available: {sorted(PROJECTIONS)}.")
        return PROJECTIONS[spec]()
    if isinstance(spec, type) and issubclass(spec, ProjectionModel):
        return spec()
    if isinstance(spec, ProjectionModel):
        return spec
    raise TypeError("A projection is a name, a projector class or a projector "
                    "(mowl.projection).")


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
