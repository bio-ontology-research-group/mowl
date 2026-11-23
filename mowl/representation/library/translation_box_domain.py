"""The entities of a geometry of points and boxes of translations, as a finite domain.

The query evaluators of :mod:`mowl.evaluation.query_search` (compiling a query to a union of
boxes, the max-min beam over witnesses) read a finite domain of points whose relations are
unions of boxes of translations, through
:class:`~mowl.representation.library.element_box_family.ElementBoxFamily`. GroundedEL has such
a domain (its elements). So has every query embedding whose entities are points and whose
relations are boxes of translations, as in Query2Box: :class:`TranslationBoxDomain` reads one
(a :class:`~mowl.representation.library.query_methods.ComposedQueryGeometry` with point
entities, :class:`~mowl.representation.library.query_parts.PointBoxes`, and
:class:`~mowl.representation.library.boxrel_relations.TranslationBoxes` relations) as that
domain: the elements are the entities, the image of an entity under a relation is the box
:math:`[x + c_r - o_r, x + c_r + o_r]`; it has no classes (a knowledge graph's queries
follow relations).

:func:`search_scores` scores the answers of a batch of queries with one of the evaluators;
:meth:`mowl.models.QueryEmbeddingModel.score_query` takes it as ``search=``.
"""
from __future__ import annotations

import torch as th

from mowl.representation.library.box_union_algebra import Boxes
from mowl.representation.regions import Box, ConstructorUnavailable

__all__ = ["TranslationBoxDomain", "search_scores"]


class TranslationBoxDomain:
    """The domain of the module docstring (the interface
    :class:`~mowl.representation.library.element_box_family.ElementBoxFamily` reads:
    ``positions``, ``class_regions``, ``translation_regions``, ``nb_classes``,
    ``nb_elements``, ``embed_dim``).

    :param geometry: A geometry with ``entities`` (point boxes) and ``relations`` (boxes of
        translations: ``box(ids)`` and ``sign``).
    :raises ConstructorUnavailable: If the entities are not points or the relations not boxes
        of translations.
    """

    def __init__(self, geometry):
        entities, relations = geometry.entities, geometry.relations
        if getattr(entities, "region", None) is not Box:
            raise ConstructorUnavailable(
                f"The query search reads entities as points; {type(entities).__name__} gives "
                f"{getattr(entities, 'region', None)} regions.")
        if not callable(getattr(relations, "box", None)) or not hasattr(relations, "sign"):
            raise ConstructorUnavailable(
                f"The query search reads a relation as a box of translations; "
                f"{type(relations).__name__} is not one.")
        self.geometry, self.entities, self.relations = geometry, entities, relations
        self.nb_elements = len(entities)
        self.nb_classes = 0
        probe = self.positions(th.zeros(1, dtype=th.long))
        self.embed_dim = int(probe.shape[-1])
        self._like = probe

    def positions(self, ids):
        """The points of the entities ``ids`` (any shape, ``(..., d)``)."""
        flat = ids.reshape(-1)
        return self.entities(flat).centre.reshape(*ids.shape, -1)

    def translation_regions(self, ids):
        """The box of translations of every relation of ``ids`` as a union of one box:
        centre :math:`c_r` (with the sign of the convention) and half-width :math:`o_r`."""
        box = self.relations.box(ids)
        width = box.offset.clamp_min(0.0) if getattr(self.relations, "clamp_extent", True) \
            else box.offset
        active = th.ones((len(ids), 1), dtype=th.bool, device=box.centre.device)
        return Boxes(cen=(self.relations.sign * box.centre).unsqueeze(1),
                     w=width.unsqueeze(1), active=active)

    def class_regions(self, ids):
        """No classes: unions without boxes."""
        d, like = self.embed_dim, self._like
        empty = th.zeros((len(ids), 0, d), dtype=like.dtype, device=like.device)
        return Boxes(cen=empty, w=empty.clone(),
                     active=th.zeros((len(ids), 0), dtype=th.bool, device=like.device))


def search_scores(geometry, structure, queries, search="beam", k=64, max_boxes=64, tau=0.1):
    """Scores ``(len(queries), entities)`` of every entity as an answer of the queries (higher
    is better, positive inside the answer region), computed over the domain of ``geometry``
    (:class:`TranslationBoxDomain`) by an evaluator of :mod:`mowl.evaluation.query_search`.

    :param structure, queries: In the nested-tuple format of
        :mod:`mowl.evaluation.query_answering`.
    :param search: ``"beam"`` (:class:`~mowl.evaluation.query_search.MaxMinBeam` with ``k``
        witnesses per hop), ``"compile"`` (the query compiled to a union of at most
        ``max_boxes`` boxes, :class:`~mowl.evaluation.query_search.RegionCompiler`) or an
        evaluator object with ``scores(family, node)`` or ``compile(family, node)``.
    """
    from mowl.evaluation.query_search import MaxMinBeam, RegionCompiler, read_batch
    from mowl.representation.library.element_box_family import ElementBoxFamily
    domain = geometry if hasattr(geometry, "translation_regions") \
        else TranslationBoxDomain(geometry)
    family = ElementBoxFamily(domain, tau=tau)
    node = read_batch(structure, list(queries))
    if search == "beam":
        search = MaxMinBeam(k=k)
    elif search == "compile":
        search = RegionCompiler(max_boxes=max_boxes)
    elif isinstance(search, str):
        raise ValueError("search must be 'beam', 'compile' or an evaluator.")
    with th.no_grad():
        if hasattr(search, "scores"):
            return search.scores(family, node)
        return family.entity_scores(search.compile(family, node))
