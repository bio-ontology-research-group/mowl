"""The objective and the geometries of the methods that embed a hierarchy (group
``hierarchy``).

These methods score the subclass order between *named* classes (``gci0``); what the published
modules differ in is the cost of a pair that is *not* ordered. The objective
(:class:`HierarchyObjective`) therefore reads ``gci0`` through the two methods of the measure,
``inside`` for a stated pair and ``negative`` for a corrupted one, and leaves every other
form to the default reading of :class:`~mowl.representation.Objective` (which works if the
geometry has the parts it needs: a relation model for :math:`\\exists r.C`, regions closed
under intersection for :math:`C \\sqcap D`).

:class:`HierarchyGeometry` is a :class:`~mowl.representation.ComposedGeometry` that tolerates
class models without a table (text encoders, parameters that are not embeddings), keeps the
unit-cube projection available as :meth:`~HierarchyGeometry.project`, and can offer
:math:`\\exists r.C` and :math:`C \\sqcap D` by lookup for methods in which compound classes
have an embedding of their own (:class:`LatticeGeometry`).
"""

import torch as th

from mowl.nn.el.elmodule import ELModule
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.objectives import Objective
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["HierarchyObjective", "HierarchyGeometry", "LatticeGeometry", "TextGeometry"]


class HierarchyObjective(Objective):
    """The reading of ``gci0`` of the hierarchy methods.

    :param margin: The margin of the negative cost (:math:`\\alpha` of order embeddings,
        :math:`\\gamma` of the cones, :math:`\\mu` of disks); the measures that have none
        (log-probabilities, distance and depth) ignore it.
    :param inside_margin: The strict margin of the cost of a stated pair (0: none, as in all
        published losses of these methods).
    :param freeze_size_negatives: Compute the cost of a negative pair with the sizes of the
        regions detached, so that negative pairs do not update them (the box lattice model:
        :math:`\\Delta` is not updated through negative pairs). The class model must
        support it (:class:`~.hierarchy_classes.UnitCubeBoxes`).
    """

    def __init__(self, margin=0.0, inside_margin=0.0, freeze_size_negatives=False, **kwargs):
        super().__init__(margin=margin, **kwargs)
        self.inside_margin = float(inside_margin)
        self.freeze_size_negatives = bool(freeze_size_negatives)

    @property
    def m(self):
        return self.inside_margin

    def gci0(self, g, rows, neg=False):
        frozen = neg and self.freeze_size_negatives
        c, d = g.classes(rows[:, 0], frozen), g.classes(rows[:, 1], frozen)
        if neg:
            negative = getattr(g.measure, "negative", None)
            if negative is None:  # a measure of another family: the semantic reading
                return g.measure.apart(c, d, self.margin)
            return negative(c, d, self.margin)
        return g.measure.inside(c, d, self.m)


class HierarchyGeometry(ComposedGeometry):
    """A geometry of the hierarchy methods: a class model, a measure, a
    :class:`HierarchyObjective` and, where the method has one, a relation model.

    It is trained like any :class:`~mowl.nn.ELModule` (``geometry(rows, "gci0", neg)``). Only
    ``gci0`` has a negative loss. ``project()`` projects the class model's parameters back
    onto their constraint set where it has one (the unit cube); a training loop has to call
    it after each step, as the models of mOWL do for ``module.project``.

    :param exists: A function ``(relation_ids, region) -> region`` for :math:`\\exists r.C`
        when there is no relation model (lookup of a compound class that has an embedding).
    :param conjunction: A function ``(a, b) -> (region, gap)`` likewise for :math:`C \\sqcap D`.
    """

    def __init__(self, classes, relations, measure, objective=None, individuals=None,
                 regularizers=(), exists=None, conjunction=None):
        ELModule.__init__(self)
        objective = objective if objective is not None else HierarchyObjective()
        if not issubclass(classes.region, measure.accepts):
            raise TypeError(
                f"{type(measure).__name__} compares "
                f"{[c.__name__ for c in measure.accepts]} regions; "
                f"{type(classes).__name__} gives {classes.region.__name__} regions.")
        self.class_model = classes
        self.relation_model = relations
        self.individual_model = individuals
        self.measure = measure
        self.objective = objective
        self.regularizers = tuple(regularizers)
        self._exists, self._conjunction = exists, conjunction
        self.class_embed = getattr(classes, "centres", None)
        self.rel_embed = getattr(relations, "vectors", None)
        self.ind_embed = getattr(individuals, "centres", None)
        self.neg_capable_gcis = frozenset({"gci0"})
        self.margin_semantics = "strict"

    # -- Geometry -----------------------------------------------------------------------
    def classes(self, ids, frozen=False):
        """The regions of ``ids``; ``frozen`` detaches their sizes (see
        :class:`HierarchyObjective`)."""
        if frozen:
            return self.class_model(ids, detach_size=True)
        return self.class_model(ids)

    def exists(self, relation_ids, region):
        if self.relation_model is None and self._exists is not None:
            return self._exists(relation_ids, region)
        return super().exists(relation_ids, region)

    def conjunction(self, a, b):
        if self._conjunction is not None:
            return self._conjunction(a, b)
        return super().conjunction(a, b)

    @property
    def parts(self):
        parts = super().parts
        if self._conjunction is not None:
            parts["conjunction"] = "own embedding"
        elif not self.class_model.exact_conjunction:
            parts["conjunction"] = "none"
        return parts

    # -- training -----------------------------------------------------------------------
    def project(self):
        """Project the class model back onto its constraint set (a no-op without one)."""
        project = getattr(self.class_model, "project", None)
        if project is not None:
            project()

    def param_groups(self):
        """Parameter groups for :class:`~mowl.nn.order.manifolds.RiemannianSGD`: the class
        model's, with the manifold of every table that lives on one."""
        groups = list(self.class_model.param_groups())
        seen = {id(p) for g in groups for p in g["params"]}
        rest = [p for p in self.parameters() if id(p) not in seen]
        if rest:
            groups.append({"params": rest, "manifold": None})
        return groups

    def regularization_loss(self):
        total = th.zeros((), device=self.class_model.device)
        for regularizer in self.regularizers:
            term = regularizer.penalty(self)
            if term is not None:
                total = total + term
        return total


class LatticeGeometry(HierarchyGeometry):
    """A geometry whose compound classes are elements with embeddings of their own (CatE:
    the elements of the lattice that the categorical projection builds). :math:`\\exists r.C`
    and :math:`C \\sqcap D` are looked up, and a compound that is not an element raises an
    error that says so, instead of being approximated.

    The individuals are elements too (:math:`\\{a\\}`), with the same table as the classes.

    :param existentials: A function ``(role_ids, filler_ids) -> element ids`` (``-1`` where
        :math:`\\exists r.C` is not an element), e.g. ``CatEModule.existential_ids``.
    :param conjunctions: A dictionary ``{(i, j): k}`` from the (unordered) pair of element ids
        of :math:`C` and :math:`D` to the element of :math:`C \\sqcap D`.
    """

    def __init__(self, classes, measure, objective=None, existentials=None,
                 conjunctions=None, regularizers=()):
        self._existential_ids = existentials
        self._conjunction_ids = dict(conjunctions or {})
        super().__init__(classes, None, measure, objective, individuals=classes,
                         regularizers=regularizers, exists=self._lookup_exists,
                         conjunction=self._lookup_conjunction)

    @staticmethod
    def _error(message):
        try:
            from mowl.projection.categorical.lattice import UnseenConceptError
        except ImportError:  # the lattice needs the JVM; the error type is only a LookupError
            UnseenConceptError = LookupError
        return UnseenConceptError(message)

    @staticmethod
    def _element_ids(region):
        if region.ids is None:
            raise ConstructorUnavailable(
                "Compound classes are looked up by element id: use regions returned by "
                "this geometry's classes().")
        return region.ids

    def _lookup_exists(self, relation_ids, region):
        if self._existential_ids is None:
            raise ConstructorUnavailable("No existential restrictions were registered.")
        nodes = self._existential_ids(relation_ids, self._element_ids(region))
        if (nodes < 0).any():
            bad = th.nonzero(nodes < 0)[0, 0].item()
            raise self._error(
                f"The existential restriction of relation {int(relation_ids[bad])} and "
                f"filler element {int(region.ids[bad])} is not an element of the lattice; "
                f"it is scored only if it was added before training.")
        return self.class_model(nodes)

    def _lookup_conjunction(self, a, b):
        ia, ib = self._element_ids(a), self._element_ids(b)
        nodes = []
        for i, j in zip(ia.tolist(), ib.tolist()):
            key = (min(i, j), max(i, j))
            if key not in self._conjunction_ids:
                raise self._error(f"The conjunction of the elements {i} and {j} is not an "
                                  f"element of the lattice.")
            nodes.append(self._conjunction_ids[key])
        nodes = th.as_tensor(nodes, dtype=th.long, device=ia.device)
        return self.class_model(nodes), th.zeros(len(nodes), 1, device=ia.device)


class TextGeometry(HierarchyGeometry):
    """A geometry whose class model is a text encoder (HiT, OnT). Classes are given as
    lists of verbalisations (or as ids, if the class model has the list of texts), so classes
    that were never seen in training have regions too.

    Compound classes are embedded from their own verbalisation (:meth:`compound`), as OnT does.
    The training losses of the papers are triplet losses with sampled negatives;
    :meth:`hierarchy_loss` is that of HiT/OnT (Eq. 5) with the margins and weights given here.

    :param clustering_margin, centripetal_margin, clustering_weight, centripetal_weight:
        :math:`\\alpha`, :math:`\\beta` and the weights of :math:`L_\\prec`.
    """

    def __init__(self, classes, relations, measure, objective=None,
                 clustering_margin=5.0, centripetal_margin=0.1, clustering_weight=1.0,
                 centripetal_weight=1.0, **kwargs):
        super().__init__(classes, relations, measure, objective, **kwargs)
        self.clustering_margin = clustering_margin
        self.centripetal_margin = centripetal_margin
        self.clustering_weight = clustering_weight
        self.centripetal_weight = centripetal_weight

    def compound(self, texts):
        """The regions of compound classes, from their verbalisations."""
        return self.class_model(list(texts))

    def hierarchy_loss(self, child, parent, negative):
        """:math:`L_\\prec` of HiT (Eq. 5) for regions, ``(B, 1)``."""
        return self.measure.triplet(child, parent, negative, self.clustering_margin,
                                    self.centripetal_margin, self.clustering_weight,
                                    self.centripetal_weight)

    def existential_loss(self, exists, relation_ids, filler, negative_1, negative_2):
        """:math:`L_r` of OnT (Eq. 6) for the regions of :math:`\\exists r.D` (from its
        verbalisation), the relations and the regions of the fillers :math:`D`: both directions
        of the order between :math:`\\exists r.D` and :math:`f_r(D)`."""
        image = self.exists(relation_ids, filler)
        forward = self.hierarchy_loss(exists, image, negative_1)
        backward = self.hierarchy_loss(image, exists, negative_2)
        return 0.5 * (forward + backward)
