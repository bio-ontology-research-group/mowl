"""The geometries of the methods that embed a hierarchy (group ``hierarchy``).

These methods score the subclass order between *named* classes (``gci0``); what the published
modules differ in is the cost of a pair that is *not* ordered. The core objective reads it
through ``measure.negative`` (:meth:`Objective.hierarchy
<mowl.representation.Objective.hierarchy>`),
and every other form by the default reading (which works if the geometry has the parts it
needs: a relation model for :math:`\\exists r.C`, regions closed under intersection for
:math:`C \\sqcap D`). :func:`hierarchy_geometry` is the
:class:`~mowl.representation.ComposedGeometry` of such a method.

Two geometries have a reading of their own: :class:`LatticeGeometry` (compound classes are
elements with embeddings of their own, looked up) and :class:`TextGeometry` (class models that
are text encoders, with the triplet losses of HiT and OnT).
"""

import torch as th

from mowl.representation.geometry import ComposedGeometry
from mowl.representation.objectives import Objective
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["hierarchy_geometry", "LatticeGeometry", "TextGeometry", "ExistentialElements"]


def hierarchy_geometry(classes, relations, measure, objective=None, individuals=None,
                       regularizers=(), **options):
    """The :class:`~mowl.representation.ComposedGeometry` of a hierarchy method: a class
    model, a measure, an objective (default :meth:`Objective.hierarchy
    <mowl.representation.Objective.hierarchy>`) and, where the method has one, a relation
    model. It reads :math:`C \\sqsubseteq D` between named classes (``supported_forms``), the
    only form with a negative loss, and is trained like any :class:`~mowl.nn.ELModule`
    (``geometry(rows, "gci0", neg)``). ``project()`` projects the class model onto its
    constraint set (the unit cube) after a step, which a training loop has to call.
    """
    objective = objective if objective is not None else Objective.hierarchy()
    options.setdefault("supported_forms", ("gci0",))
    options.setdefault("neg_capable_gcis", {"gci0"})
    return ComposedGeometry(classes, relations, measure, objective, individuals=individuals,
                            regularizers=regularizers, **options)


class ExistentialElements:
    """Mixin (before the geometry class) for a lattice geometry whose elements include the
    compound classes :math:`\\exists r.D`: a sorted table of the keys ``role * n + filler``
    and the element each key stands for, kept as buffers (``existential_keys`` and
    ``existential_nodes``, saved with the module). Call :meth:`_init_existentials` after the
    module is initialised; :meth:`existential_ids` is what a
    :class:`LatticeGeometry` takes as ``existentials``."""

    def _init_existentials(self, num_nodes):
        self.num_nodes = num_nodes
        self.register_buffer("existential_keys", th.zeros(0, dtype=th.long))
        self.register_buffer("existential_nodes", th.zeros(0, dtype=th.long))
        self._existential_ids = self.existential_ids

    def set_existentials(self, role_ids, filler_ids, node_ids):
        """Register the elements :math:`\\exists r.D`: ``node_ids[i]`` is the element for
        role ``role_ids[i]`` and filler element ``filler_ids[i]``."""
        role_ids = th.as_tensor(role_ids, dtype=th.long).reshape(-1)
        filler_ids = th.as_tensor(filler_ids, dtype=th.long).reshape(-1)
        node_ids = th.as_tensor(node_ids, dtype=th.long).reshape(-1)
        keys = role_ids * self.num_nodes + filler_ids
        keys, order = th.sort(keys)
        device = self.existential_keys.device
        self.existential_keys = keys.to(device)
        self.existential_nodes = node_ids[order].to(device)

    def existential_ids(self, role_ids, filler_ids):
        """Element ids of :math:`\\exists r.D`; ``-1`` where it is not an element."""
        keys = role_ids.long() * self.num_nodes + filler_ids.long()
        if len(self.existential_keys) == 0:
            return th.full_like(keys, -1)
        last = len(self.existential_keys) - 1
        pos = th.searchsorted(self.existential_keys, keys).clamp(max=last)
        found = self.existential_keys[pos] == keys
        return th.where(found, self.existential_nodes[pos], th.full_like(keys, -1))


class LatticeGeometry(ComposedGeometry):
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
        objective = objective if objective is not None else Objective.hierarchy()
        super().__init__(classes, None, measure, objective, individuals=classes,
                         regularizers=regularizers, supported_forms=("gci0",),
                         neg_capable_gcis={"gci0"})

    @property
    def parts(self):
        parts = super().parts
        parts["conjunction"] = "own embedding"
        return parts

    def exists(self, relation_ids, region):
        return self._lookup_exists(relation_ids, region)

    def conjunction(self, a, b):
        return self._lookup_conjunction(a, b)

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


class TextGeometry(ComposedGeometry):
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
        kwargs.setdefault("supported_forms", ("gci0",))
        kwargs.setdefault("neg_capable_gcis", {"gci0"})
        objective = objective if objective is not None else Objective.hierarchy()
        super().__init__(classes, relations, measure, objective, **kwargs)
        self.clustering_margin = clustering_margin
        self.centripetal_margin = centripetal_margin
        self.clustering_weight = clustering_weight
        self.centripetal_weight = centripetal_weight

    @property
    def parts(self):
        parts = super().parts
        if self.relation_model is not None:
            # OnT trains the embeddings of compound classes (Eq. 6 and 7); HiT has none
            parts["conjunction"] = "own embedding"
        return parts

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

    def conjunction_loss(self, conjunction, conjuncts, negatives):
        """:math:`L_\\sqcap` of OnT (Eq. 7) for the regions of :math:`C_1 \\sqcap \\dots \\sqcap
        C_n` (from its verbalisation) and, one entry per conjunct position, the regions of the
        conjuncts and of the negatives: the mean over :math:`i` of :math:`L_\\prec(x_{C_1 \\sqcap
        \\dots \\sqcap C_n} \\prec x_{C_i})`, ``(B, 1)``. For :math:`n = 2` this is Eq. 7."""
        return th.stack([self.hierarchy_loss(conjunction, x, n)
                         for x, n in zip(conjuncts, negatives)]).mean(dim=0)
