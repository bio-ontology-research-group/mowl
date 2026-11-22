"""Relation models: what a relation is. A relation model maps regions to regions:
:meth:`~Relations.image` gives the things related *from* a region,
:meth:`~Relations.preimage` the things related *to* it, which is the region of
:math:`\\exists r.C`.

Because only the relation model knows what a relation is, it also answers the axioms about
relations, through one interface: :meth:`~Relations.role_inclusion`
(:math:`r \\sqsubseteq s`) and :meth:`~Relations.role_chain`
(:math:`r \\circ t \\sqsubseteq s`). Both take the measure and the margin of the objective,
which a relation model that compares regions (boxes of translations) uses and one that
compares parameters (shift vectors) may ignore. A relation model that defines them lets a
:class:`~mowl.representation.ComposedGeometry` train on the EL++ role axioms.
"""

import torch as th
import torch.nn as nn

from mowl.representation.regions import ConstructorUnavailable

__all__ = ["Relations", "Translation"]


class Relations(nn.Module):
    """Base class of relation models.

    :cvar part: The vocabulary entry (:class:`mowl.representation.RelationModel` value).
    """

    part = ""
    #: The region classes whose regions :meth:`image` and :meth:`preimage` map (``None``:
    #: every region that can be translated or whatever the model needs; a model that maps
    #: only some families says which, so that a combination that cannot read
    #: :math:`\\exists r.C` raises :class:`ConstructorUnavailable` instead of failing inside
    #: a formula).
    accepts = None
    #: Whether a relation is a function (every point has exactly one successor, given by
    #: :meth:`successor`); the preimage of a region is then the set of points whose
    #: successor lies in it.
    functional = False

    def check_region(self, region_class):
        """Raise :class:`ConstructorUnavailable` if the regions of ``region_class`` are not
        mapped by this relation model."""
        if self.accepts is not None and not issubclass(region_class, self.accepts):
            raise ConstructorUnavailable(
                f"{type(self).__name__} maps {[c.__name__ for c in self.accepts]} regions, "
                f"not {region_class.__name__}: this combination of parts cannot read "
                f"exists r.C.")

    def related(self, ids, x, y):
        """``(B, P)`` boolean: whether the points ``x[p]`` and ``y[p]`` (``(P, d)`` each) are
        a pair of the relation of row ``b`` of ``ids``. This is the meaning of the relation
        model: :meth:`preimage` must be the set of points with a successor in the region,
        :meth:`image` the set of successors of its points, and the role axioms must hold
        for these pairs. A relation model that is not a set of pairs of points (a relation
        between regions, a ranking score) has no such test."""
        raise ConstructorUnavailable(
            f"{type(self).__name__} does not say which pairs of points it relates.")

    def image(self, ids, region):
        """:math:`\\{y : (x, y) \\in r, x \\in \\text{region}\\}` for the relations
        ``ids``."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no image.")

    def preimage(self, ids, region):
        """:math:`\\{x : (x, y) \\in r, y \\in \\text{region}\\}`: the region of
        :math:`\\exists r.C`."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no preimage.")

    def tables(self):
        """The raw parameter tables by attribute name (``{"vectors": ...}`` for a
        translation; see :func:`~mowl.representation.class_models.raw_tables`)."""
        from mowl.representation.class_models import raw_tables
        return raw_tables(self)

    def sharp_preimage(self, ids, region):
        """The preimage without any tolerance of the relation (what a *negative* axiom is
        read against). Default: the preimage."""
        return self.preimage(ids, region)

    def tolerance(self, ids):
        """How much the relation gives up of the exact image, per relation (``(B, 1)`` or
        ``0.0``); an objective may charge it (``tolerance_weight``). Default: none."""
        return 0.0

    def subsumed(self, measure, sub, ids, filler, margin=0.0):
        """Cost of :math:`C \\sqsubseteq \\exists r.D` for the region ``sub`` of
        :math:`C` and the region ``filler`` of :math:`D`. Default: ``sub`` lies inside the
        preimage of ``filler``. A relation model whose :math:`\\exists r.D` needs a second
        condition (head and tail boxes) defines its own reading."""
        return measure.inside(sub, self.preimage(ids, filler), margin)

    def role_inclusion(self, sub, sup, measure, margin=0.0):
        """Cost ``(B, 1)`` of :math:`r \\sqsubseteq s` for the relation ids ``sub``,
        ``sup``; ``measure`` and ``margin`` are those of the objective."""
        raise ConstructorUnavailable(
            f"{type(self).__name__} does not define role_inclusion(...), so role "
            f"inclusions cannot be read.")

    def role_chain(self, first, second, sup, measure, margin=0.0):
        """Cost ``(B, 1)`` of :math:`r_1 \\circ r_2 \\sqsubseteq s` (first :math:`r_1`, then
        :math:`r_2`)."""
        raise ConstructorUnavailable(
            f"{type(self).__name__} does not define role_chain(...), so role chains "
            f"cannot be read.")

    @property
    def reads_role_axioms(self):
        """Whether the relation model defines both role axioms."""
        cls = type(self)
        return (cls.role_inclusion is not Relations.role_inclusion
                and cls.role_chain is not Relations.role_chain)

    def __len__(self):
        return self.vectors.num_embeddings


class Translation(Relations):
    """A relation is one shift vector: :math:`(x, y) \\in r` iff :math:`y = x + v_r`
    (TransE; ELEmbeddings, ELBE and many others). A function: every point has exactly one
    successor, so one-to-many relations cannot be represented.

    Role axioms: :math:`r \\sqsubseteq s` holds iff :math:`v_r = v_s` and
    :math:`r_1 \\circ r_2 \\sqsubseteq s` iff :math:`v_{r_1} + v_{r_2} = v_s`. The cost is the
    distance of the two sides; equality has no strict margin, so only a slack (a negative
    margin) is read.

    :param vectors: An existing table to use instead of a new one.
    :param direction: ``None`` (default): the cost of a role axiom is the distance of the two
        sides. ``"row"``: the term :math:`|1 - \\cos(a, b)|` of EmEL++ (Mondal et al. 2021,
        Eq. 8-9) is added per row; ``"batch"``: the version of the authors' Keras code,
        normalised over the batch, one number added to every row.
    """

    part = "translation"
    functional = True

    def __init__(self, size=None, embed_dim=None, vectors=None, direction=None):
        super().__init__()
        if direction not in (None, "row", "batch"):
            raise ValueError("direction must be None, 'row' or 'batch'.")
        self.direction = direction
        if vectors is None:
            vectors = nn.Embedding(size, embed_dim)
            with th.no_grad():
                nn.init.uniform_(vectors.weight, a=-1, b=1)
                vectors.weight /= th.linalg.norm(vectors.weight, dim=1, keepdim=True)
        self.vectors = vectors

    def successor(self, ids, x):
        """The successor :math:`x + v_r` of every point ``x`` (``(P, d)``) for every relation
        of ``ids``: ``(B, P, d)``."""
        return x.unsqueeze(0) + self.vectors(ids).unsqueeze(1)

    def related(self, ids, x, y, tolerance=1e-5):
        """:math:`y = x + v_r` (up to ``tolerance``, a translation relates sets of measure
        zero)."""
        gap = th.linalg.norm(y.unsqueeze(0) - self.successor(ids, x), dim=-1)
        return gap <= tolerance

    def image(self, ids, region):
        return region.translate(self.vectors(ids))

    def preimage(self, ids, region):
        return region.translate(-self.vectors(ids))

    def _role_cost(self, left, right, margin):
        cost = th.relu(th.linalg.norm(right - left, dim=1, keepdim=True) + min(margin, 0.0))
        if self.direction is not None:
            from mowl.nn.el.emelpp.losses import _direction_term
            cost = cost + _direction_term(left, right, self.direction)
        return cost

    def role_inclusion(self, sub, sup, measure=None, margin=0.0):
        return self._role_cost(self.vectors(sub), self.vectors(sup), margin)

    def role_chain(self, first, second, sup, measure=None, margin=0.0):
        return self._role_cost(self.vectors(first) + self.vectors(second), self.vectors(sup),
                               margin)
