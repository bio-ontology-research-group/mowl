"""The geometry of Embed2Reason (E2R): classes are axis-parallel subspaces.

E2R is **not** rebuilt from swappable parts here. Its objective is a list of equations of the
paper (a class *equals* the meet, join or complement of others; a relation is a subspace of
pairs; structure terms keep the indicators in :math:`\\{0, 1\\}` and are clipped after every
step), not a reading of the normal-form axioms by a measure between regions, and it has no
existential restriction. What fits the common interface is the geometry itself, which
:class:`E2RGeometry` gives over the parameters of an
:class:`~mowl.nn.alc.e2r.E2RModule`:

* the region of a class is its 0/1 indicator over the :math:`2d` real axes
  (:class:`Subspace`), the region of an individual its vector (:class:`Point`);
* conjunction is the intersection of subspaces (product of indicators), disjunction the span
  (maximum), negation the orthogonal complement inside the concept universe (the real half);
* ``inside`` is eq. (5) of the paper, ``apart`` the disjointness loss (eq. (6) with
  :math:`\\bot = \\{0\\}`), ``member`` the residual of the projection of the vector onto the
  subspace (eq. (4) and Sec. 5).

There is no ``exists``: the paper has no existential restriction (a relation is a subspace of
pairs, :meth:`E2RGeometry.relation_member` scores a pair of individuals against it).
"""

from dataclasses import dataclass

import torch as th

from mowl.representation.adapters import register_adapter
from mowl.representation.geometry import Geometry
from mowl.representation.regions import Region

__all__ = ["Subspace", "Point", "E2RGeometry"]


@dataclass(frozen=True)
class Subspace(Region):
    """An axis-parallel subspace: the 0/1 (or relaxed) indicator of its axes, ``(B, 2d)``."""

    indicator: th.Tensor

    def __len__(self):
        return self.indicator.shape[0]

    def intersect(self, other):
        meet = self.indicator * other.indicator
        # the intersection is {0} (empty in the sense of the paper) unless one axis is left
        gap = th.relu(1 - meet.sum(-1, keepdim=True))
        return Subspace(meet), gap


@dataclass(frozen=True)
class Point(Region):
    """The vector of an individual, ``(B, 2d)``."""

    vector: th.Tensor

    def __len__(self):
        return self.vector.shape[0]


def _losses():
    from mowl.nn.alc.e2r import losses
    return losses


class E2RGeometry(Geometry):
    """The geometry of an :class:`~mowl.nn.alc.e2r.E2RModule`; shares its parameters.

    :param module: The module.
    :param unit_length: Add the unit-length term :math:`(1 - x^\\top x)^2` of eq. (4) to the
        membership cost, which makes ``member`` the module's ``class_assertion_loss``. The
        default leaves it out: ``member`` is then the squared residual that
        :meth:`~mowl.models.E2R.rank_members` sorts by, and it does not depend on how long an
        individual's vector is.
    """

    def __init__(self, module, unit_length=False):
        self.module = module
        self.unit_length = unit_length

    def classes(self, ids):
        return Subspace(self.module.concept(ids))

    def individuals(self, ids):
        return Point(self.module.entity(ids))

    def relations(self, ids):
        """The subspaces of pairs of the relations ``ids``."""
        return Subspace(self.module.relation(ids))

    def conjunction(self, a, b):
        return a.intersect(b)

    def disjunction(self, a, b):
        return Subspace(_losses().join(a.indicator, b.indicator))

    def negation(self, a):
        d = self.module.dim
        out = th.zeros_like(a.indicator)
        out[..., :d] = 1 - a.indicator[..., :d]
        return Subspace(out)

    def inside(self, a, b):
        L = _losses()
        if isinstance(a, Point):
            cost = L.residual_sq(b.indicator, a.vector)
            if self.unit_length:
                cost = cost + L.unit_length_loss(a.vector)
            return cost.unsqueeze(-1)
        return L.inclusion_loss(a.indicator, b.indicator).unsqueeze(-1)

    def apart(self, a, b):
        return _losses().disjointness_loss(a.indicator, b.indicator).unsqueeze(-1)

    def member(self, individual_ids, region):
        return self.inside(self.individuals(individual_ids), region)

    def relation_member(self, subject_ids, relation_ids, object_ids):
        """Cost of the pairs being in the relations (eq. (4) for role assertions)."""
        L, m = _losses(), self.module
        return L.role_membership_loss(
            m.relation(relation_ids), m.entity(subject_ids), m.entity(object_ids), m.dim,
            unit_length=self.unit_length).unsqueeze(-1)

    @property
    def num_classes(self):
        return self.module.c_embedding.num_embeddings

    @property
    def num_individuals(self):
        return self.module.e_embedding.num_embeddings

    @property
    def parts(self):
        return {"classes": "subspace", "relations": "subspace", "individuals": "point",
                "measure": "point-to-region distance", "conjunction": "exact",
                "negation": "exact", "disjunction": "exact"}


@register_adapter("E2RModule")
def _e2r(module):
    return E2RGeometry(module)
