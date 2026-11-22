"""Methods with a geometry of their own: ``grounded``, ``canonical_mixture``, ``faithel`` and
``boxlite``.

None of them is a combination of the parts of :mod:`mowl.representation`; each is given the
common interface, sharing the parameters of the existing module and giving the method's own
numbers:

``grounded`` (:class:`GroundedGeometry`)
    A class is a union of boxes (:class:`MaskedUnionOfBoxes`), an individual a point, a relation (in
    ``role_mode="translation"``) a set of translation vectors. Conjunction (all pairs of
    boxes) and disjunction (all boxes) are exact, :math:`\\exists r.C` is the module's
    ``exists_box``. ``inside`` is a *sufficient* test (every active box of ``A`` lies in
    one active box of ``B``): zero implies inclusion, but a box covered only by the union
    of several is not recognised; the module itself trains on sampled points.
    ``member`` is the module's own ``class_assertion`` score, minus the signed depth.
``canonical_mixture`` (:class:`CanonicalMixtureGeometry`)
    A class is the set of canonical elements that belong to it (:class:`ElementSet`), an
    individual a distribution over canonical elements (:class:`Mixture`). Inclusion is
    exact on the sets (0.0 or 1.0); membership is the module's own ``-log`` probability.
``faithel`` (:class:`FaithELGeometry`)
    The TBox is the fixed strongly faithful convex model (costs 0.0 or 1.0, those of
    ``tbox_costs``); the ABox is the learned individual vectors (``member`` is
    ``class_assertion_costs`` and ``role_assertion_costs``).
``boxlite`` (:class:`BoxLitEGeometry`)
    DL-Lite boxes of the solved convex programme: concept names and :math:`\\exists R`,
    :math:`\\exists R^-` are boxes, the negation is the complement box, individuals are
    points with a bump. ``member`` is the signed distance of the paper (negative inside).
    Nothing is trained (a solver computes the embedding).
"""

from dataclasses import dataclass

import torch as th

from mowl.representation.adapters import register_adapter
from mowl.representation.geometry import Geometry
from mowl.representation.library.other_constructed import (
    ConstructedGeometry, NormalFormBackend)
from mowl.representation.library.other_graph import Point, Related
from mowl.representation.measures import Bounds
from mowl.representation.regions import Box, ConstructorUnavailable, Region

__all__ = ["MaskedUnionOfBoxes", "GroundedGeometry", "ElementSet", "Mixture",
           "CanonicalMixtureGeometry", "FaithELGeometry", "BoxLitEGeometry", "Universe"]

#: Cost of a region that is not inside an empty one (the finite floor of the module).
_BIG = 1e6


# -- grounded ---------------------------------------------------------------------------------
class MaskedUnionOfBoxes(Region):
    """A union of boxes per row, some of which may be *inactive* (the grounded model of
    queries and defined classes), unlike
    :class:`~mowl.representation.library.multibox_parts.UnionOfBoxes`, whose boxes are all
    there and whose intersection keeps every pair with its distance. Here the intersection is
    the exact set intersection: a pair that does not intersect is dropped (inactive) and the
    cost is the distance of the nearest pair (a large number if no pair is active). A
    ``MaskedUnionOfBoxes`` is read by the loss functions of the grounded module (it carries
    the :class:`~mowl.nn.el.structural.Regions` of the module, ``structural``). Used by:
    the grounded model. ``cen`` and ``w`` are ``(B, K, d)`` (centre and
    half-width), ``active`` ``(B, K)``. ``ids`` are the class ids the region was made from
    (for named classes), else ``None``."""

    def __init__(self, cen, w, active, ids=None):
        self.cen, self.w, self.active, self.ids = cen, w, active, ids

    def __len__(self):
        return self.cen.shape[0]

    @classmethod
    def of(cls, regions, ids=None):
        return cls(regions.cen, regions.w, regions.active, ids)

    @property
    def structural(self):
        from mowl.nn.el.structural import Regions
        return Regions(cen=self.cen, w=self.w, active=self.active)

    def intersect(self, other):
        """Exact: the union of the intersections of all pairs of boxes."""
        lo = th.maximum((self.cen - self.w)[:, :, None], (other.cen - other.w)[:, None])
        hi = th.minimum((self.cen + self.w)[:, :, None], (other.cen + other.w)[:, None])
        b, k1, k2, d = lo.shape
        w = ((hi - lo) / 2).reshape(b, k1 * k2, d)
        cen = ((hi + lo) / 2).reshape(b, k1 * k2, d)
        active = (self.active[:, :, None] & other.active[:, None]).reshape(b, -1) & \
            (w.amin(dim=-1) > 0)
        gap = th.linalg.norm(th.relu(lo - hi), dim=-1).reshape(b, -1)
        gap = th.where((self.active[:, :, None] & other.active[:, None]).reshape(b, -1),
                       gap, th.full_like(gap, _BIG)).amin(dim=1, keepdim=True)
        gap = th.where(active.any(dim=1, keepdim=True), th.zeros_like(gap), gap)
        w = th.where(active[..., None], w, th.zeros_like(w))
        return MaskedUnionOfBoxes(cen, w, active), gap


class GroundedGeometry(Geometry):
    """The geometry of a :class:`~mowl.nn.el.grounded.GroundedELModule`.

    The parameters are the module's (the regions are computed from them on every call).
    Elements ``0 .. nb_classes - 1`` are classes, the rest individuals; a class id is also
    its element id.
    """

    def __init__(self, module):
        self.module = module

    def classes(self, ids):
        return MaskedUnionOfBoxes.of(self.module.class_regions(ids), ids)

    def individuals(self, ids):
        return Point(self.module.positions(self.module.nb_classes + ids))

    def exists(self, relation_ids, region):
        if self.module.role_mode != "translation":
            raise ConstructorUnavailable(
                "Only role_mode='translation' has a region for exists r.C; in 'product' "
                "mode a role is a set of pairs of boxes (see GroundedELModule.role_depth).")
        if region.ids is None:
            raise ConstructorUnavailable("The filler of exists must be a named class.")
        return MaskedUnionOfBoxes.of(self.module.exists_box(relation_ids, region.ids))

    def conjunction(self, a, b):
        return a.intersect(b)

    def disjunction(self, a, b):
        return MaskedUnionOfBoxes(th.cat([a.cen, b.cen], 1), th.cat([a.w, b.w], 1),
                                  th.cat([a.active, b.active], 1))

    def inside(self, a, b):
        """Zero if every active box of ``a`` lies in one active box of ``b`` (sufficient
        for inclusion, not necessary), otherwise the smallest total violation over the
        choices of box; ``a`` may also be an individual (a point): then the hinge of the
        negative depth."""
        if isinstance(a, Point):
            return th.relu(-self._depth(a.centre, b)).reshape(-1, 1)
        viol = th.linalg.norm(th.relu(
            (a.cen[:, :, None] - b.cen[:, None]).abs() + a.w[:, :, None] - b.w[:, None]),
            dim=-1)
        viol = th.where(b.active[:, None], viol, th.full_like(viol, _BIG)).amin(dim=2)
        return th.where(a.active, viol, th.zeros_like(viol)).sum(dim=1, keepdim=True)

    def apart(self, a, b):
        """Exact: zero iff no active box of ``a`` meets one of ``b`` (the smallest overlap
        over the coordinates, summed over the pairs of boxes)."""
        overlap = (a.w[:, :, None] + b.w[:, None] - (a.cen[:, :, None] - b.cen[:, None]).abs()
                   ).amin(dim=-1)
        both = a.active[:, :, None] & b.active[:, None]
        return th.where(both, th.relu(overlap), th.zeros_like(overlap)).sum(
            dim=(1, 2)).reshape(-1, 1)

    def _depth(self, points, region):
        if region.ids is not None:
            return self.module.depth(points, region.ids, crisp=True)
        from mowl.nn.el.structural import inside_measure
        return inside_measure(points, region.structural)

    def member(self, individual_ids, region):
        """The module's own score of a class assertion: minus the signed depth of the
        individual in the region (negative inside, zero on the boundary)."""
        points = self.module.positions(self.module.nb_classes + individual_ids)
        return -self._depth(points, region).reshape(-1, 1)

    def soft_membership(self, individual_ids, region):
        """``sigmoid(depth / tau)`` (the module's :meth:`membership`), named classes only."""
        if region.ids is None:
            raise ConstructorUnavailable("Soft membership is defined for named classes.")
        points = self.module.positions(self.module.nb_classes + individual_ids)
        return self.module.membership(points, region.ids).reshape(-1, 1)

    @property
    def num_classes(self):
        return self.module.nb_classes

    @property
    def num_individuals(self):
        return self.module.nb_elements - self.module.nb_classes

    @property
    def parts(self):
        return {"classes": "union of boxes",
                "relations": "region over pairs" if self.module.role_mode == "product"
                else "box of translations",
                "individuals": "point", "measure": "bounds per coordinate",
                "conjunction": "exact"}


register_adapter("GroundedELModule")(GroundedGeometry)


# -- canonical mixture -----------------------------------------------------------------------
class ElementSet(Region):
    """A set of canonical elements: ``mask`` is ``(B, |S|)`` with entries 0.0 or 1.0."""

    def __init__(self, mask):
        self.mask = mask

    def __len__(self):
        return self.mask.shape[0]


class Mixture(Region):
    """A distribution over the canonical elements: ``probs`` is ``(B, |S|)``; ``ids`` are
    the individuals."""

    def __init__(self, probs, ids):
        self.probs, self.ids = probs, ids

    def __len__(self):
        return self.probs.shape[0]


class CanonicalMixtureGeometry(Geometry):
    """The geometry of a :class:`~mowl.nn.el.canonical_mixture.CanonicalMixtureModule`.

    ``inside``, ``apart`` and the constructors are exact on sets of canonical elements
    (costs 0.0 / 1.0); ``member`` is the module's own ``-log`` of the probability that the
    individual's distribution puts on the class (``inf`` for probability zero), and on
    :math:`\\exists r.b` the module's ``-log`` of ``p_a . (E_r p_b)``.
    """

    def __init__(self, module):
        self.module = module
        self.structure = module.structure
        self._dense = None

    @property
    def matrix(self):
        """Dense membership matrix ``(|S|, n_classes)`` of the canonical elements."""
        if self._dense is None:
            self._dense = self.structure.M.to_dense().to(th.float32)
        return self._dense.to(self.module.U.device)

    def classes(self, ids):
        return ElementSet(self.matrix[:, ids.to(self.matrix.device)].T)

    def individuals(self, ids):
        return Mixture(th.exp(self.module.log_p(ids)), ids)

    def _role(self, relation_id):
        return self.structure.roles[int(relation_id)].to_dense().to(th.float32).to(
            self.matrix.device)

    def exists(self, relation_ids, region):
        if isinstance(region, Mixture):
            return Related(self, relation_ids, region)
        out = [(self._role(r) @ m > 0).to(th.float32)
               for r, m in zip(relation_ids.tolist(), region.mask)]
        return ElementSet(th.stack(out))

    def conjunction(self, a, b):
        mask = a.mask * b.mask
        return ElementSet(mask), (mask.sum(dim=1, keepdim=True) == 0).to(th.float32)

    def disjunction(self, a, b):
        return ElementSet(th.maximum(a.mask, b.mask))

    def negation(self, a):
        """The complement within the canonical elements."""
        return ElementSet(1 - a.mask)

    def inside(self, a, b):
        """0.0 where every canonical element of ``a`` belongs to ``b``, else 1.0 (for an
        individual: the module's ``-log`` probability)."""
        if isinstance(a, Mixture):
            return self._mass_cost(a.probs, b)
        return (a.mask > b.mask).any(dim=1, keepdim=True).to(th.float32)

    def apart(self, a, b):
        return ((a.mask * b.mask).sum(dim=1, keepdim=True) > 0).to(th.float32)

    def _mass_cost(self, probs, region):
        if isinstance(region, Related):
            tail = region.filler.probs
            reach = th.stack([self._role(r) @ p for r, p in zip(region.ids.tolist(), tail)])
            mass = (probs * reach).sum(dim=1, keepdim=True)
        else:
            mass = (probs * region.mask).sum(dim=1, keepdim=True)
        return -th.log(mass.clamp(max=1.0)) + 0.0

    def member(self, individual_ids, region):
        return self._mass_cost(self.individuals(individual_ids).probs, region)

    @property
    def num_classes(self):
        return self.module.n_classes

    @property
    def num_individuals(self):
        return self.module.n_ind

    @property
    def parts(self):
        return {"classes": "canonical element", "relations": "fixed by reasoning",
                "individuals": "distribution", "measure": "exact", "conjunction": "exact"}


register_adapter("CanonicalMixtureModule")(CanonicalMixtureGeometry)


# -- faithel ---------------------------------------------------------------------------------
class FaithELGeometry(ConstructedGeometry):
    """The geometry of a :class:`~mowl.models.faithel.FaithEL`: the fixed strongly faithful
    convex TBox model (inclusion, conjunction and existential restrictions as costs 0.0 /
    1.0, as ``tbox_costs``) and the learned individuals (``member`` is
    ``class_assertion_costs`` for named classes and ``role_assertion_costs`` for
    :math:`\\exists r.b`). Names outside the signature of the TBox cost 1.0.

    The parameters of the individuals are the model's own (``module.individuals``).
    """

    def __init__(self, model):
        self.model = model
        classes = {i: n for n, i in model.class_index_dict.items()}
        roles = {i: n for n, i in model.object_property_index_dict.items()}
        individuals = {i: n for n, i in model.individual_index_dict.items()}
        backend = _FaithBackend(model.tbox, names=[classes[i] for i in sorted(classes)],
                                roles=[roles[i] for i in sorted(roles)],
                                individuals=[individuals[i] for i in sorted(individuals)],
                                key="faithel")
        super().__init__(backend)

    def member(self, individual_ids, region):
        model = self.model
        cost = []
        for a, e in zip(individual_ids.tolist(), region.exprs):
            if e[0] == "name":
                row = th.tensor([[a, model.class_index_dict[e[1]]]])
                cost.append(model.class_assertion_costs(row).reshape(-1)[0])
            elif e[0] == "exists" and e[2][0] == "nominal":
                row = th.tensor([[a, model.object_property_index_dict[e[1]],
                                  model.individual_index_dict[e[2][1]]]])
                cost.append(model.role_assertion_costs(row).reshape(-1)[0])
            else:
                raise ConstructorUnavailable("FaithEL scores class assertions of named "
                                             "classes and role assertions.")
        return th.stack(cost).reshape(-1, 1)

    @property
    def parts(self):
        return {"classes": "fixed by reasoning", "relations": "fixed by reasoning",
                "individuals": "point", "measure": "exact", "conjunction": "exact"}


def _atoms(expr):
    if expr[0] in ("and", "or"):
        return _atoms(expr[1]) + _atoms(expr[2])
    if expr[0] in ("exists",):
        return _atoms(expr[2])
    if expr[0] == "not":
        return _atoms(expr[1])
    return [expr]


class _FaithBackend(NormalFormBackend):
    """The TBox model of FaithEL: an entity outside its signature costs 1.0."""

    def _known(self, expr):
        from mowl.geometric.normal_forms import TOP
        sig = set(self.model.signature) | {TOP}       # the top class is known (as in tbox_costs)
        return all(not (e[0] == "name" and e[1] not in sig) for e in _atoms(expr))

    def _roles_known(self, *exprs):
        return all(e[1] in self.model.role_signature for e in exprs if e[0] == "exists")

    def subsumed(self, a, b):
        if not (self._known(a) and self._known(b) and self._roles_known(a, b)):
            return False
        return super().subsumed(a, b)


@register_adapter("FaithEL")
def _faithel(model):
    """The geometry of a FaithEL: its own module where the model is built from parts, the
    adapter geometry (reading the tables of the direct implementation) otherwise."""
    module = getattr(model, "module", None)
    if isinstance(module, Geometry):
        return module
    return FaithELGeometry(model)


# -- boxlite ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Universe(Box):
    """The universe box :math:`[-s_\\Omega, s_\\Omega]^d`: the filler of an unqualified
    :math:`\\exists R` of DL-Lite."""


class BoxLitEGeometry(Geometry):
    """The geometry of a :class:`~mowl.models.boxlite.BoxLitE` embedding (or of the
    :class:`~mowl.models.boxlite.embedding.BoxLitEEmbedding` itself).

    Class ids ``0 .. n_concepts - 1`` are concept names, then :math:`\\exists R` for every
    role, then :math:`\\exists R^-` (the basic concepts of DL-Lite, in the order of
    ``BoxLitEEmbedding.basic_bounds``). A box is ``[L + eps, U - eps]`` (``eps`` the
    margin of the embedding). ``exists`` is only defined for the universe (DL-Lite has
    unqualified existentials); ``negation`` is the complement box
    :math:`[-s_\\Omega - L, s_\\Omega - U]`; ``inside`` and ``apart`` use the measure
    :class:`~mowl.representation.Bounds`. ``member`` is the paper's concept loss (a
    signed distance, negative inside the box), and the role loss ``max`` of the four
    distances for :math:`\\exists R.\\{b\\}`.
    """

    def __init__(self, embedding):
        self.embedding = embedding
        self.measure = Bounds()
        self._lower, self._upper = (th.as_tensor(x, dtype=th.float32)
                                    for x in embedding.basic_bounds())

    def _box(self, lower, upper):
        m = self.embedding.margin
        lo, hi = lower + m, upper - m
        return Box((lo + hi) / 2, (hi - lo) / 2)

    def classes(self, ids):
        return self._box(self._lower[ids], self._upper[ids])

    def universe(self, n=1):
        d = self.embedding.dim
        return Universe(th.zeros(n, d), th.full((n, d), float(self.embedding.s_omega)))

    def individuals(self, ids):
        return _Individuals(th.as_tensor(self.embedding.pos[ids.numpy()], dtype=th.float32),
                            th.as_tensor(self.embedding.bump[ids.numpy()],
                                         dtype=th.float32), ids)

    def exists(self, relation_ids, region, inverse=False):
        if isinstance(region, _Individuals):
            return Related(self, relation_ids, region)
        if not isinstance(region, Universe):
            raise ConstructorUnavailable("DL-Lite has unqualified existentials: the filler "
                                         "must be geometry.universe().")
        nc, nr = self.embedding.n_concepts, self.embedding.n_roles
        return self.classes(nc + (nr if inverse else 0) + relation_ids)

    def conjunction(self, a, b):
        return a.intersect(b)

    def negation(self, a):
        """The complement box :math:`[-s_\\Omega - L, s_\\Omega - U]` of the embedding's
        bounds ``[L, U]``, with the margin applied as for every box."""
        s, m = float(self.embedding.s_omega), self.embedding.margin
        lower, upper = a.lower - m, a.upper + m           # the raw bounds L and U
        return self._box(-s - lower, s - upper)

    def inside(self, a, b):
        return self.measure.inside(a, b)

    def apart(self, a, b):
        return self.measure.apart(a, b)

    def member(self, individual_ids, region):
        """The concept loss of the paper, ``(B, 1)``: the signed distance of the position
        to the box (``<= 0`` inside). For :math:`\\exists R.\\{b\\}` the role loss."""
        ids = individual_ids.numpy()
        if isinstance(region, Related):
            r, obj = region.ids.numpy(), region.filler.ids.numpy()
            out = [float(self.embedding.role_loss(int(rr), [a], [o])[0])
                   for rr, a, o in zip(r, ids, obj)]
            return th.tensor(out, dtype=th.float32).reshape(-1, 1)
        from mowl.representation.library.solved_boxlite import box_distance
        lower, upper = region.lower.numpy(), region.upper.numpy()
        x = self.embedding.pos[ids]
        return th.as_tensor(box_distance(lower, upper, x, 0.0), dtype=th.float32
                            ).reshape(-1, 1)

    @property
    def num_classes(self):
        return int(self._lower.shape[0])

    @property
    def num_individuals(self):
        return int(self.embedding.pos.shape[0])

    @property
    def parts(self):
        return {"classes": "box", "relations": "head and tail boxes",
                "individuals": "point with bump", "measure": "bounds per coordinate",
                "conjunction": "exact"}


class _Individuals(Region):
    def __init__(self, pos, bump, ids):
        self.pos, self.bump, self.ids = pos, bump, ids

    @property
    def centre(self):
        return self.pos

    def __len__(self):
        return self.pos.shape[0]


def _boxlite(model):
    module = getattr(model, "module", None)
    if isinstance(module, Geometry):          # the model is built from parts: its own geometry
        return module
    embedding = getattr(model, "embedding", model)
    if embedding is None:
        raise RuntimeError("BoxLitE has no embedding yet; call train() first.")
    return BoxLitEGeometry(embedding)


register_adapter("BoxLitE", "BoxLitEEmbedding")(_boxlite)
