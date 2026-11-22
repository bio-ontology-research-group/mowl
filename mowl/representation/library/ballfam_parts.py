"""Parts of EmEL++, EmEL(var) and RegD, and the four methods built from them.

New parts:

* :class:`ToleranceTranslation` (``"translation with tolerance"``): a translation whose
  image and preimage are fuzzy by a learned tolerance per relation (EmEL(var)). Its role
  axioms are those of :class:`~mowl.representation.Translation` (the vector sum of a chain,
  with the cosine term of EmEL++ as an option ``direction``).
* :class:`BoundaryDepth` (``"boundary and depth"``): the RegD inclusion energy, for boxes and
  for balls.

EmEL++ and EmEL(var) are the core :class:`~mowl.representation.Objective` with its options
(``unit_norm``, ``tolerance_weight``, ``gci0_shift``) on balls
(:class:`~mowl.representation.BallClasses` with ``radius="relu"`` or ``"softplus"``) and a
translation; the role axioms are read by the relation model. :class:`RegDObjective` is the
one reading that remains here: RegD for the inclusion forms, a host objective for the others.

Methods: ``emelpp``, ``emelvar``, ``regd`` (boxes) and ``regd_elem`` (balls), registered with
:func:`~mowl.representation.register_method` and, for the existing modules, as adapters.

The contrastive training loss of RegD (it needs the sampled negatives of a positive pair
together) is the regime :class:`~mowl.training.GroupedLogSumExpContrastiveLoss`, which the
methods name in ``default_regimes``. What is *not* a part:  the shift vector of
the EmEL++ Keras code (an optional learned translation of ``C`` in ``C subClassOf D``, set
on the module by a training regime) is only read, through ``gci0_shift``.
"""

from dataclasses import replace
from functools import partial

import torch as th
import torch.nn as nn

from mowl.representation.library import boundary_depth as D
from mowl.representation.class_models import BallClasses, BoxClasses
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.measures import RegionMeasure, CentreDistance, Bounds
from mowl.representation.methods import (register_measure, register_method,
                                         register_relation_model, DEFAULT_MEASURE)
from mowl.representation.adapters import register_adapter
from mowl.representation.objectives import Objective
from mowl.training.losses import GroupedLogSumExpContrastiveLoss, MSETargetLoss
from mowl.representation.regions import Ball, Box, ConstructorUnavailable
from mowl.representation.relation_models import Translation

__all__ = ["ToleranceTranslation", "BoundaryDepth", "RegDObjective",
           "emelpp", "emelvar", "regd", "regd_elem"]


class _View:
    """A geometry with some attributes replaced (the objectives use it to read a form with
    another measure or another existential restriction)."""

    def __init__(self, geometry, **overrides):
        self._geometry = geometry
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._geometry, name)


# -- relation models ------------------------------------------------------------------------


@register_relation_model("translation with tolerance")
class ToleranceTranslation(Translation):
    """A translation that is exact up to a learned tolerance :math:`\\sigma_r \\ge 0` per
    relation (EmEL(var), Mohapatra et al. 2021): :math:`y` is related to :math:`x` if it lies
    within :math:`\\sigma_r` of :math:`x + v_r`. The image and the preimage of a region are
    the translated region enlarged by :math:`\\sigma_r` in every direction.

    :param tolerances: An existing table ``(size, 1)`` to use instead of a new one (stored
        signed, used through its absolute value).
    :param sigma_init: Initial tolerance of a new table.

    Part (vocabulary value): ``translation with tolerance``. Used by: ``emelvar``.
    """

    part = "translation with tolerance"
    functional = False

    def __init__(self, size=None, embed_dim=None, vectors=None, tolerances=None,
                 sigma_init=0.1, direction=None):
        super().__init__(size, embed_dim, vectors, direction)
        if tolerances is None:
            tolerances = nn.Embedding(len(self), 1)
            nn.init.constant_(tolerances.weight, sigma_init)
        self.tolerances = tolerances

    def tolerance(self, ids):
        """:math:`\\sigma_r` of the relations ``ids``, ``(B, 1)``."""
        return th.abs(self.tolerances(ids))

    def related(self, ids, x, y, region=Ball, tolerance=1e-5):
        """Whether :math:`y` lies within :math:`\\sigma_r` of :math:`x + v_r`, ``(B, P)``.
        The distance is that of the region family, so that the preimage of a region is a
        region of the same family: the Euclidean distance in a geometry of balls (EmEL(var))
        and the largest coordinate difference in a geometry of boxes (``region=Box``)."""
        gap = y.unsqueeze(0) - self.successor(ids, x)
        if issubclass(region, Box):
            distance = th.abs(gap).amax(dim=-1)
        else:
            distance = th.linalg.norm(gap, dim=-1)
        return distance <= self.tolerance(ids) + tolerance

    def image(self, ids, region):
        return super().image(ids, region).grow(self.tolerance(ids))

    def preimage(self, ids, region):
        return super().preimage(ids, region).grow(self.tolerance(ids))

    def sharp_preimage(self, ids, region):
        """The preimage without the tolerance (what a negative axiom is read against)."""
        return super().preimage(ids, region)


# -- measure ----------------------------------------------------------------------------------


def _default_measure(region_class):
    """The default measure of a region class or of its nearest base class that has one."""
    for cls in region_class.__mro__:
        if cls in DEFAULT_MEASURE:
            return DEFAULT_MEASURE[cls]()
    raise ConstructorUnavailable(f"There is no measure for {region_class.__name__} regions.")


@register_measure("boundary and depth")
class BoundaryDepth(RegionMeasure):
    """The inclusion energy of RegD (Yang et al. 2025): boundary dissimilarity plus
    :math:`\\lambda` times depth dissimilarity, of the child inside the parent, for boxes and
    for balls. :meth:`inside` is :math:`\\max\\{E + \\text{margin}, \\gamma_1\\}`, zero (with
    :math:`\\gamma_1 = 0` and the default ``depth_on_containment=False``) iff the region lies
    inside the other; an empty box lies inside every region. The raw, signed energy is
    :meth:`energy`.

    RegD defines inclusion only. :meth:`outside` is its negative term (the child is not
    inside the parent, by :math:`\\gamma_2`). Every other relation (``meets``, ``apart``,
    ``empty``, ``nonempty``, the intersection of balls) is that of the ``host`` measure,
    by default the default measure of the region family.

    :param lamda: :math:`\\lambda`.
    :param p: Exponent of the depth dissimilarity; default 1 for boxes, 2 for balls.
    :param gamma1, gamma2: The margins of the paper's loss.
    :param eps: Lower bound of sizes (they are divided by).
    :param depth_on_containment: ``True`` is the formula of the paper (Eq. 9: the depth term is
        always added); a region strictly inside another then still has a positive cost, so a
        zero cost is not what "inside" means, and the methods ``regd`` and ``regd_elem`` select
        it by default. ``False`` (the default of the part) is the reading of the authors'
        released code in training (the depth term only where the child is not inside, so a
        contained region costs nothing); the code itself adds the depth term always when it
        scores (evaluation mode). The text of the paper was not available to check Eq. 9
        against: the statement about it is that of the method's card and of the audit of the
        parts' defaults.
    :param host: Measure for the other relations.

    Part (vocabulary value): ``boundary and depth``. Used by: ``regd``, ``regd_elem``.
    """

    accepts = (Ball, Box)
    part = "boundary and depth"
    exact_intersection = False

    def __init__(self, lamda=1.0, p=None, gamma1=0.0, gamma2=0.0, eps=1e-4,
                 depth_on_containment=False, host=None):
        if lamda < 0:
            raise ValueError("lamda must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.lamda, self.p, self.gamma1, self.gamma2, self.eps = lamda, p, gamma1, gamma2, eps
        self.depth_on_containment = depth_on_containment
        self.host = host
        #: ``True``: :meth:`inside` returns the raw, signed energy (no margin clamp), to rank
        #: axioms by it (the module's ``scoring()`` context sets it).
        self.raw = False
        #: ``"empty"``: an empty box (negative half-width) lies inside everything; set to
        #: ``"formula"`` by the methods of RegD, whose energy floors the sizes at ``eps``.
        self.empty_boxes = "empty"

    def host_for(self, region_class):
        """The measure of the relations RegD does not define."""
        if self.host is not None:
            return self.host
        return _default_measure(region_class)

    def regularise(self, region):
        """The region with its sizes at least ``eps``."""
        if isinstance(region, Ball):
            return replace(region, radius=region.radius.clamp_min(self.eps))
        return Box(region.centre, region.offset.clamp_min(self.eps))

    def _args(self, parent, child):
        self.check(parent, child)
        if isinstance(parent, Ball) != isinstance(child, Ball):
            raise ConstructorUnavailable(
                f"BoundaryDepth compares two balls or two boxes, not a "
                f"{type(child).__name__} with a {type(parent).__name__}.")
        parent, child = self.regularise(parent), self.regularise(child)
        if isinstance(parent, Ball):
            return (parent.centre, parent.radius.squeeze(-1),
                    child.centre, child.radius.squeeze(-1))
        return parent.centre, parent.offset, child.centre, child.offset

    def energy(self, a, b):
        """:math:`E(b, a)`, ``(B, 1)``: ``a`` is the child, ``b`` the parent. Signed."""
        args = self._args(b, a)
        if isinstance(a, Ball):
            e = D.ball_energy(*args, lamda=self.lamda, p=2 if self.p is None else self.p,
                              depth_on_containment=self.depth_on_containment)
        else:
            e = D.box_energy(*args, lamda=self.lamda, p=1 if self.p is None else self.p,
                             depth_on_containment=self.depth_on_containment)
        return e.unsqueeze(-1)

    def boundary(self, a, b):
        """Boundary dissimilarity of the child ``a`` in the parent ``b``, ``(B, 1)``."""
        args = self._args(b, a)
        f = D.ball_boundary_dissimilarity if isinstance(a, Ball) \
            else D.box_boundary_dissimilarity
        return f(*args).unsqueeze(-1)

    def inside(self, a, b, margin=0.0):
        cost = self.energy(a, b) + margin
        if not self.raw:
            cost = th.clamp(cost, min=self.gamma1)
        if isinstance(a, Box) and self.empty_boxes == "empty":
            # the size floor eps would make an empty box a tiny one
            cost = th.where(Bounds.is_empty(a), th.zeros_like(cost), cost)
        return cost

    def outside(self, a, b, margin=0.0):
        """Zero iff ``a`` is not inside ``b`` (by :math:`\\gamma_2` + ``margin``)."""
        return th.clamp(self.gamma2 + margin - self.boundary(a, b), min=0.0)

    # RegD has no other relations: the host measure answers.
    def meets(self, a, b, margin=0.0):
        return self.host_for(type(a)).meets(a, b, margin)

    def apart(self, a, b, margin=0.0):
        return self.host_for(type(a)).apart(a, b, margin)

    def empty(self, a, margin=0.0):
        return self.host_for(type(a)).empty(a, margin)

    def nonempty(self, a, epsilon=0.0):
        return self.host_for(type(a)).nonempty(a, epsilon)

    def intersection_inside(self, a, b, c, margin=0.0):
        if isinstance(a, Ball):
            return self.host_for(Ball).intersection_inside(a, b, c, margin)
        # the box between two boxes that are apart, plus the gap (a zero cost needs a
        # non-empty intersection: sufficient for the axiom, as in RegD)
        region, gap = self.regularise(a).intersect(self.regularise(b), signed=False)
        return self.inside(region, c, margin) + gap

    def intersection_apart(self, a, b, c, margin=0.0):
        return self.host_for(type(a)).intersection_apart(a, b, c, margin)

    def intersection_nonempty(self, a, b, margin=0.0, epsilon=0.0):
        return self.host_for(type(a)).intersection_nonempty(a, b, margin, epsilon)


# -- objectives -------------------------------------------------------------------------------


class RegDObjective(Objective):
    """RegD (Yang et al. 2025): the inclusion forms are read with the RegD energy, every
    other form by a ``host`` objective (ELBE's or ELEmbeddings', with its own margin).

    The negative reading of :math:`C \\sqsubseteq D` and :math:`C \\sqsubseteq \\exists r.D`
    is *non-inclusion* (:meth:`BoundaryDepth.outside`), not disjointness; the other
    inclusion forms have no negative loss.

    :param host: The :class:`~mowl.representation.Objective` of the other forms. Its margin
        is this objective's :attr:`margin`.
    :param regd_forms: The forms read with the energy. The conjunction of balls is not a
        ball, so ``gci1`` is only in the list for boxes.
    :param flip_gci2: Read :math:`C \\sqsubseteq \\exists r.D` as the reverse inclusion
        :math:`D \\sqsubseteq \\exists r.C` (the prediction-task code of the authors).
    :param consistent_radius: ``True`` (default): every form reads the radius of a class
        through the class model, so a class has one radius. ``False``: the host forms of
        bottoms and class assertions, and the negatives of the conjunctions, read
        :math:`|\\text{raw radius}|` of the same tables whatever the class model's radius is
        (one class, two radii: the earlier :class:`~mowl.nn.RegDELEmModule` with
        ``radius="softplus"``, ``consistent_radius=False``). It makes a difference only when
        the class model's radius is not ``"abs"``.
    """

    #: The host forms read with the absolute radius when ``consistent_radius`` is ``False``;
    #: ``True``: always, ``"negative"``: only their negative loss.
    MIXED_FORMS = {"gci0_bot": True, "gci3_bot": True, "class_assertion": True,
                   "gci1": "negative", "gci1_bot": "negative"}

    def __init__(self, host, regd_forms=("gci0", "gci1", "gci2", "gci3"), flip_gci2=False,
                 consistent_radius=True):
        self.host = host
        self.consistent_radius = bool(consistent_radius)
        super().__init__(host.margin, host.slack, host.epsilon, host.gci3_reading,
                         host.gci2_negative, host.negatives, host.empty, host.gci2_offset,
                         strict_negatives=host.strict_negatives)
        self.regd_forms = frozenset(regd_forms)
        # the RegD negative of gci0 (a non-inclusion) is one more form with a negative loss
        self.neg_forms = host.neg_forms | ({"gci0"} & self.regd_forms)
        self.flip_gci2 = bool(flip_gci2)

    @property
    def margin(self):
        return self.host.margin

    @margin.setter
    def margin(self, value):
        self.host.margin = float(value)

    def link_margin(self, owner, attribute="margin"):
        self.host.link_margin(owner, attribute)
        return self

    @property
    def m(self):
        """The energy is compared without margin."""
        return 0.0

    def _delegate(self, name, g, rows, neg):
        overrides = {"measure": g.measure.host_for(g.class_model.region)}
        mixed = self.MIXED_FORMS.get(name)
        if not self.consistent_radius and (mixed is True or (mixed and neg)) \
                and getattr(g.class_model, "radius_form", "abs") != "abs":
            cm = g.class_model
            absolute = BallClasses(centres=cm.centres, radii=cm.radii, radius="abs")
            overrides["class_model"] = absolute
            overrides["classes"] = lambda ids, frozen=False: absolute(ids)
        return getattr(self.host, name)(_View(g, **overrides), rows, neg)

    def gci0(self, g, rows, neg=False):
        if "gci0" not in self.regd_forms:
            return self._delegate("gci0", g, rows, neg)
        c, d = g.classes(rows[:, 0]), g.classes(rows[:, 1])
        return g.measure.outside(c, d) if neg else g.measure.inside(c, d)

    def gci1(self, g, rows, neg=False):
        if "gci1" not in self.regd_forms:
            return self._delegate("gci1", g, rows, neg)
        c, d, e = (g.classes(rows[:, i]) for i in range(3))
        return g.measure.intersection_inside(c, d, e)

    def gci2(self, g, rows, neg=False):
        if "gci2" not in self.regd_forms:
            return self._delegate("gci2", g, rows, neg)
        c, r, d = g.classes(rows[:, 0]), rows[:, 1], g.classes(rows[:, 2])
        if self.flip_gci2:
            sub, target = d, g.exists(r, c)
        else:
            sub, target = c, g.exists(r, d)
        return g.measure.outside(sub, target) if neg else g.measure.inside(sub, target)

    def gci3(self, g, rows, neg=False):
        if "gci3" not in self.regd_forms:
            return self._delegate("gci3", g, rows, neg)
        exists = g.exists(rows[:, 0], g.classes(rows[:, 1]))
        return g.measure.inside(exists, g.classes(rows[:, 2]))


def _host_forward(name):
    def form(self, g, rows, neg=False):
        return self._delegate(name, g, rows, neg)
    form.__name__ = name
    form.__doc__ = f"``{name}`` as the host objective reads it."
    return form


for _name in ("gci0_bot", "gci1_bot", "gci3_bot", "class_assertion",
              "object_property_assertion"):
    setattr(RegDObjective, _name, _host_forward(_name))
del _name


# -- methods ----------------------------------------------------------------------------------


def _ball_tables(t, radius="abs"):
    classes = BallClasses(centres=t.class_embed, radii=t.class_rad, radius=radius)
    individuals = None
    if getattr(t, "ind_embed", None) is not None:
        individuals = BallClasses(centres=t.ind_embed, radii=t.ind_rad)
    return classes, individuals


def _norm_regularizers(reg_norm, individuals):
    from mowl.training.regularization import NormPenalty
    names = ("class_embed", "ind_embed") if individuals else ("class_embed",)
    return (NormPenalty(parameters=names, target=reg_norm),)


def _elpp(tolerant, nb_classes, nb_relations, nb_individuals, embed_dim, margin,
          linear_negative, gci1_radius_term, radius, role_direction, sigma_init,
          sigma_reg_weight, gci0_shift, tables):
    t = tables
    if t is None:
        # (the tables are drawn in the order of the earlier modules: classes, role vectors,
        # individuals, then the tolerances)
        classes = BallClasses(nb_classes, embed_dim, radius=radius)
        vectors = Translation(nb_relations, embed_dim).vectors
        individuals = BallClasses(nb_individuals, embed_dim, radii_init="uniform") \
            if nb_individuals else None
        tolerances = None
    else:
        classes, individuals = _ball_tables(t, radius)
        vectors = t.rel_embed
        tolerances = getattr(t, "rel_sigma", None)
    if tolerant:
        relations = ToleranceTranslation(nb_relations, embed_dim, vectors, tolerances,
                                         sigma_init, direction=role_direction)
    else:
        relations = Translation(nb_relations, embed_dim, vectors, direction=role_direction)
    objective = Objective(margin, slack=True, gci3="overlap",
                          gci2_negative="linear" if linear_negative else "apart",
                          negatives="gci2", negative_margin="strict", unit_norm=True,
                          gci3_bot="filler", strict_negatives=False,
                          tolerance_weight=sigma_reg_weight, gci0_shift=gci0_shift)
    return ComposedGeometry(classes, relations,
                            CentreDistance(gci1_radius_term, conjunction="published"),
                            objective, individuals=individuals)


@register_method("emelpp")
def emelpp(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1,
           linear_negative=False, gci1_radius_term=True, radius="abs", role_direction="row",
           gci0_shift=None, tables=None):
    """EmEL++ (Mondal et al. 2021): ELEmbeddings with the norm terms inside every axiom
    loss, the fourth term of the conjunction loss and role axioms on the shift vectors.

    :param role_direction: ``"row"`` as published; ``"batch"`` is the Keras code's reading;
        ``None`` drops the cosine term.
    :param tables: A module with ``class_embed``, ``class_rad``, ``rel_embed`` (and
        ``ind_embed``, ``ind_rad``) to share parameters with.
    """
    return _elpp(False, nb_classes, nb_relations, nb_individuals, embed_dim, margin,
                 linear_negative, gci1_radius_term, radius, role_direction, 0.1, 1.0,
                 gci0_shift, tables)


@register_method("emelvar")
def emelvar(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1,
            linear_negative=False, gci1_radius_term=True, radius="abs",
            role_direction="row", sigma_init=0.1, sigma_reg_weight=1.0, gci0_shift=None,
            tables=None):
    """EmEL(var) (Mohapatra et al. 2021): EmEL++ with a learned tolerance per relation.

    :param tables: As for :func:`emelpp`, plus ``rel_sigma``.
    """
    return _elpp(True, nb_classes, nb_relations, nb_individuals, embed_dim, margin,
                 linear_negative, gci1_radius_term, radius, role_direction, sigma_init,
                 sigma_reg_weight, gci0_shift, tables)


def _regd_measure(host_measure, lamda, p, gamma1, gamma2, eps, depth_on_containment):
    measure = BoundaryDepth(lamda, p, gamma1, gamma2, eps, depth_on_containment,
                            host=host_measure)
    measure.empty_boxes = "formula"
    return measure


@register_method("regd")
def regd(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1,
         all_negatives=False, epsilon=0.01, lamda=1.0, p=1, gamma1=0.0, gamma2=0.0,
         eps=1e-4, depth_on_containment=True, flip_gci2=False, tables=None):
    """RegD on ELBE (Yang et al. 2025): boxes, translations, the RegD energy for the
    inclusion forms, ELBE's losses (with ``margin``) for the others.

    :param tables: A module with ``class_embed``, ``class_offset``, ``rel_embed`` (and
        ``ind_embed``, ``ind_offset``) to share parameters with.
    """
    t = tables
    classes = BoxClasses(nb_classes, embed_dim) if t is None else BoxClasses(
        centres=t.class_embed, offsets=t.class_offset)
    relations = Translation(nb_relations, embed_dim) if t is None else Translation(
        vectors=t.rel_embed)
    individuals = None
    if t is None and nb_individuals:
        individuals = BoxClasses(nb_individuals, embed_dim)
    elif t is not None and getattr(t, "ind_embed", None) is not None:
        individuals = BoxClasses(centres=t.ind_embed, offsets=t.ind_offset)
    host = Objective.elbe(margin, all_negatives=all_negatives, epsilon=epsilon)
    measure = _regd_measure(Bounds(separation="every", extent="norm",
                                   conjunction="nonempty"), lamda, p, gamma1, gamma2, eps,
                            depth_on_containment)
    geometry = ComposedGeometry(classes, relations, measure,
                                RegDObjective(host, flip_gci2=flip_gci2),
                                individuals=individuals)
    # Eq. 10 for gci0 and gci2, ELBE's loss for the other forms
    geometry.default_regimes = (partial(GroupedLogSumExpContrastiveLoss, gamma1,
                                        normal_forms=("gci0", "gci2")),
                                partial(MSETargetLoss, 0.0, 1.0, reduce="mean_first"))
    return geometry


@register_method("regd_elem")
def regd_elem(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1,
              reg_norm=1, all_negatives=False, epsilon=0.01, gci1_radius_term=True,
              lamda=1.0, p=2, gamma1=0.0, gamma2=0.0, eps=1e-4, depth_on_containment=True,
              flip_gci2=False, radius="abs", consistent_radius=True, tables=None):
    """RegD on ELEmbeddings: balls, translations, the RegD energy for ``gci0``, ``gci2`` and
    ``gci3``, ELEmbeddings' losses (slack ``margin``, unit-norm penalty) for the others.

    :param radius: ``"abs"`` or ``"softplus"`` (see :class:`~mowl.representation.BallClasses`).
    :param consistent_radius: With ``radius="softplus"``: ``True`` (default) every form reads
        the softplus radius of a class; ``False`` the bottom forms, class assertions and the
        negatives of conjunctions read :math:`|r|`, see :class:`RegDObjective`. (The RegD code
        uses softplus everywhere.)
    :param tables: A module with ``class_embed``, ``class_rad``, ``rel_embed`` (and
        ``ind_embed``, ``ind_rad``) to share parameters with.
    """
    t = tables
    if t is None:
        classes = BallClasses(nb_classes, embed_dim, radius=radius)
        relations = Translation(nb_relations, embed_dim)
        individuals = BallClasses(nb_individuals, embed_dim, radii_init="uniform") \
            if nb_individuals else None
    else:
        classes, individuals = _ball_tables(t, radius)
        relations = Translation(vectors=t.rel_embed)
    host = Objective.elem(margin, all_negatives=all_negatives, epsilon=epsilon)
    if radius != "abs" and consistent_radius:
        # ELEmbeddings' code reads the signed radius parameter as the cost of C below
        # bottom, which is the radius where the radius is |r|; with another radius form the
        # cost is the radius of the class (as RegD-ELEm does with every form)
        host.empty = "size"
    measure = _regd_measure(CentreDistance(gci1_radius_term, conjunction="published"), lamda,
                            p, gamma1, gamma2, eps, depth_on_containment)
    objective = RegDObjective(host, regd_forms=("gci0", "gci2", "gci3"),
                              flip_gci2=flip_gci2, consistent_radius=consistent_radius)
    geometry = ComposedGeometry(classes, relations, measure, objective,
                                individuals=individuals,
                                regularizers=_norm_regularizers(reg_norm,
                                                                individuals is not None))
    geometry.default_regimes = (partial(GroupedLogSumExpContrastiveLoss, gamma1,
                                        normal_forms=("gci0", "gci2")),)
    return geometry


# -- adapters ---------------------------------------------------------------------------------


@register_adapter("EmELPPModule")
def _emelpp_adapter(module):
    return emelpp(None, None, embed_dim=module.embed_dim, margin=module.margin,
                  linear_negative=module.negative_loss_form == "linear",
                  gci1_radius_term=module.intersection_radius_term,
                  radius=module.radius_form, role_direction=module.role_direction,
                  gci0_shift=lambda: getattr(module, "subclass_relation", None),
                  tables=module)


@register_adapter("EmELVarModule")
def _emelvar_adapter(module):
    return emelvar(None, None, embed_dim=module.embed_dim, margin=module.margin,
                   linear_negative=module.negative_loss_form == "linear",
                   gci1_radius_term=module.intersection_radius_term,
                   radius=module.radius_form, role_direction=module.role_direction,
                   sigma_reg_weight=module.sigma_reg_weight,
                   gci0_shift=lambda: getattr(module, "subclass_relation", None),
                   tables=module)


def _regd_options(module):
    return dict(margin=module.margin, all_negatives=module.all_negative_losses,
                epsilon=module.epsilon, lamda=module.lamda, p=module.p,
                gamma1=module.gamma1, gamma2=module.gamma2, eps=module.eps,
                depth_on_containment=module.depth_on_containment,
                flip_gci2=module.flip_gci2, tables=module)


@register_adapter("RegDELBEModule")
def _regd_adapter(module):
    return regd(None, None, embed_dim=module.embed_dim, **_regd_options(module))


@register_adapter("RegDELEmModule")
def _regd_elem_adapter(module):
    return regd_elem(None, None, embed_dim=module.embed_dim, reg_norm=module.reg_norm,
                     gci1_radius_term=module.gci1_radius_term, radius=module.radius,
                     consistent_radius=module.consistent_radius, **_regd_options(module))
