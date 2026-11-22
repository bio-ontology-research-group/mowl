"""Box2EL, MultiBoxEL, MultiELBE and the volume-based MultiBoxEL as combinations of parts, and
the geometry of their modules.

========== ==================== ====================== ======================= ===============
key        classes              relations              measure                 objective
========== ==================== ====================== ======================= ===============
box2el     box with bump        head and tail boxes    bounds (every)          Box2EL
multiboxel union of boxes, bump head and tail boxes    union of bounds         Box2EL
multielbe  union of boxes       translation            union of bounds         ELBE
mcmultibox union of boxes       union of boxes / pairs sampled volume          semantic
========== ==================== ====================== ======================= ===============

Each function builds new tables, or (``tables=module``) shares those of an existing module;
the adapters of :func:`mowl.representation.geometry_of` are these functions called with
``tables=module``. Options of the modules that are not covered are listed in each docstring.
"""

import torch as th

from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.multibox_objective import (
    ComponentSpread, DiversityPenalty, bump_penalty)
from mowl.representation.objectives import Objective
from mowl.representation.library.multibox_parts import (
    BumpedBoxClasses, HeadTailRelations, PackedUnionClasses, PairUnions, SampledVolume,
    UnionClasses, UnionMeasure, _packed_init)
from mowl.representation.measures import Bounds
from mowl.representation.adapters import register_adapter
from mowl.representation.methods import register_method
from mowl.representation.relation_models import Translation

__all__ = ["box2el", "multiboxel", "multielbe", "mcmultibox"]


def _head_tail(nb_relations, embed_dim, t):
    if t is None:
        return HeadTailRelations(nb_relations, embed_dim)
    return HeadTailRelations(head_centres=t.head_center, head_offsets=t.head_offset,
                             tail_centres=t.tail_center, tail_offsets=t.tail_offset)


@register_method("box2el")
def box2el(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.0, delta=2.0,
           reg_factor=0.05, all_negatives=False, epsilon=0.01, tables=None):
    """Box2EL (Jackermeier et al. 2024) with the negative losses of DELE.

    Boxes with a bump vector, relations as a head box and a tail box, bounds per coordinate
    with a slack margin ``margin`` (the paper's :math:`\\gamma`), positive losses squared,
    the negative loss of :math:`C \\sqsubseteq \\exists r.D` is :math:`(\\delta - \\mu)^2`
    per argument, and the norm penalty on the bump vectors.

    Verified equal to :class:`~mowl.nn.BoxSquaredELModule`: every positive loss of the forms
    ``gci0`` ... ``gci3_bot``, ``class_assertion``, ``object_property_assertion``, the
    role inclusion and role chain losses, the negative losses of ``gci2`` and
    ``object_property_assertion`` and, with ``all_negatives=True``, the DELE negative losses
    of all forms, and the regularisation loss.

    :param tables: A :class:`~mowl.nn.BoxSquaredELModule` to share parameters with.
    """
    t = tables
    individuals = None
    if t is None:
        classes = BumpedBoxClasses(nb_classes, embed_dim)
        if nb_individuals:
            individuals = BumpedBoxClasses(nb_individuals, embed_dim)
    else:
        classes = BumpedBoxClasses(centres=t.class_center, offsets=t.class_offset,
                                   bumps=t.bump_classes)
        if t.ind_center is not None:
            individuals = BumpedBoxClasses(centres=t.ind_center, offsets=t.ind_offset,
                                           bumps=t.bump_individuals)
    objective = Objective(
        margin, slack=True, epsilon=epsilon, gci3="inclusion", gci2_negative="delta",
        negatives="all" if all_negatives else "gci2", squared=True, delta=delta,
        gci3_bot_negative="filler")
    return ComposedGeometry(classes, _head_tail(nb_relations, embed_dim, t),
                            Bounds(separation="every"), objective, individuals=individuals,
                            regularizers=(bump_penalty(reg_factor),))


def _union_parts(nb_classes, embed_dim, n_components, individuals_count, t, names, bump):
    """Class and individual model of a union of boxes; ``names`` are the attributes of the
    module's tables."""
    centres, offsets, bumps, ind_centres, ind_offsets, ind_bumps = (
        None if t is None else getattr(t, n, None) for n in names)
    individuals = None
    if t is None:
        classes = UnionClasses(nb_classes, embed_dim, n_components, bump=bump)
        if individuals_count:
            individuals = UnionClasses(individuals_count, embed_dim, 1, bump=bump)
    else:
        classes = UnionClasses(components=n_components, centres=centres, offsets=offsets,
                               bumps=bumps)
        if ind_centres is not None:
            individuals = UnionClasses(components=1, centres=ind_centres,
                                       offsets=ind_offsets, bumps=ind_bumps)
    return classes, individuals


def _union_regularizers(component_reg, diversity_reg):
    out = []
    if component_reg > 0:
        out.append(ComponentSpread(component_reg))
    if diversity_reg > 0:
        out.append(DiversityPenalty(diversity_reg))
    return out


@register_method("multiboxel")
def multiboxel(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.0,
               n_components=2, delta=2.0, reg_factor=0.05, component_reg=0.0,
               component_aggregation="hard", tau=1.0, diversity_reg=0.0, epsilon=0.01,
               tables=None):
    """MultiBoxEL (Federico, Leemhuis and Kutz 2026): Box2EL with every class a union of
    ``n_components`` boxes (one bump vector per class).

    Covered, and verified equal to :class:`~mowl.nn.MultiBoxELModule` at ``n_components`` 1
    and 2 to 3: ``objective="axiom"`` (the default) with ``component_aggregation`` ``"hard"``
    and ``"soft"`` (``tau``), ``component_reg``, ``diversity_reg``, ``reg_factor``, ``delta``
    and ``gamma`` (here ``margin``); all forms with their positive loss, the negative losses
    of ``gci2`` and ``object_property_assertion``, and the regularisation loss.

    Not covered: ``objective`` ``"point"``, ``"mixed"``, ``"hybrid"`` (point supervision) and
    ``"structural"`` (definition tables, existence and witness losses, re-seeding), and with
    them their parameters (``beta``, ``neg_weight``, ``volume_reg``, ``ind_halfwidth``,
    ``margin`` as the tightening margin of the structural objective, ``max_components``,
    ``structural_definitions``, ``existence_weight``, ``size_reg``, ``reseed_*``, ``bot_agg``,
    ``witness_weight``). The DELE negative losses of the other forms are not offered (the
    module has none).

    :param tables: A :class:`~mowl.nn.MultiBoxELModule` to share parameters with.
    """
    t = tables
    classes, individuals = _union_parts(
        nb_classes, embed_dim, n_components, nb_individuals, t,
        ("class_center", "class_offset", "bump_classes", "ind_center", "ind_offset",
         "bump_individuals"), bump=True)
    if t is not None:
        tau = t.tau
    objective = Objective(
        margin, slack=True, epsilon=epsilon, gci3="inclusion", gci2_negative="delta",
        negatives="gci2", squared=True, delta=delta, gci3_bot_negative="filler")
    measure = UnionMeasure(Bounds(separation="every"), component_aggregation, tau)
    regularizers = [bump_penalty(reg_factor)] + _union_regularizers(component_reg,
                                                                    diversity_reg)
    return ComposedGeometry(classes, _head_tail(nb_relations, embed_dim, t), measure,
                            objective, individuals=individuals, regularizers=regularizers)


@register_method("multielbe")
def multielbe(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1,
              n_components=1, component_reg=0.0, component_aggregation="hard", tau=1.0,
              diversity_reg=0.0, epsilon=0.01, gci3="overlap", tables=None):
    """MultiELBE (Federico, Leemhuis and Kutz 2026): ELBE with every class a union of
    ``n_components`` boxes (translations as relations, strict margin, overlap reading of
    :math:`\\exists r.C \\sqsubseteq D`).

    Covered, and verified equal to :class:`~mowl.nn.MultiELBEModule` at ``n_components`` 1
    and 2 to 3: ``objective="axiom"`` with ``component_aggregation`` ``"hard"`` and
    ``"soft"`` (``tau``), ``component_reg``, ``diversity_reg``, ``margin``; all forms with
    their positive loss, the negative losses of ``gci2`` and ``object_property_assertion``
    (mean over all pairs of components), and the regularisation loss. ``gci3="inclusion"``
    gives the corrected reading, which the module does not have.

    Not covered: ``objective`` ``"point"``, ``"mixed"``, ``"hybrid"`` and their parameters
    (``beta``, ``neg_weight``, ``volume_reg``, ``ind_halfwidth``).

    :param tables: A :class:`~mowl.nn.MultiELBEModule` to share parameters with.
    """
    t = tables
    classes, individuals = _union_parts(
        nb_classes, embed_dim, n_components, nb_individuals, t,
        ("class_embed", "class_offset", "none", "ind_embed", "ind_offset", "none"),
        bump=False)
    relations = Translation(nb_relations, embed_dim) if t is None else Translation(
        vectors=t.rel_embed)
    if t is not None:
        tau = t.tau
    objective = Objective(margin, slack=False, epsilon=epsilon, gci3=gci3,
                          gci2_negative="overlap", negatives="gci2")
    measure = UnionMeasure(Bounds(separation="every"), component_aggregation, tau)
    return ComposedGeometry(classes, relations, measure, objective, individuals=individuals,
                            regularizers=_union_regularizers(component_reg, diversity_reg))


@register_method("mcmultibox")
def mcmultibox(nb_classes, nb_relations, nb_individuals=0, embed_dim=5, margin=0.0,
               num_boxes=5, bound=1.0, num_points=20000, temperature=1.0, aggregate="min",
               seed=0, point_chunk_size=2048, init_seed=None, tables=None):
    """The volume-based MultiBoxEL (Aharoni 2024): classes are unions of boxes, relations are
    unions of boxes over pairs, and every loss is :math:`1 - \\mathrm{Vol}(A \\cap B) /
    \\mathrm{Vol}(A)` estimated on a fixed set of sampled points.

    Rebuilt from parts: the class model (:class:`PackedUnionClasses`), the relation model
    (:class:`PairUnions`: projection on the domain, region of pairs) and the measure
    (:class:`SampledVolume`, which works for any region with a membership function). The
    objective is the semantic one: ``gci0`` ... ``gci3`` and the role inclusion read as in the
    module, the role chain is the thesis' simplified loss. Verified equal to
    :class:`~mowl.nn.VolumeMultiBoxELModule` for these six losses, also with fresh tables from
    the same ``seed``.

    Not covered: the evaluation distance of Section 7.1.2
    (:meth:`~mowl.nn.VolumeMultiBoxELModule.distance`, a metric, not a loss), and
    ``eval_points_per_box``. ``gci0_bot`` and ``gci3_bot`` are not defined by the module and
    raise an error; ``margin`` is ignored. ``gci1_bot``, ``class_assertion``,
    ``object_property_assertion`` and the negative loss of ``gci2`` are not defined by the
    module either (it raises), but exist here as extensions (the fraction of :math:`C`
    covered by :math:`D`; the inclusion of the individual boxes), because the measure has
    ``apart`` and the geometry has individuals. They are not verified against anything.

    :param tables: A :class:`~mowl.nn.VolumeMultiBoxELModule` to share parameters (and
        point sets) with.
    """
    t = tables
    if t is not None:
        classes = PackedUnionClasses(table=t.class_embed)
        relations = PairUnions(table=t.rel_embed, bound=t.bound)
        individuals = PackedUnionClasses(table=t.ind_embed) if t.ind_embed is not None \
            else None
        measure = SampledVolume([t.points, t.role_points], bound=t.bound,
                                temperature=t.temperature, aggregate=t.aggregate,
                                chunk_size=t.point_chunk_size)
    else:
        gen = th.Generator().manual_seed(seed if init_seed is None else init_seed)
        classes = PackedUnionClasses(nb_classes, embed_dim, num_boxes, bound, generator=gen)
        relations = PairUnions(nb_relations, embed_dim, num_boxes, bound, generator=gen)
        individuals = None
        if nb_individuals:
            individuals = PackedUnionClasses(table=th.nn.Parameter(
                _packed_init(nb_individuals, 1, embed_dim, bound, gen)))
        from mowl.nn.el.mcmultibox import volume as V
        points = [V.sobol_points(num_points, embed_dim, bound, seed),
                  V.sobol_points(num_points, 2 * embed_dim, bound, seed + 1)]
        measure = SampledVolume(points, bound=bound, temperature=temperature,
                                aggregate=aggregate, chunk_size=point_chunk_size)
    geometry = ComposedGeometry(classes, relations, measure, Objective(negatives="gci2"),
                                individuals=individuals, neg_capable_gcis=frozenset())
    geometry.margin_semantics = None
    return geometry


# ------------------------------------------------------------------------------------
# adapters
# ------------------------------------------------------------------------------------

@register_adapter("BoxSquaredELModule")
def _box2el(module):
    return box2el(None, None, embed_dim=module.embed_dim, margin=module.gamma,
                  delta=module.delta, reg_factor=module.reg_factor,
                  all_negatives=module.all_negative_losses, epsilon=module.epsilon,
                  tables=module)


@register_adapter("MultiBoxELModule")
def _multiboxel(module):
    return multiboxel(None, None, embed_dim=module.embed_dim, margin=module.gamma,
                      n_components=module.n_components, delta=module.delta,
                      reg_factor=module.reg_factor, component_reg=module.component_reg,
                      component_aggregation=module.component_aggregation,
                      diversity_reg=module.diversity_reg, tables=module)


@register_adapter("MultiELBEModule")
def _multielbe(module):
    return multielbe(None, None, embed_dim=module.embed_dim, margin=module.margin,
                     n_components=module.n_components, component_reg=module.component_reg,
                     component_aggregation=module.component_aggregation,
                     diversity_reg=module.diversity_reg, tables=module)


@register_adapter("VolumeMultiBoxELModule")
def _mcmultibox(module):
    return mcmultibox(None, None, embed_dim=module.embed_dim, tables=module)
