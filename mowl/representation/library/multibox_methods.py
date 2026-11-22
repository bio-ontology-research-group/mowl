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
``tables=module`` (for the direct implementations in :mod:`mowl.baselines`). The public modules
(:class:`~mowl.nn.BoxSquaredELModule`, :class:`~mowl.nn.MultiBoxELModule`,
:class:`~mowl.nn.MultiELBEModule`, :class:`~mowl.nn.VolumeMultiBoxELModule`) are these
combinations: each initialises itself with the function of its method (``into=module``) and keeps
the names of its tables (:mod:`~mowl.representation.library.multibox_modules`).
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
from mowl.representation.library.structural_measure import SlotMeasure
from mowl.representation.library.structural_methods import StructuralGeometry
from mowl.representation.library.structural_points import PointReading, PointTerms, SoftMembership
from mowl.representation.library.structural_reading import StructuralReading
from mowl.representation.library.structural_regions import SlotClasses

__all__ = ["box2el", "multiboxel", "multielbe", "mcmultibox"]


#: The forms that have a negative loss with ``all_negatives=True`` (DELE: the concept forms,
#: not the class assertion).
_DELE_NEGATIVES = ("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
                   "object_property_assertion")


def _build(geometry_class, into, *args, **kwargs):
    """The geometry of a method: a new ``geometry_class``, or, with ``into``, the existing
    uninitialised public module (a subclass of ``geometry_class``, such as
    :class:`~mowl.nn.BoxSquaredELModule`) initialised with these parts, so that the module
    *is* the combination."""
    if into is None:
        return geometry_class(*args, **kwargs)
    geometry_class.__init__(into, *args, **kwargs)
    return into


def _BOUNDS():
    """Bounds as the union methods use them (the published readings)."""
    return Bounds(separation="every", extent="norm", conjunction="nonempty")


def _head_tail(nb_relations, embed_dim, t):
    if t is None:
        return HeadTailRelations(nb_relations, embed_dim)
    return HeadTailRelations(head_centres=t.head_center, head_offsets=t.head_offset,
                             tail_centres=t.tail_center, tail_offsets=t.tail_offset)


@register_method("box2el")
def box2el(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.0, delta=2.0,
           reg_factor=0.05, all_negatives=False, epsilon=0.01, tables=None, into=None):
    """Box2EL (Jackermeier et al. 2024) with the negative losses of DELE.

    Boxes with a bump vector, relations as a head box and a tail box, bounds per coordinate
    with a slack margin ``margin`` (the paper's :math:`\\gamma`), positive losses squared,
    the negative loss of :math:`C \\sqsubseteq \\exists r.D` is :math:`(\\delta - \\mu)^2`
    per argument, and the norm penalty on the bump vectors.

    This is the module :class:`~mowl.nn.BoxSquaredELModule` and the class
    :class:`~mowl.models.BoxSquaredEL`; both are tested against the direct implementation in
    :mod:`mowl.baselines` (``tests/baselines``): every positive loss of the forms ``gci0`` ...
    ``gci3_bot``, ``class_assertion``, ``object_property_assertion``, the role inclusion and
    role chain losses, the negative losses of ``gci2`` and ``object_property_assertion`` and,
    with ``all_negatives=True``, the DELE negative losses of all forms, the regularisation
    loss and three epochs of training are equal.

    :param tables: A module with the tables of Box2EL (such as the direct implementation in
        :mod:`mowl.baselines`) to share parameters with.
    :param into: The public module (:class:`~mowl.nn.BoxSquaredELModule`) to initialise with
        these parts.
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
        negatives=_DELE_NEGATIVES if all_negatives else "gci2", squared=True, delta=delta,
        gci3_bot_negative="filler", strict_negatives=False)
    return _build(ComposedGeometry, into, classes, _head_tail(nb_relations, embed_dim, t),
                  _BOUNDS(), objective, individuals=individuals,
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


def _points(reading, t, embed_dim, beta, neg_weight, ind_halfwidth, volume_reg,
            supervision):
    """The point terms of a point reading: the soft membership, the weights, and (from the
    module or ``supervision``) the points of the canonical elements."""
    terms = PointTerms(None, embed_dim, SoftMembership(beta), neg_weight, ind_halfwidth,
                       volume_reg)
    if t is not None and getattr(t, "point_supervision", None) is not None:
        terms.set_supervision(t.point_supervision, getattr(t, "elem_embed", None))
    elif supervision is not None:
        terms.set_supervision(supervision)
    return terms


@register_method("multiboxel")
def multiboxel(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.0,
               n_components=2, delta=2.0, reg_factor=0.05, component_reg=0.0,
               component_aggregation="hard", tau=1.0, diversity_reg=0.0, epsilon=0.01,
               reading="axiom", beta=0.1, neg_weight=0.1, ind_halfwidth=None, volume_reg=0.0,
               max_components=None, existence_weight=1.0, size_reg=1e-3, bot_agg="mean",
               witness_weight=0.0, supervision=None, size_reg_without_tables=False,
               structural_definitions=True, tables=None, into=None):
    """MultiBoxEL (Federico, Leemhuis and Kutz 2026): Box2EL with every class a union of
    ``n_components`` boxes (one bump vector per class), in the five readings of the module's
    ``objective`` (here ``reading``):

    ``"axiom"`` (default)
        The module :class:`~mowl.nn.MultiBoxELModule` and the class
        :class:`~mowl.models.MultiBoxEL` are this combination; they are tested against the
        direct implementation in :mod:`mowl.baselines` at ``n_components`` 1 to 3 with
        ``component_aggregation`` ``"hard"`` and ``"soft"`` (``tau``), ``component_reg``,
        ``diversity_reg``, ``reg_factor``, ``delta`` and ``gamma`` (here ``margin``); all
        forms with their positive loss, the negative losses of ``gci2`` and
        ``object_property_assertion``, and the regularisation loss.
    ``"structural"``
        The structural reading (:class:`StructuralReading`: squared hinges with
        non-emptiness gates) on slots of boxes (:class:`SlotClasses`, ``max_components``
        slots) with a definition table (:meth:`StructuralGeometry.set_structural_tables`:
        defined classes are the intersections of their definers,
        :class:`DefinedClasses`) and the penalties existence (``existence_weight``), size
        (``size_reg``, applied once the tables are installed; ``size_reg_without_tables=True``
        applies it before that too, which shrinks ``owl:Thing``: the module's earlier
        behaviour) and witness (``witness_weight``). ``margin`` is the tightening margin
        :math:`m` of the module (``margin``, not ``gamma``); ``bot_agg``. Hard aggregation
        only. Re-seeding is a regime (see :mod:`structural_methods`).
    ``"point"``, ``"mixed"``, ``"hybrid"``
        The axiom reading with the forms the module switches off while training
        (:class:`PointReading`), the class assertion read as :math:`-\\log m_C(a)`, and the
        point loss in ``geometry.point_terms`` (:class:`PointTerms`: ``beta``,
        ``neg_weight``, ``ind_halfwidth``, ``volume_reg``; the supervision from the module or
        ``supervision``).

    The DELE negative losses of the other forms are not offered (the module has none).

    :param structural_definitions: ``"structural"``: whether the rows that state a definition
        :math:`A \\equiv X \\sqcap Y` are skipped by the losses (the objective's ``definitions``).
    :param tables: A module with the tables of MultiBoxEL (the direct implementation in
        :mod:`mowl.baselines`) to share parameters with (and, for ``"structural"``, its
        definition table and targets).
    :param into: The public module (:class:`~mowl.nn.MultiBoxELModule`) to initialise with
        these parts.
    """
    t = tables
    if reading not in ("axiom", "structural", "point", "mixed", "hybrid"):
        raise ValueError("reading must be 'axiom', 'structural', 'point', 'mixed' or "
                         "'hybrid'.")
    names = ("class_center", "class_offset", "bump_classes", "ind_center", "ind_offset",
             "bump_individuals")
    regularizers = [bump_penalty(reg_factor)] + _union_regularizers(component_reg,
                                                                    diversity_reg)
    if reading == "structural":
        if component_aggregation != "hard":
            raise ValueError("component_aggregation must be 'hard' for the structural "
                             "reading.")
        centres, offsets, bumps, ind_centres, ind_offsets, ind_bumps = (
            None if t is None else getattr(t, n, None) for n in names)
        classes = SlotClasses(nb_classes, embed_dim, n_components, max_components,
                              bump=True, centres=centres, offsets=offsets, bumps=bumps) \
            if t is None else SlotClasses(components=n_components,
                                          max_components=max_components, centres=centres,
                                          offsets=offsets, bumps=bumps)
        individuals = None
        if t is None and nb_individuals:
            individuals = UnionClasses(nb_individuals, embed_dim, 1, bump=True)
        elif ind_centres is not None:
            individuals = UnionClasses(components=1, centres=ind_centres,
                                       offsets=ind_offsets, bumps=ind_bumps)
        objective = StructuralReading(
            margin, bot_agg=bot_agg, definitions=structural_definitions, epsilon=epsilon,
            gci2_negative="delta", delta=delta, squared=("gci3", "gci3_bot"),
            gci3_bot_negative="filler")
        geometry = _build(
            StructuralGeometry, into, classes, _head_tail(nb_relations, embed_dim, t),
            SlotMeasure(_BOUNDS(), reading="structural"), objective, individuals=individuals,
            base_regularizers=regularizers, existence_weight=existence_weight,
            size_reg=size_reg, witness_weight=witness_weight,
            size_reg_without_tables=size_reg_without_tables)
        if into is not None:
            geometry.point_terms = _points(reading, t, embed_dim, beta, neg_weight,
                                           ind_halfwidth, volume_reg, supervision)
        if t is not None:
            geometry.tables_installed = bool(getattr(t, "_tables_installed", False))
            geometry.reseed_options.update(mode=t.reseed_mode, size=t.reseed_size,
                                           batch=t.reseed_batch)
            geometry.witness_embed = t.witness_embed
            if t.definition_table is not None or t.existence_targets is not None:
                geometry.set_structural_tables(t.definition_table, t.existence_targets,
                                               t.top_idx)
        return geometry
    classes, individuals = _union_parts(nb_classes, embed_dim, n_components, nb_individuals,
                                        t, names, bump=True)
    if t is not None:
        tau = t.tau
    options = dict(slack=True, epsilon=epsilon, gci3="inclusion", gci2_negative="delta",
                   negatives="gci2", squared=True, delta=delta, gci3_bot_negative="filler",
                   strict_negatives=False)
    terms = None
    if reading == "axiom":
        objective = Objective(margin, **options)
    if reading != "axiom" or into is not None:
        terms = _points(reading, t, embed_dim, beta if t is None else t.beta, neg_weight
                        if t is None else t.neg_weight, ind_halfwidth if t is None
                        else t.ind_halfwidth, volume_reg if t is None else t.volume_reg,
                        supervision)
    if reading != "axiom":
        objective = PointReading(terms, reading, margin, **options)
    measure = UnionMeasure(_BOUNDS(), component_aggregation, tau, meets_reading="every")
    geometry = _build(ComposedGeometry, into, classes,
                      _head_tail(nb_relations, embed_dim, t), measure, objective,
                      individuals=individuals, regularizers=regularizers)
    if terms is not None:
        geometry.point_terms = terms
    return geometry


@register_method("multielbe")
def multielbe(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1,
              n_components=1, component_reg=0.0, component_aggregation="hard", tau=1.0,
              diversity_reg=0.0, epsilon=0.01, gci3="overlap", reading="axiom", beta=0.1,
              neg_weight=0.1, ind_halfwidth=None, volume_reg=0.0, supervision=None,
              tables=None, into=None):
    """MultiELBE (Federico, Leemhuis and Kutz 2026): ELBE with every class a union of
    ``n_components`` boxes (translations as relations, strict margin, overlap reading of
    :math:`\\exists r.C \\sqsubseteq D`).

    The module :class:`~mowl.nn.MultiELBEModule` and the class
    :class:`~mowl.models.MultiELBE` are this combination, tested against the direct
    implementation in :mod:`mowl.baselines` at ``n_components`` 1
    and 2 to 3: ``objective="axiom"`` with ``component_aggregation`` ``"hard"`` and
    ``"soft"`` (``tau``), ``component_reg``, ``diversity_reg``, ``margin``; all forms with
    their positive loss, the negative losses of ``gci2`` and ``object_property_assertion``
    (mean over all pairs of components), and the regularisation loss. ``gci3="inclusion"``
    gives the corrected reading, which the module does not have.

    ``reading`` ``"point"``, ``"mixed"``, ``"hybrid"``: the axiom reading with the forms the
    module switches off while training (:class:`PointReading`), the class assertion read as
    :math:`-\\log m_C(a)`, the point loss in ``geometry.point_terms`` (:class:`PointTerms`:
    the membership of :math:`x + r` in the target for a role triple; ``beta``, ``neg_weight``,
    ``ind_halfwidth``, ``volume_reg``). There is no structural reading of this module.

    :param tables: A module with the tables of MultiELBE (the direct implementation in
        :mod:`mowl.baselines`) to share parameters with.
    :param into: The public module (:class:`~mowl.nn.MultiELBEModule`) to initialise with
        these parts.
    """
    t = tables
    if reading not in ("axiom", "point", "mixed", "hybrid"):
        raise ValueError("reading must be 'axiom', 'point', 'mixed' or 'hybrid'.")
    classes, individuals = _union_parts(
        nb_classes, embed_dim, n_components, nb_individuals, t,
        ("class_embed", "class_offset", "none", "ind_embed", "ind_offset", "none"),
        bump=False)
    relations = Translation(nb_relations, embed_dim) if t is None else Translation(
        vectors=t.rel_embed)
    if t is not None:
        tau = t.tau
    options = dict(slack=False, epsilon=epsilon, gci3=gci3, gci2_negative="overlap",
                   negatives="gci2", strict_negatives=False)
    terms = None
    if reading == "axiom":
        objective = Objective(margin, **options)
    if reading != "axiom" or into is not None:
        terms = _points(reading, t, embed_dim, beta if t is None else t.beta, neg_weight
                        if t is None else t.neg_weight, ind_halfwidth if t is None
                        else t.ind_halfwidth, volume_reg if t is None else t.volume_reg,
                        supervision)
    if reading != "axiom":
        objective = PointReading(terms, reading, margin, **options)
    measure = UnionMeasure(_BOUNDS(), component_aggregation, tau, meets_reading="every")
    geometry = _build(ComposedGeometry, into, classes, relations, measure, objective,
                      individuals=individuals,
                      regularizers=_union_regularizers(component_reg, diversity_reg))
    if terms is not None:
        geometry.point_terms = terms
    return geometry


@register_method("mcmultibox")
def mcmultibox(nb_classes, nb_relations, nb_individuals=0, embed_dim=5, margin=0.0,
               num_boxes=5, bound=1.0, num_points=20000, temperature=1.0, aggregate="min",
               seed=0, point_chunk_size=2048, init_seed=None, tables=None, into=None,
               num_batches=5):
    """The volume-based MultiBoxEL (Aharoni 2024): classes are unions of boxes, relations are
    unions of boxes over pairs, and every loss is :math:`1 - \\mathrm{Vol}(A \\cap B) /
    \\mathrm{Vol}(A)` estimated on a fixed set of sampled points.

    The class model is :class:`PackedUnionClasses`, the relation model :class:`PairUnions`
    (projection on the domain, region of pairs) and the measure :class:`SampledVolume`, which
    works for any region with a membership function. The objective is the semantic one:
    ``gci0`` ... ``gci3`` and the role inclusion read as in the thesis, the role chain is the
    thesis' simplified loss. The module :class:`~mowl.nn.VolumeMultiBoxELModule` and the
    class :class:`~mowl.models.VolumeMultiBoxEL` are this combination; they are tested
    against the direct implementation in :mod:`mowl.baselines` for these six losses, the
    evaluation distance and the training run, also with fresh tables from the same ``seed``.

    The evaluation distance of Section 7.1.2 (:meth:`~mowl.nn.VolumeMultiBoxELModule.distance`,
    a metric, not a loss) is
    :func:`~mowl.representation.library.multibox_distance.sampled_distance` on these parts.
    ``gci0_bot`` and ``gci3_bot`` are not defined by the thesis and raise an error in the
    module; ``margin`` is ignored. ``gci1_bot``, ``class_assertion``,
    ``object_property_assertion`` and the negative loss of ``gci2`` are not defined by the
    module either (it raises), but exist here as extensions (the fraction of :math:`C`
    covered by :math:`D`; the inclusion of the individual boxes), because the measure has
    ``apart`` and the geometry has individuals. They are not verified against anything.

    :param num_batches: Steps per epoch of the default regime
        :class:`~mowl.training.splits.ShuffledSplits`.
    :param tables: A module with the tables of the volume-based MultiBoxEL (the direct
        implementation in :mod:`mowl.baselines`) to share parameters (and point sets) with.
    :param into: The public module (:class:`~mowl.nn.VolumeMultiBoxELModule`) to initialise
        with these parts.
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
    objective = Objective(negatives="gci2", strict_negatives=False)
    geometry = _build(ComposedGeometry, into, classes, relations, measure, objective,
                      individuals=individuals, neg_capable_gcis=frozenset())
    geometry.margin_semantics = None
    # an epoch is num_batches steps on a shuffled split of the axioms (a step of a model that
    # trains with ComposedEL is one epoch of the generic loop: train for epochs * num_batches)
    from functools import partial
    from mowl.training.elkem import ValidationCheckpoint
    from mowl.training.splits import ShuffledSplits
    geometry.default_regimes = (
        partial(ShuffledSplits, num_batches, seed,
                forms=("gci0", "gci1", "gci2", "gci3", "role_inclusion", "role_chain"),
                what="the thesis"),
        partial(ValidationCheckpoint, "loss", steps_per_epoch=num_batches))
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
    """The geometry of the module in its own objective (``module.objective``); with the
    structural objective the geometry has the module's definition table, targets and
    witness points as they are *now* (call again after ``set_structural_tables``)."""
    structural = module.objective == "structural"
    return multiboxel(None, None, embed_dim=module.embed_dim,
                      margin=module.margin if structural else module.gamma,
                      n_components=module.n_components, delta=module.delta,
                      reg_factor=module.reg_factor, component_reg=module.component_reg,
                      component_aggregation=module.component_aggregation,
                      diversity_reg=module.diversity_reg, reading=module.objective,
                      max_components=module.max_components,
                      existence_weight=module.existence_weight, size_reg=module.size_reg,
                      bot_agg=module.bot_agg, witness_weight=module.witness_weight,
                      size_reg_without_tables=getattr(module, "size_reg_without_tables", False),
                      structural_definitions=getattr(module, "structural_definitions", True),
                      tables=module)


@register_adapter("MultiELBEModule")
def _multielbe(module):
    return multielbe(None, None, embed_dim=module.embed_dim, margin=module.margin,
                     n_components=module.n_components, component_reg=module.component_reg,
                     component_aggregation=module.component_aggregation,
                     diversity_reg=module.diversity_reg, reading=module.objective,
                     tables=module)


@register_adapter("VolumeMultiBoxELModule")
def _mcmultibox(module):
    return mcmultibox(None, None, embed_dim=module.embed_dim, tables=module)
