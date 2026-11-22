"""The structural reading of the axioms, and its existence, size and witness terms.

The reading (:class:`StructuralReading`) is an :class:`~mowl.representation.Objective` that
reads the concept forms as the structural objective of MultiBoxEL and PrismEL does:

* :math:`C \\sqsubseteq D` and :math:`C \\sqcap D \\sqsubseteq E`: a *squared hinge* of the
  tightened inclusion (margin :math:`m \\ge 0` tightens), gated by the **non-emptiness** of the
  source (detached: an inclusion never pushes a region to empty); a region with several slots
  averages over its non-empty slots and each is included in *some* slot of the target;
* :math:`C \\sqsubseteq \\bot` and :math:`C \\sqcap D \\sqsubseteq \\bot`: the squared positive part
  of the non-emptiness measure (of the intersection, plus the margin, for the latter; with
  several slots the mean or sum over the pairs of non-empty slots, ``bot_agg``);
* rows that *state a definition* are zero (the class model says which:
  :class:`~mowl.representation.library.structural_regions.DefinedClasses`);
* the forms about relations and assertions are read by the relation model and the measure as
  in the axiom reading, with the margin as a tightening one and the negative reading of the
  module (the margin of a negative is the same number, ``gci2_negative`` an option).

Non-emptiness of the classes that must be satisfiable is **not** a term of the axioms: it is
:class:`ExistencePenalty` (and :class:`WitnessPenalty`, :class:`SizePenalty`), penalties in the
sense of :class:`mowl.training.Regularizer`, which a geometry sums with its other
regularisers.

The reading works for every region family that has a *violation vector* of the inclusion and
a *size* (the signed non-emptiness measure): boxes (and so unions of boxes and slots, whose
boxes are the *pieces* of the region), prisms, products of octagons and any family registered
with :func:`register_family`. Every form is written once, on pieces: a region of a family is
one piece; a union of boxes with slots has a box per slot, of which the active ones count.
"""

import torch as th

from dataclasses import fields, replace

from mowl.representation.library.multibox_parts import UnionOfBoxes
from mowl.representation.library.structural_regions import (intersect_signed, largest_side,
                                                             side, slot_depth, slot_pairs,
                                                             to_slots)
from mowl.representation.objectives import Objective
from mowl.representation.regions import Box
from mowl.training.regularization import Regularizer

__all__ = ["StructuralReading", "register_family", "ExistencePenalty", "SizePenalty",
           "WitnessPenalty", "size_of", "pair_size_of"]

#: Region class -> ``(violation, size)`` for the families that are not unions of boxes.
FAMILIES = {}


def register_family(region_class, violation, size):
    """Make the structural reading available for a region family.

    :param violation: ``(a, b, margin) -> (B, n)``: the non-negative violations of
        :math:`a \\subseteq b` with margin (zero iff included; their squared norm is the loss).
    :param size: ``(a) -> (B,)``: the signed non-emptiness measure, positive iff the region
        has non-empty interior, negative (by how much) where it is empty.
    """
    FAMILIES[region_class] = (violation, size)


def _prism():
    from mowl.representation.library.shapes_regions import Prism

    def violation(a, b, margin):
        return th.cat([th.relu(a.supports - b.supports + margin),
                       th.relu(th.abs(a.centre - b.centre) + a.offset - b.offset + margin)],
                      dim=-1)

    def size(a):
        return th.minimum(a.supports.sum(dim=-1), 2.0 * a.offset.amin(dim=-1))

    register_family(Prism, violation, size)


def _octagon():
    import mowl.nn.el.octagon.geometry as G
    from mowl.representation.library.shapes_regions import OctagonProduct

    def violation(a, b, margin):
        return th.relu(G.tighten(a.bounds) - b.bounds + margin).flatten(1)

    def size(a):
        return G.product_emptiness_slack(a.bounds)

    register_family(OctagonProduct, violation, size)


def _box():
    def violation(a, b, margin):
        return th.relu((a.centre - b.centre).abs() + a.offset - b.offset + margin)

    def size(a):
        return side(a.offset)

    register_family(Box, violation, size)


_box()
_prism()
_octagon()


def _family(region):
    for cls in type(region).__mro__:
        if cls in FAMILIES:
            return FAMILIES[cls]
    return None


def _is_slots(region):
    return isinstance(region, (UnionOfBoxes, Box))


# -- pieces ----------------------------------------------------------------------------

def _count(region):
    for f in fields(region):
        value = getattr(region, f.name)
        if isinstance(value, th.Tensor):
            return value.shape[0]
    raise TypeError(f"{type(region).__name__} has no tensors.")


def _pieces(region, counting_only=False):
    """``(pieces, active)``: the region as ``k`` pieces per row, ``pieces`` a flat batch of
    ``B * k`` regions of one family (row-major) and ``active`` ``(B, k)`` which count. A box,
    union of boxes or slots: its boxes of signed half-width (``counting_only``: the active
    slots with interior only); any other family: the region itself, one piece per row."""
    if _is_slots(region):
        s = to_slots(region)
        batch, k, dim = s.centre.shape
        active = s.active & (side(s.w.detach()) > 0.0) if counting_only else s.active
        return Box(s.centre.reshape(-1, dim), s.w.reshape(-1, dim)), active
    return region, th.ones(_count(region), 1, dtype=th.bool,
                           device=_first_tensor(region).device)


def _first_tensor(region):
    return next(getattr(region, f.name) for f in fields(region)
                if isinstance(getattr(region, f.name), th.Tensor))


def _meet(a, b, counting_only=False):
    """``(pieces, active)`` of the intersections of every piece of ``a`` with every piece of
    ``b`` (a pair counts iff both pieces do)."""
    if _is_slots(a):
        sa, sb = to_slots(a), to_slots(b)
        if counting_only:
            sa = replace(sa, active=sa.active & (side(sa.w.detach()) > 0.0))
            sb = replace(sb, active=sb.active & (side(sb.w.detach()) > 0.0))
        centre, w, active = slot_pairs(sa, sb)
        dim = centre.shape[-1]
        return Box(centre.reshape(-1, dim), w.reshape(-1, dim)), active
    inter, _ = intersect_signed(a, b)
    return inter, th.ones(_count(inter), 1, dtype=th.bool,
                          device=_first_tensor(inter).device)


def _take(region, index):
    """The rows ``index`` of every tensor of a region."""
    return replace(region, **{f.name: getattr(region, f.name)[index] for f in fields(region)
                              if isinstance(getattr(region, f.name), th.Tensor)})


def _sizes(pieces, active):
    """``(B, k)``: the size of every piece."""
    return _family_of(pieces)[1](pieces).reshape(active.shape)


def _covered(source, source_active, target, target_active, margin):
    """``(B, ks)``: for every piece of the source, the squared violation of its inclusion in
    the best counting piece of the target."""
    batch, ks = source_active.shape
    kt = target_active.shape[1]
    rows = th.arange(batch, device=source_active.device)
    i = (rows[:, None, None] * ks + th.arange(ks, device=rows.device)[None, :, None])
    j = (rows[:, None, None] * kt + th.arange(kt, device=rows.device)[None, None, :])
    i, j = i.expand(batch, ks, kt).reshape(-1), j.expand(batch, ks, kt).reshape(-1)
    violation = _family_of(source)[0]
    if ks == kt == 1:
        squares = violation(source, target, margin).square().sum(dim=-1)
    else:
        squares = violation(_take(source, i), _take(target, j), margin).square().sum(dim=-1)
    squares = squares.view(batch, ks, kt)
    if batch and not bool(target_active.any(dim=1).all()):
        raise ValueError("The inclusion needs at least one active slot of the target in "
                         "every row.")
    return squares.masked_fill(~target_active[:, None, :], float("inf")).amin(dim=2)


def _mean_where(values, gate):
    """``(B,)``: the mean of ``values`` ``(B, k)`` over the entries of ``gate`` (zero where
    there is none)."""
    total = th.where(gate, values, th.zeros_like(values)).sum(dim=1)
    return total / gate.to(values.dtype).sum(dim=1).clamp_min(1.0)


def _hinge_mean(values):
    """The mean of ``relu(values)**2``, zero for an empty tensor."""
    if values.numel() == 0:
        return values.new_zeros(())
    return th.relu(values).square().mean()


def size_of(region):
    """``(B,)``: the non-emptiness measure of the class regions: the largest size among its
    counting pieces (for slots the largest of the slots' smallest signed sides, for the other
    families their registered ``size``)."""
    pieces, active = _pieces(region)
    return largest_side(_sizes(pieces, active), active)


def pair_size_of(a, b):
    """``(B,)``: the non-emptiness measure of :math:`a \\cap b`: the largest size among the
    intersections of a counting piece of ``a`` with one of ``b``."""
    pieces, active = _meet(a, b)
    return largest_side(_sizes(pieces, active), active)


def _family_of(region):
    found = _family(region)
    if found is None:
        raise TypeError(
            f"The structural reading has no violation and size for {type(region).__name__}; "
            f"register_family(...) provides them.")
    return found


class _View:
    """The geometry as an objective sees it, with the class regions passed through ``fix``."""

    def __init__(self, geometry, fix):
        self._geometry, self._fix = geometry, fix

    def __getattr__(self, name):
        return getattr(self._geometry, name)

    def classes(self, ids, frozen=False):
        return self._fix(self._geometry.classes(ids, frozen))


def _abs_offsets(region):
    from dataclasses import replace
    offset = getattr(region, "offset", None)
    if offset is None or isinstance(region, UnionOfBoxes):
        return region
    return replace(region, offset=th.abs(offset))


class StructuralReading(Objective):
    """The structural reading of the concept forms (see the module docstring); the other
    forms and all options are those of :class:`~mowl.representation.Objective`.

    :param margin: :math:`m`, the tightening margin of every form (strict: the objective's
        ``slack`` is ``False``).
    :param bot_agg: ``"mean"`` or ``"sum"``: the aggregation over pairs of slots of the loss
        of :math:`C \\sqcap D \\sqsubseteq \\bot` (unions of boxes).
    :param definitions: Skip the rows that state a definition (zero loss) where the class
        model has definitions (``masked_gci0``, ``masked_gci1``).
    :param empty_regions: How the forms about relations and assertions see a region of
        negative extent (a defined class that is empty): ``"keep"`` (default: the region as
        it is, signed; a region that is not slots and is empty makes the rows with it as
        source (``gci2``) or filler (``gci3``) vacuous, a negative ``gci2`` of an empty source
        or target has no cost, and a union of boxes with slots ignores the empty slots by
        itself) or ``"abs"`` (the absolute value of the half-widths, so an empty box is a
        phantom non-empty one: the earlier reading of the prism module, which the ``prism``
        builder selects with ``empty_box_reading="absolute"``).
    """

    def __init__(self, margin=0.02, *, bot_agg="mean", definitions=True,
                 empty_regions="keep", **options):
        if bot_agg not in ("mean", "sum"):
            raise ValueError("bot_agg must be 'mean' or 'sum'.")
        if empty_regions not in ("keep", "abs"):
            raise ValueError("empty_regions must be 'keep' or 'abs'.")
        options.setdefault("strict_negatives", False)
        options.setdefault("negatives", "gci2")
        super().__init__(margin, **options)
        self.bot_agg = bot_agg
        self.definitions = bool(definitions)
        self.empty_regions = empty_regions

    # -- helpers --------------------------------------------------------------------------
    def _masked(self, g, name, rows):
        mask = getattr(g.class_model, name, None)
        if mask is None or not self.definitions:
            return None
        return mask(rows)

    def _view(self, g):
        return _View(g, _abs_offsets) if self.empty_regions == "abs" else g

    # -- the concept forms: written once, on pieces ------------------------------------
    def _included(self, g, rows, source, source_active, target, extra_gate=None):
        """Every counting piece of the source with interior (detached) included in some
        counting piece of the target: the mean of the squared violations."""
        target_pieces, target_active = _pieces(target)
        best = _covered(source, source_active, target_pieces, target_active, self.m)
        gate = source_active & (_sizes(_detached(source), source_active) > 0.0)
        if extra_gate is not None:
            gate = gate & extra_gate[:, None]
        return _mean_where(best, gate)

    def gci0(self, g, rows, neg=False):
        c, d = g.classes(rows[:, 0]), g.classes(rows[:, 1])
        source, active = _pieces(c)
        cost = self._included(g, rows, source, active, d)
        masked = self._masked(g, "masked_gci0", rows)
        if masked is not None:
            cost = th.where(masked, th.zeros_like(cost), cost)
        return cost.reshape(-1, 1)

    def gci0_bot(self, g, rows, neg=False):
        pieces, active = _pieces(g.classes(rows[:, 0]))
        cost = _mean_where(th.relu(_sizes(pieces, active)).square(), active)
        return cost.reshape(-1, 1)

    def gci1(self, g, rows, neg=False):
        c, d, e = (g.classes(rows[:, i]) for i in range(3))
        masked = self._masked(g, "masked_gci1", rows)
        source, active = _meet(c, d)
        free = None if masked is None else ~masked.bool()
        cost = self._included(g, rows, source, active, e, extra_gate=free)
        if free is not None:
            cost = th.where(free, cost, th.zeros_like(cost))
        return cost.reshape(-1, 1)

    def gci1_bot(self, g, rows, neg=False):
        c, d = g.classes(rows[:, 0]), g.classes(rows[:, 1])
        pieces, active = _meet(c, d, counting_only=_is_slots(c))
        terms = th.relu(_sizes(pieces, active) + self.m).square()
        if self.bot_agg == "sum":
            cost = th.where(active, terms, th.zeros_like(terms)).sum(dim=1)
        else:
            cost = _mean_where(terms, active)
        return cost.reshape(-1, 1)

    # -- the forms about relations and assertions: the axiom reading on the region view ----
    def _nonempty(self, g, region):
        """``(B, 1)`` boolean (detached): the region is not empty (``empty_regions="keep"``
        for a family that is not slots, whose empty slots are ignored by themselves);
        ``None`` where no gate applies."""
        if self.empty_regions != "keep" or _is_slots(region):
            return None
        return (_family_of(region)[1](_detached(region)) > 0.0).reshape(-1, 1)

    @staticmethod
    def _gated(cost, gate):
        return cost if gate is None else th.where(gate, cost, th.zeros_like(cost))

    def gci2(self, g, rows, neg=False):
        cost = super().gci2(self._view(g), rows, neg)
        # An empty source is included in everything and nothing lies in an empty target: the
        # row is vacuous (positive), and a negative of it asks for what cannot hold (no cost).
        gate = self._nonempty(g, g.classes(rows[:, 0]))
        if neg and gate is not None:
            gate = gate & self._nonempty(g, g.classes(rows[:, 2]))
        return self._gated(cost, gate)

    def gci3(self, g, rows, neg=False):
        cost = super().gci3(self._view(g), rows, neg)
        # an empty filler: the region of (exists r.C) is empty and lies in everything
        return self._gated(cost, self._nonempty(g, g.classes(rows[:, 1])))

    def gci3_bot(self, g, rows, neg=False):
        cost = super().gci3_bot(self._view(g), rows, neg)
        filler = g.classes(rows[:, 1])
        if self.empty_regions == "keep" and hasattr(filler, "offset") \
                and hasattr(filler, "supports"):
            # a prism whose box is empty (the box part of the filler has no point)
            gate = (filler.offset.detach().amin(dim=-1) > 0.0).reshape(-1, 1)
            return self._gated(cost, gate)
        return cost

    def class_assertion(self, g, rows, neg=False):
        return super().class_assertion(self._view(g), rows, neg)

    def object_property_assertion(self, g, rows, neg=False):
        return super().object_property_assertion(self._view(g), rows, neg)


def _detached(region):
    """The region with detached tensors (a gate must not carry gradients)."""
    updates = {f.name: getattr(region, f.name).detach() for f in fields(region)
               if isinstance(getattr(region, f.name), th.Tensor)}
    return replace(region, **updates)


# ====================================================================================
# penalties: existence, size, witnesses
# ====================================================================================

def _number(value):
    return float(value() if callable(value) else value)


class ExistencePenalty(Regularizer):
    """The structural existence loss: every class in ``classes`` and every conjunction
    :math:`C \\sqcap D` of ``pairs`` that is satisfiable has to have a region of size at least
    ``margin``: :math:`\\operatorname{mean}_C \\max(0, m - \\mathrm{NE}(C))^2 +
    \\operatorname{mean}_{(C,D)} \\max(0, m - \\mathrm{NE}(C \\sqcap D))^2` (zero for an empty
    list), times ``weight``. Works for every family of :func:`size_of`.

    :param classes: ``(C,)`` class ids.
    :param pairs: ``(P, 2)`` class ids.
    :param margin: A number, or a function returning it (for a margin that lives elsewhere).
    :param weight: A number or a function.
    """

    kind = "penalty"

    def __init__(self, classes, pairs, margin, weight=1.0):
        super().__init__(parameters=(), weight=weight)
        self.classes_ids, self.pairs, self.margin = classes, pairs, margin

    def penalty(self, model):
        weight = _number(self.weight)
        if weight <= 0.0:
            return None
        device = model.device
        class_sizes = th.empty(0, device=device)
        pair_sizes = th.empty(0, device=device)
        if self.classes_ids.numel() > 0:
            class_sizes = size_of(model.classes(self.classes_ids.to(device)))
        if self.pairs.numel() > 0:
            ids = self.pairs.to(device)
            pair_sizes = pair_size_of(model.classes(ids[:, 0]), model.classes(ids[:, 1]))
        margin = _number(self.margin)
        return weight * (_hinge_mean(margin - class_sizes) + _hinge_mean(margin - pair_sizes))


class SizePenalty(Regularizer):
    """A small pull of the *primitive* classes towards small regions: ``weight`` times the
    mean of ``scales[i] * |table_i|`` (the tables concatenated per class) over the rows that
    ``rows(model)`` selects (the classes that are neither defined nor the top class).

    :param tables: ``model -> [tensor (N, n_i)]``: the raw half-width parameters.
    :param scales: One factor per table (``2`` for a half-width: the side length).
    :param rows: ``model -> (N,) boolean``.
    """

    kind = "penalty"

    def __init__(self, tables, scales, rows, weight=1e-3):
        super().__init__(parameters=(), weight=weight)
        self.tables, self.scales, self.rows = tables, tuple(scales), rows

    def penalty(self, model):
        weight = _number(self.weight)
        mask = self.rows(model)
        if weight <= 0.0 or not bool(mask.any()):
            return None
        parts = [scale * th.abs(t[mask]) for t, scale in zip(self.tables(model), self.scales)]
        return weight * th.cat(parts, dim=-1).mean()


class WitnessPenalty(Regularizer):
    """The structural witness loss. Every class :math:`C` that is satisfiable has a witness
    point :math:`p_C` (the rows of ``model.witness_embed``, parameters of the geometry); the
    penalty asks :math:`p_C` to lie at least :math:`m` inside the classes it belongs to
    (``targets.pos``) and at least :math:`m` outside those it must not (``targets.neg``):
    :math:`\\operatorname{mean} \\max(0, m - \\mathrm{inside})^2 + \\operatorname{mean}
    \\max(0, \\mathrm{inside} + m)^2`, with the signed depth
    :func:`~mowl.representation.library.structural_regions.slot_depth` in the classes (as
    slots: through defined classes the gradient reaches their definers).

    :param targets: A :class:`~mowl.nn.el.structural.WitnessTargets`.
    :param margin: A number or a function.
    :param weight: A number or a function (zero: no penalty).
    """

    kind = "penalty"

    def __init__(self, targets, margin, weight=1.0):
        super().__init__(parameters=(), weight=weight)
        self.targets, self.margin = targets, margin

    def penalty(self, model):
        weight = _number(self.weight)
        embed = getattr(model, "witness_embed", None)
        if weight <= 0.0 or embed is None:
            return None
        device = model.device
        depth = []
        for pairs in (self.targets.pos, self.targets.neg):
            pairs = pairs.to(device)
            if pairs.numel() == 0:
                depth.append(th.empty(0, device=device))
                continue
            depth.append(slot_depth(embed(pairs[:, 0]), to_slots(model.classes(pairs[:, 1]))))
        margin = _number(self.margin)
        return weight * (_hinge_mean(margin - depth[0]) + _hinge_mean(depth[1] + margin))
