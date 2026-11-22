"""The point objectives of MultiBoxEL and MultiELBE: a measure "sampled points", and the
terms that read the regions through points.

Under ``objective="point"`` (and ``"mixed"``, ``"hybrid"``) the modules do not compare regions
for the axioms. They draw points (the canonical elements of the ontology: a table of
coordinates ``elem_embed``, or the individuals) and ask whether a point belongs to a class
(soft membership, binary cross-entropy against the canonical model) and whether a pair of
points is in a relation:

:class:`SoftMembership`
    A measure (:class:`~mowl.representation.measures.RegionMeasure`): ``member(point, region)``
    is :math:`-\\log m`, ``apart(point, region)`` is :math:`-\\log(1 - m)` of the soft membership
    :math:`m` of the point in the region (the minimum over the coordinates of a sigmoid of the
    depth, the noisy-or of the components of a union; a ball: a sigmoid of the depth). It works
    for every region that has a soft membership (boxes, unions, balls).
:class:`PointTerms`
    The loss on a batch of sampled pairs and role triples (:meth:`~PointTerms.loss`) and the
    volume penalty, for any geometry whose classes are boxes or unions of boxes with
    (optional) bump vectors and whose relations are head and tail boxes.
:class:`PointReading`
    The objective that switches the forms: ``"point"``: while training, every concept form
    and the role assertion are zero (the points carry the supervision) and the class assertion
    is :math:`-\\log m_C(a)`; ``"mixed"``: the same for the concept forms of ``gci0``-``gci3``
    and role assertions, the forms with bottom keep their axiom loss; ``"hybrid"``: only the
    class assertion is the point score.
:class:`PointPenalty`
    :class:`~mowl.training.Regularizer` that adds the point loss to the regularisation loss of
    the geometry (what the model's ``extra_loss`` does).
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from mowl.representation.library.multibox_parts import UnionOfBoxes
from mowl.representation.measures import RegionMeasure
from mowl.representation.methods import register_measure
from mowl.representation.objectives import Objective
from mowl.representation.regions import Ball, Box
from mowl.training.regularization import Regularizer

__all__ = ["SoftMembership", "PointTerms", "PointReading", "PointPenalty", "VolumePenalty",
           "POINT_READINGS", "log_volume"]

#: The readings of the axioms of MultiBoxEL and MultiELBE besides ``"structural"``.
POINT_READINGS = ("axiom", "point", "mixed", "hybrid")


def log_volume(offsets, eps=1e-6):
    """The mean over boxes and coordinates of :math:`\\log(2|o| + \\varepsilon)`: the log
    side length (zero for no box). Small boxes make it very negative: as a penalty it pulls
    the classes towards small regions."""
    if offsets.numel() == 0:
        return offsets.new_zeros(())
    sides = th.log(offsets.abs().mul(2.0).add(eps))
    return sides.mean(dim=-1).mean()


def _bce(probability, target, weight=1.0, eps=1e-6):
    """Weighted binary cross-entropy of probabilities clamped to ``[eps, 1 - eps]`` (zero for
    an empty batch)."""
    if probability.numel() == 0:
        return probability.new_zeros(())
    p = probability.clamp(eps, 1.0 - eps)
    return F.binary_cross_entropy(p, th.full_like(p, float(target)),
                                  weight=th.full_like(p, float(weight)))


def _point(region):
    """The coordinates ``(B, d)`` of a point given as a region (the centre of a box, or of
    the only component of a union)."""
    centre = region.centre
    return centre[:, 0] if centre.dim() == 3 else centre


@register_measure("soft membership")
class SoftMembership(RegionMeasure):
    """Soft membership of points in regions; see the module docstring.

    :param beta: Temperature :math:`\\beta` of the sigmoid.
    :param eps: The membership is clamped to ``[eps, 1 - eps]`` in the logarithm.

    Part (vocabulary value): ``sampled volume``. Used by: ``multiboxel`` and ``multielbe`` with
    the point readings (``reading="point"``, ``"mixed"``, ``"hybrid"``).
    """

    accepts = (UnionOfBoxes, Box, Ball)
    part = "sampled volume"

    def __init__(self, beta=0.1, eps=1e-6):
        if beta <= 0:
            raise ValueError("beta must be positive.")
        self.beta, self.eps = float(beta), float(eps)

    def probability(self, x, region):
        """``(B,)``: the soft membership of the points ``x`` ``(B, d)`` in ``region`` (one
        region per point)."""
        if isinstance(region, Ball):
            depth = region.radius.reshape(-1) - th.linalg.norm(x - region.centre, dim=-1)
            return th.sigmoid(depth / self.beta)
        if not isinstance(region, UnionOfBoxes):
            return self.box_probability(x, region.centre[:, None], region.offset[:, None])[:, 0]
        m = self.box_probability(x, region.centre, region.offset)
        active = getattr(region, "active", None)
        if active is not None:
            m = th.where(active, m, th.zeros_like(m))
        # the noisy-or of the boxes of the union
        return 1.0 - (1.0 - m).prod(dim=-1)

    def box_probability(self, x, centre, offset):
        """``(B, k)``: the soft membership of the points ``x`` ``(B, d)`` in each of the ``k``
        boxes ``(B, k, d)`` of their row: the smallest, over the coordinates, sigmoid of the
        depth :math:`(|o| - |x - c|) / \\beta` (the minimum, not the product, so that it does
        not decay with the dimension)."""
        depth = offset.abs() - (x[:, None, :] - centre).abs()
        return th.sigmoid(depth / self.beta).amin(dim=-1)

    def _clamped(self, a, b):
        x = a if isinstance(a, th.Tensor) else _point(a)
        return self.probability(x, b).clamp(min=self.eps, max=1.0 - self.eps)

    def member(self, a, b, margin=0.0):
        return -th.log(self._clamped(a, b)).unsqueeze(1)

    def inside(self, a, b, margin=0.0):
        return self.member(a, b, margin)

    def apart(self, a, b, margin=0.0):
        return -th.log(1.0 - self._clamped(a, b)).unsqueeze(1)


class PointTerms(nn.Module):
    """The loss on sampled points of the point objectives.

    :param supervision: A :class:`~mowl.datasets.el.points.PointSupervision`.
    :param embed_dim: Dimension of the points.
    :param measure: The :class:`SoftMembership` (default ``beta=0.1``).
    :param neg_weight: Weight of the random negatives.
    :param ind_halfwidth: Half-width of the virtual box of an individual that is the filler of
        a role triple (default :math:`5\\beta`).
    :param volume_reg: Weight of :func:`log_volume` of the classes
        ``volume_classes``.
    :param volume_classes: Class ids the volume penalty covers (the named classes except top
        and bottom).
    :param elem_embed: The table of the coordinates of the canonical elements (default: a
        new one, ``(n_elements, embed_dim)``, uniform in :math:`[-1, 1]` and normalised).
    """

    def __init__(self, supervision, embed_dim, measure=None, neg_weight=0.1, ind_halfwidth=None,
                 volume_reg=0.0, volume_classes=None, elem_embed=None):
        super().__init__()
        self.supervision = supervision
        self.measure = measure if measure is not None else SoftMembership()
        self.neg_weight = float(neg_weight)
        self.ind_halfwidth = 5.0 * self.measure.beta if ind_halfwidth is None \
            else float(ind_halfwidth)
        self.volume_reg = float(volume_reg)
        self.volume_classes = volume_classes
        self.embed_dim = embed_dim
        self.elem_embed = None
        if supervision is not None:
            self.set_supervision(supervision, elem_embed)

    def set_supervision(self, supervision, elem_embed=None):
        """Attach a :class:`~mowl.datasets.el.points.PointSupervision` and (unless given) a
        new table of coordinates for its elements."""
        self.supervision = supervision
        if elem_embed is None:
            elem_embed = nn.Embedding(supervision.n_elements, self.embed_dim)
            with th.no_grad():
                nn.init.uniform_(elem_embed.weight, a=-1, b=1)
                elem_embed.weight.div_(th.linalg.norm(elem_embed.weight, dim=1, keepdim=True))
        self.elem_embed = elem_embed
        return self

    # -- elements -------------------------------------------------------------------------
    def points(self, g, elements):
        """Coordinates of canonical elements: classes' from ``elem_embed``, individuals' from
        the table of the individual model."""
        table = self.elem_embed
        idx = elements.to(table.weight.device, th.long)
        is_individual = self.supervision.element_is_individual.to(idx.device)[idx]
        if not bool(is_individual.any()):
            return table(idx)
        individuals = getattr(g.individual_model, "centres", None)
        if individuals is None:
            raise ValueError("PointSupervision contains individual elements, but the "
                             "geometry has no individuals.")
        own = self.supervision.element_class_index.to(idx.device)[idx]
        if bool(is_individual.all()):
            return individuals(own)
        out = table(idx.masked_fill(is_individual, 0))
        return th.where(is_individual[:, None], individuals(own.masked_fill(~is_individual, 0)),
                        out)

    def bumps(self, g, elements):
        """The bump vector of the class or individual each element stands for."""
        ps = self.supervision
        idx = elements.to(self.elem_embed.weight.device, th.long)
        is_individual = ps.element_is_individual.to(idx.device)[idx]
        own = ps.element_class_index.to(idx.device)[idx]
        classes = g.class_model.bumps
        out = classes(own.clamp(max=len(g.class_model) - 1))
        if bool(is_individual.any()):
            individuals = g.individual_model.bumps
            out = out.clone()
            out[is_individual] = individuals(own[is_individual])
        return out

    # -- the losses -----------------------------------------------------------------------
    def _pair_probability(self, g, pairs):
        p = pairs.to(self.elem_embed.weight.device, th.long)
        return self.measure.probability(self.points(g, p[:, 0]), g.classes(p[:, 1]))

    def membership_loss(self, g, batch, neg=False, weight=None):
        """BCE of the soft membership of element points in classes: a sample dictionary
        (``pos`` target 1, ``hard`` target 0, ``rand`` target 0 with weight ``neg_weight``)
        or a ``(B, 2)`` tensor of ``[element, class]`` (target 0 if ``neg``)."""
        device = self.elem_embed.weight.device
        if isinstance(batch, dict):
            loss = th.zeros((), device=device)
            for key, target, w in (("pos", 1.0, 1.0), ("hard", 0.0, 1.0),
                                   ("rand", 0.0, self.neg_weight if weight is None
                                    else float(weight))):
                pairs = batch.get(key)
                if pairs is None or pairs.numel() == 0 or w <= 0.0:
                    continue
                loss = loss + _bce(self._pair_probability(g, pairs), target, weight=w)
            return loss
        w = (self.neg_weight if neg else 1.0) if weight is None else float(weight)
        return _bce(self._pair_probability(g, batch), 0.0 if neg else 1.0, weight=w)

    def target_membership(self, g, points, elements):
        """Soft membership of ``points`` ``(B, d)`` in the target elements: in the region of
        the class (noisy-or of its boxes), or in the virtual box of half-width
        ``ind_halfwidth`` around the individual."""
        device = points.device
        ps = self.supervision
        y = elements.to(device, th.long)
        is_individual = ps.element_is_individual.to(device)[y]
        own = ps.element_class_index.to(device)[y]
        out = points.new_zeros(points.shape[0])
        classes = ~is_individual
        if bool(classes.any()):
            out = out.clone()
            out[classes] = self.measure.probability(points[classes],
                                                    g.classes(own[classes]))
        if bool(is_individual.any()):
            centre = g.individual_model.centres(own[is_individual])
            virtual = Box(centre, th.full_like(centre, self.ind_halfwidth))
            out = out.clone()
            out[is_individual] = self.measure.probability(points[is_individual], virtual)
        return out

    def role_loss(self, g, batch, neg=False, weight=None):
        """BCE of the membership of a role triple ``[x, r, y]`` of elements.

        Head and tail boxes (Box2EL, MultiBoxEL): the product of the membership of
        :math:`x + b_y` in the head box and of :math:`y + b_x` in the tail box (for an
        individual :math:`y`: of the tail centre moved by :math:`-b_x` in the virtual box
        around :math:`y`). Translations (MultiELBE): the membership of :math:`x + r` in
        :math:`y`."""
        device = self.elem_embed.weight.device
        triples = batch.get("roles") if isinstance(batch, dict) else batch
        if triples is None or triples.numel() == 0:
            return th.zeros((), device=device)
        t = triples.to(device, th.long)
        x, r, y = t[:, 0], t[:, 1], t[:, 2]
        relations = g.relation_model
        if hasattr(relations, "head") and hasattr(relations, "tail"):
            bx, by = self.bumps(g, x), self.bumps(g, y)
            head, tail = relations.head(r), relations.tail(r)
            m_head = self.measure.probability(self.points(g, x) + by, head)
            is_individual = self.supervision.element_is_individual.to(device)[y]
            m_tail = self.measure.probability(self.points(g, y) + bx, tail)
            if bool(is_individual.any()):
                m_tail = m_tail.clone()
                m_tail[is_individual] = self.target_membership(
                    g, (tail.centre - bx)[is_individual], y[is_individual])
            m = m_head * m_tail
        else:
            m = self.target_membership(g, self.points(g, x) + relations.vectors(r), y)
        w = (self.neg_weight if neg else 1.0) if weight is None else float(weight)
        return _bce(m, 0.0 if neg else 1.0, weight=w)

    def volume(self, g):
        """:func:`log_volume` of the offsets of the classes ``volume_classes`` (zero if there
        are none)."""
        if self.volume_classes is None or len(self.volume_classes) == 0:
            return th.zeros((), device=self.elem_embed.weight.device)
        ids = th.as_tensor(self.volume_classes, dtype=th.long,
                           device=self.elem_embed.weight.device)
        return log_volume(g.classes(ids).offset)

    @th.no_grad()
    def modelness(self, g, supervision=None, chunk_size=8192):
        """How far the points are a model of the supervision: the fraction of the positive
        pairs with membership at least 0.5 (``pos_frac``) and of the hard negatives with less
        (``hardneg_frac``); 1 for none."""
        supervision = self.supervision if supervision is None else supervision
        out = {}
        for key, pairs, positive in (("pos_frac", supervision.pos_pairs, True),
                                     ("hardneg_frac", supervision.hard_neg_pairs, False)):
            if pairs.shape[0] == 0:
                out[key] = 1.0
                continue
            hits = 0
            for chunk in th.split(pairs, max(int(chunk_size), 1)):
                m = self._pair_probability(g, chunk)
                hits += int(((m >= 0.5) if positive else (m < 0.5)).sum())
            out[key] = hits / pairs.shape[0]
        return out

    def loss(self, g, batch):
        """The extra loss of a point objective for a sample ``batch``: membership, role, and
        ``volume_reg`` times the volume penalty."""
        total = self.membership_loss(g, batch) + self.role_loss(g, batch)
        if self.volume_reg > 0.0:
            total = total + self.volume_reg * self.volume(g)
        return total

    def class_assertion_score(self, g, data, eps=1e-12):
        """:math:`-\\log m_C(a)` for ``data`` ``[individual, class]``, ``(B, 1)``."""
        a = g.individual_model.centres(data[:, 0])
        m = self.measure.probability(a, g.classes(data[:, 1]))
        return -th.log(m.clamp(min=eps)).unsqueeze(1)


_ZERO = {
    "point": ("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
              "object_property_assertion"),
    "mixed": ("gci0", "gci1", "gci2", "gci3", "object_property_assertion"),
    "hybrid": (),
}


class PointReading(Objective):
    """The forms of a point objective (see the module docstring): the axiom reading
    (``options`` are those of :class:`~mowl.representation.Objective`) with the forms of
    ``mode`` zero while the geometry is training, and the class assertion read as
    :math:`-\\log m_C(a)` in every mode.

    :param terms: The :class:`PointTerms` (for the measure of the class assertion).
    :param mode: ``"point"``, ``"mixed"`` or ``"hybrid"``.
    """

    def __init__(self, terms, mode="point", margin=0.0, **options):
        if mode not in _ZERO:
            raise ValueError(f"mode must be one of {sorted(_ZERO)}.")
        options.setdefault("strict_negatives", False)
        options.setdefault("negatives", "gci2")
        super().__init__(margin, **options)
        self.terms, self.mode = terms, mode

    def _form_cost(self, g, form, rows, neg):
        if form in _ZERO[self.mode] and g.training:
            return th.zeros(rows.shape[0], 1, device=rows.device,
                            dtype=g.class_model.centres.weight.dtype)
        return super()._form_cost(g, form, rows, neg)

    def class_assertion(self, g, rows, neg=False):
        return self.terms.class_assertion_score(g, rows)


class VolumePenalty(Regularizer):
    """:func:`log_volume` of the boxes of the classes ``classes`` (a function of the geometry
    or a list of class ids; default all), times ``weight``: a pull towards small regions for
    any class model whose regions are boxes or unions of boxes (the volume term of the point
    objectives of MultiBoxEL and MultiELBE, without the top and bottom classes).

    :param weight: Factor (zero: no penalty).
    :param classes: Class ids, or ``geometry -> class ids``, or ``None`` (all classes).
    """

    kind = "penalty"

    def __init__(self, weight=1e-3, classes=None):
        super().__init__(parameters=(), weight=weight)
        self.classes = classes

    def penalty(self, model):
        if float(self.weight) <= 0.0:
            return None
        ids = self.classes(model) if callable(self.classes) else self.classes
        if ids is None:
            ids = th.arange(len(model.class_model), device=model.device)
        ids = th.as_tensor(ids, dtype=th.long, device=model.device)
        if ids.numel() == 0:
            return None
        return float(self.weight) * log_volume(model.classes(ids).offset)


class PointPenalty(Regularizer):
    """The point loss as a penalty of the geometry: each call draws a batch with
    ``sampler(call_index)`` (default: ``supervision.sample(points_per_epoch, rng)`` with the
    generator seeded ``seed + call_index``, as the model's per-epoch sampling) and returns
    ``terms.loss(geometry, batch)``.

    :param terms: The :class:`PointTerms`.
    :param points_per_epoch: Size of a sample.
    :param seed: Base seed.
    :param sampler: ``call_index -> batch``, replacing the default.
    """

    kind = "penalty"

    def __init__(self, terms, points_per_epoch=1024, seed=0, sampler=None):
        super().__init__(parameters=(), weight=1.0)
        self.terms, self.points_per_epoch, self.seed = terms, points_per_epoch, seed
        self.sampler = sampler
        self.calls = 0

    def penalty(self, model):
        index = self.calls
        self.calls += 1
        if self.sampler is not None:
            batch = self.sampler(index)
        else:
            rng = np.random.default_rng(self.seed + index)
            batch = self.terms.supervision.sample(self.points_per_epoch, rng=rng)
        return self.terms.loss(model, batch)
