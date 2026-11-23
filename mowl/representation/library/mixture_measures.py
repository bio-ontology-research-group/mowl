"""The probability-mass measure of the canonical mixture, and the entropy of a distribution as
a regulariser.

:class:`ProbabilityMass` compares the distribution :math:`p_a` of an individual over the
elements of a constructed model with a set of elements :math:`C` by the mass it puts on the
set,

.. math:: P(a \\in C) = 1 - \\sum_{x \\in S_a,\\, x \\notin C} p_a(x),

computed as one minus the mass *outside* :math:`C` (so that it is exactly 1 when the whole
support lies in :math:`C`, and exactly 0 when no element of the support does), and with
:math:`\\exists r.\\{b\\}` by :math:`P(r(a, b)) = p_a \\cdot (E_r p_b)`. A loss reads the mass as
a Bernoulli probability (binary cross entropy, ``loss="bce"``): a positive assertion costs
:math:`-\\log P`, a negative one :math:`-\\log(1 - P)`, with :math:`P` clamped to
:math:`[\\varepsilon, 1 - \\varepsilon]`. The score of an assertion (:meth:`ProbabilityMass.score`)
is :math:`-\\log P` without the clamp: 0.0 exactly for an entailed assertion, ``inf`` for a
contradicted one. Sets of elements are compared exactly (cost 0.0 or 1.0).
"""

import torch as th

from mowl.representation.library.mixture_individuals import (ElementMixture, RelatedMixture,
                                                             expand)
from mowl.representation.library.mixture_tables import SparseElementSets
from mowl.representation.measures import RegionMeasure
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["ProbabilityMass", "EntropyPenalty", "class_mass", "role_mass"]


def _flag(value):
    return value.to(th.get_default_dtype()).reshape(-1, 1)


# -- the mass of a set of elements -------------------------------------------------------------
def _named_mass_chunk(model, table, ind, cls):
    """``(B,)``: :math:`P(a \\in C)` for rows ``(ind, cls)`` of named classes."""
    device, dtype = model.U.device, model.U.dtype
    out = th.zeros((len(ind),), dtype=dtype, device=device)
    unique, inverse = th.unique(ind, return_inverse=True)
    for s, e in model.chunks(unique):
        sub = unique[s:e]
        rows = th.nonzero((inverse >= s) & (inverse < e), as_tuple=False).reshape(-1)
        if rows.numel() == 0:
            continue
        d = model.distribution(sub)
        local = inverse.index_select(0, rows) - s
        row_full = d.full.index_select(0, local)
        # individuals whose support is S: the mass of C by a sparse product with M[:, C]
        full_rows = rows[row_full]
        if full_rows.numel():
            to_dense = th.full((len(sub),), -1, dtype=th.long, device=device)
            to_dense[d.dense_pos] = th.arange(d.dense_pos.numel(), device=device)
            pos = to_dense.index_select(0, inverse.index_select(0, full_rows) - s)
            probs = th.exp(d.dense_logp)
            c_unique, c_inv = th.unique(cls.index_select(0, full_rows), return_inverse=True)
            sizes = table.column_sizes(c_unique)
            if int(sizes.sum()) > 0:
                inside = th.sparse.mm(table.columns_t(c_unique).to(dtype), probs.t())
                mass_in = inside[c_inv, pos]
                size = sizes.index_select(0, c_inv)
                total = probs.sum(dim=1).index_select(0, pos)
                outside = th.where(size >= table.n_elements, th.zeros_like(mass_in),
                                   (total - mass_in).clamp(min=0.0))
                value = th.where(size > 0, (1.0 - outside).clamp(0.0, 1.0),
                                 th.zeros_like(mass_in))
                out = out.scatter(0, full_rows, value)
        # the others: one minus the mass of the support entries outside C
        part_rows = rows[~row_full]
        if part_rows.numel():
            to_part = th.full((len(sub),), -1, dtype=th.long, device=device)
            to_part[d.part_pos] = th.arange(d.part_pos.numel(), device=device)
            local_part = to_part.index_select(0, inverse.index_select(0, part_rows) - s)
            row, element, p = expand(d, local_part)
            if row.numel():
                member = table.contains(element, cls.index_select(0, part_rows)
                                        .index_select(0, row))
                outside = th.zeros((part_rows.numel(),), dtype=p.dtype, device=device)
                if bool((~member).any()):
                    outside.scatter_add_(0, row[~member], p[~member])
                some = th.zeros((part_rows.numel(),), dtype=th.bool, device=device)
                if bool(member.any()):
                    some[row[member]] = True
                value = th.where(some, (1.0 - outside).clamp(0.0, 1.0),
                                 th.zeros_like(outside))
                out = out.scatter(0, part_rows, value)
    return out


def class_mass(mixture, sets):
    """``(B,)``: :math:`P(a \\in C)` of the distributions ``mixture`` and the sets of
    elements ``sets`` (row by row)."""
    model = mixture.model
    ind = mixture.ids
    if sets.named and len(sets) == len(mixture):
        cls = sets.ids.to(ind.device)
        step = model.row_chunk(len(ind))
        if len(ind) <= step:
            return _named_mass_chunk(model, sets.table, ind, cls)
        return th.cat([_named_mass_chunk(model, sets.table, ind[s:s + step], cls[s:s + step])
                       for s in range(0, len(ind), step)])
    # a compound set: one minus the mass outside it
    probs = model.probs(ind)
    mask = sets.mask.to(probs.dtype)
    if mask.shape[0] == 1:
        mask = mask.expand_as(probs)
    support = probs > 0
    outside = (probs * (1 - mask)).sum(dim=1)
    some = (support & (mask > 0)).any(dim=1)
    return th.where(some, (1.0 - outside).clamp(0.0, 1.0), th.zeros_like(outside))


def role_mass(head, related):
    """``(B,)``: :math:`P(r(a, b)) = p_a \\cdot (E_r p_b)` for the distributions ``head`` of
    :math:`a` and ``related`` (:math:`\\exists r.\\{b\\}`)."""
    model = head.model
    heads, tails = head.ids, related.filler.ids
    rels = related.ids.to(heads.device)
    device, dtype = model.U.device, model.U.dtype
    n = model.n_elements
    out = th.zeros((len(heads),), dtype=dtype, device=device)
    for key in th.unique(rels).tolist():
        e = related.relations.matrix(key)
        if e is None:
            continue
        rows = th.nonzero(rels == key, as_tuple=False).reshape(-1)
        sub_rows = th.stack([heads.index_select(0, rows), rels.index_select(0, rows),
                             tails.index_select(0, rows)], dim=1)
        for s, t in model.pair_chunks(sub_rows):
            c_rows = rows[s:t]
            m = c_rows.numel()
            both = th.cat([heads.index_select(0, c_rows), tails.index_select(0, c_rows)])
            unique, inverse = th.unique(both, return_inverse=True)
            u_head, u_tail = inverse[:m], inverse[m:]
            d = model.distribution(unique)
            dense_probs = th.exp(d.dense_logp)
            to_dense = th.full((len(unique),), -1, dtype=th.long, device=device)
            to_dense[d.dense_pos] = th.arange(d.dense_pos.numel(), device=device)
            to_part = th.full((len(unique),), -1, dtype=th.long, device=device)
            to_part[d.part_pos] = th.arange(d.part_pos.numel(), device=device)
            # p_b as columns (|S|, m)
            p_b = th.zeros((n, m), dtype=dtype, device=device)
            tail_full = d.full.index_select(0, u_tail)
            if bool(tail_full.any()):
                cols = th.nonzero(tail_full, as_tuple=False).reshape(-1)
                p_b[:, cols] = dense_probs.index_select(
                    0, to_dense.index_select(0, u_tail.index_select(0, cols))).t()
            if bool((~tail_full).any()):
                cols = th.nonzero(~tail_full, as_tuple=False).reshape(-1)
                row, element, p = expand(d, to_part.index_select(0, u_tail.index_select(0,
                                                                                        cols)))
                if row.numel():
                    p_b[element, cols.index_select(0, row)] = p
            q = th.sparse.mm(e, p_b)                                     # (|S|, m)
            value = th.zeros((m,), dtype=dtype, device=device)
            head_full = d.full.index_select(0, u_head)
            if bool(head_full.any()):
                cols = th.nonzero(head_full, as_tuple=False).reshape(-1)
                p_a = dense_probs.index_select(
                    0, to_dense.index_select(0, u_head.index_select(0, cols)))
                value[cols] = (p_a * q.index_select(1, cols).t()).sum(dim=1)
            if bool((~head_full).any()):
                cols = th.nonzero(~head_full, as_tuple=False).reshape(-1)
                row, element, p = expand(d, to_part.index_select(0, u_head.index_select(0,
                                                                                        cols)))
                if row.numel():
                    at = cols.index_select(0, row)
                    value.scatter_add_(0, at, p * q[element, at])
            out = out.scatter_add(0, c_rows, value)
    return out.clamp(0.0, 1.0)


class ProbabilityMass(RegionMeasure):
    """The probability mass an individual's distribution puts on a set of elements (module
    docstring); sets of elements are compared exactly.

    :param loss: ``"bce"`` (default): the losses of an assertion are the binary cross
        entropy of the mass clamped to :math:`[\\varepsilon, 1 - \\varepsilon]`
        (:math:`-\\log P` and :math:`-\\log(1 - P)`); ``"log"``: :math:`-\\log P` and
        :math:`-\\log(1 - P)` with only the arguments of the logarithm kept above
        :math:`\\varepsilon`.
    :param eps: :math:`\\varepsilon`.

    ``member`` and ``inside`` of an individual are its positive loss, ``apart`` its
    negative loss; :meth:`score` is :math:`-\\log P` itself.

    Part (vocabulary value): ``exact``. Used by: ``canonical_mixture``.
    """

    accepts = (SparseElementSets, ElementMixture, RelatedMixture)
    part = "exact"
    # the loss of an entailed assertion is -log(1 - eps), not zero; the score is exact
    reading = "score"

    def __init__(self, loss="bce", eps=1e-7):
        if loss not in ("bce", "log"):
            raise ValueError(f"loss must be 'bce' or 'log', not {loss!r}.")
        self.loss, self.eps = loss, float(eps)

    @staticmethod
    def _margin(margin):
        if margin:
            from mowl.representation.regions import MarginUnavailable
            raise MarginUnavailable("The probability-mass measure has no margin (a "
                                    "membership of a finite model holds or does not).")

    def mass(self, a, b):
        """``(B,)``: :math:`P(a \\in b)` for an individual ``a`` (:class:`ElementMixture`)."""
        if not isinstance(a, ElementMixture):
            raise ConstructorUnavailable("The mass is that of an individual's distribution.")
        if isinstance(b, RelatedMixture):
            return role_mass(a, b)
        return class_mass(a, b)

    def score(self, a, b):
        """``(B, 1)``: :math:`-\\log P(a \\in b)` (0.0 when entailed, ``inf`` when
        excluded)."""
        return (-th.log(self.mass(a, b)) + 0.0).reshape(-1, 1)

    def positive(self, p):
        if self.loss == "bce":
            return -th.log(p.clamp(self.eps, 1.0 - self.eps))
        return -th.log(p.clamp(min=self.eps))

    def negative(self, p, *args):
        if self.loss == "bce":
            return -th.log(1.0 - p.clamp(self.eps, 1.0 - self.eps))
        return -th.log((1.0 - p).clamp(min=self.eps))

    # -- the relations ------------------------------------------------------------------------
    def member(self, a, b, margin=0.0):
        self._margin(margin)
        if isinstance(a, ElementMixture):
            return self.positive(self.mass(a, b)).reshape(-1, 1)
        return self.inside(a, b)

    def inside(self, a, b, margin=0.0):
        self._margin(margin)
        if isinstance(a, ElementMixture):
            return self.member(a, b)
        return _flag((a.mask > b.mask).any(dim=1))

    def apart(self, a, b, margin=0.0):
        self._margin(margin)
        if isinstance(a, ElementMixture):
            return self.negative(self.mass(a, b)).reshape(-1, 1)
        return _flag((a.mask * b.mask).sum(dim=1) > 0)

    def meets(self, a, b, margin=0.0):
        self._margin(margin)
        if isinstance(a, ElementMixture):
            return self.member(a, b)
        return _flag((a.mask * b.mask).sum(dim=1) == 0)

    def outside(self, a, b, margin=0.0):
        self._margin(margin)
        if isinstance(a, ElementMixture):
            return self.apart(a, b)
        return _flag(~(a.mask > b.mask).any(dim=1))

    def empty(self, a, margin=0.0):
        self._margin(margin)
        return _flag(~a.is_empty())

    def nonempty(self, a, epsilon=0.0):
        return _flag(a.is_empty())

    # an empty conjunction is included in every class: no gap is added
    def intersection_inside(self, a, b, c, margin=0.0):
        region, _ = a.intersect(b)
        return self.inside(region, c, margin)

    def intersection_apart(self, a, b, c, margin=0.0):
        region, _ = a.intersect(b)
        return self.apart(region, c, margin)

    def intersection_nonempty(self, a, b, margin=0.0, epsilon=0.0):
        region, _ = a.intersect(b)
        return self.nonempty(region)


class EntropyPenalty:
    """:math:`\\lambda \\operatorname{mean}_a H(p_a)` over a batch of individuals: favours
    concentrated distributions (``weight`` :math:`\\lambda > 0`) or spread ones
    (:math:`\\lambda < 0`). Reads the individual model of a geometry (``entropy``).

    :meth:`penalty` takes the individuals of the batch (``ids``; default: all); the training
    regime of the canonical mixture passes them (``batch_penalty``).
    """

    kind = "penalty"
    batch_penalty = True

    def __init__(self, weight=0.0):
        self.weight = float(weight)

    def penalty(self, geometry, ids=None):
        model = geometry.individual_model
        if ids is None:
            ids = th.arange(len(model), device=model.U.device)
        if self.weight == 0.0 or len(ids) == 0:
            return model.U.sum() * 0.0
        return self.weight * model.entropy(ids).mean()

    def project(self, model):
        """A penalty moves nothing."""

    def __repr__(self):
        return f"EntropyPenalty(weight={self.weight!r})"
