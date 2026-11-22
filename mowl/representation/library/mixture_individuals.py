"""Individuals as distributions over the elements of a constructed model (the trained part of
the canonical mixture).

An individual :math:`a` is a probability distribution
:math:`p_a = \\operatorname{softmax}_{S_a}(\\ell_a)` over its *admissible support*
:math:`S_a`: the elements of the constructed model that belong to every class the model
entails for :math:`a` (the intersection of the element sets of the classes of :math:`a`'s own
element). The logits are low rank,
:math:`\\ell_a = U_a V^\\top + b + \\delta_a`, with a fixed offset
:math:`\\delta_a(x_a) = \\log(\\frac{\\alpha}{1 - \\alpha}(|S_a| - 1))` on the individual's own
element (so that it starts with mass :math:`\\alpha`, ``init_self_mass``) and :math:`0`
elsewhere (and for :math:`|S_a| = 1`). An individual without an element (one that occurs only
in validation or test data) has the support :math:`S`.

Because :math:`p_a` is a distribution over :math:`S_a` only, the probability mass of a class
that the model entails for :math:`a` is 1 and that of a class that excludes :math:`S_a` is 0,
for every value of the parameters (the guarantees of the canonical mixture).

The supports are stored as one concatenated index vector with offsets; logits are computed on
the support entries only (an individual whose support is all of :math:`S` takes a dense row,
which is cheaper). Nothing of size ``n_individuals x |S|`` is allocated.
"""

import math

import torch as th
import torch.nn as nn

from mowl.representation.class_models import Classes
from mowl.representation.regions import Region

__all__ = ["ElementMixture", "RelatedMixture", "Distribution", "MixtureIndividuals",
           "MAX_SUPPORT_ENTRIES"]

#: Largest number of support entries computed at once (by the chunked scores and losses).
MAX_SUPPORT_ENTRIES = 524_288


class ElementMixture(Region):
    """The distributions of the individuals ``ids`` (computed when a measure reads them).

    :param model: The :class:`MixtureIndividuals`.
    :param ids: ``(B,)`` individual ids.
    """

    def __init__(self, model, ids):
        self.model, self.ids = model, ids

    def __len__(self):
        return self.ids.shape[0]


class RelatedMixture(Region):
    """:math:`\\exists r.\\{b\\}`: the elements related by the roles ``ids`` to the
    distribution ``filler`` of the individuals :math:`b` (read as :math:`E_r p_b`)."""

    def __init__(self, relations, ids, filler):
        self.relations, self.ids, self.filler = relations, ids, filler

    def __len__(self):
        return self.ids.shape[0]


class Distribution:
    """The distributions of distinct individuals ``ids``: ``full`` ``(k,)`` says which have
    the whole of :math:`S` as support; their log-probabilities are the dense rows
    ``dense_logp`` (``dense_pos`` gives their positions in ``ids``); the others are support
    entries ``(rows, elements, log_probs, probs)`` with ``rows`` positions in ``ids`` and
    ``offsets`` the start of the entries of every partial individual (positions
    ``part_pos``)."""

    def __init__(self, ids, full, dense_pos, dense_logp, part_pos, rows, elements, log_probs,
                 probs, lengths):
        self.ids, self.full = ids, full
        self.dense_pos, self.dense_logp = dense_pos, dense_logp
        self.part_pos, self.rows, self.elements = part_pos, rows, elements
        self.log_probs, self.probs, self.lengths = log_probs, probs, lengths
        self.offsets = th.zeros((lengths.numel() + 1,), dtype=th.long, device=ids.device)
        self.offsets[1:] = th.cumsum(lengths, dim=0)


def _segments(starts, lengths, device):
    """Positions of the concatenated segments ``[starts[i], starts[i] + lengths[i])`` and the
    segment of every position."""
    total = int(lengths.sum())
    seg = th.repeat_interleave(th.arange(lengths.numel(), device=device), lengths)
    flat = (th.arange(total, device=device)
            - th.repeat_interleave(th.cumsum(lengths, 0) - lengths, lengths)
            + th.repeat_interleave(starts, lengths))
    return seg, flat


def expand(dist, local, value_of="probs"):
    """For the partial individuals at positions ``local`` of ``dist.part_pos`` (one per
    row), the support entries of every row: ``(row, element, value)``."""
    device = local.device
    starts = dist.offsets.index_select(0, local)
    lengths = dist.offsets.index_select(0, local + 1) - starts
    if int(lengths.sum()) == 0:
        empty = th.empty((0,), dtype=th.long, device=device)
        return empty, empty, th.empty((0,), dtype=dist.probs.dtype, device=device)
    row, flat = _segments(starts, lengths, device)
    return row, dist.elements.index_select(0, flat), \
        getattr(dist, value_of).index_select(0, flat)


class MixtureIndividuals(Classes):
    """The individual model of the canonical mixture: ``U`` ``(n_ind, rank)``, ``V``
    ``(|S|, rank)``, ``b`` ``(|S|,)`` and the admissible supports (module docstring).

    :param table: The :class:`~mowl.representation.library.mixture_tables.MembershipTable`
        of the constructed model.
    :param individual_element: ``(n_ind,)`` the element of every individual (``-1``: none).
    :param rank: :math:`r`.
    :param init_self_mass: :math:`\\alpha \\in (0, 1)`, the initial mass of the own element.
    :param training_class_assertions: ``(N, 2)`` rows ``(individual, class)``; checked to be
        entailed by the constructed model (a training assertion the model does not entail
        is an error).
    :param dense_full_support: Compute individuals whose support is all of :math:`S` as
        dense rows (faster; the same numbers up to rounding).

    Part (vocabulary value): ``distribution``. Used by: ``canonical_mixture``.
    """

    region = ElementMixture
    part = "distribution"
    max_support_entries = MAX_SUPPORT_ENTRIES

    def __init__(self, table, individual_element, rank=50, init_self_mass=0.9,
                 training_class_assertions=None, dense_full_support=True):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"rank must be an int >= 1, got {rank!r}")
        if not 0.0 < float(init_self_mass) < 1.0:
            raise ValueError(f"init_self_mass must be in (0, 1), got {init_self_mass!r}")
        individual_element = th.as_tensor(individual_element, dtype=th.long).cpu()
        if individual_element.numel() == 0:
            raise ValueError("CanonicalMixture needs individuals")
        self.rank, self.init_self_mass = int(rank), float(init_self_mass)
        self.n_elements = table.n_elements
        self.n_classes = table.n_classes
        self.n_ind = int(individual_element.numel())
        self.dense_full_support = bool(dense_full_support)
        self.U = nn.Parameter(th.empty((self.n_ind, self.rank)))
        self.V = nn.Parameter(th.empty((self.n_elements, self.rank)))
        self.b = nn.Parameter(th.zeros((self.n_elements,)))
        nn.init.normal_(self.U, mean=0.0, std=1e-5 / math.sqrt(self.rank))
        nn.init.normal_(self.V, mean=0.0, std=1.0 / math.sqrt(self.rank))
        self.register_buffer("individual_element", individual_element.clone(),
                             persistent=False)
        self._check_assertions(table, training_class_assertions)
        offsets, indices, own = self._supports(table)
        self.register_buffer("support_offsets", offsets, persistent=False)
        self.register_buffer("support_indices", indices, persistent=False)
        self.register_buffer("own_logit_offset", own, persistent=False)
        self.register_buffer("is_full_support", (offsets[1:] - offsets[:-1]) == self.n_elements,
                             persistent=False)

    # -- construction of the supports ---------------------------------------------------------
    def _check_assertions(self, table, rows):
        if rows is None or rows.numel() == 0:
            return
        rows = rows.detach().cpu().to(th.long)
        if rows.ndim != 2 or rows.shape[1] != 2:
            raise ValueError(
                f"training_class_assertions must have shape (N, 2), got {tuple(rows.shape)}")
        ind, cls = rows[:, 0], rows[:, 1]
        if bool(((ind < 0) | (ind >= self.n_ind)).any()):
            raise ValueError("Individual index in training_class_assertions is out of bounds.")
        if bool(((cls < 0) | (cls >= self.n_classes)).any()):
            raise ValueError("Class index in training_class_assertions is out of bounds.")
        ok = table.contains(self.individual_element.index_select(0, ind), cls)
        if not bool(ok.all()):
            bad = int(th.nonzero(~ok, as_tuple=False)[0])
            raise ValueError(f"Training class assertion (individual={int(ind[bad])}, "
                             f"class={int(cls[bad])}) is not entailed by M.")

    def _supports(self, table):
        """CSR supports: the elements of every class of the own element, intersected
        (smallest first); all of :math:`S` without an own element or without classes."""
        import numpy as np
        ccol = table.ccol.cpu().numpy()
        members = table.members.cpu().numpy()
        crow = table.crow.cpu().numpy()
        types = table.types.cpu().numpy()
        odds = self.init_self_mass / (1.0 - self.init_self_mass)
        n = self.n_elements
        supports, lengths, own = [], [], []
        for i, el in enumerate(self.individual_element.tolist()):
            if el < 0:
                supports.append(np.arange(n))
                lengths.append(n)
                own.append(0.0)
                continue
            classes = sorted(types[crow[el]:crow[el + 1]].tolist(),
                             key=lambda c: int(ccol[c + 1] - ccol[c]))
            if not classes:
                elems = np.arange(n)
            else:
                elems = members[ccol[classes[0]]:ccol[classes[0] + 1]]
                for c in classes[1:]:
                    if elems.size <= 1:
                        break
                    elems = np.intersect1d(elems, members[ccol[c]:ccol[c + 1]],
                                           assume_unique=True)
                pos = int(np.searchsorted(elems, el))
                if pos >= elems.size or int(elems[pos]) != el:
                    raise ValueError(f"Own element {el} of individual {i} is not in its "
                                     f"admissible support.")
            supports.append(elems)
            lengths.append(int(elems.size))
            own.append(math.log(odds * float(elems.size - 1)) if elems.size > 1 else 0.0)
        lengths = th.tensor(lengths, dtype=th.long)
        offsets = th.zeros((self.n_ind + 1,), dtype=th.long)
        offsets[1:] = th.cumsum(lengths, dim=0)
        indices = th.from_numpy(np.concatenate(supports).astype(np.int64))
        return offsets, indices, th.tensor(own, dtype=th.float32)

    # -- sizes and chunks -----------------------------------------------------------------------
    def __len__(self):
        return self.n_ind

    @property
    def embed_dim(self):
        return self.rank

    @property
    def support_lengths(self):
        """``(n_ind,)``: :math:`|S_a|`."""
        return self.support_offsets[1:] - self.support_offsets[:-1]

    def tables(self):
        return {"U": self.U, "V": self.V, "b": self.b}

    def row_chunk(self, rows):
        """Rows per chunk such that a dense ``(rows, |S|)`` block stays below 8 MB."""
        if rows <= 0:
            return 1
        limit = 8 * 1024 * 1024
        if self.n_elements * rows * 4 > limit:
            return min(128, max(1, limit // max(self.n_elements * 4, 1)))
        return rows

    def costs(self, ids):
        """``(B,)``: the support entries of every individual (a dense row counts a tenth)."""
        lengths = self.support_lengths.index_select(0, ids)
        if not self.dense_full_support:
            return lengths
        full = self.is_full_support.index_select(0, ids)
        return th.where(full, th.full_like(lengths, max(1, self.n_elements // 10)), lengths)

    @staticmethod
    def _ranges(costs, limit, max_rows=None):
        n = int(costs.numel())
        if n == 0:
            return []
        if int(costs.sum()) <= limit and (max_rows is None or n <= max_rows):
            return [(0, n)]
        ranges, start, acc = [], 0, 0
        for i, c in enumerate(costs.tolist()):
            if (acc + c > limit or (max_rows is not None and i - start >= max_rows)) \
                    and i > start:
                ranges.append((start, i))
                start, acc = i, c
            else:
                acc += c
        ranges.append((start, n))
        return ranges

    def chunks(self, ids, limit=MAX_SUPPORT_ENTRIES):
        """``(start, end)`` slices of ``ids`` with at most ``limit`` support entries each."""
        return self._ranges(self.costs(ids), limit)

    def pair_chunks(self, rows, limit=MAX_SUPPORT_ENTRIES):
        """``(start, end)`` slices of assertion ``rows`` (individuals in column 0, and 2 for
        role assertions) with at most ``limit`` support entries and :meth:`row_chunk` rows."""
        cost = self.costs(rows[:, 0])
        if rows.shape[1] >= 3:
            cost = cost + self.costs(rows[:, 2])
        return self._ranges(cost, limit, self.row_chunk(int(rows.shape[0])))

    # -- the distributions -----------------------------------------------------------------------
    def forward(self, ids, detach_size=False):
        return ElementMixture(self, th.as_tensor(ids, dtype=th.long,
                                                 device=self.U.device).reshape(-1))

    def dense_log_p(self, ids):
        """``(k, |S|)``: :math:`\\log p_a` of individuals whose support is :math:`S`."""
        if ids.numel() == 0:
            return th.empty((0, self.n_elements), dtype=self.U.dtype, device=self.U.device)
        logits = th.addmm(self.b, self.U.index_select(0, ids), self.V.t())
        own = self.individual_element.index_select(0, ids)
        has = own >= 0
        if bool(has.any()):
            r = th.nonzero(has, as_tuple=False).reshape(-1)
            c = own.index_select(0, r)
            logits[r, c] = logits[r, c] + self.own_logit_offset.index_select(0, ids)[r].to(
                logits.dtype)
        return th.log_softmax(logits, dim=1)

    def support_log_p(self, ids):
        """Support entries of the individuals ``ids``: ``(rows, elements, log_probs, probs,
        lengths)``, ``rows`` positions in ``ids`` (a softmax over each support only)."""
        device, dtype = self.U.device, self.U.dtype
        starts = self.support_offsets.index_select(0, ids)
        lengths = self.support_offsets.index_select(0, ids + 1) - starts
        if int(lengths.sum()) == 0:
            e = th.empty((0,), dtype=th.long, device=device)
            f = th.empty((0,), dtype=dtype, device=device)
            return e, e, f, f, lengths
        rows, flat = _segments(starts, lengths, device)
        elements = self.support_indices.index_select(0, flat)
        u = self.U.index_select(0, ids)
        logits = (u.index_select(0, rows) * self.V.index_select(0, elements)).sum(dim=1) \
            + self.b.index_select(0, elements)
        own = self.individual_element.index_select(0, ids).index_select(0, rows)
        offset = self.own_logit_offset.index_select(0, ids).to(dtype).index_select(0, rows)
        logits = th.where((own >= 0) & (elements == own), logits + offset, logits)
        top = th.full((len(ids),), float("-inf"), dtype=dtype, device=device)
        top.scatter_reduce_(0, rows, logits.detach(), reduce="amax", include_self=False)
        shifted = logits - top.index_select(0, rows)
        e = th.exp(shifted)
        total = th.zeros((len(ids),), dtype=dtype, device=device).scatter_add_(0, rows, e)
        return rows, elements, shifted - th.log(total).index_select(0, rows), \
            e / total.index_select(0, rows), lengths

    def distribution(self, ids):
        """The :class:`Distribution` of the (distinct) individuals ``ids``."""
        device = self.U.device
        if self.dense_full_support:
            full = self.is_full_support.index_select(0, ids)
        else:
            full = th.zeros((len(ids),), dtype=th.bool, device=device)
        dense_pos = th.nonzero(full, as_tuple=False).reshape(-1)
        part_pos = th.nonzero(~full, as_tuple=False).reshape(-1)
        dense = self.dense_log_p(ids.index_select(0, dense_pos))
        rows, elements, logp, probs, lengths = self.support_log_p(
            ids.index_select(0, part_pos))
        return Distribution(ids, full, dense_pos, dense, part_pos, rows, elements, logp,
                            probs, lengths)

    def log_p(self, ids):
        """``(B, |S|)``: :math:`\\log p_a`, ``-inf`` outside the support."""
        ids = th.as_tensor(ids, dtype=th.long, device=self.U.device).reshape(-1)
        out = th.full((len(ids), self.n_elements), float("-inf"), dtype=self.U.dtype,
                      device=self.U.device)
        for s, e in self.chunks(ids):
            d = self.distribution(ids[s:e])
            out[d.dense_pos + s] = d.dense_logp
            out[d.part_pos.index_select(0, d.rows) + s, d.elements] = d.log_probs
        return out

    def probs(self, ids):
        """``(B, |S|)``: :math:`p_a`, zero outside the support."""
        ids = th.as_tensor(ids, dtype=th.long, device=self.U.device).reshape(-1)
        out = th.zeros((len(ids), self.n_elements), dtype=self.U.dtype, device=self.U.device)
        for s, e in self.chunks(ids):
            d = self.distribution(ids[s:e])
            out[d.dense_pos + s] = th.exp(d.dense_logp)
            out[d.part_pos.index_select(0, d.rows) + s, d.elements] = d.probs
        return out

    def entropy(self, ids):
        """``(B,)``: :math:`H(p_a)`."""
        ids = th.as_tensor(ids, dtype=th.long, device=self.U.device).reshape(-1)
        parts = []
        for s, e in self.chunks(ids):
            d = self.distribution(ids[s:e])
            out = th.zeros((e - s,), dtype=self.U.dtype, device=self.U.device)
            if d.dense_pos.numel():
                out[d.dense_pos] = -(th.exp(d.dense_logp) * d.dense_logp).sum(dim=1)
            if d.part_pos.numel():
                part = th.zeros((d.part_pos.numel(),), dtype=out.dtype, device=out.device)
                part.scatter_add_(0, d.rows, -d.probs * d.log_probs)
                out[d.part_pos] = part
            parts.append(out)
        return th.cat(parts) if parts else th.zeros((0,), device=self.U.device)
