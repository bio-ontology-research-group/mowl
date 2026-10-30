"""CanonicalMixtureModule neural module for the canonical-element mixture model (R4)."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mowl.geometric.membership import MembershipStructure
from mowl.nn import ELModule

MAX_SUPPORT_ENTRIES: int = 524_288


class CanonicalMixtureModule(ELModule):
    """Canonical-element mixture neural module (R4).

    Represents each individual :math:`a` as a probability distribution
    :math:`p_a = \\operatorname{softmax}_{S_a}(\\ell_a)` over its admissible
    canonical support
    :math:`S_a = \\{x \\in S : x \\in C^{\\mathcal{I}} \\text{ for every entailed class } C \\text{ of } a\\}`
    (computed by intersecting the sorted CSC columns of :math:`M` over the
    minimal training-entailed types of :math:`a` derived from :math:`M`'s rows),
    where logits are factored
    as :math:`\\ell_a = U_a V^\\top + b + \\delta_a` with
    :math:`U \\in \\mathbb{R}^{n_{\\text{ind}} \\times r}`,
    :math:`V \\in \\mathbb{R}^{|S| \\times r}`, :math:`b \\in \\mathbb{R}^{|S|}`, and a
    fixed own-element logit offset
    :math:`\\delta_a(x_a) = \\log(9 \\cdot (|S_a| - 1))` when :math:`|S_a| > 1`
    (generalised to :math:`\\log(\\frac{\\alpha}{1-\\alpha} \\cdot (|S_a| - 1))` for
    ``init_self_mass = alpha``, and :math:`0` when :math:`|S_a| = 1` where
    :math:`p_a(x_a) = 1`).

    Guarantees:
    - **Faithful by construction:** For every parameter setting, :math:`p_a` is a
      convex combination of elements of the canonical model :math:`\\mathcal{I}`
      restricted to :math:`S_a` (computed by gathering logits on :math:`S_a` only
      before ``logsumexp``). Every ontology-entailed class assertion :math:`C(a)`
      (including those entailed through roles) has empty complement
      :math:`\\{x \\in S_a : x \\notin C^{\\mathcal{I}}\\} = \\emptyset`, so
      :math:`1 - \\sum_{x \\in S_a, x \\notin C^{\\mathcal{I}}} p_a(x) = 1` and
      :meth:`class_assertion_score` equals ``0.0`` exactly. Every contradicted
      class assertion satisfies :math:`M[x, C] = 0` for all :math:`x \\in S_a`,
      so its probability is ``0.0`` and :meth:`class_assertion_score` equals
      ``+inf``.
    - **Zero hierarchy violation rate:** Whenever :math:`\\mathcal{O} \\models C \\sqsubseteq D`,
      :math:`M[x, C] \\le M[x, D]` for all :math:`x \\in S`, so
      :math:`\\{x \\in S_a : x \\notin D^{\\mathcal{I}}\\} \\subseteq \\{x \\in S_a : x \\notin C^{\\mathcal{I}}\\}`,
      :math:`s(a, C) \\le s(a, D)`, and the negative-log score satisfies
      :math:`\\operatorname{score}(a, D) \\le \\operatorname{score}(a, C)` for all
      individuals :math:`a` and all parameters.
    - **Sparse CSR memory discipline:** Never allocates a dense ``(|S|, |S|)``,
      ``(n_ind, |S|)``, or ``(n_ind, n_classes)`` buffer; admissible supports
      :math:`S_a` are stored CSR-style as concatenated ``support_indices`` plus
      ``support_offsets``, negatives are drawn by rejection sampling against the
      individual's sparse :math:`M` row, and role scores
      :math:`p_a \\cdot (E_r p_b)` use :func:`torch.sparse.mm`.

    Non-guarantees:
    - TBox GCI loss methods (``gci0_loss`` .. ``gci3_bot_loss``) and role-axiom
      losses are not used for training (named classes have no learnable
      parameters; a class is its column of :attr:`MembershipStructure.M`).
    """

    neg_capable_gcis = frozenset({"class_assertion", "object_property_assertion"})
    role_axiom_capable = False
    MAX_SUPPORT_ENTRIES = MAX_SUPPORT_ENTRIES

    def __init__(
        self,
        structure: MembershipStructure,
        rank: int = 50,
        entropy_reg: float = 0.0,
        init_self_mass: float = 0.9,
        training_class_assertions: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if not isinstance(structure, MembershipStructure):
            raise TypeError(
                f"structure must be a MembershipStructure, got {type(structure).__name__}"
            )
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"rank must be an int >= 1, got {rank!r}")
        if float(entropy_reg) < 0.0:
            raise ValueError(f"entropy_reg must be >= 0, got {entropy_reg!r}")
        if not (0.0 < float(init_self_mass) < 1.0):
            raise ValueError(
                f"init_self_mass must be in (0, 1), got {init_self_mass!r}"
            )

        self.structure = structure
        self.rank = int(rank)
        self.entropy_reg = float(entropy_reg)
        self.init_self_mass = float(init_self_mass)

        n_elements, n_classes = structure.M.shape
        n_ind = int(structure.individual_element.numel())
        if n_ind == 0:
            raise ValueError("CanonicalMixture needs individuals")

        self.n_elements = int(n_elements)
        self.n_classes = int(n_classes)
        self.n_ind = int(n_ind)

        # Learnable parameters: U (n_ind x r), V (|S| x r), b (|S|)
        self.U = nn.Parameter(torch.empty((self.n_ind, self.rank), dtype=torch.float32))
        self.V = nn.Parameter(
            torch.empty((self.n_elements, self.rank), dtype=torch.float32)
        )
        self.b = nn.Parameter(torch.zeros((self.n_elements,), dtype=torch.float32))
        nn.init.normal_(self.U, mean=0.0, std=1e-5 / math.sqrt(self.rank))
        nn.init.normal_(self.V, mean=0.0, std=1.0 / math.sqrt(self.rank))

        self.register_buffer(
            "individual_element",
            structure.individual_element.clone().to(dtype=torch.long),
        )

        # Coalesce M once at construction and build CSR of M (elements -> classes),
        # CSC of M (classes -> sorted element indices), sorted flat keys
        # (row * n_classes + col), and float sparse M^T for spmm.
        m_coal = structure.M.coalesce()
        m_indices = m_coal.indices().cpu()
        m_values = m_coal.values().bool().cpu()
        if m_values.numel() > 0:
            raw_rows = m_indices[0][m_values]
            raw_cols = m_indices[1][m_values]
            flat_keys_raw = raw_rows * self.n_classes + raw_cols
            row_order = torch.argsort(flat_keys_raw, stable=True)
            sorted_rows = raw_rows[row_order]
            sorted_cols = raw_cols[row_order]
            sorted_flat_keys = flat_keys_raw[row_order]
            row_counts = torch.bincount(sorted_rows, minlength=self.n_elements)
            m_crow_offsets = torch.zeros((self.n_elements + 1,), dtype=torch.long)
            m_crow_offsets[1:] = torch.cumsum(row_counts, dim=0)

            col_flat_keys = raw_cols * self.n_elements + raw_rows
            col_order = torch.argsort(col_flat_keys, stable=True)
            csc_cols = raw_cols[col_order]
            csc_rows = raw_rows[col_order]
            col_counts = torch.bincount(csc_cols, minlength=self.n_classes)
            m_ccol_offsets = torch.zeros((self.n_classes + 1,), dtype=torch.long)
            m_ccol_offsets[1:] = torch.cumsum(col_counts, dim=0)
        else:
            sorted_rows = torch.empty((0,), dtype=torch.long)
            sorted_cols = torch.empty((0,), dtype=torch.long)
            sorted_flat_keys = torch.empty((0,), dtype=torch.long)
            m_crow_offsets = torch.zeros((self.n_elements + 1,), dtype=torch.long)
            csc_cols = torch.empty((0,), dtype=torch.long)
            csc_rows = torch.empty((0,), dtype=torch.long)
            m_ccol_offsets = torch.zeros((self.n_classes + 1,), dtype=torch.long)

        self.register_buffer("_m_crow_offsets", m_crow_offsets)
        self.register_buffer("_m_col_indices", sorted_cols)
        self.register_buffer("_m_flat_keys", sorted_flat_keys)
        self.register_buffer("_m_ccol_offsets", m_ccol_offsets)
        self.register_buffer("_m_csc_row_indices", csc_rows)

        # Store M^T as a coalesced float sparse tensor (n_classes x |S|) for rank_classes spmm
        mt_indices = (
            torch.stack([csc_cols, csc_rows], dim=0)
            if csc_cols.numel() > 0
            else torch.empty((2, 0), dtype=torch.long)
        )
        mt_vals = torch.ones((mt_indices.shape[1],), dtype=torch.float32)
        self.register_buffer(
            "_m_float_t",
            torch.sparse_coo_tensor(
                mt_indices,
                mt_vals,
                size=(self.n_classes, self.n_elements),
                dtype=torch.float32,
            ).coalesce(),
        )

        # Store role matrices as coalesced float32 sparse tensors (|S| x |S|) for sparse.mm
        self._role_keys: List[int] = sorted(structure.roles.keys())
        for r_idx in self._role_keys:
            r_coal = structure.roles[r_idx].coalesce()
            r_ind = r_coal.indices()
            r_val = r_coal.values().bool()
            edges = (
                r_ind[:, r_val].to(dtype=torch.long)
                if r_val.numel() > 0
                else torch.empty((2, 0), dtype=torch.long)
            )
            r_float_vals = torch.ones((edges.shape[1],), dtype=torch.float32)
            r_sparse = torch.sparse_coo_tensor(
                edges,
                r_float_vals,
                size=(self.n_elements, self.n_elements),
                dtype=torch.float32,
            ).coalesce()
            self.register_buffer(f"_role_mat_{r_idx}", r_sparse)

        # Precompute CSR admissible supports S_a from full membership rows M[x_a, :]
        support_offsets, support_indices, own_offsets = self._build_admissible_supports(
            training_class_assertions
        )
        self.register_buffer("support_offsets", support_offsets)
        self.register_buffer("support_indices", support_indices)
        self.register_buffer("own_logit_offset", own_offsets)
        self.register_buffer(
            "is_full_support",
            (support_offsets[1:] - support_offsets[:-1]) == self.n_elements,
        )
        self._use_full_support_fast_path: bool = True

    @property
    def support_lengths(self) -> torch.Tensor:
        """Return 1-D ``LongTensor`` of shape ``(n_ind,)`` with ``|S_a|`` per individual."""
        return self.support_offsets[1:] - self.support_offsets[:-1]

    def _is_entailed_for_elem(
        self, elem_idx: torch.Tensor, cls_idx: torch.Tensor
    ) -> torch.Tensor:
        """Check ``M[elem_idx, cls_idx] == True`` via binary search on :attr:`_m_flat_keys`."""
        valid = (
            (elem_idx >= 0)
            & (elem_idx < self.n_elements)
            & (cls_idx >= 0)
            & (cls_idx < self.n_classes)
        )
        if self._m_flat_keys.numel() == 0 or not bool(valid.any().item()):
            return torch.zeros_like(valid, dtype=torch.bool)

        safe_elem = elem_idx.clamp(min=0, max=max(self.n_elements - 1, 0))
        safe_cls = cls_idx.clamp(min=0, max=max(self.n_classes - 1, 0))
        keys = safe_elem * self.n_classes + safe_cls
        pos = torch.searchsorted(self._m_flat_keys, keys)
        max_pos = self._m_flat_keys.numel() - 1
        found = self._m_flat_keys.index_select(0, pos.clamp(max=max_pos)) == keys
        return valid & (pos <= max_pos) & found

    def _validate_training_class_assertions(
        self, training_class_assertions: Optional[torch.Tensor]
    ) -> None:
        """Validate shape, bounds, and M-entailment of ``training_class_assertions``."""
        if (
            training_class_assertions is not None
            and training_class_assertions.numel() > 0
        ):
            ca = training_class_assertions.detach().cpu().to(dtype=torch.long)
            if ca.ndim != 2 or ca.shape[1] != 2:
                raise ValueError(
                    f"training_class_assertions must have shape (N, 2), got {tuple(ca.shape)}"
                )
            ind_col = ca[:, 0]
            cls_col = ca[:, 1]
            if bool(((ind_col < 0) | (ind_col >= self.n_ind)).any().item()):
                raise ValueError("Individual index in training_class_assertions is out of bounds.")
            if bool(((cls_col < 0) | (cls_col >= self.n_classes)).any().item()):
                raise ValueError("Class index in training_class_assertions is out of bounds.")
            own_elems_ca = self.individual_element.cpu().index_select(0, ind_col)
            entailed_mask = self._is_entailed_for_elem(own_elems_ca, cls_col)
            if not bool(entailed_mask.all().item()):
                bad_pos = int(torch.nonzero(~entailed_mask, as_tuple=False)[0].item())
                bad_ind = int(ind_col[bad_pos].item())
                bad_cls = int(cls_col[bad_pos].item())
                raise ValueError(
                    f"Training class assertion (individual={bad_ind}, class={bad_cls}) is not entailed by M."
                )

    def _build_admissible_supports(
        self, training_class_assertions: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute CSR admissible element offsets, concatenated indices, and own-element logit offsets.

        Computes :math:`S_a` as the intersection of the sorted CSC column sets of
        :math:`M` over the *minimal* (most specific) training-entailed types of
        individual :math:`a` (types none of whose strict subclasses is also a type
        of :math:`a`, derived from :math:`M`'s rows).
        """
        self._validate_training_class_assertions(training_class_assertions)

        support_lists: List[torch.Tensor] = []
        lengths = torch.empty((self.n_ind,), dtype=torch.long)
        own_offsets = torch.empty((self.n_ind,), dtype=torch.float32)
        odds = self.init_self_mass / (1.0 - self.init_self_mass)

        m_crow_np = self._m_crow_offsets.cpu().numpy()
        m_cols_np = self._m_col_indices.cpu().numpy()
        m_ccol_np = self._m_ccol_offsets.cpu().numpy()
        m_csc_rows_np = self._m_csc_row_indices.cpu().numpy()
        ind_elems_np = self.individual_element.cpu().numpy()

        # Map each class index c to its canonical element row in S (or -1 if absent, e.g. TOP)
        class_to_elem = np.full((self.n_classes,), -1, dtype=np.int64)
        if self.structure.class_index_dict and self.structure.element_index:
            for c_iri, c_idx in self.structure.class_index_dict.items():
                if 0 <= c_idx < self.n_classes:
                    el_idx = self.structure.element_index.get(c_iri, -1)
                    if el_idx >= 0:
                        class_to_elem[c_idx] = el_idx

        for ind_i in range(self.n_ind):
            own_el = int(ind_elems_np[ind_i])
            if own_el < 0:
                # Individual occurs only in validation/test (no training assertions): support = all of S
                elems = torch.arange(self.n_elements, dtype=torch.long)
                s_len = int(elems.numel())
                lengths[ind_i] = s_len
                own_offsets[ind_i] = 0.0
                support_lists.append(elems)
                continue

            row_start = int(m_crow_np[own_el])
            row_end = int(m_crow_np[own_el + 1])
            t_arr = m_cols_np[row_start:row_end]

            if t_arr.size == 0:
                if not (0 <= own_el < self.n_elements):
                    raise ValueError(
                        f"Own element {own_el} of individual {ind_i} is not in its admissible support."
                    )
                elems = torch.arange(self.n_elements, dtype=torch.long)
            else:
                if t_arr.size == 1:
                    c0 = int(t_arr[0])
                    elems_np = m_csc_rows_np[m_ccol_np[c0] : m_ccol_np[c0 + 1]]
                else:
                    # Find minimal types of a: types in t_arr none of whose strict subclasses is in t_arr.
                    # Class c is a strict subclass of d iff d is in M[r_c, :] and c is not in M[r_d, :].
                    non_minimal = set()
                    for c in t_arr:
                        c_int = int(c)
                        if c_int in non_minimal:
                            continue
                        r_c = int(class_to_elem[c_int])
                        if r_c >= 0:
                            sups_c = m_cols_np[m_crow_np[r_c] : m_crow_np[r_c + 1]]
                            for d in sups_c:
                                d_int = int(d)
                                if d_int == c_int or d_int in non_minimal:
                                    continue
                                r_d = int(class_to_elem[d_int])
                                if r_d < 0:
                                    non_minimal.add(d_int)
                                else:
                                    sups_d = m_cols_np[
                                        m_crow_np[r_d] : m_crow_np[r_d + 1]
                                    ]
                                    if not np.any(sups_d == c_int):
                                        non_minimal.add(d_int)
                    min_types = [int(c) for c in t_arr if int(c) not in non_minimal]
                    if not min_types:
                        min_types = [int(c) for c in t_arr]
                    # Intersect sorted CSC columns starting from the smallest column
                    min_types.sort(
                        key=lambda c: int(m_ccol_np[c + 1] - m_ccol_np[c])
                    )
                    c0 = min_types[0]
                    elems_np = m_csc_rows_np[m_ccol_np[c0] : m_ccol_np[c0 + 1]]
                    for c in min_types[1:]:
                        col_c = m_csc_rows_np[m_ccol_np[c] : m_ccol_np[c + 1]]
                        elems_np = np.intersect1d(
                            elems_np, col_c, assume_unique=True
                        )
                        if elems_np.size <= 1:
                            break

                pos = int(np.searchsorted(elems_np, own_el))
                if pos >= elems_np.size or int(elems_np[pos]) != own_el:
                    raise ValueError(
                        f"Own element {own_el} of individual {ind_i} is not in its admissible support."
                    )
                elems = torch.from_numpy(elems_np.copy()).to(dtype=torch.long)

            support_lists.append(elems)
            s_len = int(elems.numel())
            lengths[ind_i] = s_len
            # Own-element logit offset: log(9 * (|S_a| - 1)) when |S_a| > 1, 0 when |S_a| == 1
            if s_len > 1:
                own_offsets[ind_i] = math.log(odds * float(s_len - 1))
            else:
                own_offsets[ind_i] = 0.0

        support_offsets = torch.zeros((self.n_ind + 1,), dtype=torch.long)
        support_offsets[1:] = torch.cumsum(lengths, dim=0)
        support_indices = (
            torch.cat(support_lists, dim=0)
            if support_lists
            else torch.empty((0,), dtype=torch.long)
        )
        return support_offsets, support_indices, own_offsets

    def _build_admissible_supports_reference(
        self, training_class_assertions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Private reference implementation of support construction via :meth:`MembershipStructure.type_mask`."""
        self._validate_training_class_assertions(training_class_assertions)

        support_lists: List[torch.Tensor] = []
        lengths = torch.empty((self.n_ind,), dtype=torch.long)
        own_offsets = torch.empty((self.n_ind,), dtype=torch.float32)
        odds = self.init_self_mass / (1.0 - self.init_self_mass)

        m_crow = self._m_crow_offsets.cpu()
        m_cols = self._m_col_indices.cpu()
        ind_elems_cpu = self.individual_element.cpu()

        for ind_i in range(self.n_ind):
            own_el = int(ind_elems_cpu[ind_i].item())
            if own_el < 0:
                elems = torch.arange(self.n_elements, dtype=torch.long)
                s_len = int(elems.numel())
                lengths[ind_i] = s_len
                own_offsets[ind_i] = 0.0
                support_lists.append(elems)
                continue

            row_start = int(m_crow[own_el].item())
            row_end = int(m_crow[own_el + 1].item())
            t_tensor = m_cols[row_start:row_end]
            mask = self.structure.type_mask(t_tensor).cpu()
            if not bool(mask[own_el].item()):
                raise ValueError(
                    f"Own element {own_el} of individual {ind_i} is not in its admissible support."
                )
            elems = torch.nonzero(mask, as_tuple=False).reshape(-1).to(dtype=torch.long)
            support_lists.append(elems)
            s_len = int(elems.numel())
            lengths[ind_i] = s_len
            if s_len > 1:
                own_offsets[ind_i] = math.log(odds * float(s_len - 1))
            else:
                own_offsets[ind_i] = 0.0

        support_offsets = torch.zeros((self.n_ind + 1,), dtype=torch.long)
        support_offsets[1:] = torch.cumsum(lengths, dim=0)
        support_indices = (
            torch.cat(support_lists, dim=0)
            if support_lists
            else torch.empty((0,), dtype=torch.long)
        )
        return support_offsets, support_indices, own_offsets

    def _batch_chunk_size(self, bsz: int) -> int:
        """Return row chunk size: at most 128 when ``|S| * bsz * 4`` bytes exceeds 8 MB."""
        if bsz <= 0:
            return 1
        max_bytes = 8 * 1024 * 1024
        if self.n_elements * bsz * 4 > max_bytes:
            return min(128, max(1, max_bytes // max(self.n_elements * 4, 1)))
        return bsz

    def _effective_support_costs(self, ind_idx: torch.Tensor) -> torch.Tensor:
        """Return per-individual support-entry cost (dense full-support rows cost 4 B/entry vs ~40 B gathered)."""
        starts = self.support_offsets.index_select(0, ind_idx)
        ends = self.support_offsets.index_select(0, ind_idx + 1)
        lengths = ends - starts
        if not self._use_full_support_fast_path:
            return lengths
        is_full = self.is_full_support.index_select(0, ind_idx)
        full_cost = max(1, self.n_elements // 10)
        return torch.where(is_full, torch.full_like(lengths, full_cost), lengths)

    def _ind_chunk_ranges(
        self, ind_idx: torch.Tensor, max_sup_entries: int = MAX_SUPPORT_ENTRIES
    ) -> List[Tuple[int, int]]:
        """Partition ``ind_idx`` into ``(start, end)`` slices each holding ``<= max_sup_entries`` support items."""
        bsz = int(ind_idx.shape[0])
        if bsz == 0:
            return []
        costs = self._effective_support_costs(ind_idx)
        total = int(costs.sum().item())
        if total <= max_sup_entries:
            return [(0, bsz)]
        cost_list = costs.tolist()
        ranges: List[Tuple[int, int]] = []
        s = 0
        acc = 0
        for i, L in enumerate(cost_list):
            if acc + L > max_sup_entries and i > s:
                ranges.append((s, i))
                s = i
                acc = L
            else:
                acc += L
        if s < bsz:
            ranges.append((s, bsz))
        return ranges

    def _pair_chunk_ranges(
        self, data: torch.Tensor, max_sup_entries: int = MAX_SUPPORT_ENTRIES
    ) -> List[Tuple[int, int]]:
        """Partition 2-D ``data`` rows so each slice's endpoints have ``<= max_sup_entries`` total support items and ``<= _batch_chunk_size`` rows."""
        bsz = int(data.shape[0])
        if bsz == 0:
            return []
        max_rows = self._batch_chunk_size(bsz)
        h_idx = data[:, 0]
        h_costs = self._effective_support_costs(h_idx)
        if data.shape[1] >= 3:
            t_idx = data[:, 2]
            t_costs = self._effective_support_costs(t_idx)
            row_costs = h_costs + t_costs
        else:
            row_costs = h_costs

        total_cost = int(row_costs.sum().item())
        if bsz <= max_rows and total_cost <= max_sup_entries:
            return [(0, bsz)]

        cost_list = row_costs.tolist()
        ranges: List[Tuple[int, int]] = []
        s = 0
        acc = 0
        for i, c in enumerate(cost_list):
            if (acc + c > max_sup_entries or (i - s) >= max_rows) and i > s:
                ranges.append((s, i))
                s = i
                acc = c
            else:
                acc += c
        if s < bsz:
            ranges.append((s, bsz))
        return ranges

    def _full_support_log_p(self, full_inds: torch.Tensor) -> torch.Tensor:
        """Compute dense ``(k, |S|)`` log-probabilities ``log_softmax(U[a] @ V^T + b + delta_a)`` for full-support individuals."""
        device = self.U.device
        dtype = self.U.dtype
        k = int(full_inds.shape[0])
        if k == 0:
            return torch.empty((0, self.n_elements), dtype=dtype, device=device)
        u_batch = self.U.index_select(0, full_inds)  # (k, r)
        logits = torch.addmm(self.b, u_batch, self.V.t())  # (k, |S|)
        own_elems = self.individual_element.index_select(0, full_inds)
        has_own = own_elems >= 0
        if bool(has_own.any().item()):
            own_rows = torch.nonzero(has_own, as_tuple=False).reshape(-1)
            own_cols = own_elems.index_select(0, own_rows)
            own_offs = (
                self.own_logit_offset.index_select(0, full_inds)
                .index_select(0, own_rows)
                .to(dtype=dtype)
            )
            logits[own_rows, own_cols] = logits[own_rows, own_cols] + own_offs
        return torch.log_softmax(logits, dim=1)

    def _expand_batch_supports(
        self,
        inv_idx: torch.Tensor,
        u_offsets: torch.Tensor,
        u_elem_ids: torch.Tensor,
        u_sup_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Expand segmented support entries ``(col_ids, elem_ids, probs, total)`` for ``inv_idx``."""
        c_bsz = inv_idx.shape[0]
        device = inv_idx.device
        starts = u_offsets.index_select(0, inv_idx)
        ends = u_offsets.index_select(0, inv_idx + 1)
        lengths = ends - starts
        total = int(lengths.sum().item())
        if total == 0:
            empty_l = torch.empty((0,), dtype=torch.long, device=device)
            empty_f = torch.empty((0,), dtype=u_sup_probs.dtype, device=device)
            return empty_l, empty_l, empty_f, 0
        col_ids = torch.repeat_interleave(
            torch.arange(c_bsz, device=device, dtype=torch.long), lengths
        )
        cum_len = torch.cumsum(lengths, dim=0) - lengths
        flat_pos = (
            torch.arange(total, device=device, dtype=torch.long)
            - torch.repeat_interleave(cum_len, lengths)
            + torch.repeat_interleave(starts, lengths)
        )
        elems = u_elem_ids.index_select(0, flat_pos)
        probs = u_sup_probs.index_select(0, flat_pos)
        return col_ids, elems, probs, total

    def _sup_log_p_and_probs_single(
        self, ind_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather logits on :math:`S_a` for one bounded slice of individuals."""
        bsz = ind_idx.shape[0]
        device = self.U.device
        dtype = self.U.dtype
        if bsz == 0:
            empty_l = torch.empty((0,), dtype=torch.long, device=device)
            empty_f = torch.empty((0,), dtype=dtype, device=device)
            return empty_l, empty_l, empty_f, empty_f, empty_l

        starts = self.support_offsets.index_select(0, ind_idx)
        ends = self.support_offsets.index_select(0, ind_idx + 1)
        lengths = ends - starts
        total_sup = int(lengths.sum().item())
        if total_sup == 0:
            empty_l = torch.empty((0,), dtype=torch.long, device=device)
            empty_f = torch.empty((0,), dtype=dtype, device=device)
            return empty_l, empty_l, empty_f, empty_f, lengths

        row_ids = torch.repeat_interleave(
            torch.arange(bsz, device=device, dtype=torch.long), lengths
        )
        cum_len = torch.cumsum(lengths, dim=0) - lengths
        flat_pos = (
            torch.arange(total_sup, device=device, dtype=torch.long)
            - torch.repeat_interleave(cum_len, lengths)
            + torch.repeat_interleave(starts, lengths)
        )
        elem_ids = self.support_indices.index_select(0, flat_pos)

        # Compute logits exclusively on the gathered support entries (in chunks if large)
        u_batch = self.U.index_select(0, ind_idx)  # (B, r)
        sup_chunk = 262_144
        if total_sup <= sup_chunk:
            sup_logits = (
                u_batch.index_select(0, row_ids) * self.V.index_select(0, elem_ids)
            ).sum(dim=1) + self.b.index_select(0, elem_ids)
        else:
            logit_parts = []
            for s in range(0, total_sup, sup_chunk):
                r_c = row_ids[s : s + sup_chunk]
                e_c = elem_ids[s : s + sup_chunk]
                logit_parts.append(
                    (u_batch.index_select(0, r_c) * self.V.index_select(0, e_c)).sum(
                        dim=1
                    )
                    + self.b.index_select(0, e_c)
                )
            sup_logits = torch.cat(logit_parts, dim=0)

        own_elems = self.individual_element.index_select(0, ind_idx)
        own_offs = self.own_logit_offset.index_select(0, ind_idx).to(dtype=dtype)
        row_own_elem = own_elems.index_select(0, row_ids)
        is_own = (row_own_elem >= 0) & (elem_ids == row_own_elem)
        sup_logits = torch.where(
            is_own, sup_logits + own_offs.index_select(0, row_ids), sup_logits
        )

        # Segmented log-softmax over gathered support entries only
        row_max = torch.full((bsz,), float("-inf"), dtype=dtype, device=device)
        row_max.scatter_reduce_(
            0, row_ids, sup_logits.detach(), reduce="amax", include_self=False
        )
        shifted = sup_logits - row_max.index_select(0, row_ids)
        exp_shifted = torch.exp(shifted)
        sum_exp = torch.zeros((bsz,), dtype=dtype, device=device)
        sum_exp.scatter_add_(0, row_ids, exp_shifted)
        sup_log_probs = shifted - torch.log(sum_exp).index_select(0, row_ids)
        sup_probs = exp_shifted / sum_exp.index_select(0, row_ids)
        return row_ids, elem_ids, sup_log_probs, sup_probs, lengths

    def _sup_log_p_and_probs(
        self, ind_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather logits on :math:`S_a` and compute segmented ``(row_ids, elem_ids, sup_log_probs, sup_probs, lengths)``."""
        ind_idx = torch.as_tensor(
            ind_idx, dtype=torch.long, device=self.U.device
        ).reshape(-1)
        bsz = ind_idx.shape[0]
        device = self.U.device
        dtype = self.U.dtype
        if bsz == 0:
            empty_l = torch.empty((0,), dtype=torch.long, device=device)
            empty_f = torch.empty((0,), dtype=dtype, device=device)
            return empty_l, empty_l, empty_f, empty_f, empty_l

        ranges = self._ind_chunk_ranges(ind_idx)
        if len(ranges) == 1:
            return self._sup_log_p_and_probs_single(ind_idx)

        row_parts: List[torch.Tensor] = []
        elem_parts: List[torch.Tensor] = []
        logp_parts: List[torch.Tensor] = []
        prob_parts: List[torch.Tensor] = []
        len_parts: List[torch.Tensor] = []
        for s, e in ranges:
            r_i, el_i, lp_i, pr_i, l_i = self._sup_log_p_and_probs_single(
                ind_idx[s:e]
            )
            row_parts.append(r_i + s)
            elem_parts.append(el_i)
            logp_parts.append(lp_i)
            prob_parts.append(pr_i)
            len_parts.append(l_i)
        return (
            torch.cat(row_parts, dim=0),
            torch.cat(elem_parts, dim=0),
            torch.cat(logp_parts, dim=0),
            torch.cat(prob_parts, dim=0),
            torch.cat(len_parts, dim=0),
        )

    def _probs(self, ind_idx: torch.Tensor) -> torch.Tensor:
        """Return masked probability distribution ``p_a`` of shape ``(B, |S|)``."""
        ind_idx = torch.as_tensor(
            ind_idx, dtype=torch.long, device=self.U.device
        ).reshape(-1)
        bsz = ind_idx.shape[0]
        device = self.U.device
        probs = torch.zeros(
            (bsz, self.n_elements), dtype=self.U.dtype, device=device
        )
        if bsz == 0:
            return probs
        for s, e in self._ind_chunk_ranges(ind_idx):
            sub_inds = ind_idx[s:e]
            if self._use_full_support_fast_path:
                sub_is_full = self.is_full_support.index_select(0, sub_inds)
            else:
                sub_is_full = torch.zeros(
                    (sub_inds.shape[0],), dtype=torch.bool, device=device
                )
            if bool(sub_is_full.any().item()):
                full_pos = torch.nonzero(sub_is_full, as_tuple=False).reshape(-1)
                probs[full_pos + s] = torch.exp(
                    self._full_support_log_p(sub_inds.index_select(0, full_pos))
                )
            if bool((~sub_is_full).any().item()):
                part_pos = torch.nonzero(~sub_is_full, as_tuple=False).reshape(-1)
                row_ids, elem_ids, _, sup_probs, _ = (
                    self._sup_log_p_and_probs_single(
                        sub_inds.index_select(0, part_pos)
                    )
                )
                probs[part_pos.index_select(0, row_ids) + s, elem_ids] = sup_probs
        return probs

    def log_p(self, ind_idx: torch.Tensor) -> torch.Tensor:
        """Compute masked log-probabilities ``log p_a`` of shape ``(B, |S|)`` for ``ind_idx``.

        Log-softmax is computed exclusively over the admissible support
        :math:`S_a`, and entries outside :math:`S_a` are ``-inf``.
        """
        ind_idx = torch.as_tensor(
            ind_idx, dtype=torch.long, device=self.U.device
        ).reshape(-1)
        bsz = ind_idx.shape[0]
        device = self.U.device
        out = torch.full(
            (bsz, self.n_elements),
            float("-inf"),
            dtype=self.U.dtype,
            device=device,
        )
        if bsz == 0:
            return out
        for s, e in self._ind_chunk_ranges(ind_idx):
            sub_inds = ind_idx[s:e]
            if self._use_full_support_fast_path:
                sub_is_full = self.is_full_support.index_select(0, sub_inds)
            else:
                sub_is_full = torch.zeros(
                    (sub_inds.shape[0],), dtype=torch.bool, device=device
                )
            if bool(sub_is_full.any().item()):
                full_pos = torch.nonzero(sub_is_full, as_tuple=False).reshape(-1)
                out[full_pos + s] = self._full_support_log_p(
                    sub_inds.index_select(0, full_pos)
                )
            if bool((~sub_is_full).any().item()):
                part_pos = torch.nonzero(~sub_is_full, as_tuple=False).reshape(-1)
                row_ids, elem_ids, sup_log_probs, _, _ = (
                    self._sup_log_p_and_probs_single(
                        sub_inds.index_select(0, part_pos)
                    )
                )
                out[part_pos.index_select(0, row_ids) + s, elem_ids] = sup_log_probs
        return out

    def _class_assertion_prob_chunk(self, data: torch.Tensor) -> torch.Tensor:
        """Compute class-membership probability for a single row chunk of ``data``."""
        ind_idx = data[:, 0]
        cls_idx = data[:, 1]
        bsz = data.shape[0]
        device = self.U.device
        dtype = self.U.dtype

        unique_inds, inv_ind = torch.unique(ind_idx, return_inverse=True)
        out = torch.zeros((bsz,), dtype=dtype, device=device)

        for s, e in self._ind_chunk_ranges(unique_inds):
            sub_u = unique_inds[s:e]
            mask_b = (inv_ind >= s) & (inv_ind < e)
            rows_b = torch.nonzero(mask_b, as_tuple=False).reshape(-1)
            if rows_b.numel() == 0:
                continue
            local_inv = inv_ind.index_select(0, rows_b) - s

            if self._use_full_support_fast_path:
                u_is_full = self.is_full_support.index_select(0, sub_u)
            else:
                u_is_full = torch.zeros(
                    (sub_u.numel(),), dtype=torch.bool, device=device
                )

            # 1. Dense fast path for full-support individuals (S_a = S):
            # log_softmax(U[a] @ V^T + b) multiplied with M[:, C] (sparse) for class score and complement sum.
            if bool(u_is_full.any().item()):
                full_u_pos = torch.nonzero(u_is_full, as_tuple=False).reshape(-1)
                full_u = sub_u.index_select(0, full_u_pos)
                row_is_full = u_is_full.index_select(0, local_inv)
                full_rows_b = rows_b.index_select(
                    0, torch.nonzero(row_is_full, as_tuple=False).reshape(-1)
                )
                if full_rows_b.numel() > 0:
                    u_to_full = torch.full(
                        (sub_u.numel(),), -1, dtype=torch.long, device=device
                    )
                    u_to_full[full_u_pos] = torch.arange(
                        full_u.numel(), dtype=torch.long, device=device
                    )
                    full_local_inv = u_to_full.index_select(
                        0, inv_ind.index_select(0, full_rows_b) - s
                    )
                    full_probs = torch.exp(self._full_support_log_p(full_u))  # (k, |S|)
                    c_queries = cls_idx.index_select(0, full_rows_b)
                    c_uniq, c_inv = torch.unique(c_queries, return_inverse=True)
                    c_starts = self._m_ccol_offsets.index_select(0, c_uniq)
                    c_ends = self._m_ccol_offsets.index_select(0, c_uniq + 1)
                    c_lens = c_ends - c_starts
                    total_nnz = int(c_lens.sum().item())

                    if total_nnz > 0:
                        n_c = int(c_uniq.numel())
                        col_ids = torch.repeat_interleave(
                            torch.arange(n_c, device=device, dtype=torch.long),
                            c_lens,
                        )
                        cum_c = torch.cumsum(c_lens, dim=0) - c_lens
                        flat_c_pos = (
                            torch.arange(total_nnz, device=device, dtype=torch.long)
                            - torch.repeat_interleave(cum_c, c_lens)
                            + torch.repeat_interleave(c_starts, c_lens)
                        )
                        row_elem_ids = self._m_csc_row_indices.index_select(
                            0, flat_c_pos
                        )
                        m_sub_t = torch.sparse_coo_tensor(
                            torch.stack([col_ids, row_elem_ids], dim=0),
                            torch.ones((total_nnz,), dtype=dtype, device=device),
                            size=(n_c, self.n_elements),
                            dtype=dtype,
                            device=device,
                        ).coalesce()
                        # M[:, c_uniq]^T @ full_probs^T -> (n_c, k)
                        in_c_mat = torch.sparse.mm(m_sub_t, full_probs.t())
                        in_c_mass = in_c_mat[c_inv, full_local_inv]
                        row_c_len = c_lens.index_select(0, c_inv)
                        row_prob_sum = full_probs.sum(dim=1).index_select(
                            0, full_local_inv
                        )
                        sum_not_c = torch.where(
                            row_c_len >= self.n_elements,
                            torch.zeros_like(in_c_mass),
                            (row_prob_sum - in_c_mass).clamp(min=0.0),
                        )
                        c_probs_full = torch.where(
                            row_c_len > 0,
                            (1.0 - sum_not_c).clamp(min=0.0, max=1.0),
                            torch.zeros_like(in_c_mass),
                        )
                        out = out.scatter(0, full_rows_b, c_probs_full)

            # 2. Gathered path for proper-subset support individuals (S_a != S):
            if bool((~u_is_full).any().item()):
                part_u_pos = torch.nonzero(~u_is_full, as_tuple=False).reshape(-1)
                part_u = sub_u.index_select(0, part_u_pos)
                row_is_part = ~u_is_full.index_select(0, local_inv)
                part_rows_b = rows_b.index_select(
                    0, torch.nonzero(row_is_part, as_tuple=False).reshape(-1)
                )
                if part_rows_b.numel() > 0:
                    c_bsz = int(part_rows_b.numel())
                    u_to_part = torch.full(
                        (sub_u.numel(),), -1, dtype=torch.long, device=device
                    )
                    u_to_part[part_u_pos] = torch.arange(
                        part_u.numel(), dtype=torch.long, device=device
                    )
                    part_local_inv = u_to_part.index_select(
                        0, inv_ind.index_select(0, part_rows_b) - s
                    )
                    _, u_elem_ids, _, u_sup_probs, u_lengths = (
                        self._sup_log_p_and_probs_single(part_u)
                    )
                    u_offsets = torch.zeros(
                        (part_u.numel() + 1,), dtype=torch.long, device=device
                    )
                    u_offsets[1:] = torch.cumsum(u_lengths, dim=0)

                    batch_ids, x_elems, p_vals, total_pairs = (
                        self._expand_batch_supports(
                            part_local_inv, u_offsets, u_elem_ids, u_sup_probs
                        )
                    )
                    if total_pairs > 0:
                        c_vals = cls_idx.index_select(0, part_rows_b).index_select(
                            0, batch_ids
                        )
                        in_c = self._is_entailed_for_elem(x_elems, c_vals)
                        not_in_c = ~in_c

                        sum_not_c = torch.zeros(
                            (c_bsz,), dtype=p_vals.dtype, device=device
                        )
                        if bool(not_in_c.any().item()):
                            sum_not_c.scatter_add_(
                                0, batch_ids[not_in_c], p_vals[not_in_c]
                            )

                        has_in_c = torch.zeros(
                            (c_bsz,), dtype=torch.bool, device=device
                        )
                        if bool(in_c.any().item()):
                            has_in_c[batch_ids[in_c]] = True

                        c_probs_part = torch.where(
                            has_in_c,
                            (1.0 - sum_not_c).clamp(min=0.0, max=1.0),
                            torch.zeros_like(sum_not_c),
                        )
                        out = out.scatter(0, part_rows_b, c_probs_part)

        return out

    def _class_assertion_prob(self, data: torch.Tensor) -> torch.Tensor:
        """Return class-membership probability :math:`1 - \\sum_{x \\in S_a, x \\notin C} p_a(x) \\in [0, 1]`."""
        data = torch.as_tensor(data, dtype=torch.long, device=self.U.device)
        if data.numel() == 0:
            return torch.empty((0,), dtype=self.U.dtype, device=self.U.device)

        bsz = data.shape[0]
        chunk_size = self._batch_chunk_size(bsz)
        if bsz <= chunk_size:
            return self._class_assertion_prob_chunk(data)
        return torch.cat(
            [
                self._class_assertion_prob_chunk(data[s : s + chunk_size])
                for s in range(0, bsz, chunk_size)
            ],
            dim=0,
        )

    def class_assertion_score(self, data: torch.Tensor) -> torch.Tensor:
        """Return negative log probability :math:`-\\log(1 - \\sum_{x \\in S_a, x \\notin C} p_a(x))` of shape ``(B,)``.

        Parameters
        ----------
        data : torch.LongTensor
            2-D tensor of shape ``(B, 2)`` with ``[ind_idx, class_idx]`` rows.
        """
        return -torch.log(self._class_assertion_prob(data)) + 0.0

    def _object_property_assertion_prob(self, data: torch.Tensor) -> torch.Tensor:
        """Return role-assertion probability :math:`p_a \\cdot (E_r p_b) \\in [0, 1]` of shape ``(B,)`` via ``sparse.mm``."""
        data = torch.as_tensor(data, dtype=torch.long, device=self.U.device)
        if data.numel() == 0:
            return torch.empty((0,), dtype=self.U.dtype, device=self.U.device)

        head_idx = data[:, 0]
        rel_idx = data[:, 1]
        tail_idx = data[:, 2]
        bsz = data.shape[0]
        device = self.U.device
        dtype = self.U.dtype

        probs = torch.zeros((bsz,), dtype=dtype, device=device)
        for r_key in torch.unique(rel_idx).tolist():
            r_int = int(r_key)
            buf_name = f"_role_mat_{r_int}"
            if not hasattr(self, buf_name):
                continue
            e_r: torch.Tensor = getattr(self, buf_name)
            if e_r._nnz() == 0:
                continue
            mask = rel_idx == r_int
            rows_b = torch.nonzero(mask, as_tuple=False).reshape(-1)
            sub_data = data.index_select(0, rows_b)

            for s_idx, e_idx in self._pair_chunk_ranges(sub_data):
                c_rows = rows_b[s_idx:e_idx]
                c_bsz = int(c_rows.numel())
                c_heads = head_idx.index_select(0, c_rows)
                c_tails = tail_idx.index_select(0, c_rows)

                c_all = torch.cat([c_heads, c_tails], dim=0)
                c_uniq, c_inv = torch.unique(c_all, return_inverse=True)
                u_h = c_inv[:c_bsz]
                u_t = c_inv[c_bsz:]

                if self._use_full_support_fast_path:
                    u_is_full = self.is_full_support.index_select(0, c_uniq)
                else:
                    u_is_full = torch.zeros(
                        (c_uniq.numel(),), dtype=torch.bool, device=device
                    )

                n_uniq = int(c_uniq.numel())
                has_full = bool(u_is_full.any().item())
                has_part = bool((~u_is_full).any().item())

                full_probs: Optional[torch.Tensor] = None
                u_to_full: Optional[torch.Tensor] = None
                if has_full:
                    full_u_pos = torch.nonzero(u_is_full, as_tuple=False).reshape(-1)
                    full_inds = c_uniq.index_select(0, full_u_pos)
                    full_probs = torch.exp(
                        self._full_support_log_p(full_inds)
                    )  # (n_full, |S|)
                    u_to_full = torch.full(
                        (n_uniq,), -1, dtype=torch.long, device=device
                    )
                    u_to_full[full_u_pos] = torch.arange(
                        full_inds.numel(), dtype=torch.long, device=device
                    )

                part_elem_ids: Optional[torch.Tensor] = None
                part_sup_probs: Optional[torch.Tensor] = None
                part_offsets: Optional[torch.Tensor] = None
                u_to_part: Optional[torch.Tensor] = None
                if has_part:
                    part_u_pos = torch.nonzero(~u_is_full, as_tuple=False).reshape(-1)
                    part_inds = c_uniq.index_select(0, part_u_pos)
                    _, part_elem_ids, _, part_sup_probs, part_lengths = (
                        self._sup_log_p_and_probs_single(part_inds)
                    )
                    part_offsets = torch.zeros(
                        (part_inds.numel() + 1,), dtype=torch.long, device=device
                    )
                    part_offsets[1:] = torch.cumsum(part_lengths, dim=0)
                    u_to_part = torch.full(
                        (n_uniq,), -1, dtype=torch.long, device=device
                    )
                    u_to_part[part_u_pos] = torch.arange(
                        part_inds.numel(), dtype=torch.long, device=device
                    )

                p_b_col = torch.zeros(
                    (self.n_elements, c_bsz), dtype=dtype, device=device
                )
                t_is_full = u_is_full.index_select(0, u_t)
                if has_full and bool(t_is_full.any().item()):
                    assert full_probs is not None and u_to_full is not None
                    t_full_cols = torch.nonzero(t_is_full, as_tuple=False).reshape(-1)
                    t_full_idx = u_to_full.index_select(
                        0, u_t.index_select(0, t_full_cols)
                    )
                    p_b_col[:, t_full_cols] = full_probs.index_select(0, t_full_idx).t()

                if has_part and bool((~t_is_full).any().item()):
                    assert (
                        part_offsets is not None
                        and part_elem_ids is not None
                        and part_sup_probs is not None
                        and u_to_part is not None
                    )
                    t_part_cols = torch.nonzero(~t_is_full, as_tuple=False).reshape(-1)
                    t_part_inv = u_to_part.index_select(
                        0, u_t.index_select(0, t_part_cols)
                    )
                    t_sub_cols, t_elems, t_probs, total_t = (
                        self._expand_batch_supports(
                            t_part_inv, part_offsets, part_elem_ids, part_sup_probs
                        )
                    )
                    if total_t > 0:
                        p_b_col[
                            t_elems, t_part_cols.index_select(0, t_sub_cols)
                        ] = t_probs

                # q = E_r @ p_b of shape (|S|, c_bsz), then dot with p_a
                q = torch.sparse.mm(e_r, p_b_col)
                c_probs = torch.zeros((c_bsz,), dtype=dtype, device=device)

                h_is_full = u_is_full.index_select(0, u_h)
                if has_full and bool(h_is_full.any().item()):
                    assert full_probs is not None and u_to_full is not None
                    h_full_cols = torch.nonzero(h_is_full, as_tuple=False).reshape(-1)
                    h_full_idx = u_to_full.index_select(
                        0, u_h.index_select(0, h_full_cols)
                    )
                    p_a_full = full_probs.index_select(0, h_full_idx)
                    q_full = q.index_select(1, h_full_cols).t()
                    c_probs[h_full_cols] = (p_a_full * q_full).sum(dim=1)

                if has_part and bool((~h_is_full).any().item()):
                    assert (
                        part_offsets is not None
                        and part_elem_ids is not None
                        and part_sup_probs is not None
                        and u_to_part is not None
                    )
                    h_part_cols = torch.nonzero(~h_is_full, as_tuple=False).reshape(-1)
                    h_part_inv = u_to_part.index_select(
                        0, u_h.index_select(0, h_part_cols)
                    )
                    h_sub_cols, h_elems, h_probs, total_h = (
                        self._expand_batch_supports(
                            h_part_inv, part_offsets, part_elem_ids, part_sup_probs
                        )
                    )
                    if total_h > 0:
                        orig_h_cols = h_part_cols.index_select(0, h_sub_cols)
                        q_at_h = q[h_elems, orig_h_cols]
                        c_probs.scatter_add_(0, orig_h_cols, h_probs * q_at_h)

                probs = probs.scatter_add(0, c_rows, c_probs)
                del p_b_col, q

        return probs.clamp(min=0.0, max=1.0)

    def object_property_assertion_score(self, data: torch.Tensor) -> torch.Tensor:
        """Return negative log probability :math:`-\\log (p_a \\cdot (E_r p_b))` of shape ``(B,)``.

        Computes :math:`p_a \\cdot (E_r p_b)` via :func:`torch.sparse.mm` without
        allocating a dense :math:`(|S|, |S|)` or :math:`(B, \\operatorname{nnz}(E_r))` tensor.

        Parameters
        ----------
        data : torch.LongTensor
            2-D tensor of shape ``(B, 3)`` with ``[head_ind, role_idx, tail_ind]`` rows.
        """
        return -torch.log(self._object_property_assertion_prob(data)) + 0.0

    def class_assertion_loss(
        self, data: torch.Tensor, neg: bool = False
    ) -> torch.Tensor:
        """BCE loss on :math:`s = \\exp(-\\operatorname{score})` for class assertions."""
        probs = self._class_assertion_prob(data).clamp(min=1e-7, max=1.0 - 1e-7)
        if neg:
            return -torch.log(1.0 - probs)
        return -torch.log(probs)

    def object_property_assertion_loss(
        self, data: torch.Tensor, neg: bool = False
    ) -> torch.Tensor:
        """BCE loss on :math:`s = \\exp(-\\operatorname{score})` for object property assertions."""
        probs = self._object_property_assertion_prob(data).clamp(
            min=1e-7, max=1.0 - 1e-7
        )
        if neg:
            return -torch.log(1.0 - probs)
        return -torch.log(probs)

    def sample_class_negatives(self, pos_data: torch.Tensor) -> torch.Tensor:
        """Sample one non-entailed class per positive ``(ind_idx, class_idx)`` row via rejection sampling."""
        pos_data = torch.as_tensor(pos_data, dtype=torch.long, device=self.U.device)
        if pos_data.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long, device=self.U.device)
        ind_col = pos_data[:, 0]
        bsz = ind_col.shape[0]
        device = self.U.device

        own_elems = self.individual_element.index_select(0, ind_col)
        neg_classes = torch.randint(
            0, self.n_classes, (bsz,), device=device, dtype=torch.long
        )
        bad = self._is_entailed_for_elem(own_elems, neg_classes)

        # Resample up to 20 times in total (1 initial + 19 retries)
        for _ in range(19):
            if not bool(bad.any().item()):
                break
            n_bad = int(bad.sum().item())
            neg_classes[bad] = torch.randint(
                0, self.n_classes, (n_bad,), device=device, dtype=torch.long
            )
            bad = self._is_entailed_for_elem(own_elems, neg_classes)

        # Fallback: find any class with M[x_a, C] = 0 via sparse lookup into _m_crow_offsets / _m_col_indices
        if bool(bad.any().item()):
            bad_rows = torch.nonzero(bad, as_tuple=False).reshape(-1).tolist()
            for r_idx in bad_rows:
                el = int(own_elems[r_idx].item())
                if el < 0:
                    continue
                start = int(self._m_crow_offsets[el].item())
                end = int(self._m_crow_offsets[el + 1].item())
                entailed_set = set(self._m_col_indices[start:end].tolist())
                if len(entailed_set) < self.n_classes:
                    c0 = int(
                        torch.randint(
                            0, self.n_classes, (1,), device=device, dtype=torch.long
                        ).item()
                    )
                    for step in range(self.n_classes):
                        cand = (c0 + step) % self.n_classes
                        if cand not in entailed_set:
                            neg_classes[r_idx] = cand
                            break

        return torch.stack([ind_col, neg_classes], dim=1)

    def sample_role_negatives(self, pos_data: torch.Tensor) -> torch.Tensor:
        """Sample one corrupted tail individual per positive ``(head_ind, role_idx, tail_ind)`` row."""
        pos_data = torch.as_tensor(pos_data, dtype=torch.long, device=self.U.device)
        if pos_data.numel() == 0:
            return torch.empty((0, 3), dtype=torch.long, device=self.U.device)
        bsz = pos_data.shape[0]
        chunk_size = self._batch_chunk_size(bsz)
        if bsz <= chunk_size:
            corrupt_tail = torch.randint(
                0, self.n_ind, (bsz,), device=self.U.device, dtype=torch.long
            )
            return torch.stack([pos_data[:, 0], pos_data[:, 1], corrupt_tail], dim=1)
        out = torch.empty((bsz, 3), dtype=torch.long, device=self.U.device)
        for start in range(0, bsz, chunk_size):
            end = min(bsz, start + chunk_size)
            out[start:end, 0] = pos_data[start:end, 0]
            out[start:end, 1] = pos_data[start:end, 1]
            out[start:end, 2] = torch.randint(
                0, self.n_ind, (end - start,), device=self.U.device, dtype=torch.long
            )
        return out

    def regularization_loss(self, ind_idx: torch.Tensor) -> torch.Tensor:
        """Return the entropy regularisation term :math:`\\lambda \\cdot \\operatorname{mean}_a H(p_a)` for ``ind_idx``."""
        if ind_idx is None:
            raise TypeError("regularization_loss requires an ind_idx batch tensor.")
        if self.entropy_reg == 0.0:
            return self.U.sum() * 0.0
        ind_idx = torch.as_tensor(
            ind_idx, dtype=torch.long, device=self.U.device
        ).reshape(-1)
        bsz = ind_idx.shape[0]
        if bsz == 0:
            return self.U.sum() * 0.0
        device = self.U.device
        dtype = self.U.dtype
        ent_parts: List[torch.Tensor] = []
        for s, e in self._ind_chunk_ranges(ind_idx):
            sub_inds = ind_idx[s:e]
            c_bsz = int(sub_inds.shape[0])
            ent_c = torch.zeros((c_bsz,), dtype=dtype, device=device)
            if self._use_full_support_fast_path:
                sub_is_full = self.is_full_support.index_select(0, sub_inds)
            else:
                sub_is_full = torch.zeros((c_bsz,), dtype=torch.bool, device=device)

            if bool(sub_is_full.any().item()):
                full_pos = torch.nonzero(sub_is_full, as_tuple=False).reshape(-1)
                lp_full = self._full_support_log_p(sub_inds.index_select(0, full_pos))
                p_full = torch.exp(lp_full)
                ent_c[full_pos] = -(p_full * lp_full).sum(dim=1)

            if bool((~sub_is_full).any().item()):
                part_pos = torch.nonzero(~sub_is_full, as_tuple=False).reshape(-1)
                row_ids, _, sup_log_probs, sup_probs, _ = (
                    self._sup_log_p_and_probs_single(sub_inds.index_select(0, part_pos))
                )
                ent_part = torch.zeros(
                    (part_pos.numel(),), dtype=dtype, device=device
                )
                ent_part.scatter_add_(0, row_ids, -sup_probs * sup_log_probs)
                ent_c[part_pos] = ent_part

            ent_parts.append(ent_c)
        ent_per_ind = torch.cat(ent_parts, dim=0)
        return self.entropy_reg * ent_per_ind.mean()

    def rank_classes(self, ind_idx: torch.Tensor) -> torch.Tensor:
        """Return ``(B, n_classes)`` negative-log class-membership scores for ``ind_idx``.

        Only the endpoints ``0`` and ``1`` of the membership probability are
        exact: entailed classes (all :math:`x \\in S_a` belong to :math:`C`) are
        forced to probability ``1.0`` (score ``0.0``) and classes with zero
        support overlap (no :math:`x \\in S_a` belongs to :math:`C`) are forced
        to probability ``0.0`` (score ``+inf``) via exact support counts from
        :func:`torch.sparse.mm`. For intermediate values, the complement mass is
        computed as ``row_sum - in_c`` rather than a direct complement sum, so
        intermediate scores are approximate; :meth:`class_assertion_score` is
        the exact path.
        """
        ind_idx = torch.as_tensor(
            ind_idx, dtype=torch.long, device=self.U.device
        ).reshape(-1)
        bsz = ind_idx.shape[0]
        device = self.U.device
        dtype = self.U.dtype
        if bsz == 0:
            return torch.empty((0, self.n_classes), dtype=dtype, device=device)

        chunk_size = self._batch_chunk_size(bsz)
        if bsz > chunk_size:
            return torch.cat(
                [
                    self.rank_classes(ind_idx[s : s + chunk_size])
                    for s in range(0, bsz, chunk_size)
                ],
                dim=0,
            )

        probs_elem = torch.zeros((bsz, self.n_elements), dtype=dtype, device=device)
        sup_mask_float = torch.zeros((bsz, self.n_elements), dtype=dtype, device=device)
        lengths = (
            self.support_offsets.index_select(0, ind_idx + 1)
            - self.support_offsets.index_select(0, ind_idx)
        )

        if self._use_full_support_fast_path:
            sub_is_full = self.is_full_support.index_select(0, ind_idx)
        else:
            sub_is_full = torch.zeros((bsz,), dtype=torch.bool, device=device)

        if bool(sub_is_full.any().item()):
            full_pos = torch.nonzero(sub_is_full, as_tuple=False).reshape(-1)
            probs_elem[full_pos] = torch.exp(
                self._full_support_log_p(ind_idx.index_select(0, full_pos))
            )
            sup_mask_float[full_pos] = 1.0

        if bool((~sub_is_full).any().item()):
            part_pos = torch.nonzero(~sub_is_full, as_tuple=False).reshape(-1)
            row_ids, elem_ids, _, sup_probs, _ = self._sup_log_p_and_probs(
                ind_idx.index_select(0, part_pos)
            )
            orig_rows = part_pos.index_select(0, row_ids)
            probs_elem[orig_rows, elem_ids] = sup_probs
            sup_mask_float[orig_rows, elem_ids] = 1.0

        # M is (|S|, n_classes), _m_float_t is (n_classes, |S|)
        in_c_probs = torch.sparse.mm(self._m_float_t, probs_elem.t()).t()  # (B, n_classes)
        in_c_counts = torch.sparse.mm(self._m_float_t, sup_mask_float.t()).t()  # (B, n_classes)

        # Complement mass sum_{x in S_a, x notin C} p_a(x) via sparse.mm on (sup_probs - in_c)
        # When all x in S_a belong to C (in_c_counts == |S_a|), complement is empty -> prob = 1.0.
        # When no x in S_a belongs to C (in_c_counts == 0), prob = 0.0.
        s_lens = lengths.to(dtype=dtype).unsqueeze(1)
        all_in_c = in_c_counts >= (s_lens - 0.5)
        none_in_c = in_c_counts < 0.5

        row_prob_sum = probs_elem.sum(dim=1, keepdim=True)
        complement_probs = (row_prob_sum - in_c_probs).clamp(min=0.0, max=1.0)
        class_probs = (1.0 - complement_probs).clamp(min=0.0, max=1.0)
        class_probs = torch.where(all_in_c, torch.ones_like(class_probs), class_probs)
        class_probs = torch.where(none_in_c, torch.zeros_like(class_probs), class_probs)
        return -torch.log(class_probs) + 0.0
