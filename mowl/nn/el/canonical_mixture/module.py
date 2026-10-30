"""CanonicalMixtureModule neural module for the canonical-element mixture model (R4)."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from mowl.geometric.membership import MembershipStructure
from mowl.nn import ELModule


class CanonicalMixtureModule(ELModule):
    """Canonical-element mixture neural module (R4).

    Represents each individual :math:`a` as a probability distribution
    :math:`p_a = \\operatorname{softmax}_{S_a}(\\ell_a)` over its admissible
    canonical support
    :math:`S_a = \\{x \\in S : x \\in C^{\\mathcal{I}} \\text{ for every entailed class } C \\text{ of } a\\}`
    (computed via :meth:`~mowl.geometric.membership.MembershipStructure.type_mask`
    over the full membership row :math:`M[x_a, :]`), where logits are factored
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
        # sorted flat keys (row * n_classes + col), and float sparse M^T for spmm.
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
        else:
            sorted_rows = torch.empty((0,), dtype=torch.long)
            sorted_cols = torch.empty((0,), dtype=torch.long)
            sorted_flat_keys = torch.empty((0,), dtype=torch.long)
            m_crow_offsets = torch.zeros((self.n_elements + 1,), dtype=torch.long)

        self.register_buffer("_m_crow_offsets", m_crow_offsets)
        self.register_buffer("_m_col_indices", sorted_cols)
        self.register_buffer("_m_flat_keys", sorted_flat_keys)

        # Store M^T as a coalesced float sparse tensor (n_classes x |S|) for rank_classes spmm
        mt_indices = (
            torch.stack([sorted_cols, sorted_rows], dim=0)
            if sorted_cols.numel() > 0
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

    def _build_admissible_supports(
        self, training_class_assertions: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute CSR admissible element offsets, concatenated indices, and own-element logit offsets."""
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
                # Individual occurs only in validation/test (no training assertions): support = all of S
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

        starts = self.support_offsets.index_select(0, ind_idx)
        ends = self.support_offsets.index_select(0, ind_idx + 1)
        lengths = ends - starts
        total_sup = int(lengths.sum().item())

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

        # Compute logits for the batch and gather support entries only
        u_batch = self.U.index_select(0, ind_idx)  # (B, r)
        raw_logits = u_batch @ self.V.t() + self.b.unsqueeze(0)  # (B, |S|)
        sup_logits = raw_logits[row_ids, elem_ids]  # (total_sup,)

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

    def _probs(self, ind_idx: torch.Tensor) -> torch.Tensor:
        """Return masked probability distribution ``p_a`` of shape ``(B, |S|)``."""
        ind_idx = torch.as_tensor(
            ind_idx, dtype=torch.long, device=self.U.device
        ).reshape(-1)
        bsz = ind_idx.shape[0]
        probs = torch.zeros(
            (bsz, self.n_elements), dtype=self.U.dtype, device=self.U.device
        )
        if bsz == 0:
            return probs
        row_ids, elem_ids, _, sup_probs, _ = self._sup_log_p_and_probs(ind_idx)
        probs[row_ids, elem_ids] = sup_probs
        return probs

    def log_p(self, ind_idx: torch.Tensor) -> torch.Tensor:
        """Compute masked log-probabilities ``log p_a`` of shape ``(B, |S|)`` for ``ind_idx``.

        Log-softmax is computed exclusively over the gathered admissible support
        :math:`S_a`, and entries outside :math:`S_a` are ``-inf``.
        """
        ind_idx = torch.as_tensor(
            ind_idx, dtype=torch.long, device=self.U.device
        ).reshape(-1)
        bsz = ind_idx.shape[0]
        out = torch.full(
            (bsz, self.n_elements),
            float("-inf"),
            dtype=self.U.dtype,
            device=self.U.device,
        )
        if bsz == 0:
            return out
        row_ids, elem_ids, sup_log_probs, _, _ = self._sup_log_p_and_probs(ind_idx)
        out[row_ids, elem_ids] = sup_log_probs
        return out

    def _class_assertion_prob(self, data: torch.Tensor) -> torch.Tensor:
        """Return class-membership probability :math:`1 - \\sum_{x \\in S_a, x \\notin C} p_a(x) \\in [0, 1]`."""
        data = torch.as_tensor(data, dtype=torch.long, device=self.U.device)
        if data.numel() == 0:
            return torch.empty((0,), dtype=self.U.dtype, device=self.U.device)

        ind_idx = data[:, 0]
        cls_idx = data[:, 1]
        bsz = data.shape[0]
        device = self.U.device

        unique_inds, inv_ind = torch.unique(ind_idx, return_inverse=True)
        _, u_elem_ids, _, u_sup_probs, u_lengths = self._sup_log_p_and_probs(
            unique_inds
        )
        u_offsets = torch.zeros(
            (unique_inds.numel() + 1,), dtype=torch.long, device=device
        )
        u_offsets[1:] = torch.cumsum(u_lengths, dim=0)

        b_starts = u_offsets.index_select(0, inv_ind)
        b_ends = u_offsets.index_select(0, inv_ind + 1)
        b_lengths = b_ends - b_starts
        total_pairs = int(b_lengths.sum().item())

        if total_pairs == 0:
            return torch.zeros((bsz,), dtype=self.U.dtype, device=device)

        batch_ids = torch.repeat_interleave(
            torch.arange(bsz, device=device, dtype=torch.long), b_lengths
        )
        cum_b = torch.cumsum(b_lengths, dim=0) - b_lengths
        flat_u_pos = (
            torch.arange(total_pairs, device=device, dtype=torch.long)
            - torch.repeat_interleave(cum_b, b_lengths)
            + torch.repeat_interleave(b_starts, b_lengths)
        )
        x_elems = u_elem_ids.index_select(0, flat_u_pos)
        p_vals = u_sup_probs.index_select(0, flat_u_pos)
        c_vals = cls_idx.index_select(0, batch_ids)

        in_c = self._is_entailed_for_elem(x_elems, c_vals)
        not_in_c = ~in_c

        sum_not_c = torch.zeros((bsz,), dtype=p_vals.dtype, device=device)
        if bool(not_in_c.any().item()):
            sum_not_c.scatter_add_(0, batch_ids[not_in_c], p_vals[not_in_c])

        has_in_c = torch.zeros((bsz,), dtype=torch.bool, device=device)
        if bool(in_c.any().item()):
            has_in_c[batch_ids[in_c]] = True

        return torch.where(
            has_in_c,
            (1.0 - sum_not_c).clamp(min=0.0, max=1.0),
            torch.zeros_like(sum_not_c),
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

        all_inds = torch.cat([head_idx, tail_idx], dim=0)
        unique_inds, inv = torch.unique(all_inds, return_inverse=True)
        p_unique = self._probs(unique_inds)
        bsz = data.shape[0]
        inv_head = inv[:bsz]
        inv_tail = inv[bsz:]

        probs = torch.zeros((bsz,), dtype=p_unique.dtype, device=p_unique.device)
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
            u_h = inv_head.index_select(0, rows_b)
            u_t = inv_tail.index_select(0, rows_b)
            p_a = p_unique.index_select(0, u_h)  # (B_r, |S|)
            p_b = p_unique.index_select(0, u_t)  # (B_r, |S|)
            er_pb = torch.sparse.mm(e_r, p_b.t()).t()  # (B_r, |S|)
            probs = probs.scatter_add(0, rows_b, (p_a * er_pb).sum(dim=1))

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
        ind_col = pos_data[:, 0]
        bsz = ind_col.shape[0]
        device = self.U.device
        if bsz == 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)

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
        corrupt_tail = torch.randint(
            0, self.n_ind, (pos_data.shape[0],), device=self.U.device, dtype=torch.long
        )
        return torch.stack([pos_data[:, 0], pos_data[:, 1], corrupt_tail], dim=1)

    def regularization_loss(
        self, ind_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return the entropy regularisation term :math:`\\lambda \\cdot \\operatorname{mean}_a H(p_a)`."""
        if self.entropy_reg == 0.0:
            return self.U.sum() * 0.0
        if ind_idx is None:
            ind_idx = torch.arange(self.n_ind, device=self.U.device, dtype=torch.long)
        ind_idx = torch.as_tensor(
            ind_idx, dtype=torch.long, device=self.U.device
        ).reshape(-1)
        bsz = ind_idx.shape[0]
        if bsz == 0:
            return self.U.sum() * 0.0
        row_ids, _, sup_log_probs, sup_probs, _ = self._sup_log_p_and_probs(ind_idx)
        ent_per_ind = torch.zeros(
            (bsz,), dtype=sup_probs.dtype, device=sup_probs.device
        )
        ent_per_ind.scatter_add_(0, row_ids, -sup_probs * sup_log_probs)
        return self.entropy_reg * ent_per_ind.mean()

    def rank_classes(self, ind_idx: torch.Tensor) -> torch.Tensor:
        """Return ``(B, n_classes)`` negative-log class-membership scores for ``ind_idx``.

        Computes :math:`1 - \\sum_{x \\in S_a, x \\notin C} p_a(x)` via
        :func:`torch.sparse.mm` so entailed classes score ``0.0`` exactly and
        classes with zero probability mass under :math:`p_a` get ``+inf``.
        """
        ind_idx = torch.as_tensor(
            ind_idx, dtype=torch.long, device=self.U.device
        ).reshape(-1)
        bsz = ind_idx.shape[0]
        device = self.U.device
        dtype = self.U.dtype
        if bsz == 0:
            return torch.empty((0, self.n_classes), dtype=dtype, device=device)

        row_ids, elem_ids, _, sup_probs, lengths = self._sup_log_p_and_probs(ind_idx)
        probs_elem = torch.zeros((bsz, self.n_elements), dtype=dtype, device=device)
        probs_elem[row_ids, elem_ids] = sup_probs
        sup_mask_float = torch.zeros((bsz, self.n_elements), dtype=dtype, device=device)
        sup_mask_float[row_ids, elem_ids] = 1.0

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
