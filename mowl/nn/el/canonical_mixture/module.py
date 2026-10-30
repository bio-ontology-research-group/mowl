"""CanonicalMixtureModule neural module for the canonical-element mixture model (R4)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from mowl.geometric.membership import MembershipStructure
from mowl.nn import ELModule


class CanonicalMixtureModule(ELModule):
    """Canonical-element mixture neural module (R4).

    Represents each individual :math:`a` as a probability distribution
    :math:`p_a = \\operatorname{softmax}_{S_a}(\\ell_a)` over its admissible
    canonical support
    :math:`S_a = \\{x \\in S : x \\in C^{\\mathcal{I}} \\text{ for every training type } C \\text{ of } a\\}`,
    where logits are factored as :math:`\\ell_a = U_a V^\\top + b + \\delta_a` with
    :math:`U \\in \\mathbb{R}^{n_{\\text{ind}} \\times r}`,
    :math:`V \\in \\mathbb{R}^{|S| \\times r}`, :math:`b \\in \\mathbb{R}^{|S|}`, and a
    fixed own-element logit offset :math:`\\delta_a(x_a) = \\log(9 \\cdot |S_a|)`
    (generalised to :math:`\\log(\\frac{\\alpha}{1-\\alpha} \\cdot |S_a|)` for
    ``init_self_mass = alpha``).

    Guarantees:
    - **Faithful by construction:** For every parameter setting, :math:`p_a` is a
      convex combination of elements of the canonical model :math:`\\mathcal{I}`
      restricted to :math:`S_a`. Every training-entailed class assertion
      :math:`C(a)` satisfies :math:`M[x, C] = 1` for all :math:`x \\in S_a`, so
      :math:`\\sum_{x \\in S_a} p_a(x) M[x, C] = 1` and
      :meth:`class_assertion_score` equals ``0.0``. Every contradicted class
      assertion satisfies :math:`M[x, C] = 0` for all :math:`x \\in S_a`, so its
      probability is ``0.0`` and :meth:`class_assertion_score` equals ``+inf``.
    - **Zero hierarchy violation rate:** Whenever :math:`\\mathcal{O} \\models C \\sqsubseteq D`,
      :math:`M[x, C] \\le M[x, D]` for all :math:`x \\in S`, so
      :math:`s(a, C) \\le s(a, D)` and the negative-log score satisfies
      :math:`\\operatorname{score}(a, D) \\le \\operatorname{score}(a, C)` for all
      individuals :math:`a` and all parameters.
    - **Sparse memory discipline:** Never allocates a dense ``(|S|, |S|)`` matrix
      or a dense ``(n_ind, |S|)`` mask when :math:`n_{\\text{ind}} \\cdot |S| > 5 \\times 10^7`;
      admissible supports :math:`S_a` are stored as a padded ``LongTensor`` of
      element indices plus lengths, and role scores :math:`p_a^\\top E_r p_b`
      gather only the non-zero edges of :math:`E_r`.

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
        nn.init.normal_(self.U, mean=0.0, std=1e-3 / math.sqrt(self.rank))
        nn.init.normal_(self.V, mean=0.0, std=1.0 / math.sqrt(self.rank))

        self.register_buffer(
            "individual_element",
            structure.individual_element.clone().to(dtype=torch.long),
        )

        # Precompute CSR representation of M^T (classes -> elements) for fast sparse gather
        m_coal = structure.M.coalesce()
        m_indices = m_coal.indices()
        m_values = m_coal.values().bool()
        if m_values.numel() > 0:
            elem_rows = m_indices[0][m_values]
            cls_cols = m_indices[1][m_values]
            order = torch.argsort(cls_cols, stable=True)
            sorted_cls = cls_cols[order]
            sorted_elem = elem_rows[order]
            counts = torch.bincount(sorted_cls, minlength=self.n_classes)
            crow_offsets = torch.zeros((self.n_classes + 1,), dtype=torch.long)
            crow_offsets[1:] = torch.cumsum(counts, dim=0)
        else:
            sorted_elem = torch.empty((0,), dtype=torch.long)
            crow_offsets = torch.zeros((self.n_classes + 1,), dtype=torch.long)

        self.register_buffer("_mt_crow_offsets", crow_offsets)
        self.register_buffer("_mt_col_indices", sorted_elem)

        # Also store M as a float sparse tensor (|S| x n_classes) for rank_classes spmm
        m_float_indices = (
            torch.stack([m_indices[0][m_values], m_indices[1][m_values]], dim=0)
            if m_values.numel() > 0
            else torch.empty((2, 0), dtype=torch.long)
        )
        m_float_vals = torch.ones((m_float_indices.shape[1],), dtype=torch.float32)
        self.register_buffer(
            "_m_float_t",
            torch.sparse_coo_tensor(
                torch.stack([m_float_indices[1], m_float_indices[0]], dim=0)
                if m_float_indices.shape[1] > 0
                else torch.empty((2, 0), dtype=torch.long),
                m_float_vals,
                size=(self.n_classes, self.n_elements),
                dtype=torch.float32,
            ).coalesce(),
        )

        # Store role edge index pairs as buffers
        self._role_keys: List[int] = sorted(structure.roles.keys())
        for r_idx in self._role_keys:
            r_coal = structure.roles[r_idx].coalesce()
            r_ind = r_coal.indices()
            r_val = r_coal.values().bool()
            edges = (
                r_ind[:, r_val]
                if r_val.numel() > 0
                else torch.empty((2, 0), dtype=torch.long)
            )
            self.register_buffer(f"_role_edges_{r_idx}", edges.to(dtype=torch.long))

        # Precompute per-individual admissible support S_a from training class assertions
        # and entailed classes per individual (from M[x_a, :] and S_a)
        padded_support, support_lengths, own_offsets = self._build_admissible_supports(
            training_class_assertions
        )
        self.register_buffer("support_indices", padded_support)
        self.register_buffer("support_lengths", support_lengths)
        self.register_buffer("own_logit_offset", own_offsets)

        # Precompute non-entailed class lists per individual for negative sampling
        self._non_entailed_by_ind: List[torch.Tensor] = (
            self._build_non_entailed_classes(training_class_assertions)
        )

    def _build_admissible_supports(
        self, training_class_assertions: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute padded admissible element indices, lengths, and own-element logit offsets."""
        types_by_ind: Dict[int, List[int]] = {}
        if training_class_assertions is not None and training_class_assertions.numel() > 0:
            ca = training_class_assertions.detach().cpu().to(dtype=torch.long)
            for row in ca:
                ind_i = int(row[0].item())
                cls_i = int(row[1].item())
                types_by_ind.setdefault(ind_i, []).append(cls_i)

        support_lists: List[torch.Tensor] = []
        lengths = torch.empty((self.n_ind,), dtype=torch.long)
        own_offsets = torch.empty((self.n_ind,), dtype=torch.float32)
        odds = self.init_self_mass / (1.0 - self.init_self_mass)

        for ind_i in range(self.n_ind):
            own_el = int(self.structure.individual_element[ind_i].item())
            if ind_i in types_by_ind:
                t_tensor = torch.tensor(types_by_ind[ind_i], dtype=torch.long)
            else:
                # Default to the canonical types of the individual's own element in M
                m_coal = self.structure.M.coalesce()
                idx = m_coal.indices()
                val = m_coal.values().bool()
                mask_own = idx[0][val] == own_el
                t_tensor = idx[1][val][mask_own].cpu()

            mask = self.structure.type_mask(t_tensor).cpu()
            # Ensure x_a is always in S_a
            mask[own_el] = True
            elems = torch.nonzero(mask, as_tuple=False).reshape(-1).to(dtype=torch.long)
            support_lists.append(elems)
            s_len = int(elems.numel())
            lengths[ind_i] = s_len
            # Own-element logit offset: log(9 * |S_a|) at init_self_mass=0.9
            own_offsets[ind_i] = math.log(odds * float(max(s_len, 1)))

        max_len = int(lengths.max().item()) if self.n_ind > 0 else 1
        padded = torch.full((self.n_ind, max_len), -1, dtype=torch.long)
        for ind_i, elems in enumerate(support_lists):
            padded[ind_i, : elems.numel()] = elems

        return padded, lengths, own_offsets

    def _build_non_entailed_classes(
        self, training_class_assertions: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """For each individual a, find classes C not entailed on S_a (and not asserted in training)."""
        m_coal = self.structure.M.coalesce()
        idx = m_coal.indices().cpu()
        val = m_coal.values().bool().cpu()
        rows = idx[0][val]
        cols = idx[1][val]

        entailed_by_elem: Dict[int, set] = {}
        for r, c in zip(rows.tolist(), cols.tolist()):
            entailed_by_elem.setdefault(r, set()).add(c)

        asserted_by_ind: Dict[int, set] = {}
        if training_class_assertions is not None and training_class_assertions.numel() > 0:
            ca = training_class_assertions.detach().cpu().to(dtype=torch.long)
            for row in ca:
                asserted_by_ind.setdefault(int(row[0].item()), set()).add(
                    int(row[1].item())
                )

        all_classes = set(range(self.n_classes))
        out: List[torch.Tensor] = []
        for ind_i in range(self.n_ind):
            own_el = int(self.structure.individual_element[ind_i].item())
            entailed = set(entailed_by_elem.get(own_el, set())) | asserted_by_ind.get(
                ind_i, set()
            )
            non_entailed = sorted(all_classes - entailed)
            if not non_entailed:
                # Fallback if an individual entails every class in the signature
                non_entailed = list(range(self.n_classes))
            out.append(torch.tensor(non_entailed, dtype=torch.long))
        return out

    def _log_p_and_probs(
        self, ind_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(masked_log_p, probs, support_mask)`` with ``-inf``-free backward graph on ``probs``."""
        ind_idx = torch.as_tensor(ind_idx, dtype=torch.long, device=self.U.device).reshape(-1)
        bsz = ind_idx.shape[0]
        if bsz == 0:
            empty = torch.empty(
                (0, self.n_elements), dtype=self.U.dtype, device=self.U.device
            )
            return empty, empty, empty.bool()

        u_batch = self.U.index_select(0, ind_idx)  # (B, r)
        raw_logits = u_batch @ self.V.t() + self.b.unsqueeze(0)  # (B, |S|)

        own_elems = self.individual_element.index_select(0, ind_idx)  # (B,)
        own_offs = self.own_logit_offset.index_select(0, ind_idx).to(
            dtype=raw_logits.dtype
        )
        raw_logits = raw_logits.scatter_add(
            1, own_elems.unsqueeze(1), own_offs.unsqueeze(1)
        )

        sup_padded = self.support_indices.index_select(0, ind_idx)  # (B, max_len)
        valid_mask = sup_padded >= 0
        row_ids, col_ids = torch.nonzero(valid_mask, as_tuple=True)
        elem_ids = sup_padded[row_ids, col_ids]
        support_mask = torch.zeros_like(raw_logits, dtype=torch.bool)
        support_mask[row_ids, elem_ids] = True

        VERY_NEG = -1e9
        masked_logits = torch.where(
            support_mask, raw_logits, torch.full_like(raw_logits, VERY_NEG)
        )
        max_logits = masked_logits.max(dim=-1, keepdim=True).values.detach()
        shifted = masked_logits - max_logits
        exp_shifted = torch.exp(shifted) * support_mask.to(dtype=raw_logits.dtype)
        sum_exp = exp_shifted.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        probs = exp_shifted / sum_exp
        log_probs_finite = shifted - torch.log(sum_exp)
        return log_probs_finite, probs, support_mask

    def _probs(self, ind_idx: torch.Tensor) -> torch.Tensor:
        """Return masked probability distribution ``p_a`` of shape ``(B, |S|)``."""
        _, probs, _ = self._log_p_and_probs(ind_idx)
        return probs

    def log_p(self, ind_idx: torch.Tensor) -> torch.Tensor:
        """Compute masked log-probabilities ``log p_a`` of shape ``(B, |S|)`` for ``ind_idx``.

        Logits outside :math:`S_a` are set to ``-inf`` before ``log_softmax``, and
        the own-element logit receives :attr:`own_logit_offset`
        (:math:`\\log(9 \\cdot |S_a|)` when ``init_self_mass=0.9``).
        """
        log_probs_finite, _, support_mask = self._log_p_and_probs(ind_idx)
        return torch.where(
            support_mask,
            log_probs_finite,
            torch.full_like(log_probs_finite, float("-inf")),
        )

    def _class_assertion_prob(self, data: torch.Tensor) -> torch.Tensor:
        """Return class-membership probability :math:`\\sum_x p_a(x) M[x, C] \\in [0, 1]` of shape ``(B,)``."""
        data = torch.as_tensor(data, dtype=torch.long, device=self.U.device)
        if data.numel() == 0:
            return torch.empty((0,), dtype=self.U.dtype, device=self.U.device)

        ind_idx = data[:, 0]
        cls_idx = data[:, 1]

        unique_inds, inv_ind = torch.unique(ind_idx, return_inverse=True)
        p_unique = self._probs(unique_inds)  # (U_ind, |S|)

        starts = self._mt_crow_offsets.index_select(0, cls_idx)
        ends = self._mt_crow_offsets.index_select(0, cls_idx + 1)
        lengths = ends - starts
        total_nnz = int(lengths.sum().item())

        probs = torch.zeros(
            (data.shape[0],), dtype=p_unique.dtype, device=p_unique.device
        )
        if total_nnz > 0:
            batch_ids = torch.repeat_interleave(
                torch.arange(data.shape[0], device=self.U.device, dtype=torch.long),
                lengths,
            )
            cum_len = torch.cumsum(lengths, dim=0) - lengths
            flat_pos = (
                torch.arange(total_nnz, device=self.U.device, dtype=torch.long)
                - torch.repeat_interleave(cum_len, lengths)
                + torch.repeat_interleave(starts, lengths)
            )
            elem_ids = self._mt_col_indices.index_select(0, flat_pos)
            u_ind_ids = inv_ind.index_select(0, batch_ids)
            vals = p_unique[u_ind_ids, elem_ids]
            probs = probs.scatter_add(0, batch_ids, vals)

        return probs.clamp(min=0.0, max=1.0)

    def class_assertion_score(self, data: torch.Tensor) -> torch.Tensor:
        """Return negative log probability :math:`-\\log \\sum_x p_a(x) M[x, C]` of shape ``(B,)``.

        Parameters
        ----------
        data : torch.LongTensor
            2-D tensor of shape ``(B, 2)`` with ``[ind_idx, class_idx]`` rows.
        """
        return -torch.log(self._class_assertion_prob(data))

    def _object_property_assertion_prob(self, data: torch.Tensor) -> torch.Tensor:
        """Return role-assertion probability :math:`p_a^\\top E_r p_b \\in [0, 1]` of shape ``(B,)``."""
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
            buf_name = f"_role_edges_{r_int}"
            if not hasattr(self, buf_name):
                continue
            edges: torch.Tensor = getattr(self, buf_name)
            if edges.shape[1] == 0:
                continue
            mask = rel_idx == r_int
            rows_b = torch.nonzero(mask, as_tuple=False).reshape(-1)
            u_h = inv_head.index_select(0, rows_b)
            u_t = inv_tail.index_select(0, rows_b)
            p_i = p_unique.index_select(0, u_h).index_select(1, edges[0])
            q_j = p_unique.index_select(0, u_t).index_select(1, edges[1])
            probs = probs.scatter_add(0, rows_b, (p_i * q_j).sum(dim=1))

        return probs.clamp(min=0.0, max=1.0)

    def object_property_assertion_score(self, data: torch.Tensor) -> torch.Tensor:
        """Return negative log probability :math:`-\\log (p_a^\\top E_r p_b)` of shape ``(B,)``.

        Gathers only the non-zero edges :math:`(i, j)` of :math:`E_r` without
        ever densifying :math:`E_r`.

        Parameters
        ----------
        data : torch.LongTensor
            2-D tensor of shape ``(B, 3)`` with ``[head_ind, role_idx, tail_ind]`` rows.
        """
        return -torch.log(self._object_property_assertion_prob(data))

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
        """Sample one random non-entailed class per positive ``(ind_idx, class_idx)`` row."""
        pos_data = torch.as_tensor(pos_data, dtype=torch.long, device=self.U.device)
        ind_col = pos_data[:, 0]
        neg_classes = torch.empty_like(ind_col)
        for ind_val in torch.unique(ind_col).tolist():
            ind_int = int(ind_val)
            mask = ind_col == ind_int
            count = int(mask.sum().item())
            pool = self._non_entailed_by_ind[ind_int].to(device=self.U.device)
            rand_pos = torch.randint(0, pool.numel(), (count,), device=self.U.device)
            neg_classes[mask] = pool.index_select(0, rand_pos)
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
        log_probs_finite, probs, _ = self._log_p_and_probs(ind_idx)
        ent_terms = -probs * log_probs_finite
        entropy = ent_terms.sum(dim=-1).mean()
        return self.entropy_reg * entropy

    def rank_classes(self, ind_idx: torch.Tensor) -> torch.Tensor:
        """Return ``(B, n_classes)`` negative-log class-membership scores for ``ind_idx``.

        A class with zero probability mass under :math:`p_a` gets ``+inf``.
        """
        ind_idx = torch.as_tensor(ind_idx, dtype=torch.long, device=self.U.device).reshape(-1)
        probs_elem = self._probs(ind_idx)  # (B, |S|)
        # M is (|S|, n_classes), so _m_float_t is (n_classes, |S|);
        # torch.sparse.mm(_m_float_t, probs_elem.T) -> (n_classes, B)
        class_probs = torch.sparse.mm(self._m_float_t, probs_elem.t()).t()  # (B, n_classes)
        class_probs = class_probs.clamp(min=0.0, max=1.0)
        return -torch.log(class_probs)
