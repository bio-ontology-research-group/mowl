"""The evaluation regime of GroundedEL: crisp violation rates and the exact-model check.

Mixed into :class:`mowl.models.GroundedEL`; reads the module and the axioms of the model."""
from __future__ import annotations

from typing import Any, Dict, List
import torch as th
from mowl.nn.el.grounded import (
    RegionSample,
    class_assertion,
    crisp_membership,
    gci0,
    gci0_bot,
    gci1,
    gci1_bot,
    gci2,
    gci3,
    gci3_bot,
    role_assertion,
    role_chain,
    role_inclusion,
    sample_class_region,
    sample_pair_region,
    sample_role_region,
)
from mowl.nn.el.grounded.module import _EMPTY_DEPTH_FLOOR, _validate_int_ge
from mowl.representation.library.grounded_logic import validate_nonneg_weight as _validate_nonneg_weight
from mowl.owlapi.defaults import BOT as BOTTOM, TOP
from mowl.models.grounded.common import _TOL_DEPTH_LEVELS, _ToleranceCountDict, _short_iri_map

class GroundedEvaluation:
    """Crisp element and region violation rates (:meth:`crisp_report`) and the exact-model
    check of the finite domain (:meth:`evaluate_model`)."""

    def _box_corner_samples(
        self,
        cen: th.Tensor,
        w: th.Tensor,
        active: th.Tensor,
        gen: th.Generator,
        max_corners_per_box: int = 32,
    ) -> th.Tensor:
        """Return interior-nudged box corner points ``c ± (1 - 1e-4) * w`` for active non-empty boxes."""
        valid_k = active & (th.amin(w, dim=-1) > 0.0)
        if not bool(valid_k.any().item()):
            return th.empty((0, self.embed_dim), dtype=cen.dtype, device=cen.device)
        c_act = cen[valid_k]  # (M, d)
        w_act = w[valid_k] * (1.0 - 1e-4)  # (M, d)
        m_cnt, dim = c_act.shape
        if dim <= 5 and (1 << dim) <= max_corners_per_box:
            signs = th.tensor(
                [
                    [(1.0 if (mask >> bit) & 1 else -1.0) for bit in range(dim)]
                    for mask in range(1 << dim)
                ],
                dtype=cen.dtype,
                device=cen.device,
            )  # (2^d, d)
            pts = c_act.unsqueeze(1) + w_act.unsqueeze(1) * signs.unsqueeze(0)
            return pts.reshape(-1, dim)
        u_bits = (
            th.randint(
                0,
                2,
                (m_cnt, max_corners_per_box, dim),
                device=gen.device,
                generator=gen,
            ).to(device=cen.device, dtype=cen.dtype)
            * 2.0
            - 1.0
        )
        pts = c_act.unsqueeze(1) + w_act.unsqueeze(1) * u_bits
        return pts.reshape(-1, dim)


    def _eval_elem_depths_chunked(
        self,
        eval_pos: th.Tensor,
        mentioned_classes: th.Tensor,
        chunk_size: int = 512,
    ) -> th.Tensor:
        """Crisp depth ``(L, M)`` of ``eval_pos`` in ``mentioned_classes`` (the class model's
        depth table)."""
        return self.module.class_model.depth_table(eval_pos, mentioned_classes,
                                                   chunk_size=chunk_size)


    def _eval_elem_memberships_chunked(
        self,
        eval_pos: th.Tensor,
        mentioned_classes: th.Tensor,
        chunk_size: int = 512,
    ) -> th.Tensor:
        """Compute crisp membership ``(L, M)`` only for ``eval_pos`` and ``mentioned_classes`` in chunks."""
        return crisp_membership(
            self._eval_elem_depths_chunked(
                eval_pos, mentioned_classes, chunk_size=chunk_size
            )
        )


    def crisp_report(
        self,
        n_eval: int = 10000,
        tol: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute reasoner-free crisp violation metrics, per-axiom worst depth, and depth-bucket violation counts (§7).

        A sampled point :math:`z` is a counterexample for a region axiom at
        tolerance ``tol >= 0`` only if its violating depth exceeds ``tol``:
        ``depth_RHS(z) < -tol`` (``<= 0.0`` at ``tol == 0.0``) for containment
        forms (``gci0``, ``gci1``, ``gci2``, ``gci3``, ``role_inclusion``,
        ``role_chain``) and ``min(depth_A, depth_B) > tol`` (analogously for
        ``gci0_bot`` and ``gci3_bot``) for disjointness/bottom forms.

        Guarantees:

        - Reasoner-free (R1): never imports or calls any reasoner or
          :class:`~mowl.geometric.completion.CanonicalModel`.
        - Never materialises a :math:`(|\\Delta| \\times |\\mathcal{C}|)` tensor;
          computes memberships only for the sampled evaluation elements and the
          classes mentioned by the evaluated axioms, in chunks.
        - ``tol=0.0`` reproduces exact strict-boundary ``elem_viol`` and
          ``region_viol`` numbers bit-for-bit.
        - Returns ``{"elem_viol": float, "region_viol": float, "worst_depth": dict, "n_axioms_violated_by_depth": dict, "n_empty_classes": int, "n_dead_elements": int}``
          where ``n_axioms_violated_by_depth`` reports violated axiom counts at
          ``tol`` in ``{0, 1e-5, 1e-4, 1e-3, 1e-2}`` and is non-increasing in ``tol``.

        Non-guarantees:

        - Does not compute reasoner-dependent metrics (``exist``, ``dead_ok``,
          ``n_false_dead``, ``faithful_named_*``, ``exact``).

        Parameters
        ----------
        n_eval : int, default 10000
            Total evaluation sample budget across axioms for ``region_viol``.
        tol : float, default 0.0
            Non-negative boundary tolerance :math:`\\ge 0` for region counterexamples.

        Returns
        -------
        Dict[str, Any]
            ``{"elem_viol": ..., "region_viol": ..., "worst_depth": ..., "n_axioms_violated_by_depth": ..., "n_empty_classes": ..., "n_dead_elements": ...}``.
        """
        _validate_int_ge("n_eval", n_eval, 1)
        tol_f = _validate_nonneg_weight("tol", tol)
        mod = self.module
        device = self.device
        td = self.training_datasets
        gen = th.Generator(device="cpu")
        gen.manual_seed(self.seed + 9999)

        cls_names = _short_iri_map(self.class_index_dict)
        rel_names = _short_iri_map(self.object_property_index_dict)
        worst_depth: Dict[str, Dict[str, float]] = {}
        tol_counts: Dict[str, int] = {k_str: 0 for k_str, _ in _TOL_DEPTH_LEVELS}

        def _record_containment_axiom(
            ax_type: str,
            ax_str: str,
            has_lhs: bool,
            min_rhs_dep: float,
            viol_zero: bool,
        ) -> bool:
            bucket = worst_depth.setdefault(ax_type, {})
            w_val = float(min_rhs_dep) if has_lhs else 0.0
            if ax_str in bucket:
                bucket[ax_str] = min(bucket[ax_str], w_val)
            else:
                bucket[ax_str] = w_val
            for k_str, t_val in _TOL_DEPTH_LEVELS:
                if t_val == 0.0:
                    is_v = viol_zero
                else:
                    is_v = has_lhs and (w_val < -t_val)
                if is_v:
                    tol_counts[k_str] += 1
            if tol_f == 0.0:
                return viol_zero
            return has_lhs and (w_val < -tol_f)

        def _record_bot_axiom(
            ax_type: str,
            ax_str: str,
            max_lhs_dep: float,
            viol_zero: bool,
        ) -> bool:
            bucket = worst_depth.setdefault(ax_type, {})
            w_val = float(max_lhs_dep)
            if ax_str in bucket:
                bucket[ax_str] = max(bucket[ax_str], w_val)
            else:
                bucket[ax_str] = w_val
            for k_str, t_val in _TOL_DEPTH_LEVELS:
                if t_val == 0.0:
                    is_v = viol_zero
                else:
                    is_v = w_val > t_val
                if is_v:
                    tol_counts[k_str] += 1
            if tol_f == 0.0:
                return viol_zero
            return w_val > tol_f

        with th.no_grad():
            pos_all = mod.positions()  # (nb_elements, d)
            dead_c = mod.dead() > 0.5
            if self.bot_id is not None:
                dead_c[self.bot_id] = True
            dead_f = dead_c.to(dtype=th.float32)
            live_mask = ~dead_c
            live_ids = th.nonzero(live_mask, as_tuple=False).view(-1)

            if live_ids.numel() > 256:
                perm = th.randperm(live_ids.numel(), device="cpu", generator=gen)[:256].to(
                    device=device
                )
                eval_elem_ids = live_ids[perm]
            else:
                eval_elem_ids = live_ids
            eval_pos = pos_all[eval_elem_ids]
            eval_dead_f = dead_f[eval_elem_ids]

            # Collect unique classes mentioned by the evaluated axioms
            cls_parts: List[th.Tensor] = []
            for k_ax, cols in (
                ("gci0", (0, 1)),
                ("gci0_bot", (0,)),
                ("gci1", (0, 1, 2)),
                ("gci1_bot", (0, 1)),
                ("gci2", (0, 2)),
                ("gci3", (1, 2)),
                ("gci3_bot", (1,)),
            ):
                if k_ax in td and len(td[k_ax]) > 0:
                    t_ax = td[k_ax][:].to(device=device, dtype=th.long)
                    for col_i in cols:
                        cls_parts.append(t_ax[:, col_i])
            if cls_parts:
                mentioned_classes = th.unique(th.cat(cls_parts))
            else:
                mentioned_classes = th.empty((0,), dtype=th.long, device=device)

            cls_to_col = th.full(
                (self.nb_classes,), -1, dtype=th.long, device=device
            )
            if mentioned_classes.numel() > 0:
                cls_to_col[mentioned_classes] = th.arange(
                    mentioned_classes.numel(), dtype=th.long, device=device
                )

            # Compute signed depths and crisp memberships ONLY for eval_elem_ids and mentioned_classes in chunks
            elem_dep_live = self._eval_elem_depths_chunked(
                eval_pos, mentioned_classes, chunk_size=512
            )  # (L, M)
            elem_m_live = crisp_membership(elem_dep_live)  # (L, M)

            elem_viol_count = 0
            elem_total_count = 0

            # Count total TBox/RBox axioms to divide n_eval budget cleanly
            total_region_axioms = sum(
                len(td[k])
                for k in (
                    "gci0",
                    "gci0_bot",
                    "gci1",
                    "gci1_bot",
                    "gci2",
                    "gci3",
                    "gci3_bot",
                    "role_inclusion",
                    "role_chain",
                )
                if k in td and len(td[k]) > 0
            )
            reg_viol_axioms = 0
            reg_total_axioms = 0

            if total_region_axioms <= 512:
                n_per_ax = max(16, min(2000, n_eval // max(total_region_axioms, 1)))

                # 1. GCI0: A <= B
                if "gci0" in td and len(td["gci0"]) > 0:
                    for row in td["gci0"][:].detach().cpu().tolist():
                        a_id, b_id = int(row[0]), int(row[1])
                        col_a = int(cls_to_col[a_id].item())
                        col_b = int(cls_to_col[b_id].item())
                        if eval_elem_ids.numel() > 0:
                            h_e = gci0(
                                elem_m_live[:, col_a],
                                elem_m_live[:, col_b],
                                dead=eval_dead_f,
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        s_reg = sample_class_region(
                            mod, a_id, generator=gen, n_samples=n_per_ax
                        )
                        reg_a = mod.class_regions(a_id)
                        corners = self._box_corner_samples(
                            reg_a.cen, reg_a.w, reg_a.active, gen
                        )
                        pts_list = [eval_pos]
                        if bool(s_reg.valid.any().item()):
                            pts_list.append(s_reg.points[s_reg.valid])
                        if corners.numel() > 0:
                            pts_list.append(corners)
                        pts = th.cat(pts_list, dim=0)
                        has_lhs = False
                        min_rhs = 0.0
                        viol_0 = False
                        if pts.shape[0] > 0:
                            da = mod.depth(pts, a_id, crisp=True)
                            db = mod.depth(pts, b_id, crisp=True)
                            mask = da > 0.0
                            has_lhs = bool(mask.any().item())
                            if has_lhs:
                                min_rhs = float(db[mask].min().item())
                                viol_0 = bool((db[mask] <= 0.0).any().item())
                        ax_s = f"{cls_names.get(a_id, str(a_id))} <= {cls_names.get(b_id, str(b_id))}"
                        if _record_containment_axiom("gci0", ax_s, has_lhs, min_rhs, viol_0):
                            reg_viol_axioms += 1

                # 2. GCI0_bot: A <= bot
                if "gci0_bot" in td and len(td["gci0_bot"]) > 0:
                    for row in td["gci0_bot"][:].detach().cpu().tolist():
                        a_id = int(row[0])
                        col_a = int(cls_to_col[a_id].item())
                        if eval_elem_ids.numel() > 0:
                            h_e = gci0_bot(elem_m_live[:, col_a], dead=eval_dead_f)
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        s_reg = sample_class_region(
                            mod, a_id, generator=gen, n_samples=n_per_ax
                        )
                        pts_list = [eval_pos]
                        if bool(s_reg.valid.any().item()):
                            pts_list.append(s_reg.points[s_reg.valid])
                        pts = th.cat(pts_list, dim=0)
                        if pts.shape[0] > 0:
                            da = mod.depth(pts, a_id, crisp=True)
                            max_lhs = float(da.max().item())
                            viol_0 = bool(s_reg.valid.any().item()) or bool(
                                (da > 0.0).any().item()
                            )
                        else:
                            max_lhs = float(_EMPTY_DEPTH_FLOOR)
                            viol_0 = bool(s_reg.valid.any().item())
                        ax_s = f"{cls_names.get(a_id, str(a_id))} <= bot"
                        if _record_bot_axiom("gci0_bot", ax_s, max_lhs, viol_0):
                            reg_viol_axioms += 1

                # 3. GCI1: A cap B <= C
                if "gci1" in td and len(td["gci1"]) > 0:
                    for row in td["gci1"][:].detach().cpu().tolist():
                        a_id, b_id, c_id = int(row[0]), int(row[1]), int(row[2])
                        col_a = int(cls_to_col[a_id].item())
                        col_b = int(cls_to_col[b_id].item())
                        col_c = int(cls_to_col[c_id].item())
                        if eval_elem_ids.numel() > 0:
                            h_e = gci1(
                                elem_m_live[:, col_a],
                                elem_m_live[:, col_b],
                                elem_m_live[:, col_c],
                                dead=eval_dead_f,
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        s_reg = sample_pair_region(
                            mod, a_id, b_id, generator=gen, n_samples=n_per_ax
                        )
                        pts_list = [eval_pos]
                        if bool(s_reg.valid.any().item()):
                            pts_list.append(s_reg.points[s_reg.valid])
                        pts = th.cat(pts_list, dim=0)
                        has_lhs = False
                        min_rhs = 0.0
                        viol_0 = False
                        if pts.shape[0] > 0:
                            da = mod.depth(pts, a_id, crisp=True)
                            db = mod.depth(pts, b_id, crisp=True)
                            dc = mod.depth(pts, c_id, crisp=True)
                            mask = (da > 0.0) & (db > 0.0)
                            has_lhs = bool(mask.any().item())
                            if has_lhs:
                                min_rhs = float(dc[mask].min().item())
                                viol_0 = bool((dc[mask] <= 0.0).any().item())
                        ax_s = (
                            f"{cls_names.get(a_id, str(a_id))} cap "
                            f"{cls_names.get(b_id, str(b_id))} <= "
                            f"{cls_names.get(c_id, str(c_id))}"
                        )
                        if _record_containment_axiom("gci1", ax_s, has_lhs, min_rhs, viol_0):
                            reg_viol_axioms += 1

                # 4. GCI1_bot: A cap B <= bot
                if "gci1_bot" in td and len(td["gci1_bot"]) > 0:
                    for row in td["gci1_bot"][:].detach().cpu().tolist():
                        a_id, b_id = int(row[0]), int(row[1])
                        col_a = int(cls_to_col[a_id].item())
                        col_b = int(cls_to_col[b_id].item())
                        if eval_elem_ids.numel() > 0:
                            h_e = gci1_bot(
                                elem_m_live[:, col_a],
                                elem_m_live[:, col_b],
                                dead=eval_dead_f,
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        s_inter = sample_pair_region(
                            mod, a_id, b_id, generator=gen, n_samples=n_per_ax
                        )
                        if bool(s_inter.valid.any().item()):
                            pts_list = [eval_pos, s_inter.points[s_inter.valid]]
                        else:
                            s_a = sample_class_region(
                                mod, a_id, generator=gen, n_samples=n_per_ax
                            )
                            pts_list = [eval_pos]
                            if bool(s_a.valid.any().item()):
                                pts_list.append(s_a.points[s_a.valid])
                        pts = th.cat(pts_list, dim=0)
                        if pts.shape[0] > 0:
                            da = mod.depth(pts, a_id, crisp=True)
                            db = mod.depth(pts, b_id, crisp=True)
                            ab_dep = th.minimum(da, db)
                            max_lhs = float(ab_dep.max().item())
                            viol_0 = bool(s_inter.valid.any().item()) or bool(
                                (ab_dep > 0.0).any().item()
                            )
                        else:
                            max_lhs = float(_EMPTY_DEPTH_FLOOR)
                            viol_0 = bool(s_inter.valid.any().item())
                        ax_s = (
                            f"{cls_names.get(a_id, str(a_id))} cap "
                            f"{cls_names.get(b_id, str(b_id))} <= bot"
                        )
                        if _record_bot_axiom("gci1_bot", ax_s, max_lhs, viol_0):
                            reg_viol_axioms += 1

                # 5. GCI2: A <= exists r.B
                if "gci2" in td and len(td["gci2"]) > 0:
                    for row in td["gci2"][:].detach().cpu().tolist():
                        a_id, r_id, b_id = int(row[0]), int(row[1]), int(row[2])
                        col_a = int(cls_to_col[a_id].item())
                        pb = pos_all[b_id]
                        if eval_elem_ids.numel() > 0:
                            rho_e = crisp_membership(
                                mod.role_depth(eval_pos, pb, r_id, crisp=True)
                            )
                            db_e = dead_f[b_id].expand_as(eval_dead_f)
                            h_e = gci2(
                                elem_m_live[:, col_a],
                                rho_e,
                                dead=eval_dead_f,
                                dead_filler=db_e,
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        s_reg = sample_class_region(
                            mod, a_id, generator=gen, n_samples=n_per_ax
                        )
                        reg_a = mod.class_regions(a_id)
                        corners = self._box_corner_samples(
                            reg_a.cen, reg_a.w, reg_a.active, gen
                        )
                        pts_list = [eval_pos]
                        if bool(s_reg.valid.any().item()):
                            pts_list.append(s_reg.points[s_reg.valid])
                        if corners.numel() > 0:
                            pts_list.append(corners)
                        pts = th.cat(pts_list, dim=0)
                        has_lhs = False
                        min_rhs = 0.0
                        viol_0 = False
                        if pts.shape[0] > 0:
                            da = mod.depth(pts, a_id, crisp=True)
                            r_dep = mod.role_depth(pts, pb, r_id, crisp=True)
                            mask = da > 0.0
                            has_lhs = bool(mask.any().item())
                            filler_dead = bool(dead_c[b_id].item())
                            if has_lhs:
                                rhs_dep = -da[mask] if filler_dead else r_dep[mask]
                                min_rhs = float(rhs_dep.min().item())
                                viol_0 = filler_dead or bool(
                                    (rhs_dep <= 0.0).any().item()
                                )
                        ax_s = (
                            f"{cls_names.get(a_id, str(a_id))} <= exists "
                            f"{rel_names.get(r_id, str(r_id))}."
                            f"{cls_names.get(b_id, str(b_id))}"
                        )
                        if _record_containment_axiom("gci2", ax_s, has_lhs, min_rhs, viol_0):
                            reg_viol_axioms += 1

                # 6. GCI3: exists r.B <= C
                if "gci3" in td and len(td["gci3"]) > 0:
                    n_y = min(int(eval_elem_ids.numel()), 32)
                    y_eval_ids = eval_elem_ids[:n_y]
                    for row in td["gci3"][:].detach().cpu().tolist():
                        r_id, b_id, c_id = int(row[0]), int(row[1]), int(row[2])
                        col_b = int(cls_to_col[b_id].item())
                        col_c = int(cls_to_col[c_id].item())
                        if eval_elem_ids.numel() > 0 and n_y > 0:
                            for y_idx in range(n_y):
                                y_id = int(y_eval_ids[y_idx].item())
                                rho_xy = crisp_membership(
                                    mod.role_depth(
                                        eval_pos, pos_all[y_id], r_id, crisp=True
                                    )
                                )
                                mb_y = elem_m_live[y_idx, col_b].expand_as(rho_xy)
                                h_e = gci3(
                                    rho_xy,
                                    mb_y,
                                    elem_m_live[:, col_c],
                                    dead=eval_dead_f,
                                )
                                elem_viol_count += int((h_e > 0.0).sum().item())
                                elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        has_lhs = False
                        min_rhs = 0.0
                        viol_0 = False
                        if mod.role_mode == "translation":
                            if eval_elem_ids.numel() > 0 and n_y > 0:
                                l_cnt = int(eval_elem_ids.numel())
                                zx = (
                                    eval_pos.unsqueeze(1)
                                    .expand(l_cnt, n_y, self.embed_dim)
                                    .reshape(-1, self.embed_dim)
                                )
                                zy = (
                                    pos_all[y_eval_ids]
                                    .unsqueeze(0)
                                    .expand(l_cnt, n_y, self.embed_dim)
                                    .reshape(-1, self.embed_dim)
                                )
                                r_dep = mod.role_depth(zx, zy, r_id, crisp=True)
                                db_z = mod.depth(zy, b_id, crisp=True)
                                dc_z = mod.depth(zx, c_id, crisp=True)
                                mask = (r_dep > 0.0) & (db_z > 0.0)
                                has_lhs = bool(mask.any().item())
                                if has_lhs:
                                    min_rhs = float(dc_z[mask].min().item())
                                    viol_0 = bool((dc_z[mask] <= 0.0).any().item())
                        else:
                            h_s, t_s = sample_role_region(
                                mod, r_id, generator=gen, n_samples=n_per_ax
                            )
                            if bool(h_s.valid.any().item()):
                                zx = h_s.points[h_s.valid]
                                zy = t_s.points[t_s.valid]
                                r_dep = mod.role_depth(zx, zy, r_id, crisp=True)
                                db_z = mod.depth(zy, b_id, crisp=True)
                                dc_z = mod.depth(zx, c_id, crisp=True)
                                mask = (r_dep > 0.0) & (db_z > 0.0)
                                has_lhs = bool(mask.any().item())
                                if has_lhs:
                                    min_rhs = float(dc_z[mask].min().item())
                                    viol_0 = bool((dc_z[mask] <= 0.0).any().item())
                        ax_s = (
                            f"exists {rel_names.get(r_id, str(r_id))}."
                            f"{cls_names.get(b_id, str(b_id))} <= "
                            f"{cls_names.get(c_id, str(c_id))}"
                        )
                        if _record_containment_axiom("gci3", ax_s, has_lhs, min_rhs, viol_0):
                            reg_viol_axioms += 1

                # 7. GCI3_bot: exists r.B <= bot
                if "gci3_bot" in td and len(td["gci3_bot"]) > 0:
                    n_y = min(int(eval_elem_ids.numel()), 32)
                    y_eval_ids = eval_elem_ids[:n_y]
                    for row in td["gci3_bot"][:].detach().cpu().tolist():
                        r_id, b_id = int(row[0]), int(row[1])
                        col_b = int(cls_to_col[b_id].item())
                        if eval_elem_ids.numel() > 0 and n_y > 0:
                            for y_idx in range(n_y):
                                y_id = int(y_eval_ids[y_idx].item())
                                rho_xy = crisp_membership(
                                    mod.role_depth(
                                        eval_pos, pos_all[y_id], r_id, crisp=True
                                    )
                                )
                                mb_y = elem_m_live[y_idx, col_b].expand_as(rho_xy)
                                h_e = gci3_bot(rho_xy, mb_y, dead=eval_dead_f)
                                elem_viol_count += int((h_e > 0.0).sum().item())
                                elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        if mod.role_mode == "translation":
                            if eval_elem_ids.numel() > 0 and n_y > 0:
                                l_cnt = int(eval_elem_ids.numel())
                                zx = (
                                    eval_pos.unsqueeze(1)
                                    .expand(l_cnt, n_y, self.embed_dim)
                                    .reshape(-1, self.embed_dim)
                                )
                                zy = (
                                    pos_all[y_eval_ids]
                                    .unsqueeze(0)
                                    .expand(l_cnt, n_y, self.embed_dim)
                                    .reshape(-1, self.embed_dim)
                                )
                                r_dep = mod.role_depth(zx, zy, r_id, crisp=True)
                                db_z = mod.depth(zy, b_id, crisp=True)
                                rb_dep = th.minimum(r_dep, db_z)
                                max_lhs = float(rb_dep.max().item())
                                viol_0 = bool((rb_dep > 0.0).any().item())
                            else:
                                max_lhs = float(_EMPTY_DEPTH_FLOOR)
                                viol_0 = False
                        else:
                            h_s, t_s = sample_role_region(
                                mod, r_id, generator=gen, n_samples=n_per_ax
                            )
                            if bool(h_s.valid.any().item()):
                                zx = h_s.points[h_s.valid]
                                zy = t_s.points[t_s.valid]
                                r_dep = mod.role_depth(zx, zy, r_id, crisp=True)
                                db_z = mod.depth(zy, b_id, crisp=True)
                                rb_dep = th.minimum(r_dep, db_z)
                                max_lhs = float(rb_dep.max().item())
                                viol_0 = bool((rb_dep > 0.0).any().item())
                            else:
                                max_lhs = float(_EMPTY_DEPTH_FLOOR)
                                viol_0 = False
                        ax_s = (
                            f"exists {rel_names.get(r_id, str(r_id))}."
                            f"{cls_names.get(b_id, str(b_id))} <= bot"
                        )
                        if _record_bot_axiom("gci3_bot", ax_s, max_lhs, viol_0):
                            reg_viol_axioms += 1
            else:
                # Chunked vectorised evaluation for large ontologies (e.g. GO)
                n_per_ax = max(1, min(16, n_eval // max(total_region_axioms, 1)))
                ax_chunk = 512

                if "gci0" in td and len(td["gci0"]) > 0:
                    data_g0 = td["gci0"][:].to(device=device, dtype=th.long)
                    for s_i in range(0, data_g0.shape[0], ax_chunk):
                        rows = data_g0[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        a_c = rows[:, 0]
                        b_c = rows[:, 1]
                        ca = cls_to_col[a_c]
                        cb = cls_to_col[b_c]
                        min_dep = th.full((bsz,), float("inf"), dtype=th.float32, device=device)
                        has_lhs = th.zeros((bsz,), dtype=th.bool, device=device)
                        if eval_elem_ids.numel() > 0:
                            ma_e = elem_m_live[:, ca]  # (L, B)
                            mb_e = elem_m_live[:, cb]  # (L, B)
                            h_e = gci0(
                                ma_e,
                                mb_e,
                                dead=eval_dead_f.unsqueeze(1).expand_as(ma_e),
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                            da_e = elem_dep_live[:, ca]
                            db_e = elem_dep_live[:, cb]
                            mask_e = da_e > 0.0
                            min_dep = th.where(
                                mask_e, db_e, th.full_like(db_e, float("inf"))
                            ).amin(dim=0)
                            has_lhs = mask_e.any(dim=0)
                        a_rep = a_c.repeat_interleave(n_per_ax)
                        b_rep = b_c.repeat_interleave(n_per_ax)
                        s_reg = sample_class_region(mod, a_rep, generator=gen)
                        if bool(s_reg.valid.any().item()):
                            da_r = mod.depth(s_reg.points, a_rep, crisp=True)
                            db_r = mod.depth(s_reg.points, b_rep, crisp=True)
                            mask_r = (da_r > 0.0) & s_reg.valid
                            dep_r = th.where(
                                mask_r, db_r, th.full_like(db_r, float("inf"))
                            ).view(bsz, n_per_ax)
                            min_dep = th.minimum(min_dep, dep_r.amin(dim=1))
                            has_lhs = has_lhs | mask_r.view(bsz, n_per_ax).any(dim=1)
                        viol_0_b = has_lhs & (min_dep <= 0.0)
                        rows_list = rows.detach().cpu().tolist()
                        has_list = has_lhs.detach().cpu().tolist()
                        min_list = min_dep.detach().cpu().tolist()
                        v0_list = viol_0_b.detach().cpu().tolist()
                        for r_row, h_flag, m_val, v0_flag in zip(
                            rows_list, has_list, min_list, v0_list
                        ):
                            a_i, b_i = int(r_row[0]), int(r_row[1])
                            ax_s = f"{cls_names.get(a_i, str(a_i))} <= {cls_names.get(b_i, str(b_i))}"
                            if _record_containment_axiom(
                                "gci0", ax_s, bool(h_flag), float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

                if "gci0_bot" in td and len(td["gci0_bot"]) > 0:
                    data_g0b = td["gci0_bot"][:].to(device=device, dtype=th.long)
                    for s_i in range(0, data_g0b.shape[0], ax_chunk):
                        rows = data_g0b[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        a_c = rows[:, 0]
                        ca = cls_to_col[a_c]
                        max_dep = th.full(
                            (bsz,), _EMPTY_DEPTH_FLOOR, dtype=th.float32, device=device
                        )
                        if eval_elem_ids.numel() > 0:
                            ma_e = elem_m_live[:, ca]
                            h_e = gci0_bot(
                                ma_e,
                                dead=eval_dead_f.unsqueeze(1).expand_as(ma_e),
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                            max_dep = elem_dep_live[:, ca].amax(dim=0)
                        a_rep = a_c.repeat_interleave(n_per_ax)
                        s_reg = sample_class_region(mod, a_rep, generator=gen)
                        if bool(s_reg.valid.any().item()):
                            da_r = mod.depth(s_reg.points, a_rep, crisp=True)
                            da_r_m = th.where(
                                s_reg.valid,
                                da_r,
                                th.full_like(da_r, _EMPTY_DEPTH_FLOOR),
                            ).view(bsz, n_per_ax)
                            max_dep = th.maximum(max_dep, da_r_m.amax(dim=1))
                        viol_0_b = (max_dep > 0.0) | s_reg.valid.view(
                            bsz, n_per_ax
                        ).any(dim=1)
                        for r_row, m_val, v0_flag in zip(
                            rows.detach().cpu().tolist(),
                            max_dep.detach().cpu().tolist(),
                            viol_0_b.detach().cpu().tolist(),
                        ):
                            a_i = int(r_row[0])
                            ax_s = f"{cls_names.get(a_i, str(a_i))} <= bot"
                            if _record_bot_axiom(
                                "gci0_bot", ax_s, float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

                if "gci1" in td and len(td["gci1"]) > 0:
                    data_g1 = td["gci1"][:].to(device=device, dtype=th.long)
                    for s_i in range(0, data_g1.shape[0], ax_chunk):
                        rows = data_g1[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        a_c, b_c, c_c = rows[:, 0], rows[:, 1], rows[:, 2]
                        ca, cb, cc = cls_to_col[a_c], cls_to_col[b_c], cls_to_col[c_c]
                        min_dep = th.full((bsz,), float("inf"), dtype=th.float32, device=device)
                        has_lhs = th.zeros((bsz,), dtype=th.bool, device=device)
                        if eval_elem_ids.numel() > 0:
                            ma_e = elem_m_live[:, ca]
                            mb_e = elem_m_live[:, cb]
                            mc_e = elem_m_live[:, cc]
                            h_e = gci1(
                                ma_e,
                                mb_e,
                                mc_e,
                                dead=eval_dead_f.unsqueeze(1).expand_as(ma_e),
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                            da_e = elem_dep_live[:, ca]
                            db_e = elem_dep_live[:, cb]
                            dc_e = elem_dep_live[:, cc]
                            mask_e = (da_e > 0.0) & (db_e > 0.0)
                            min_dep = th.where(
                                mask_e, dc_e, th.full_like(dc_e, float("inf"))
                            ).amin(dim=0)
                            has_lhs = mask_e.any(dim=0)
                        a_rep = a_c.repeat_interleave(n_per_ax)
                        b_rep = b_c.repeat_interleave(n_per_ax)
                        c_rep = c_c.repeat_interleave(n_per_ax)
                        s_reg = sample_pair_region(mod, a_rep, b_rep, generator=gen)
                        if bool(s_reg.valid.any().item()):
                            da_r = mod.depth(s_reg.points, a_rep, crisp=True)
                            db_r = mod.depth(s_reg.points, b_rep, crisp=True)
                            dc_r = mod.depth(s_reg.points, c_rep, crisp=True)
                            mask_r = (da_r > 0.0) & (db_r > 0.0) & s_reg.valid
                            dep_r = th.where(
                                mask_r, dc_r, th.full_like(dc_r, float("inf"))
                            ).view(bsz, n_per_ax)
                            min_dep = th.minimum(min_dep, dep_r.amin(dim=1))
                            has_lhs = has_lhs | mask_r.view(bsz, n_per_ax).any(dim=1)
                        viol_0_b = has_lhs & (min_dep <= 0.0)
                        for r_row, h_flag, m_val, v0_flag in zip(
                            rows.detach().cpu().tolist(),
                            has_lhs.detach().cpu().tolist(),
                            min_dep.detach().cpu().tolist(),
                            viol_0_b.detach().cpu().tolist(),
                        ):
                            a_i, b_i, c_i = int(r_row[0]), int(r_row[1]), int(r_row[2])
                            ax_s = (
                                f"{cls_names.get(a_i, str(a_i))} cap "
                                f"{cls_names.get(b_i, str(b_i))} <= "
                                f"{cls_names.get(c_i, str(c_i))}"
                            )
                            if _record_containment_axiom(
                                "gci1", ax_s, bool(h_flag), float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

                if "gci1_bot" in td and len(td["gci1_bot"]) > 0:
                    data_g1b = td["gci1_bot"][:].to(device=device, dtype=th.long)
                    for s_i in range(0, data_g1b.shape[0], ax_chunk):
                        rows = data_g1b[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        a_c, b_c = rows[:, 0], rows[:, 1]
                        ca, cb = cls_to_col[a_c], cls_to_col[b_c]
                        max_dep = th.full(
                            (bsz,), _EMPTY_DEPTH_FLOOR, dtype=th.float32, device=device
                        )
                        if eval_elem_ids.numel() > 0:
                            ma_e = elem_m_live[:, ca]
                            mb_e = elem_m_live[:, cb]
                            h_e = gci1_bot(
                                ma_e,
                                mb_e,
                                dead=eval_dead_f.unsqueeze(1).expand_as(ma_e),
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                            max_dep = th.minimum(
                                elem_dep_live[:, ca], elem_dep_live[:, cb]
                            ).amax(dim=0)
                        a_rep = a_c.repeat_interleave(n_per_ax)
                        b_rep = b_c.repeat_interleave(n_per_ax)
                        s_inter = sample_pair_region(mod, a_rep, b_rep, generator=gen)
                        if bool(s_inter.valid.any().item()):
                            da_r = mod.depth(s_inter.points, a_rep, crisp=True)
                            db_r = mod.depth(s_inter.points, b_rep, crisp=True)
                            ab_r = th.where(
                                s_inter.valid,
                                th.minimum(da_r, db_r),
                                th.full_like(da_r, _EMPTY_DEPTH_FLOOR),
                            ).view(bsz, n_per_ax)
                            max_dep = th.maximum(max_dep, ab_r.amax(dim=1))
                        viol_0_b = (max_dep > 0.0) | s_inter.valid.view(
                            bsz, n_per_ax
                        ).any(dim=1)
                        for r_row, m_val, v0_flag in zip(
                            rows.detach().cpu().tolist(),
                            max_dep.detach().cpu().tolist(),
                            viol_0_b.detach().cpu().tolist(),
                        ):
                            a_i, b_i = int(r_row[0]), int(r_row[1])
                            ax_s = (
                                f"{cls_names.get(a_i, str(a_i))} cap "
                                f"{cls_names.get(b_i, str(b_i))} <= bot"
                            )
                            if _record_bot_axiom(
                                "gci1_bot", ax_s, float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

                if "gci2" in td and len(td["gci2"]) > 0:
                    data_g2 = td["gci2"][:].to(device=device, dtype=th.long)
                    for s_i in range(0, data_g2.shape[0], ax_chunk):
                        rows = data_g2[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        a_c, r_c, b_c = rows[:, 0], rows[:, 1], rows[:, 2]
                        ca = cls_to_col[a_c]
                        min_dep = th.full((bsz,), float("inf"), dtype=th.float32, device=device)
                        has_lhs = th.zeros((bsz,), dtype=th.bool, device=device)
                        if eval_elem_ids.numel() > 0:
                            l_cnt = int(eval_elem_ids.numel())
                            x_mat = eval_elem_ids.unsqueeze(0).expand(bsz, l_cnt)
                            y_mat = b_c.unsqueeze(1)
                            _, r_dep_c = self._grid_role_membership(
                                pos_all, x_mat, y_mat, r_c, need_crisp=True
                            )
                            assert r_dep_c is not None
                            r_dep_e = r_dep_c.view(bsz, l_cnt).t()
                            rho_e = crisp_membership(r_dep_e)
                            ma_e = elem_m_live[:, ca]  # (L, B)
                            db_e = dead_f[b_c].unsqueeze(0).expand_as(ma_e)
                            h_e = gci2(
                                ma_e,
                                rho_e,
                                dead=eval_dead_f.unsqueeze(1).expand_as(ma_e),
                                dead_filler=db_e,
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                            da_e = elem_dep_live[:, ca]
                            rhs_e = th.where(
                                dead_c[b_c].unsqueeze(0), -da_e, r_dep_e
                            )
                            mask_e = da_e > 0.0
                            min_dep = th.where(
                                mask_e, rhs_e, th.full_like(rhs_e, float("inf"))
                            ).amin(dim=0)
                            has_lhs = mask_e.any(dim=0)
                        a_rep = a_c.repeat_interleave(n_per_ax)
                        r_rep = r_c.repeat_interleave(n_per_ax)
                        b_rep = b_c.repeat_interleave(n_per_ax)
                        s_reg = sample_class_region(mod, a_rep, generator=gen)
                        if bool(s_reg.valid.any().item()):
                            da_r = mod.depth(s_reg.points, a_rep, crisp=True)
                            rd_r = mod.role_depth(
                                s_reg.points, pos_all[b_rep], r_rep, crisp=True
                            )
                            rhs_r = th.where(dead_c[b_rep], -da_r, rd_r)
                            mask_r = (da_r > 0.0) & s_reg.valid
                            dep_r = th.where(
                                mask_r, rhs_r, th.full_like(rhs_r, float("inf"))
                            ).view(bsz, n_per_ax)
                            min_dep = th.minimum(min_dep, dep_r.amin(dim=1))
                            has_lhs = has_lhs | mask_r.view(bsz, n_per_ax).any(dim=1)
                        viol_0_b = has_lhs & ((min_dep <= 0.0) | dead_c[b_c])
                        for r_row, h_flag, m_val, v0_flag in zip(
                            rows.detach().cpu().tolist(),
                            has_lhs.detach().cpu().tolist(),
                            min_dep.detach().cpu().tolist(),
                            viol_0_b.detach().cpu().tolist(),
                        ):
                            a_i, r_i, b_i = int(r_row[0]), int(r_row[1]), int(r_row[2])
                            ax_s = (
                                f"{cls_names.get(a_i, str(a_i))} <= exists "
                                f"{rel_names.get(r_i, str(r_i))}."
                                f"{cls_names.get(b_i, str(b_i))}"
                            )
                            if _record_containment_axiom(
                                "gci2", ax_s, bool(h_flag), float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

                if "gci3" in td and len(td["gci3"]) > 0:
                    data_g3 = td["gci3"][:].to(device=device, dtype=th.long)
                    n_y = min(int(eval_elem_ids.numel()), 32)
                    y_eval_ids = eval_elem_ids[:n_y]
                    for s_i in range(0, data_g3.shape[0], ax_chunk):
                        rows = data_g3[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        r_c, b_c, c_c = rows[:, 0], rows[:, 1], rows[:, 2]
                        cb, cc = cls_to_col[b_c], cls_to_col[c_c]
                        if eval_elem_ids.numel() > 0 and n_y > 0:
                            l_cnt = int(eval_elem_ids.numel())
                            x_mat = eval_elem_ids.unsqueeze(0).expand(bsz, l_cnt)
                            y_mat = y_eval_ids.unsqueeze(0).expand(bsz, n_y)
                            _, r_dep_c = self._grid_role_membership(
                                pos_all, x_mat, y_mat, r_c, need_crisp=True
                            )
                            assert r_dep_c is not None
                            rho_xy = crisp_membership(r_dep_c).view(bsz, l_cnt, n_y)
                            mb_y = elem_m_live[:n_y, cb].t().unsqueeze(1).expand(
                                bsz, l_cnt, n_y
                            )
                            mc_x = elem_m_live[:, cc].t().unsqueeze(2).expand(
                                bsz, l_cnt, n_y
                            )
                            dead_x = eval_dead_f.view(1, l_cnt, 1).expand(
                                bsz, l_cnt, n_y
                            )
                            h_e = gci3(rho_xy, mb_y, mc_x, dead=dead_x)
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        min_dep = th.full((bsz,), float("inf"), dtype=th.float32, device=device)
                        has_lhs = th.zeros((bsz,), dtype=th.bool, device=device)
                        if mod.role_mode == "translation":
                            if eval_elem_ids.numel() > 0 and n_y > 0:
                                l_cnt = int(eval_elem_ids.numel())
                                assert r_dep_c is not None
                                rd_grid = r_dep_c.view(bsz, l_cnt, n_y)
                                db_grid = elem_dep_live[:n_y, cb].t().unsqueeze(1).expand(
                                    bsz, l_cnt, n_y
                                )
                                dc_grid = elem_dep_live[:, cc].t().unsqueeze(2).expand(
                                    bsz, l_cnt, n_y
                                )
                                mask_r = (rd_grid > 0.0) & (db_grid > 0.0)
                                dep_r = th.where(
                                    mask_r, dc_grid, th.full_like(dc_grid, float("inf"))
                                ).view(bsz, l_cnt * n_y)
                                min_dep = dep_r.amin(dim=1)
                                has_lhs = mask_r.view(bsz, l_cnt * n_y).any(dim=1)
                        else:
                            r_rep = r_c.repeat_interleave(n_per_ax)
                            b_rep = b_c.repeat_interleave(n_per_ax)
                            c_rep = c_c.repeat_interleave(n_per_ax)
                            h_s, t_s = sample_role_region(mod, r_rep, generator=gen)
                            if bool(h_s.valid.any().item()):
                                rd_r = mod.role_depth(
                                    h_s.points, t_s.points, r_rep, crisp=True
                                )
                                db_r = mod.depth(t_s.points, b_rep, crisp=True)
                                dc_r = mod.depth(h_s.points, c_rep, crisp=True)
                                mask_r = (rd_r > 0.0) & (db_r > 0.0) & h_s.valid
                                dep_r = th.where(
                                    mask_r, dc_r, th.full_like(dc_r, float("inf"))
                                ).view(bsz, n_per_ax)
                                min_dep = dep_r.amin(dim=1)
                                has_lhs = mask_r.view(bsz, n_per_ax).any(dim=1)
                        viol_0_b = has_lhs & (min_dep <= 0.0)
                        for r_row, h_flag, m_val, v0_flag in zip(
                            rows.detach().cpu().tolist(),
                            has_lhs.detach().cpu().tolist(),
                            min_dep.detach().cpu().tolist(),
                            viol_0_b.detach().cpu().tolist(),
                        ):
                            r_i, b_i, c_i = int(r_row[0]), int(r_row[1]), int(r_row[2])
                            ax_s = (
                                f"exists {rel_names.get(r_i, str(r_i))}."
                                f"{cls_names.get(b_i, str(b_i))} <= "
                                f"{cls_names.get(c_i, str(c_i))}"
                            )
                            if _record_containment_axiom(
                                "gci3", ax_s, bool(h_flag), float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

                if "gci3_bot" in td and len(td["gci3_bot"]) > 0:
                    data_g3b = td["gci3_bot"][:].to(device=device, dtype=th.long)
                    n_y = min(int(eval_elem_ids.numel()), 32)
                    y_eval_ids = eval_elem_ids[:n_y]
                    for s_i in range(0, data_g3b.shape[0], ax_chunk):
                        rows = data_g3b[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        r_c, b_c = rows[:, 0], rows[:, 1]
                        cb = cls_to_col[b_c]
                        if eval_elem_ids.numel() > 0 and n_y > 0:
                            l_cnt = int(eval_elem_ids.numel())
                            x_mat = eval_elem_ids.unsqueeze(0).expand(bsz, l_cnt)
                            y_mat = y_eval_ids.unsqueeze(0).expand(bsz, n_y)
                            _, r_dep_c = self._grid_role_membership(
                                pos_all, x_mat, y_mat, r_c, need_crisp=True
                            )
                            assert r_dep_c is not None
                            rho_xy = crisp_membership(r_dep_c).view(bsz, l_cnt, n_y)
                            mb_y = elem_m_live[:n_y, cb].t().unsqueeze(1).expand(
                                bsz, l_cnt, n_y
                            )
                            dead_x = eval_dead_f.view(1, l_cnt, 1).expand(
                                bsz, l_cnt, n_y
                            )
                            h_e = gci3_bot(rho_xy, mb_y, dead=dead_x)
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        max_dep = th.full(
                            (bsz,), _EMPTY_DEPTH_FLOOR, dtype=th.float32, device=device
                        )
                        if mod.role_mode == "translation":
                            if eval_elem_ids.numel() > 0 and n_y > 0:
                                l_cnt = int(eval_elem_ids.numel())
                                assert r_dep_c is not None
                                rd_grid = r_dep_c.view(bsz, l_cnt, n_y)
                                db_grid = elem_dep_live[:n_y, cb].t().unsqueeze(1).expand(
                                    bsz, l_cnt, n_y
                                )
                                rb_r = th.minimum(rd_grid, db_grid).view(bsz, l_cnt * n_y)
                                max_dep = rb_r.amax(dim=1)
                        else:
                            r_rep = r_c.repeat_interleave(n_per_ax)
                            b_rep = b_c.repeat_interleave(n_per_ax)
                            h_s, t_s = sample_role_region(mod, r_rep, generator=gen)
                            if bool(h_s.valid.any().item()):
                                rd_r = mod.role_depth(
                                    h_s.points, t_s.points, r_rep, crisp=True
                                )
                                db_r = mod.depth(t_s.points, b_rep, crisp=True)
                                rb_r = th.where(
                                    h_s.valid,
                                    th.minimum(rd_r, db_r),
                                    th.full_like(rd_r, _EMPTY_DEPTH_FLOOR),
                                ).view(bsz, n_per_ax)
                                max_dep = rb_r.amax(dim=1)
                        viol_0_b = max_dep > 0.0
                        for r_row, m_val, v0_flag in zip(
                            rows.detach().cpu().tolist(),
                            max_dep.detach().cpu().tolist(),
                            viol_0_b.detach().cpu().tolist(),
                        ):
                            r_i, b_i = int(r_row[0]), int(r_row[1])
                            ax_s = (
                                f"exists {rel_names.get(r_i, str(r_i))}."
                                f"{cls_names.get(b_i, str(b_i))} <= bot"
                            )
                            if _record_bot_axiom(
                                "gci3_bot", ax_s, float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

            # 8. Role inclusion: r <= s
            n_per_role_ax = max(16, min(2000, n_eval // max(total_region_axioms, 1)))
            if "role_inclusion" in td and len(td["role_inclusion"]) > 0:
                y_eval_ids = (
                    eval_elem_ids[:32]
                    if eval_elem_ids.numel() > 32
                    else eval_elem_ids
                )
                for row in td["role_inclusion"][:].detach().cpu().tolist():
                    r_id, s_id = int(row[0]), int(row[1])
                    if eval_elem_ids.numel() > 0 and y_eval_ids.numel() > 0:
                        for y_id in y_eval_ids.tolist():
                            rr = crisp_membership(
                                mod.role_depth(eval_pos, pos_all[y_id], r_id, crisp=True)
                            )
                            rs = crisp_membership(
                                mod.role_depth(eval_pos, pos_all[y_id], s_id, crisp=True)
                            )
                            h_e = role_inclusion(rr, rs)
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    has_lhs = False
                    min_rhs = 0.0
                    viol_0 = False
                    if mod.role_mode == "translation":
                        v_s = sample_role_region(
                            mod, r_id, generator=gen, n_samples=n_per_role_ax
                        )
                        assert isinstance(v_s, RegionSample)  # NamedTuple: a plain tuple check would reject it (#177)
                        if bool(v_s.valid.any().item()):
                            zy = v_s.points[v_s.valid]
                            zx = th.zeros_like(zy)
                            dep_r = mod.role_depth(zx, zy, r_id, crisp=True)
                            dep_s = mod.role_depth(zx, zy, s_id, crisp=True)
                            mask = dep_r > 0.0
                            has_lhs = bool(mask.any().item())
                            if has_lhs:
                                min_rhs = float(dep_s[mask].min().item())
                                viol_0 = bool((dep_s[mask] <= 0.0).any().item())
                    else:
                        h_s, t_s = sample_role_region(
                            mod, r_id, generator=gen, n_samples=n_per_role_ax
                        )
                        if bool(h_s.valid.any().item()):
                            zx = h_s.points[h_s.valid]
                            zy = t_s.points[t_s.valid]
                            dep_r = mod.role_depth(zx, zy, r_id, crisp=True)
                            dep_s = mod.role_depth(zx, zy, s_id, crisp=True)
                            mask = dep_r > 0.0
                            has_lhs = bool(mask.any().item())
                            if has_lhs:
                                min_rhs = float(dep_s[mask].min().item())
                                viol_0 = bool((dep_s[mask] <= 0.0).any().item())
                    ax_s = f"{rel_names.get(r_id, str(r_id))} <= {rel_names.get(s_id, str(s_id))}"
                    if _record_containment_axiom("role_inclusion", ax_s, has_lhs, min_rhs, viol_0):
                        reg_viol_axioms += 1

            # 9. Role chain: r o s <= t
            if "role_chain" in td and len(td["role_chain"]) > 0:
                c_eval_ids = (
                    eval_elem_ids[:16]
                    if eval_elem_ids.numel() > 16
                    else eval_elem_ids
                )
                c_pos = pos_all[c_eval_ids]
                for row in td["role_chain"][:].detach().cpu().tolist():
                    r_id, s_id, t_id = int(row[0]), int(row[1]), int(row[2])
                    if c_eval_ids.numel() > 0:
                        for y_id in c_eval_ids.tolist():
                            rho_xy = crisp_membership(
                                mod.role_depth(c_pos, pos_all[y_id], r_id, crisp=True)
                            )
                            for z_id in c_eval_ids.tolist():
                                rho_yz = crisp_membership(
                                    mod.role_depth(
                                        pos_all[y_id], pos_all[z_id], s_id, crisp=True
                                    )
                                ).expand_as(rho_xy)
                                rho_xz = crisp_membership(
                                    mod.role_depth(
                                        c_pos, pos_all[z_id], t_id, crisp=True
                                    )
                                )
                                h_e = role_chain(rho_xy, rho_yz, rho_xz)
                                elem_viol_count += int((h_e > 0.0).sum().item())
                                elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    has_lhs = False
                    min_rhs = 0.0
                    viol_0 = False
                    if mod.role_mode == "translation":
                        if c_eval_ids.numel() > 0:
                            n_c = int(c_pos.shape[0])
                            zx = (
                                c_pos.view(n_c, 1, 1, mod.embed_dim)
                                .expand(n_c, n_c, n_c, mod.embed_dim)
                                .reshape(-1, mod.embed_dim)
                            )
                            zy = (
                                c_pos.view(1, n_c, 1, mod.embed_dim)
                                .expand(n_c, n_c, n_c, mod.embed_dim)
                                .reshape(-1, mod.embed_dim)
                            )
                            zz = (
                                c_pos.view(1, 1, n_c, mod.embed_dim)
                                .expand(n_c, n_c, n_c, mod.embed_dim)
                                .reshape(-1, mod.embed_dim)
                            )
                            dep_r = mod.role_depth(zx, zy, r_id, crisp=True)
                            dep_s = mod.role_depth(zy, zz, s_id, crisp=True)
                            dep_t = mod.role_depth(zx, zz, t_id, crisp=True)
                            mask = (dep_r > 0.0) & (dep_s > 0.0)
                            has_lhs = bool(mask.any().item())
                            if has_lhs:
                                min_rhs = float(dep_t[mask].min().item())
                                viol_0 = bool((dep_t[mask] <= 0.0).any().item())
                    else:
                        hx, hy = sample_role_region(
                            mod, r_id, generator=gen, n_samples=n_per_role_ax
                        )
                        _, tz = sample_role_region(
                            mod, s_id, generator=gen, n_samples=n_per_role_ax
                        )
                        v_c = hx.valid & tz.valid
                        if bool(v_c.any().item()):
                            zx = hx.points[v_c]
                            zy = hy.points[v_c]
                            zz = tz.points[v_c]
                            dep_r = mod.role_depth(zx, zy, r_id, crisp=True)
                            dep_s = mod.role_depth(zy, zz, s_id, crisp=True)
                            dep_t = mod.role_depth(zx, zz, t_id, crisp=True)
                            mask = (dep_r > 0.0) & (dep_s > 0.0)
                            has_lhs = bool(mask.any().item())
                            if has_lhs:
                                min_rhs = float(dep_t[mask].min().item())
                                viol_0 = bool((dep_t[mask] <= 0.0).any().item())
                    ax_s = (
                        f"{rel_names.get(r_id, str(r_id))} o "
                        f"{rel_names.get(s_id, str(s_id))} <= "
                        f"{rel_names.get(t_id, str(t_id))}"
                    )
                    if _record_containment_axiom("role_chain", ax_s, has_lhs, min_rhs, viol_0):
                        reg_viol_axioms += 1

            # 10. ABox assertions
            if "class_assertion" in td and len(td["class_assertion"]) > 0:
                ca_data = td["class_assertion"][:].to(device=device, dtype=th.long)
                for s_i in range(0, ca_data.shape[0], 4096):
                    rows = ca_data[s_i : s_i + 4096]
                    el_a = self.nb_classes + rows[:, 0]
                    c_id = rows[:, 1]
                    m_ca = crisp_membership(
                        mod.depth(pos_all[el_a], c_id, crisp=True)
                    )
                    h_e = class_assertion(m_ca)
                    elem_viol_count += int((h_e > 0.0).sum().item())
                    elem_total_count += int(h_e.numel())

            if (
                "object_property_assertion" in td
                and len(td["object_property_assertion"]) > 0
            ):
                opa_data = td["object_property_assertion"][:].to(
                    device=device, dtype=th.long
                )
                for s_i in range(0, opa_data.shape[0], 4096):
                    rows = opa_data[s_i : s_i + 4096]
                    el_a = self.nb_classes + rows[:, 0]
                    r_id = rows[:, 1]
                    el_b = self.nb_classes + rows[:, 2]
                    rho_ab = crisp_membership(
                        mod.role_depth(pos_all[el_a], pos_all[el_b], r_id, crisp=True)
                    )
                    h_e = role_assertion(rho_ab)
                    elem_viol_count += int((h_e > 0.0).sum().item())
                    elem_total_count += int(h_e.numel())

            elem_viol = (
                float(elem_viol_count / elem_total_count)
                if elem_total_count > 0
                else 0.0
            )
            region_viol = (
                float(reg_viol_axioms / reg_total_axioms)
                if reg_total_axioms > 0
                else 0.0
            )
            cls_reg_all = mod.class_regions()
            empty_cls_mask = ~cls_reg_all.active.any(dim=1)
            if self.bot_id is not None:
                empty_cls_mask[self.bot_id] = False
            if self.top_id is not None:
                empty_cls_mask[self.top_id] = False
            n_empty_classes = int(empty_cls_mask.sum().item())

            dead_elem_mask = (mod.dead() > 0.5).clone()
            if self.bot_id is not None:
                dead_elem_mask[self.bot_id] = False
            n_dead_elements = int(dead_elem_mask.sum().item())

            return {
                "elem_viol": elem_viol,
                "region_viol": region_viol,
                "worst_depth": worst_depth,
                "n_axioms_violated_by_depth": _ToleranceCountDict(tol_counts),
                "n_empty_classes": n_empty_classes,
                "n_dead_elements": n_dead_elements,
            }


    def evaluate_model(
        self,
        use_reasoner: bool = True,
        n_eval: int = 10000,
        tol: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute §7 evaluation metrics dictionary.

        Lazily imports :class:`~mowl.geometric.completion.CanonicalModel` inside
        this method ONLY when ``use_reasoner=True``.

        Guarantees:

        - Returns a dict with keys ``"elem_viol"``, ``"region_viol"``,
          ``"region_viol_tol1e-4"``, ``"exist"``, ``"dead_ok"``,
          ``"n_empty_classes"``, ``"n_dead_elements"``, ``"n_false_dead"``,
          ``"faithful_named_precision"``, ``"faithful_named_recall"``,
          ``"faithful_named"``, ``"exact"``, and ``"exact_tol1e-4"``.
        - ``exact`` is ``True`` if and only if ``elem_viol == 0.0``,
          ``region_viol == 0.0``, ``exist == 1.0``, and ``dead_ok is True``.
        - ``region_viol_tol1e-4`` and ``exact_tol1e-4`` are always computed at
          tolerance ``1e-4``, independent of ``tol``.

        Non-guarantees:

        - When ``use_reasoner=False``, ``exist`` is evaluated over live classes
          (``dead <= 0.5``) and ``n_false_dead`` and ``faithful_named_*`` are ``None``.

        Parameters
        ----------
        use_reasoner : bool, default True
            Whether to compute ground-truth satisfiability and subsumptions via
            :class:`~mowl.geometric.completion.CanonicalModel`.
        n_eval : int, default 10000
            Sample budget for ``region_viol`` and region inclusion checks.
        tol : float, default 0.0
            Non-negative boundary tolerance :math:`\\ge 0` passed to :meth:`crisp_report`.

        Returns
        -------
        Dict[str, Any]
            §7 evaluation metrics dictionary.
        """
        base = self.crisp_report(n_eval=n_eval, tol=tol)
        elem_viol = base["elem_viol"]
        region_viol = base["region_viol"]
        n_empty_classes = int(base["n_empty_classes"])
        n_dead_elements = int(base["n_dead_elements"])
        td = self.training_datasets
        total_region_axioms = sum(
            len(td[k])
            for k in (
                "gci0",
                "gci0_bot",
                "gci1",
                "gci1_bot",
                "gci2",
                "gci3",
                "gci3_bot",
                "role_inclusion",
                "role_chain",
            )
            if k in td and len(td[k]) > 0
        )
        region_viol_tol1e4 = (
            float(base["n_axioms_violated_by_depth"]["1e-4"] / total_region_axioms)
            if total_region_axioms > 0
            else 0.0
        )

        mod = self.module
        device = self.device
        gen = th.Generator(device="cpu")
        gen.manual_seed(self.seed + 2026)

        with th.no_grad():
            pos_all = mod.positions()
            dead_cls = (mod.dead()[: self.nb_classes] > 0.5).cpu()
            if self.bot_id is not None:
                dead_cls[self.bot_id] = True

            self_inside = th.zeros((self.nb_classes,), dtype=th.bool)
            for c_start in range(0, self.nb_classes, 8192):
                c_end = min(c_start + 8192, self.nb_classes)
                c_ids = th.arange(c_start, c_end, dtype=th.long, device=device)
                dep_c = mod.depth(pos_all[c_ids], c_ids, crisp=True).cpu()
                self_inside[c_start:c_end] = (dep_c > 0.0) & (~dead_cls[c_start:c_end])

            if not use_reasoner:
                live_cls = ~dead_cls
                exist_val = (
                    float(self_inside[live_cls].to(th.float32).mean().item())
                    if bool(live_cls.any().item())
                    else 1.0
                )
                dead_ok = True
                exact = bool(
                    elem_viol == 0.0
                    and region_viol == 0.0
                    and exist_val == 1.0
                    and dead_ok
                )
                exact_tol1e4 = bool(
                    elem_viol == 0.0
                    and region_viol_tol1e4 == 0.0
                    and exist_val == 1.0
                    and dead_ok
                )
                return {
                    "elem_viol": elem_viol,
                    "region_viol": region_viol,
                    "region_viol_tol1e-4": region_viol_tol1e4,
                    "exist": exist_val,
                    "dead_ok": dead_ok,
                    "n_empty_classes": n_empty_classes,
                    "n_dead_elements": n_dead_elements,
                    "n_false_dead": None,
                    "faithful_named_precision": None,
                    "faithful_named_recall": None,
                    "faithful_named": (None, None),
                    "exact": exact,
                    "exact_tol1e-4": exact_tol1e4,
                }

            if self.dataset is None or getattr(self.dataset, "ontology", None) is None:
                raise ValueError(
                    "use_reasoner=True requires an OWL dataset; pass use_reasoner=False for table-constructed GroundedEL models"
                )

            from mowl.geometric.completion import CanonicalModel

            cm = CanonicalModel.from_ontology(self.dataset.ontology)

            sat_mask = th.zeros((self.nb_classes,), dtype=th.bool)
            for c_iri, c_idx in self.class_index_dict.items():
                if c_iri == BOTTOM:
                    sat_mask[c_idx] = False
                elif c_iri == TOP:
                    sat_mask[c_idx] = bool(cm.is_satisfiable(TOP))
                else:
                    sat_mask[c_idx] = bool(cm.is_satisfiable(c_iri))

            unsat_mask = ~sat_mask
            dead_ok = bool(th.equal(dead_cls, unsat_mask))
            n_false_dead = int((dead_cls & sat_mask).sum().item())

            if bool(sat_mask.any().item()):
                exist_val = float(
                    self_inside[sat_mask].to(dtype=th.float32).mean().item()
                )
            else:
                exist_val = 1.0

            # Named class region inclusion faithfulness (M2)
            named_items = [
                (iri, idx)
                for iri, idx in self.named_class_index_dict.items()
                if iri not in (TOP, BOTTOM) and bool(sat_mask[idx].item())
            ]
            if len(named_items) > 128:
                named_items = named_items[:128]

            n_named = len(named_items)
            if n_named >= 2:
                n_pts_per_cls = max(16, min(256, n_eval // max(n_named, 1)))
                all_regs = mod.class_regions()
                cls_points: Dict[int, th.Tensor] = {}
                for _, c_idx in named_items:
                    s_reg = sample_class_region(
                        mod, c_idx, generator=gen, n_samples=n_pts_per_cls
                    )
                    corners = self._box_corner_samples(
                        all_regs.cen[c_idx : c_idx + 1],
                        all_regs.w[c_idx : c_idx + 1],
                        all_regs.active[c_idx : c_idx + 1],
                        gen,
                    )
                    pts_parts = [pos_all[c_idx : c_idx + 1]]
                    if bool(s_reg.valid.any().item()):
                        pts_parts.append(s_reg.points[s_reg.valid])
                    if corners.numel() > 0:
                        pts_parts.append(corners)
                    cand_pts = th.cat(pts_parts, dim=0)
                    in_c = mod.depth(cand_pts, c_idx, crisp=True) > 0.0
                    cls_points[c_idx] = cand_pts[in_c]

                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for a_iri, a_idx in named_items:
                    pts_a = cls_points[a_idx]
                    for b_iri, b_idx in named_items:
                        if a_idx == b_idx:
                            continue
                        gt_entailed = bool(
                            cm.extension(a_iri).issubset(cm.extension(b_iri))
                        )
                        if pts_a.shape[0] == 0:
                            pred_inc = True
                        else:
                            pred_inc = bool(
                                (mod.depth(pts_a, b_idx, crisp=True) > 0.0)
                                .all()
                                .item()
                            )
                        if gt_entailed and pred_inc:
                            tp += 1
                        elif not gt_entailed and pred_inc:
                            fp += 1
                        elif not gt_entailed and not pred_inc:
                            tn += 1
                        else:
                            fn += 1

                # Precision/recall of non-entailed pairs being separated & entailed pairs included
                precision = (
                    float(tn / (tn + fn))
                    if (tn + fn) > 0
                    else (1.0 if fp == 0 else 0.0)
                )
                recall = (
                    float(tn / (tn + fp))
                    if (tn + fp) > 0
                    else (1.0 if fn == 0 else 0.0)
                )
            else:
                precision = 1.0
                recall = 1.0

            exact = bool(
                elem_viol == 0.0
                and region_viol == 0.0
                and exist_val == 1.0
                and dead_ok
            )
            exact_tol1e4 = bool(
                elem_viol == 0.0
                and region_viol_tol1e4 == 0.0
                and exist_val == 1.0
                and dead_ok
            )
            return {
                "elem_viol": elem_viol,
                "region_viol": region_viol,
                "region_viol_tol1e-4": region_viol_tol1e4,
                "exist": exist_val,
                "dead_ok": dead_ok,
                "n_empty_classes": n_empty_classes,
                "n_dead_elements": n_dead_elements,
                "n_false_dead": n_false_dead,
                "faithful_named_precision": precision,
                "faithful_named_recall": recall,
                "faithful_named": (precision, recall),
                "exact": exact,
                "exact_tol1e-4": exact_tol1e4,
            }
