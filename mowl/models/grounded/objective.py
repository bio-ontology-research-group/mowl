"""The objective of GroundedEL: the grounded constraints.

A step samples a minibatch of every axiom form with candidate elements and region points
(the grounding of the axioms over the finite domain of the elements,
:mod:`mowl.nn.el.grounded.sampling`) and sums the hinge losses of the constraints
(:mod:`mowl.nn.el.grounded.losses`): :math:`L_{fact} + L_{axioms} + \\lambda_{reg}
L_{region} + L_{minimality}`. The class is the objective part of the combination: it is
mixed into :class:`mowl.models.GroundedEL`, which holds the parameters (the module)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch as th
from mowl.nn.el.grounded.sampling import (
    RegionSample,
    sample_class_region,
    sample_elements,
    sample_exists_region,
    sample_filler_elements,
    sample_inside_elements,
    sample_pair_region,
    sample_role_region,
)
from mowl.models.grounded.common import _CountInt

class GroundedConstraints:
    """The grounded constraints as the objective of the model (see the module docstring)."""

    def _sample_rows(self, data: th.Tensor, gen: th.Generator) -> th.Tensor:
        """Sample up to ``self.batch_size`` rows from ``data`` using ``gen``."""
        n = int(data.shape[0])
        if n <= self.batch_size:
            return data.to(device=self.device, dtype=th.long)
        idx = th.randint(
            0, n, (self.batch_size,), dtype=th.long, device=gen.device, generator=gen
        ).to(device=data.device)
        return data[idx].to(device=self.device, dtype=th.long)


    def _sample_corrupted_opa(
        self,
        opa_data: th.Tensor,
        gen: th.Generator,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Sample ``min(len(opa_data), batch_size) * neg_k`` rows of ``opa_data`` and corrupt head or tail with a random individual element ID."""
        n = int(opa_data.shape[0])
        n_neg = min(n, self.batch_size) * self.neg_k
        idx = th.randint(
            0, n, (n_neg,), dtype=th.long, device=gen.device, generator=gen
        ).to(device=opa_data.device)
        neg_rows = opa_data[idx].to(device=self.device, dtype=th.long)
        el_x = self.nb_classes + neg_rows[:, 0]
        r_id = neg_rows[:, 1]
        el_y = self.nb_classes + neg_rows[:, 2]
        rand_ind = (
            self.nb_classes
            + th.randint(
                0,
                self.nb_inds,
                (n_neg,),
                dtype=th.long,
                device=gen.device,
                generator=gen,
            ).to(device=self.device)
        )
        replace_head = (
            th.randint(
                0,
                2,
                (n_neg,),
                dtype=th.long,
                device=gen.device,
                generator=gen,
            ).to(device=self.device)
            == 0
        )
        x_neg = th.where(replace_head, rand_ind, el_x)
        y_neg = th.where(replace_head, el_y, rand_ind)
        return x_neg, y_neg, r_id


    def _batch_candidate_elements(
        self,
        batch_elems: th.Tensor,
        gen: th.Generator,
    ) -> th.Tensor:
        """Return candidate batch elements (all elements when ``nb_elements <= 32``, else subsampled)."""
        if self.nb_elements <= 32:
            return th.arange(self.nb_elements, dtype=th.long, device=self.device)
        uniq = th.unique(batch_elems)
        max_b = max(self.n_random_elements, 8)
        if uniq.numel() <= max_b:
            return uniq
        perm = th.randperm(uniq.numel(), device=gen.device, generator=gen)[
            :max_b
        ].to(device=uniq.device)
        return uniq[perm]


    def _inside_candidate_elements(
        self,
        lhs_classes: th.Tensor,
        gen: th.Generator,
        pos_all: th.Tensor,
    ) -> Optional[th.Tensor]:
        """Return up to ``n_inside_elements`` crisp-inside elements when not already covered by full ``_batch_candidate_elements``."""
        if self.n_inside_elements <= 0 or self.nb_elements <= 32:
            return None
        return sample_inside_elements(
            self.module,
            lhs_classes,
            self.n_inside_elements,
            gen,
            pos_all=pos_all,
        )


    def _sample_batch_y_fillers(
        self,
        rows: th.Tensor,
        gen: th.Generator,
    ) -> th.Tensor:
        """Subsample ``all_fillers`` to at most ``self.n_filler_samples`` per batch, always including each row's own filler ``rows[:, 1]``."""
        bsz = int(rows.shape[0])
        assert self._filler_mask_by_role is not None
        present = th.zeros((self.nb_rels,), dtype=th.bool, device=self.device)
        present[rows[:, 0]] = True
        mask = (self._filler_mask_by_role & present.unsqueeze(1)).any(0)
        mask[rows[:, 1]] = True
        all_fillers = th.nonzero(mask).squeeze(1)
        if all_fillers.numel() <= self.n_filler_samples:
            return sample_filler_elements(
                all_fillers,
                self.nb_elements,
                self.n_random_elements,
                generator=gen,
                batch_size=bsz,
            )
        n_shared = max(self.n_filler_samples - 1, 0)
        if n_shared > 0:
            perm = th.randperm(
                all_fillers.numel(), device=gen.device, generator=gen
            )[:n_shared].to(device=all_fillers.device)
            sub_fillers = all_fillers[perm]
        else:
            sub_fillers = th.empty((0,), dtype=th.long, device=self.device)
        y_rest = sample_filler_elements(
            sub_fillers,
            self.nb_elements,
            self.n_random_elements,
            generator=gen,
            batch_size=bsz,
        )
        return th.cat([rows[:, 1].unsqueeze(1), y_rest], dim=1)


    def _grid_role_membership(
        self,
        pos_all: th.Tensor,
        x_mat: th.Tensor,
        y_mat: th.Tensor,
        r_ids: th.Tensor,
        need_crisp: bool = False,
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """Degrees of the relation model on a ``(B, sx, sy)`` grid of elements (and the crisp
        depths when ``need_crisp``)."""
        return self.module.relation_model.grid_membership(
            pos_all[x_mat], pos_all[y_mat], r_ids, need_crisp=need_crisp)

    def _depth_and_membership_pos_grad_only(
        self,
        points: th.Tensor,
        class_ids: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """``(depth, membership)`` of the class model with its boxes held fixed (gradients
        reach only ``points``)."""
        return self.module.class_model.held_membership(points, class_ids)

    @property
    def constraint_set(self):
        """The grounded constraints of every axiom form, read in the logic of the model
        (:class:`~mowl.nn.el.grounded.constraints.ConstraintSet`)."""
        from mowl.nn.el.grounded.constraints import ConstraintSet
        rules = getattr(self, "rules", None)
        cached = self.__dict__.get("_constraint_set")
        if cached is None or cached.rules is not rules:
            cached = ConstraintSet(rules) if rules is not None else ConstraintSet()
            self.__dict__["_constraint_set"] = cached
        return cached

    def compute_batch_losses(
        self,
        generator: Optional[th.Generator] = None,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Dict[str, th.Tensor]:
        """Compute one step of per-term GroundedEL TBox/ABox losses (§3 and §4).

        Can be used directly as the TBox step of an external training loop (such as
        joint inductive protein + ontology training in DeepGO-SE / E3):

        - **Generator state:** Pass ``generator=model.generator`` (or omit
          ``generator`` to use ``self.generator`` automatically, which is seeded
          with ``self.seed + 1337`` at construction and can be reset via
          :meth:`reset_generator`). Consecutive calls advance ``self.generator`` so
          each step samples a fresh TBox mini-batch and random/adversarial points.
        - **Temperature schedule state:** Either pass ``step`` and ``total_steps``
          directly to :meth:`compute_batch_losses` (which calls
          ``self.step_tau(step, total_steps)`` before evaluating memberships) or
          call :meth:`step_tau` / ``self.module.set_tau(tau)`` once per epoch/step.
        - **Post-step deadness projection:** After ``optimizer.step()``, call
          :meth:`clamp_dead_logits` to project ``self.module.dead_logits`` onto
          ``[0, 1]`` and keep ``dead_logits[bot_id] = +100.0``.

        Guarantees:

        - Returns a dictionary containing ``"fact"``, ``"axioms"``, ``"region"``,
          ``"minimality"``, ``"neg"``, ``"total"``, and detached 0-dim tensors
          ``"w_fact"``, ``"w_axioms"``, ``"w_region"`` on ``self.device``.
        - Uses ``generator`` when provided, or ``self.generator`` when ``None``,
          for all random element and region draws.
        - When ``step`` and ``total_steps`` are provided, updates ``self.module.tau``
          via :meth:`step_tau` prior to loss evaluation.

        Non-guarantees:

        - When ``n_region_samples == 0`` or ``lambda_reg == 0``, ``"region"`` is
          exact ``0.0``.
        - When ``neg_weight > 0``, corrupted pairs are not filtered against the
          ABox (a corruption may coincide with a stated or entailed fact).

        Parameters
        ----------
        generator : Optional[torch.Generator], default None
            Explicit PyTorch RNG on CPU. Defaults to ``self.generator`` when ``None``.
        step : Optional[int], default None
            Optional 0-based schedule step index ``0 <= step < total_steps`` for
            geometric :math:`\\tau` annealing from ``tau0`` to ``tau1``.
        total_steps : Optional[int], default None
            Optional total number of schedule steps :math:`\\ge 1` for :math:`\\tau`
            annealing. Must be provided if and only if ``step`` is provided.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of scalar loss tensors.
        """
        if (step is None) != (total_steps is None):
            raise ValueError(
                "step and total_steps must both be provided or both be None"
            )
        if step is not None and total_steps is not None:
            self.step_tau(step, total_steps)
        if generator is None:
            generator = self.generator
        self._ensure_role_tables()
        self._ensure_static_id_tensors()

        mod = self.module
        device = self.device
        dtype = mod.class_model.centers.dtype
        td = self.training_datasets
        # The parts: class model, relation model, measure, shared element domain, and the
        # constraints of every form (read in the logic of the model).
        classes, relations = mod.class_model, mod.relation_model
        measure, shared = mod.measure, mod.shared
        C = self.constraint_set
        hinges = C.hinges

        def exists_depth(points, rel_ids, class_ids):
            return relations.exists_depth(points, rel_ids, class_ids, classes)

        pos_all = shared.positions()  # (nb_elements, d)
        dead_raw = shared.dead()
        if self.bot_id is not None:
            dead_all = dead_raw.clone()
            dead_all[self.bot_id] = 1.0
        else:
            dead_all = dead_raw
        # Crisp silencer gate for box/position hinges: live elements
        # pull boxes to full containment (m = 1); dead elements are silenced.
        dead_thresh = 0.75 if self.dead_from_bot_only else 0.5
        dead_gate = (dead_all.detach() > dead_thresh).to(dtype=dtype)

        # 1. Fact (existence) loss on class elements
        assert self._live_cls_ids is not None
        cls_ids = self._live_cls_ids
        if cls_ids.numel() > 0:
            if cls_ids.numel() > self.batch_size:
                cls_ids = self._sample_rows(cls_ids.unsqueeze(1), generator).squeeze(1)
            m_self = classes.membership(pos_all[cls_ids], cls_ids)
            loss_fact = C.fact(m_self, dead=dead_gate[cls_ids]).mean()
        else:
            loss_fact = th.zeros((), dtype=dtype, device=device)

        axiom_terms: List[th.Tensor] = []
        region_terms: List[th.Tensor] = []
        do_region = self.lambda_reg > 0.0 and self.n_region_samples > 0
        n_region_skipped_top = 0

        # 2. GCI0: A <= B
        if "gci0" in td and len(td["gci0"]) > 0:
            rows = self._sample_rows(td["gci0"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows.reshape(-1), generator)
            in_el = self._inside_candidate_elements(rows[:, 0], generator, pos_all)
            x_mat = sample_elements(
                rows[:, 0],
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = rows[:, 0].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            b_flat = rows[:, 1].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            ma = classes.membership(px, a_flat)
            mb = classes.membership(px, b_flat)
            if self.dead_from_bot_only:
                is_self = x_flat == a_flat
                in_b_crisp = classes.depth(px, b_flat, crisp=True) > 0.0
                ma_box = th.where(is_self, th.ones_like(ma), ma)
                mb_box = th.where(in_b_crisp, th.ones_like(mb), mb)
                h_box = C.gci0(ma_box, mb_box, dead=dead_gate[x_flat])
                a_col = rows[:, 0]
                b_col = rows[:, 1]
                d_pull = (1.0 - dead_gate[a_col]) * C.gci0_region(
                    classes.depth(pos_all[a_col], b_col), margin=self.region_margin
                )
                axiom_terms.append(h_box.mean() + d_pull.mean())
            else:
                h_box = C.gci0(ma, mb, dead=dead_gate[x_flat])
                is_self = x_flat == a_flat
                is_self_dead = is_self & (dead_gate[x_flat] > 0.5)
                crisp_viol_0 = (
                    is_self
                    & ((classes.depth(px, a_flat, crisp=True) > 0.0) | is_self_dead)
                    & (classes.depth(px, b_flat, crisp=True) <= 0.0)
                )
                ma_d = th.where(is_self_dead, th.ones_like(ma), ma.detach())
                dead_eff_0 = th.where(
                    crisp_viol_0, dead_all[x_flat], th.ones_like(dead_all[x_flat])
                )
                h_dead = C.gci0(ma_d, mb.detach(), dead=dead_eff_0)
                axiom_terms.append((h_box + h_dead).mean())
            if do_region:
                a_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                h_fn = (
                    (
                        lambda z: C.gci0_region(
                            classes.depth(z, rows[:, 1].repeat(max(z.shape[0] // bsz, 1))),
                            margin=self.region_margin,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                reg = sample_class_region(
                    mod,
                    rows[:, 0],
                    generator=generator,
                    n_samples=self.n_region_samples,
                    n_adv=self.n_adv,
                    hinge_fn=h_fn,
                )
                if bool(reg.valid.any().item()):
                    pts_raw = reg.points[reg.valid]
                    pts = pts_raw if self.region_lhs_grad else pts_raw.detach()
                    av = a_reg[reg.valid]
                    bv = b_reg[reg.valid]
                    live_a = 1.0 - dead_gate[av]
                    dead_a = dead_gate[av]
                    reg_h = live_a * C.gci0_region(
                        classes.depth(pts, bv), margin=self.region_margin
                    ) + dead_a * C.gci0_bot_region(
                        classes.depth(pts_raw.detach(), av), margin=self.region_margin
                    )
                    region_terms.append(reg_h.mean())

        # 3. GCI0_bot: A <= bot
        if "gci0_bot" in td and len(td["gci0_bot"]) > 0:
            rows = self._sample_rows(td["gci0_bot"][:], generator)
            a_col = rows[:, 0]
            bsz = int(a_col.shape[0])
            cand = self._batch_candidate_elements(a_col, generator)
            in_el = self._inside_candidate_elements(a_col, generator, pos_all)
            x_mat = sample_elements(
                a_col,
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = a_col.unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            ma = classes.membership(px, a_flat)
            if self.dead_from_bot_only:
                crisp_0b = (
                    (x_flat == a_flat)
                    | (classes.depth(px, a_flat, crisp=True) > 0.0)
                    | (dead_all[x_flat].detach() > 0.5)
                )
                dead_box_0b = th.where(
                    x_flat == a_flat, dead_gate[x_flat], th.ones_like(dead_gate[x_flat])
                )
                h_box = C.gci0_bot(ma, dead=dead_box_0b)
                ma_d = th.where(crisp_0b, th.ones_like(ma), ma.detach())
                dead_eff_0b = th.where(
                    crisp_0b, dead_all[x_flat], th.ones_like(dead_all[x_flat])
                )
                h_dead = C.gci0_bot(ma_d, dead=dead_eff_0b)
                axiom_terms.append((h_box + h_dead).mean())
            else:
                h_box = C.gci0_bot(ma, dead=dead_gate[x_flat])
                ma_d = th.where(x_flat == a_flat, th.ones_like(ma), ma.detach())
                h_dead = C.gci0_bot(ma_d, dead=dead_all[x_flat])
                axiom_terms.append((h_box + h_dead).mean())
            if do_region:
                a_reg = a_col.repeat(self.n_region_samples)
                h_fn = (
                    (
                        lambda z: C.gci0_bot_region(
                            classes.depth(z, a_col.repeat(max(z.shape[0] // bsz, 1))),
                            margin=self.region_margin,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                reg = sample_class_region(
                    mod,
                    a_col,
                    generator=generator,
                    n_samples=self.n_region_samples,
                    n_adv=self.n_adv,
                    hinge_fn=h_fn,
                )
                if bool(reg.valid.any().item()):
                    pts = reg.points[reg.valid].detach()
                    av = a_reg[reg.valid]
                    region_terms.append(
                        C.gci0_bot_region(
                            classes.depth(pts, av), margin=self.region_margin
                        ).mean()
                    )

        # 4. GCI1: A cap B <= C
        if "gci1" in td and len(td["gci1"]) > 0:
            rows = self._sample_rows(td["gci1"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows.reshape(-1), generator)
            in_el = self._inside_candidate_elements(rows[:, :2], generator, pos_all)
            x_mat = sample_elements(
                rows[:, :2],
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = rows[:, 0].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            b_flat = rows[:, 1].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            c_flat = rows[:, 2].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            ma = classes.membership(px, a_flat)
            mb = classes.membership(px, b_flat)
            mc = classes.membership(px, c_flat)
            axiom_terms.append(
                C.gci1(ma, mb, mc, dead=dead_gate[x_flat]).mean()
            )
            if do_region:
                a_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                c_reg = rows[:, 2].repeat(self.n_region_samples)
                h_fn = (
                    (
                        lambda z: C.gci1_region(
                            classes.depth(z, rows[:, 2].repeat(max(z.shape[0] // bsz, 1))),
                            margin=self.region_margin,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                reg = sample_pair_region(
                    mod,
                    rows[:, 0],
                    rows[:, 1],
                    generator=generator,
                    n_samples=self.n_region_samples,
                    n_adv=self.n_adv,
                    hinge_fn=h_fn,
                )
                if bool(reg.valid.any().item()):
                    pts_raw = reg.points[reg.valid]
                    pts = pts_raw if self.region_lhs_grad else pts_raw.detach()
                    av = a_reg[reg.valid]
                    bv = b_reg[reg.valid]
                    cv = c_reg[reg.valid]
                    live_ab = (1.0 - dead_gate[av]) * (1.0 - dead_gate[bv])
                    region_terms.append(
                        (
                            live_ab
                            * C.gci1_region(
                                classes.depth(pts, cv), margin=self.region_margin
                            )
                        ).mean()
                    )

        # 5. GCI1_bot: A cap B <= bot
        if "gci1_bot" in td and len(td["gci1_bot"]) > 0:
            rows = self._sample_rows(td["gci1_bot"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows[:, :2].reshape(-1), generator)
            in_el = self._inside_candidate_elements(rows[:, :2], generator, pos_all)
            x_mat = sample_elements(
                rows[:, :2],
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = rows[:, 0].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            b_flat = rows[:, 1].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            ma = classes.membership(px, a_flat)
            mb = classes.membership(px, b_flat)
            both_crisp = (classes.depth(px, a_flat, crisp=True) > 0.0) & (
                classes.depth(px, b_flat, crisp=True) > 0.0
            )
            if self.dead_from_bot_only:
                is_lhs_self = (x_flat == a_flat) | (x_flat == b_flat)
                dep_a_pos, ma_pos = self._depth_and_membership_pos_grad_only(px, a_flat)
                dep_b_pos, mb_pos = self._depth_and_membership_pos_grad_only(px, b_flat)
                ma_box = th.where(is_lhs_self, ma, ma_pos)
                mb_box = th.where(is_lhs_self, mb, mb_pos)
                d_push = (
                    0.5
                    * (1.0 - dead_gate[x_flat])
                    * hinges.outside(th.minimum(dep_a_pos, dep_b_pos))
                )
                h_box = C.gci1_bot(ma_box, mb_box, dead=dead_gate[x_flat]) + d_push
                a_col = rows[:, 0]
                b_col = rows[:, 1]
                d_sep = 0.5 * (
                    (1.0 - dead_gate[a_col]) * hinges.outside(classes.depth(pos_all[a_col], b_col))
                    + (1.0 - dead_gate[b_col]) * hinges.outside(classes.depth(pos_all[b_col], a_col))
                )
                dead_eff_1b = th.where(
                    both_crisp, dead_all[x_flat], th.ones_like(dead_all[x_flat])
                )
                h_dead = C.gci1_bot(
                    th.ones_like(ma), th.ones_like(mb), dead=dead_eff_1b
                )
                axiom_terms.append((h_box + h_dead).mean() + d_sep.mean())
            else:
                h_box = C.gci1_bot(ma, mb, dead=dead_gate[x_flat])
                dead_eff_1b = th.where(
                    both_crisp, dead_all[x_flat], th.ones_like(dead_all[x_flat])
                )
                h_dead = C.gci1_bot(ma.detach(), mb.detach(), dead=dead_eff_1b)
                axiom_terms.append((h_box + h_dead).mean())
            if do_region:
                a_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                h_fn = (
                    (
                        lambda z: C.gci1_bot_region(
                            classes.depth(z, rows[:, 0].repeat(max(z.shape[0] // bsz, 1))),
                            classes.depth(z, rows[:, 1].repeat(max(z.shape[0] // bsz, 1))),
                            margin=self.region_margin,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                reg = sample_pair_region(
                    mod,
                    rows[:, 0],
                    rows[:, 1],
                    generator=generator,
                    n_samples=self.n_region_samples,
                    n_adv=self.n_adv,
                    hinge_fn=h_fn,
                )
                if bool(reg.valid.any().item()):
                    pts = reg.points[reg.valid].detach()
                    av = a_reg[reg.valid]
                    bv = b_reg[reg.valid]
                    region_terms.append(
                        C.gci1_bot_region(
                            classes.depth(pts, av),
                            classes.depth(pts, bv),
                            margin=self.region_margin,
                        ).mean()
                    )

        # 6. GCI2: A <= exists r.B
        if "gci2" in td and len(td["gci2"]) > 0:
            rows = self._sample_rows(td["gci2"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows[:, [0, 2]].reshape(-1), generator)
            in_el = self._inside_candidate_elements(rows[:, 0], generator, pos_all)
            x_mat = sample_elements(
                rows[:, 0],
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = rows[:, 0].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            r_flat = rows[:, 1].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            b_flat = rows[:, 2].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            pb = pos_all[b_flat]
            ma = classes.membership(px, a_flat)
            rho_xb = relations.membership(px, pb, r_flat)
            df_b = dead_all[b_flat].detach()
            if self.dead_from_bot_only:
                crisp_2d = (
                    (x_flat == a_flat)
                    | (classes.depth(px, a_flat, crisp=True) > 0.0)
                    | (dead_all[x_flat].detach() > 0.5)
                )
                ma_d = th.where(crisp_2d, th.ones_like(ma), ma.detach())
            else:
                is_self_dead = (x_flat == a_flat) & (dead_gate[x_flat] > 0.5)
                ma_d = th.where(is_self_dead, th.ones_like(ma), ma.detach())
            edge_h = C.gci2(ma, rho_xb, dead=dead_gate[x_flat], dead_filler=None)
            dead_prop_h = C.gci2_dead(
                ma, dead_filler=df_b, dead=dead_gate[x_flat]
            ) + th.where(
                df_b > 0.5,
                C.gci2_dead(ma_d, dead_filler=df_b, dead=dead_all[x_flat]),
                th.zeros_like(ma),
            )
            axiom_terms.append((edge_h + dead_prop_h).mean())
            if do_region:
                a_reg = rows[:, 0].repeat(self.n_region_samples)
                r_reg = rows[:, 1].repeat(self.n_region_samples)
                b_reg = rows[:, 2].repeat(self.n_region_samples)
                if relations.mode == "translation":
                    h_fn = (
                        (
                            lambda z: C.gci2_region_trans(
                                classes.membership(
                                    z,
                                    rows[:, 0].repeat(max(z.shape[0] // bsz, 1)),
                                ),
                                measure.soft(
                                    exists_depth(
                                        z,
                                        rows[:, 1].repeat(max(z.shape[0] // bsz, 1)),
                                        rows[:, 2].repeat(max(z.shape[0] // bsz, 1)),
                                    ), shared.tau),
                            )
                        )
                        if self.n_adv > 0
                        else None
                    )
                    reg = sample_class_region(
                        mod,
                        rows[:, 0],
                        generator=generator,
                        n_samples=self.n_region_samples,
                        n_adv=self.n_adv,
                        hinge_fn=h_fn,
                    )
                    if bool(reg.valid.any().item()):
                        pts_raw = reg.points[reg.valid]
                        pts = pts_raw if self.region_lhs_grad else pts_raw.detach()
                        av = a_reg[reg.valid]
                        rv = r_reg[reg.valid]
                        bv = b_reg[reg.valid]
                        live_a = 1.0 - dead_gate[av]
                        dead_a = dead_gate[av]
                        m_a_z = classes.membership(pts, av)
                        ex_dep = exists_depth(pts, rv, bv)
                        m_exists_z = measure.soft(
                            ex_dep, shared.tau) * (1.0 - dead_gate[bv])
                        reg_h = live_a * C.gci2_region_trans(
                            m_a_z, m_exists_z
                        ) + dead_a * C.gci0_bot_region(
                            classes.depth(pts_raw.detach(), av), margin=self.region_margin
                        )
                        region_terms.append(reg_h.mean())
                else:
                    h_fn = (
                        (
                            lambda z: C.gci2_region(
                                relations.depth(
                                    z,
                                    pos_all[
                                        rows[:, 2].repeat(max(z.shape[0] // bsz, 1))
                                    ].detach(),
                                    rows[:, 1].repeat(max(z.shape[0] // bsz, 1)),
                                ),
                                margin=self.region_margin,
                                agg=self.agg,
                                agg_tau=self.agg_tau,
                            )
                        )
                        if self.n_adv > 0
                        else None
                    )
                    reg = sample_class_region(
                        mod,
                        rows[:, 0],
                        generator=generator,
                        n_samples=self.n_region_samples,
                        n_adv=self.n_adv,
                        hinge_fn=h_fn,
                    )
                    if bool(reg.valid.any().item()):
                        pts_raw = reg.points[reg.valid]
                        pts = pts_raw if self.region_lhs_grad else pts_raw.detach()
                        av = a_reg[reg.valid]
                        rv = r_reg[reg.valid]
                        bv = b_reg[reg.valid]
                        live_a = 1.0 - dead_gate[av]
                        dead_a = dead_gate[av]
                        reg_h = live_a * C.gci2_region(
                            relations.depth(pts, pos_all[bv], rv),
                            margin=self.region_margin,
                            agg=self.agg,
                            agg_tau=self.agg_tau,
                            depth_a=classes.depth(pts_raw.detach(), av),
                            dead_filler=dead_all[bv].detach(),
                        ) + dead_a * C.gci0_bot_region(
                            classes.depth(pts_raw.detach(), av), margin=self.region_margin
                        )
                        region_terms.append(reg_h.mean())

        # 7. GCI3: exists r.B <= C
        if "gci3" in td and len(td["gci3"]) > 0:
            rows = self._sample_rows(td["gci3"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows[:, 1:].reshape(-1), generator)
            in_el = self._inside_candidate_elements(rows[:, 2], generator, pos_all)
            x_mat = sample_elements(
                rows[:, 2],
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            y_mat = self._sample_batch_y_fillers(rows, generator)
            sx = int(x_mat.shape[1])
            sy = int(y_mat.shape[1])
            rho_xy, _ = self._grid_role_membership(
                pos_all, x_mat, y_mat, rows[:, 0], need_crisp=False
            )
            b_2d = rows[:, 1].unsqueeze(1).expand(bsz, sy).reshape(-1)
            c_2d = rows[:, 2].unsqueeze(1).expand(bsz, sx).reshape(-1)
            mb_grid = (
                classes.membership(pos_all[y_mat.reshape(-1)], b_2d)
                .view(bsz, 1, sy)
                .expand(bsz, sx, sy)
                .reshape(-1)
            )
            mc_grid = (
                classes.membership(pos_all[x_mat.reshape(-1)], c_2d)
                .view(bsz, sx, 1)
                .expand(bsz, sx, sy)
                .reshape(-1)
            )
            dead_x_grid = (
                dead_gate[x_mat].unsqueeze(2).expand(bsz, sx, sy).reshape(-1)
            )
            axiom_terms.append(
                C.gci3(rho_xy, mb_grid, mc_grid, dead=dead_x_grid).mean()
            )
            if do_region:
                if self.top_id is not None:
                    top_mask_3 = rows[:, 1] == self.top_id
                    n_skip_3 = int(top_mask_3.sum().item())
                    n_region_skipped_top += n_skip_3
                    rows_3 = rows[~top_mask_3] if n_skip_3 > 0 else rows
                else:
                    rows_3 = rows
                bsz_3 = int(rows_3.shape[0])
                if bsz_3 > 0:
                    r_reg = rows_3[:, 0].repeat(self.n_region_samples)
                    b_reg = rows_3[:, 1].repeat(self.n_region_samples)
                    c_reg = rows_3[:, 2].repeat(self.n_region_samples)
                    if relations.mode == "translation":
                        h_fn_3 = (
                            (
                                lambda z: C.gci3_region_trans(
                                    measure.soft(
                                        exists_depth(
                                            z,
                                            rows_3[:, 0].repeat(
                                                max(z.shape[0] // bsz_3, 1)
                                            ),
                                            rows_3[:, 1].repeat(
                                                max(z.shape[0] // bsz_3, 1)
                                            ),
                                        ), shared.tau),
                                    classes.membership(
                                        z,
                                        rows_3[:, 2].repeat(
                                            max(z.shape[0] // bsz_3, 1)
                                        ),
                                    ),
                                )
                            )
                            if self.n_adv > 0
                            else None
                        )
                        ex_samp = sample_exists_region(
                            mod,
                            rows_3[:, 0],
                            rows_3[:, 1],
                            generator=generator,
                            n_samples=self.n_region_samples,
                            n_adv=self.n_adv,
                            hinge_fn=h_fn_3,
                        )
                        if bool(ex_samp.valid.any().item()):
                            pts_raw = ex_samp.points[ex_samp.valid]
                            pts = (
                                pts_raw if self.region_lhs_grad else pts_raw.detach()
                            )
                            rv = r_reg[ex_samp.valid]
                            bv = b_reg[ex_samp.valid]
                            cv = c_reg[ex_samp.valid]
                            ex_dep = exists_depth(pts, rv, bv)
                            m_exists_z = measure.soft(
                                ex_dep, shared.tau) * (1.0 - dead_gate[bv])
                            m_c_z = classes.membership(pts, cv)
                            region_terms.append(
                                C.gci3_region_trans(m_exists_z, m_c_z).mean()
                            )
                    else:
                        h_fn_3 = (
                            (
                                lambda xy: C.gci3_region(
                                    relations.depth(
                                        xy[:, : self.embed_dim],
                                        xy[:, self.embed_dim :],
                                        rows_3[:, 0].repeat(
                                            max(xy.shape[0] // bsz_3, 1)
                                        ),
                                    ),
                                    classes.depth(
                                        xy[:, self.embed_dim :],
                                        rows_3[:, 1].repeat(
                                            max(xy.shape[0] // bsz_3, 1)
                                        ),
                                    ),
                                    classes.depth(
                                        xy[:, : self.embed_dim],
                                        rows_3[:, 2].repeat(
                                            max(xy.shape[0] // bsz_3, 1)
                                        ),
                                    ),
                                    margin=self.region_margin,
                                    smooth=True,
                                )
                            )
                            if self.n_adv > 0
                            else None
                        )
                        h_samp, t_samp = sample_role_region(
                            mod,
                            rows_3[:, 0],
                            generator=generator,
                            n_samples=self.n_region_samples,
                            n_adv=self.n_adv,
                            hinge_fn=h_fn_3,
                        )
                        if bool(h_samp.valid.any().item()):
                            zx_raw = h_samp.points[h_samp.valid]
                            zy_raw = t_samp.points[t_samp.valid]
                            zx = zx_raw if self.region_lhs_grad else zx_raw.detach()
                            zy = zy_raw if self.region_lhs_grad else zy_raw.detach()
                            rv = r_reg[h_samp.valid]
                            bv = b_reg[h_samp.valid]
                            cv = c_reg[h_samp.valid]
                            region_terms.append(
                                C.gci3_region(
                                    relations.depth(
                                        zx_raw.detach(), zy_raw.detach(), rv
                                    ),
                                    classes.depth(zy, bv),
                                    classes.depth(zx, cv),
                                    margin=self.region_margin,
                                ).mean()
                            )
                            # Also evaluate at z = zx with y = filler element x_B (§4)
                            py_fill = pos_all[bv]
                            region_terms.append(
                                C.gci3_region(
                                    relations.depth(zx_raw.detach(), py_fill, rv),
                                    classes.depth(py_fill, bv),
                                    classes.depth(zx, cv),
                                    margin=self.region_margin,
                                ).mean()
                            )

        # 8. GCI3_bot: exists r.B <= bot
        if "gci3_bot" in td and len(td["gci3_bot"]) > 0:
            rows = self._sample_rows(td["gci3_bot"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows[:, 1], generator)
            in_el = self._inside_candidate_elements(rows[:, 1], generator, pos_all)
            x_mat = sample_elements(
                rows[:, 1],
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            y_mat = self._sample_batch_y_fillers(rows, generator)
            sx = int(x_mat.shape[1])
            sy = int(y_mat.shape[1])
            rho_xy, r_dep_crisp = self._grid_role_membership(
                pos_all, x_mat, y_mat, rows[:, 0], need_crisp=True
            )
            assert r_dep_crisp is not None
            b_2d = rows[:, 1].unsqueeze(1).expand(bsz, sy).reshape(-1)
            py_flat = pos_all[y_mat.reshape(-1)]
            mb_grid = (
                classes.membership(py_flat, b_2d)
                .view(bsz, 1, sy)
                .expand(bsz, sx, sy)
                .reshape(-1)
            )
            b_crisp_grid = (
                (classes.depth(py_flat, b_2d, crisp=True) > 0.0)
                .view(bsz, 1, sy)
                .expand(bsz, sx, sy)
                .reshape(-1)
            )
            both_crisp = (r_dep_crisp > 0.0) & b_crisp_grid
            x_grid = x_mat.unsqueeze(2).expand(bsz, sx, sy).reshape(-1)
            h_box = C.gci3_bot(rho_xy, mb_grid, dead=dead_gate[x_grid])
            if self.dead_from_bot_only:
                dead_eff_3b = th.where(
                    both_crisp, dead_all[x_grid], th.ones_like(dead_all[x_grid])
                )
                h_dead = C.gci3_bot(
                    th.ones_like(rho_xy), th.ones_like(mb_grid), dead=dead_eff_3b
                )
            else:
                dead_eff_3b = th.where(
                    both_crisp, dead_all[x_grid], th.ones_like(dead_all[x_grid])
                )
                h_dead = C.gci3_bot(
                    rho_xy.detach(), mb_grid.detach(), dead=dead_eff_3b
                )
            axiom_terms.append((h_box + h_dead).mean())
            if do_region:
                if self.top_id is not None:
                    top_mask_3b = rows[:, 1] == self.top_id
                    n_skip_3b = int(top_mask_3b.sum().item())
                    n_region_skipped_top += n_skip_3b
                    rows_3b = rows[~top_mask_3b] if n_skip_3b > 0 else rows
                else:
                    rows_3b = rows
                bsz_3b = int(rows_3b.shape[0])
                if bsz_3b > 0:
                    r_reg = rows_3b[:, 0].repeat(self.n_region_samples)
                    b_reg = rows_3b[:, 1].repeat(self.n_region_samples)
                    if relations.mode == "translation":
                        h_fn_3b = (
                            (
                                lambda z: C.gci3_bot_region_trans(
                                    measure.soft(
                                        exists_depth(
                                            z,
                                            rows_3b[:, 0].repeat(
                                                max(z.shape[0] // bsz_3b, 1)
                                            ),
                                            rows_3b[:, 1].repeat(
                                                max(z.shape[0] // bsz_3b, 1)
                                            ),
                                        ), shared.tau)
                                )
                            )
                            if self.n_adv > 0
                            else None
                        )
                        ex_samp = sample_exists_region(
                            mod,
                            rows_3b[:, 0],
                            rows_3b[:, 1],
                            generator=generator,
                            n_samples=self.n_region_samples,
                            n_adv=self.n_adv,
                            hinge_fn=h_fn_3b,
                        )
                        if bool(ex_samp.valid.any().item()):
                            pts_raw = ex_samp.points[ex_samp.valid]
                            pts = (
                                pts_raw if self.region_lhs_grad else pts_raw.detach()
                            )
                            rv = r_reg[ex_samp.valid]
                            bv = b_reg[ex_samp.valid]
                            ex_dep = exists_depth(pts, rv, bv)
                            m_exists_z = measure.soft(
                                ex_dep, shared.tau) * (1.0 - dead_gate[bv])
                            region_terms.append(
                                C.gci3_bot_region_trans(m_exists_z).mean()
                            )
                    else:
                        h_fn_3b = (
                            (
                                lambda xy: C.gci3_bot_region(
                                    relations.depth(
                                        xy[:, : self.embed_dim],
                                        xy[:, self.embed_dim :],
                                        rows_3b[:, 0].repeat(
                                            max(xy.shape[0] // bsz_3b, 1)
                                        ),
                                    ),
                                    classes.depth(
                                        xy[:, self.embed_dim :],
                                        rows_3b[:, 1].repeat(
                                            max(xy.shape[0] // bsz_3b, 1)
                                        ),
                                    ),
                                    margin=self.region_margin,
                                )
                            )
                            if self.n_adv > 0
                            else None
                        )
                        h_samp, t_samp = sample_role_region(
                            mod,
                            rows_3b[:, 0],
                            generator=generator,
                            n_samples=self.n_region_samples,
                            n_adv=self.n_adv,
                            hinge_fn=h_fn_3b,
                        )
                        if bool(h_samp.valid.any().item()):
                            zx_raw = h_samp.points[h_samp.valid]
                            zy_raw = t_samp.points[t_samp.valid]
                            zy = zy_raw if self.region_lhs_grad else zy_raw.detach()
                            rv = r_reg[h_samp.valid]
                            bv = b_reg[h_samp.valid]
                            region_terms.append(
                                C.gci3_bot_region(
                                    relations.depth(
                                        zx_raw.detach(), zy_raw.detach(), rv
                                    ),
                                    classes.depth(zy, bv),
                                    margin=self.region_margin,
                                ).mean()
                            )

        # 9. Role inclusion: r <= s
        if "role_inclusion" in td and len(td["role_inclusion"]) > 0:
            rows = self._sample_rows(td["role_inclusion"][:], generator)
            bsz = int(rows.shape[0])
            all_el = self._batch_candidate_elements(
                th.arange(min(self.nb_elements, 16), device=device), generator
            )
            sx = int(all_el.numel())
            x_grid = all_el.view(1, sx, 1).expand(bsz, sx, sx).reshape(-1)
            y_grid = all_el.view(1, 1, sx).expand(bsz, sx, sx).reshape(-1)
            r_grid = rows[:, 0].view(bsz, 1, 1).expand(bsz, sx, sx).reshape(-1)
            s_grid = rows[:, 1].view(bsz, 1, 1).expand(bsz, sx, sx).reshape(-1)
            px = pos_all[x_grid]
            py = pos_all[y_grid]
            axiom_terms.append(
                C.role_inclusion(
                    relations.membership(px, py, r_grid),
                    relations.membership(px, py, s_grid),
                ).mean()
            )
            if do_region:
                r_reg = rows[:, 0].repeat(self.n_region_samples)
                s_reg = rows[:, 1].repeat(self.n_region_samples)
                if relations.mode == "translation":
                    h_fn_ri = (
                        (
                            lambda v: C.role_inclusion_region_trans(
                                relations.membership(
                                    th.zeros_like(v),
                                    v,
                                    rows[:, 0].repeat(max(v.shape[0] // bsz, 1)),
                                ),
                                relations.membership(
                                    th.zeros_like(v),
                                    v,
                                    rows[:, 1].repeat(max(v.shape[0] // bsz, 1)),
                                ),
                            )
                        )
                        if self.n_adv > 0
                        else None
                    )
                    v_samp = sample_role_region(
                        mod,
                        rows[:, 0],
                        generator=generator,
                        n_samples=self.n_region_samples,
                        n_adv=self.n_adv,
                        hinge_fn=h_fn_ri,
                    )
                    assert isinstance(v_samp, RegionSample)
                    if bool(v_samp.valid.any().item()):
                        v_raw = v_samp.points[v_samp.valid]
                        v_pts = v_raw if self.region_lhs_grad else v_raw.detach()
                        rv = r_reg[v_samp.valid]
                        sv = s_reg[v_samp.valid]
                        z0 = th.zeros_like(v_pts)
                        region_terms.append(
                            C.role_inclusion_region_trans(
                                relations.membership(z0, v_pts, rv),
                                relations.membership(z0, v_pts, sv),
                            ).mean()
                        )
                else:
                    h_fn_ri = (
                        (
                            lambda xy: C.role_inclusion_region(
                                relations.depth(
                                    xy[:, : self.embed_dim],
                                    xy[:, self.embed_dim :],
                                    rows[:, 1].repeat(max(xy.shape[0] // bsz, 1)),
                                ),
                                margin=self.region_margin,
                            )
                        )
                        if self.n_adv > 0
                        else None
                    )
                    h_samp, t_samp = sample_role_region(
                        mod,
                        rows[:, 0],
                        generator=generator,
                        n_samples=self.n_region_samples,
                        n_adv=self.n_adv,
                        hinge_fn=h_fn_ri,
                    )
                    if bool(h_samp.valid.any().item()):
                        zx_raw = h_samp.points[h_samp.valid]
                        zy_raw = t_samp.points[t_samp.valid]
                        zx = zx_raw if self.region_lhs_grad else zx_raw.detach()
                        zy = zy_raw if self.region_lhs_grad else zy_raw.detach()
                        sv = s_reg[h_samp.valid]
                        region_terms.append(
                            C.role_inclusion_region(
                                relations.depth(zx, zy, sv),
                                margin=self.region_margin,
                            ).mean()
                        )

        # 10. Role chain: r o s <= t
        if "role_chain" in td and len(td["role_chain"]) > 0:
            rows = self._sample_rows(td["role_chain"][:], generator)
            bsz = int(rows.shape[0])
            all_el = self._batch_candidate_elements(
                th.arange(min(self.nb_elements, 12), device=device), generator
            )
            if all_el.numel() > 10:
                all_el = all_el[:10]
            sx = int(all_el.numel())
            x_g = all_el.view(1, sx, 1, 1).expand(bsz, sx, sx, sx).reshape(-1)
            y_g = all_el.view(1, 1, sx, 1).expand(bsz, sx, sx, sx).reshape(-1)
            z_g = all_el.view(1, 1, 1, sx).expand(bsz, sx, sx, sx).reshape(-1)
            r_g = rows[:, 0].view(bsz, 1, 1, 1).expand(bsz, sx, sx, sx).reshape(-1)
            s_g = rows[:, 1].view(bsz, 1, 1, 1).expand(bsz, sx, sx, sx).reshape(-1)
            t_g = rows[:, 2].view(bsz, 1, 1, 1).expand(bsz, sx, sx, sx).reshape(-1)
            px = pos_all[x_g]
            py = pos_all[y_g]
            pz = pos_all[z_g]
            axiom_terms.append(
                C.role_chain(
                    relations.membership(px, py, r_g),
                    relations.membership(py, pz, s_g),
                    relations.membership(px, pz, t_g),
                ).mean()
            )
            if do_region:
                r_reg = rows[:, 0].repeat(self.n_region_samples)
                s_reg = rows[:, 1].repeat(self.n_region_samples)
                t_reg = rows[:, 2].repeat(self.n_region_samples)
                if relations.mode == "translation":
                    v1_samp = sample_role_region(
                        mod,
                        rows[:, 0],
                        generator=generator,
                        n_samples=self.n_region_samples,
                        n_adv=0,
                    )
                    v2_samp = sample_role_region(
                        mod,
                        rows[:, 1],
                        generator=generator,
                        n_samples=self.n_region_samples,
                        n_adv=0,
                    )
                    assert isinstance(v1_samp, RegionSample) and isinstance(
                        v2_samp, RegionSample
                    )
                    v_chain = v1_samp.valid & v2_samp.valid
                    if bool(v_chain.any().item()):
                        v1_raw = v1_samp.points[v_chain]
                        v2_raw = v2_samp.points[v_chain]
                        v1 = v1_raw if self.region_lhs_grad else v1_raw.detach()
                        v2 = v2_raw if self.region_lhs_grad else v2_raw.detach()
                        rv = r_reg[v_chain]
                        sv = s_reg[v_chain]
                        tv = t_reg[v_chain]
                        z0 = th.zeros_like(v1)
                        region_terms.append(
                            C.role_chain_region_trans(
                                relations.membership(z0, v1, rv),
                                relations.membership(z0, v2, sv),
                                relations.membership(z0, v1 + v2, tv),
                            ).mean()
                        )
                else:
                    hx, hy = sample_role_region(
                        mod,
                        rows[:, 0],
                        generator=generator,
                        n_samples=self.n_region_samples,
                        n_adv=0,
                    )
                    _, tz = sample_role_region(
                        mod,
                        rows[:, 1],
                        generator=generator,
                        n_samples=self.n_region_samples,
                        n_adv=0,
                    )
                    v_chain = hx.valid & tz.valid
                    if bool(v_chain.any().item()):
                        zx_raw = hx.points[v_chain]
                        zy_raw = hy.points[v_chain]
                        zz_raw = tz.points[v_chain]
                        zx = zx_raw if self.region_lhs_grad else zx_raw.detach()
                        zy = zy_raw if self.region_lhs_grad else zy_raw.detach()
                        zz = zz_raw if self.region_lhs_grad else zz_raw.detach()
                        rv = r_reg[v_chain]
                        sv = s_reg[v_chain]
                        tv = t_reg[v_chain]
                        region_terms.append(
                            C.role_chain_region(
                                relations.depth(
                                    zx_raw.detach(), zy_raw.detach(), rv
                                ),
                                relations.depth(zy, zz_raw.detach(), sv),
                                relations.depth(zx, zz, tv),
                                margin=self.region_margin,
                            ).mean()
                        )

        # 11. Class assertion: A(a)
        if "class_assertion" in td and len(td["class_assertion"]) > 0:
            rows = self._sample_rows(td["class_assertion"][:], generator)
            el_a = self.nb_classes + rows[:, 0]
            c_id = rows[:, 1]
            axiom_terms.append(
                C.class_assertion(classes.membership(pos_all[el_a], c_id)).mean()
            )

        # 12. Object property assertion: r(a, b)
        if (
            "object_property_assertion" in td
            and len(td["object_property_assertion"]) > 0
        ):
            rows = self._sample_rows(td["object_property_assertion"][:], generator)
            el_a = self.nb_classes + rows[:, 0]
            r_id = rows[:, 1]
            el_b = self.nb_classes + rows[:, 2]
            axiom_terms.append(
                C.role_assertion(
                    relations.membership(pos_all[el_a], pos_all[el_b], r_id)
                ).mean()
            )

        # 12b. Corrupted-fact negatives on object property assertions
        if (
            self.neg_weight > 0.0
            and self.nb_inds > 0
            and "object_property_assertion" in td
            and len(td["object_property_assertion"]) > 0
        ):
            x_neg, y_neg, r_neg = self._regime_corrupted_opa(
                td["object_property_assertion"], generator
            )
            rho_neg = relations.membership(pos_all[x_neg], pos_all[y_neg], r_neg)
            loss_neg = rho_neg.mean()
        else:
            loss_neg = th.zeros((), dtype=dtype, device=device)

        loss_axioms = (
            th.stack(axiom_terms).sum()
            if axiom_terms
            else th.zeros((), dtype=dtype, device=device)
        )
        loss_region = (
            th.stack(region_terms).sum()
            if region_terms
            else th.zeros((), dtype=dtype, device=device)
        )

        # 13. Minimality (principled negatives)
        assert self._ord_cls_ids is not None
        ord_cls = self._ord_cls_ids
        if ord_cls.numel() > 0 and self.eps_min > 0.0:
            if self.nb_elements * ord_cls.numel() <= 1024:
                all_el = th.arange(self.nb_elements, dtype=th.long, device=device)
                gx = all_el.unsqueeze(1).expand(self.nb_elements, ord_cls.numel()).reshape(-1)
                ga = ord_cls.unsqueeze(0).expand(self.nb_elements, ord_cls.numel()).reshape(-1)
            else:
                n_min = min(self.batch_size, max(256, self.nb_elements * self.n_random_elements))
                gx = th.randint(
                    0, self.nb_elements, (n_min,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
                ga_idx = th.randint(
                    0, ord_cls.numel(), (n_min,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
                ga = ord_cls[ga_idx]
            m_rand = classes.membership(pos_all[gx], ga)
        else:
            m_rand = None

        if self.nb_rels > 0 and self.eps_role > 0.0:
            if (self.nb_elements ** 2) * self.nb_rels <= 1024:
                all_el = th.arange(self.nb_elements, dtype=th.long, device=device)
                all_r = th.arange(self.nb_rels, dtype=th.long, device=device)
                rx = all_el.view(self.nb_elements, 1, 1).expand(self.nb_elements, self.nb_elements, self.nb_rels).reshape(-1)
                ry = all_el.view(1, self.nb_elements, 1).expand(self.nb_elements, self.nb_elements, self.nb_rels).reshape(-1)
                rr = all_r.view(1, 1, self.nb_rels).expand(self.nb_elements, self.nb_elements, self.nb_rels).reshape(-1)
            else:
                n_rmin = min(self.batch_size, max(256, self.nb_elements * self.n_random_elements))
                rx = th.randint(
                    0, self.nb_elements, (n_rmin,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
                ry = th.randint(
                    0, self.nb_elements, (n_rmin,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
                rr = th.randint(
                    0, self.nb_rels, (n_rmin,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
            rho_rand = relations.membership(pos_all[rx], pos_all[ry], rr)
        else:
            rho_rand = None

        assert self._dead_min_ids is not None
        dead_for_min = (
            dead_all[self._dead_min_ids]
            if self._dead_min_ids.numel() > 0
            else None
        )
        loss_min = C.minimality(
            m_rand=m_rand,
            rho_rand=rho_rand,
            dead=dead_for_min,
            eps=self.eps_min,
            eps_r=self.eps_role,
            eps_dead=self.eps_dead,
        )

        if self.constraint_mode == "fixed":
            w_fact_f = 1.0
            w_axioms_f = 1.0
            w_region_f = 1.0
            total = (
                loss_fact
                + loss_axioms
                + self.lambda_reg * loss_region
                + loss_min
            )
        else:
            w_fact_f = float(self.dual_weights["fact"])
            w_axioms_f = float(self.dual_weights["axioms"])
            w_region_f = float(self.dual_weights["region"])
            total = (
                w_fact_f * loss_fact
                + w_axioms_f * loss_axioms
                + w_region_f * self.lambda_reg * loss_region
                + loss_min
            )
        if self.neg_weight > 0.0:
            total = total + self.neg_weight * loss_neg

        return {
            "fact": loss_fact,
            "axioms": loss_axioms,
            "region": loss_region,
            "minimality": loss_min,
            "neg": loss_neg,
            "total": total,
            "w_fact": th.tensor(w_fact_f, dtype=dtype, device=device).detach(),
            "w_axioms": th.tensor(w_axioms_f, dtype=dtype, device=device).detach(),
            "w_region": th.tensor(w_region_f, dtype=dtype, device=device).detach(),
            "n_region_skipped_top": _CountInt(n_region_skipped_top),  # type: ignore[dict-item]
        }


    def _regime_corrupted_opa(
        self, opa_dataset: Any, gen: th.Generator
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Corrupted object property assertions as element ids ``(x, y, r)``: from the
        active training regime's ``negatives`` hook if it provides them, otherwise from
        the model's own sampler, and in both cases passed through the regime's
        ``filter_negatives`` (for example to remove stated facts). Without a regime
        this is exactly :meth:`_sample_corrupted_opa`."""
        regime = getattr(self, "_active_regime", None)
        if regime is None:
            regime = self.training_regime
        if regime is None:
            return self._sample_corrupted_opa(opa_dataset[:], gen)
        rows = regime.negatives("object_property_assertion", opa_dataset)
        if rows is NotImplemented or rows is None:
            x_neg, y_neg, r_neg = self._sample_corrupted_opa(opa_dataset[:], gen)
            rows = th.stack(
                [x_neg - self.nb_classes, r_neg, y_neg - self.nb_classes], dim=1)
        filtered = regime.filter_negatives("object_property_assertion", opa_dataset, rows)
        if filtered is not None:
            rows = filtered
        rows = rows.to(device=self.device, dtype=th.long)
        return self.nb_classes + rows[:, 0], self.nb_classes + rows[:, 2], rows[:, 1]
