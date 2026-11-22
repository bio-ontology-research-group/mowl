"""The objective of GroundedEL, as the model uses it: the parts that make one step.

A step is :meth:`GroundedObjective.losses
<mowl.representation.library.grounded_objective.GroundedObjective.losses>` (every axiom
form read at sampled elements and at sampled points of regions) over the geometry
(``self.module``), with the samplers
:class:`~mowl.representation.library.grounded_sampling.ElementSampler` and
:class:`~mowl.representation.library.grounded_sampling.RegionSampler`. The model's options
(``n_random_elements``, ``n_region_samples``, ``lambda_reg``, ...) are the options of these
parts; ``objective=``, ``element_sampler=`` and ``region_sampler=`` replace them. Mixed into
:class:`mowl.models.GroundedEL`."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch as th

from mowl.models.grounded.common import _CountInt
from mowl.representation.library.grounded_objective import GroundedObjective, Grounding
from mowl.representation.library.grounded_sampling import ElementSampler, RegionSampler


class GroundedConstraints:
    """The objective part of the combination (see the module docstring)."""

    #: Parts given to the constructor (``None``: built from the options at every step).
    _objective_part = None
    _element_sampler_part = None
    _region_sampler_part = None

    # -- the parts ----------------------------------------------------------------------------
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

    @property
    def objective(self) -> GroundedObjective:
        """The objective (the one given, or the grounded constraints with the options of the
        model)."""
        if self._objective_part is not None:
            return self._objective_part
        return GroundedObjective(
            self.constraint_set, lambda_reg=self.lambda_reg, region_margin=self.region_margin,
            region_lhs_grad=self.region_lhs_grad, dead_from_bot_only=self.dead_from_bot_only,
            neg_weight=self.neg_weight, neg_k=self.neg_k, eps_min=self.eps_min,
            eps_role=self.eps_role, eps_dead=self.eps_dead, agg=self.agg, agg_tau=self.agg_tau)

    @objective.setter
    def objective(self, part):
        self._objective_part = part

    @property
    def element_sampler(self) -> ElementSampler:
        """Which elements a step reads (the one given, or one with the model's options)."""
        if self._element_sampler_part is not None:
            return self._element_sampler_part
        return ElementSampler(self.batch_size, self.n_random_elements, self.n_inside_elements,
                              self.n_filler_samples)

    @element_sampler.setter
    def element_sampler(self, part):
        self._element_sampler_part = part

    @property
    def region_sampler(self) -> RegionSampler:
        """Which points of regions a step reads (the one given, or one with the model's
        options)."""
        if self._region_sampler_part is not None:
            return self._region_sampler_part
        # the published gradient of the first point of a row (see RegionSampler)
        return RegionSampler(self.n_region_samples, self.n_adv, doubled_gradient=True)

    @region_sampler.setter
    def region_sampler(self, part):
        self._region_sampler_part = part

    def grounding(self) -> Grounding:
        """The finite domain of the model (class elements, filler elements per role)."""
        self._ensure_role_tables()
        self._ensure_static_id_tensors()
        return Grounding(self._live_cls_ids, self._ord_cls_ids, self._dead_min_ids,
                         self._filler_mask_by_role, self.nb_classes, self.nb_inds,
                         self.top_id, self.bot_id)

    # -- one step -----------------------------------------------------------------------------
    def compute_batch_losses(
        self,
        generator: Optional[th.Generator] = None,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Dict[str, th.Tensor]:
        """One step of the grounded constraints (the objective over the module).

        Usable as the ontology step of an external training loop: ``generator`` defaults
        to ``self.generator`` (seeded with ``seed + 1337``, see :meth:`reset_generator`) and
        advances with every call; ``step`` and ``total_steps`` (both or neither) anneal the
        temperature first (:meth:`step_tau`); after ``optimizer.step()`` call
        :meth:`clamp_dead_logits`.

        :return: ``"fact"``, ``"axioms"``, ``"region"``, ``"minimality"``, ``"neg"``,
            ``"total"``, the detached weights ``"w_fact"``, ``"w_axioms"``, ``"w_region"``
            (the dual weights with ``constraint_mode="lagrangian"``, else 1) and
            ``"n_region_skipped_top"`` (rows with top as filler, whose region term is not
            read). ``"region"`` is 0 when ``n_region_samples == 0`` or ``lambda_reg == 0``;
            corrupted assertions are not filtered against the ABox unless the training
            regime filters them.
        """
        if (step is None) != (total_steps is None):
            raise ValueError("step and total_steps must both be provided or both be None")
        if step is not None:
            self.step_tau(step, total_steps)
        weights = None if self.constraint_mode == "fixed" else self.dual_weights
        out = self.objective.losses(
            self.module, self.training_datasets, self.grounding(),
            self.generator if generator is None else generator, self.element_sampler,
            self.region_sampler, weights=weights, negatives=self._regime_corrupted_opa)
        out["n_region_skipped_top"] = _CountInt(out["n_region_skipped_top"])
        for name in ("w_fact", "w_axioms", "w_region"):
            out[name] = out[name].detach()
        return out

    def _regime_corrupted_opa(self, opa_dataset: Any, gen: th.Generator) -> th.Tensor:
        """Corrupted role assertions ``(a, r, b)``: from the active training regime's
        ``negatives`` hook if it gives them, else from the element sampler, and in both cases
        passed through the regime's ``filter_negatives``."""
        def sampled():
            return self.element_sampler.corrupted(opa_dataset[:].to(self.device), self.nb_inds,
                                                  gen, self.neg_k)
        regime = getattr(self, "_active_regime", None)
        if regime is None:
            regime = self.training_regime
        if regime is None:
            return sampled()
        rows = regime.negatives("object_property_assertion", opa_dataset)
        if rows is NotImplemented or rows is None:
            rows = sampled()
        filtered = regime.filter_negatives("object_property_assertion", opa_dataset, rows)
        return rows if filtered is None else filtered

    # -- the samplers under the names of the earlier code -------------------------------------
    def _sample_rows(self, data, gen):
        return self.element_sampler.rows(data, gen, self.device)

    def _sample_corrupted_opa(self, opa_data, gen):
        rows = self.element_sampler.corrupted(opa_data.to(self.device), self.nb_inds, gen,
                                              self.neg_k)
        return self.nb_classes + rows[:, 0], self.nb_classes + rows[:, 2], rows[:, 1]

    def _batch_candidate_elements(self, batch_elems, gen):
        return self.element_sampler.batch(batch_elems, self.nb_elements, gen)

    def _inside_candidate_elements(self, lhs_classes, gen, pos_all):
        return self.element_sampler.inside(self.module.class_model, lhs_classes, pos_all, gen)

    def _sample_batch_y_fillers(self, rows, gen):
        self._ensure_role_tables()
        return self.element_sampler.fillers(rows, self._filler_mask_by_role, gen)

    def _grid_role_membership(self, pos_all, x_mat, y_mat, r_ids, need_crisp=False):
        return self.module.relation_model.grid_membership(pos_all[x_mat], pos_all[y_mat], r_ids,
                                                          need_crisp=need_crisp)

    def _depth_and_membership_pos_grad_only(self, points, class_ids):
        return self.module.class_model.held_membership(points, class_ids)
